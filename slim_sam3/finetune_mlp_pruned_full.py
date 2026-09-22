#!/usr/bin/env python3
"""
Feature + logit distillation fine-tuning for the FULL-32-block MLP-channel-pruned SAM3.

Teacher : original sam3.pt  (32 blocks, 4736 MLP hidden)
Student : sam3_full32_mlp_pruned_3072  (32 blocks, 3072 MLP hidden)

Strategy (v2 — fixes overdetection and mid-block divergence):
  Feature loss : MSE on DENSE anchor blocks (every --dense-every steps,
                 always including blocks 12-20 and the final block)
  Logit loss   : MSE(sigmoid(student pred_logits), sigmoid(teacher pred_logits))
                 scaled by --logit-weight  (default 0.5)
                 This directly constrains prediction confidence and eliminates
                 the overdetection / confidence inflation seen after v1.
  Only student MLP fc1 / fc2 weights are trained.

Usage (GPU node):
  python finetune_mlp_pruned_full.py \\
      --student-ckpt finetune_full32_mlp_3072/student_best.pt \\
      --student-meta finetune_full32_mlp_3072/mlp_pruned_meta.json \\
      --orig-ckpt    /ocean/projects/cis220039p/mdt2/hguo7/sam3/sam3.pt \\
C      --output-dir   finetune_full32_mlp_3072_v2 \\
      --steps 1500 --lr 1e-4 --logit-weight 0.5 --dense-every 4
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torchvision.transforms import v2

import sam3
from sam3 import build_sam3_image_model
from sam3.model.data_misc import FindStage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_slim_sam3 import build_slim_sam3_image_model

# ── prompt patterns ───────────────────────────────────────────────────────────
GLASS_FAMILY  = re.compile(r"glass|transparent|mirror|glazing", re.IGNORECASE)
WINDOW_FAMILY = re.compile(r"window|pane|skylight", re.IGNORECASE)
FALLBACK_GLASS_PROMPTS  = ["glass", "glass wall", "glass door", "glass window"]
FALLBACK_WINDOW_PROMPTS = ["window", "window pane", "skylight"]


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_prompts(sample_dir: Path) -> list[str]:
    """Return EITHER glass OR window prompts for one sample (not mixed).

    At test time we query glass and window separately; training should match
    that — so each training step uses one prompt family only.
    Priority: window prompts if present, else glass prompts, else fallback glass.
    """
    for p in sample_dir.glob("*_bboxes.json"):
        try:
            with open(p) as fh:
                bboxes = json.load(fh)
            glass_set  = set()
            window_set = set()
            for m in bboxes.get("masks", []):
                txt = m.get("prompt", "")
                if WINDOW_FAMILY.search(txt):
                    window_set.add(txt)
                elif GLASS_FAMILY.search(txt):
                    glass_set.add(txt)
            if window_set:
                return sorted(window_set)
            if glass_set:
                return sorted(glass_set)
        except Exception:
            pass
    return FALLBACK_GLASS_PROMPTS


def discover_samples(data_root: Path) -> list[tuple[Path, list[str]]]:
    """Return list of (image_path, prompts) for all valid samples."""
    samples = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir() or "train_val" not in scene_dir.name:
            continue
        if "debug" in scene_dir.name.lower():
            continue
        for s in sorted(scene_dir.iterdir()):
            if not s.is_dir():
                continue
            img_path = None
            for ext in ("*.jpg", "*.png"):
                for f in s.glob(ext):
                    if "overlay" not in f.name and "white" not in f.name:
                        img_path = f
                        break
                if img_path:
                    break
            if img_path:
                prompts = _load_prompts(s)
                samples.append((img_path, prompts))
    return samples


def discover_images(data_root: Path) -> list[Path]:
    """Legacy helper — returns image paths only (used as fallback)."""
    return [img for img, _ in discover_samples(data_root)]


def image_to_tensor(processor, image: Image.Image, device: torch.device) -> torch.Tensor:
    """Preprocess without going through @inference_mode set_image_batch."""
    t = v2.functional.to_image(image.convert("RGB"))
    t = processor.transform(t)
    return t.unsqueeze(0).to(device)


# ── hook-based feature capture ────────────────────────────────────────────────

class BlockFeatureCapture:
    def __init__(self, model):
        self._handles = []
        self.features: dict[int, torch.Tensor] = {}
        trunk = model.backbone.vision_backbone.trunk
        for i, block in enumerate(trunk.blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self.features[idx] = output
        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr_with_warmup(optimizer, step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student-ckpt", required=True,
                   help="Full-32-block MLP-pruned checkpoint (.pt)")
    p.add_argument("--student-meta", required=True,
                   help="mlp_pruned_meta.json from prune_mlp_channels.py")
    p.add_argument("--orig-ckpt", required=True,
                   help="Original sam3.pt (teacher)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="finetune_full32_mlp_3072")
    p.add_argument("--mlp-hidden-dim", type=int, default=3072)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--loss-mode", choices=["sparse", "dense", "all"], default="dense")
    p.add_argument("--intermediate-weight", type=float, default=0.5)
    p.add_argument("--dense-every", type=int, default=4,
                   help="Supervise every Nth block (dense mode). "
                        "Blocks 12-20 are always included. Default: 4")
    p.add_argument("--logit-weight", type=float, default=0.5,
                   help="Weight on the output logit MSE distillation term. "
                        "0 = feature distillation only. Default: 0.5")
    p.add_argument("--feature-weight", type=float, default=1.0,
                   help="Weight on the per-block feature-MSE distillation term. "
                        "Lower this (and raise --logit-weight) to shift from "
                        "feature-matching toward output-matching distillation. "
                        "Default: 1.0 (feature-dominant, original behavior).")
    p.add_argument("--unfreeze", type=str, default="mlp",
                   help="Comma-separated list of trunk component groups to make "
                        "trainable. Choices: mlp (.mlp.fc1/fc2), attn (.attn.*), "
                        "norm (block LayerNorms). Everything else (language "
                        "backbone, neck, decoder) stays frozen. Default: mlp "
                        "(MLP-only baseline). Example: --unfreeze mlp,attn,norm")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"device={device}  amp={args.amp}  steps={args.steps}  "
        f"lr={args.lr}  loss_mode={args.loss_mode}")

    # ── Load teacher (original sam3.pt, 32 blocks, hidden=4736) ──────────────
    log(f"\nLoading teacher from {args.orig_ckpt} ...")
    sam3_root = Path(sam3.__file__).resolve().parent.parent
    bpe_path  = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    teacher = build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(device),
        eval_mode=True,
        checkpoint_path=str(Path(args.orig_ckpt).expanduser().resolve()),
        load_from_HF=False,
        enable_segmentation=True,   # needed for pred_logits in logit distillation
        enable_inst_interactivity=False,
        compile=False,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_teacher_blocks = len(teacher.backbone.vision_backbone.trunk.blocks)
    log(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params, "
        f"{n_teacher_blocks} blocks (all frozen)")

    # ── Load student (32 blocks, hidden=3072) ─────────────────────────────────
    log(f"\nLoading student (hidden={args.mlp_hidden_dim}) from {args.student_ckpt} ...")
    student_proc = build_slim_sam3_image_model(
        slim_ckpt=args.student_ckpt,
        meta_json=args.student_meta,
        mlp_hidden_dim=args.mlp_hidden_dim,
        device=str(device),
        eval_mode=False,
    )
    student = student_proc.model

    # Freeze everything, then unfreeze the requested trunk-block component groups.
    # All groups are restricted to `trunk.blocks.` so the language backbone, neck,
    # and decoder (which also contain .attn./.norm/.mlp submodules) stay frozen.
    unfreeze_groups = {g.strip() for g in args.unfreeze.split(",") if g.strip()}
    valid_groups = {"mlp", "attn", "norm"}
    bad = unfreeze_groups - valid_groups
    if bad:
        raise ValueError(f"--unfreeze got unknown group(s) {sorted(bad)}; "
                         f"valid: {sorted(valid_groups)}")

    def group_of(name: str) -> str | None:
        if "trunk.blocks." not in name:
            return None
        if ".mlp.fc" in name:
            return "mlp"
        if ".attn." in name:
            return "attn"
        if ".norm1." in name or ".norm2." in name:
            return "norm"
        return None

    for p in student.parameters():
        p.requires_grad_(False)
    mlp_params = []                       # kept name; now = all trainable params
    group_counts = {g: 0 for g in valid_groups}
    for name, p in student.named_parameters():
        g = group_of(name)
        if g is not None and g in unfreeze_groups:
            p.requires_grad_(True)
            mlp_params.append(p)
            group_counts[g] += p.numel()

    log(f"  Unfreezing groups: {sorted(unfreeze_groups)}")
    for g in sorted(unfreeze_groups):
        log(f"    {g:5s}: {group_counts[g]/1e6:.2f}M trainable params")

    n_student_blocks = len(student.backbone.vision_backbone.trunk.blocks)
    log(f"  Student: {sum(p.numel() for p in student.parameters())/1e6:.1f}M params, "
        f"{n_student_blocks} blocks")
    log(f"  Total trainable params: {sum(p.numel() for p in mlp_params)/1e6:.2f}M")

    if n_teacher_blocks != n_student_blocks:
        raise RuntimeError(
            f"Block count mismatch: teacher={n_teacher_blocks}, student={n_student_blocks}. "
            f"Ensure --student-meta points to the full-32-block pruned meta."
        )

    # Graft language backbone into student (built with text=None)
    student.backbone.language_backbone = teacher.backbone.language_backbone
    log("  Language backbone grafted from teacher into student.")

    # ── Anchor blocks for feature distillation ────────────────────────────────
    n_blocks = n_student_blocks
    if args.loss_mode == "dense":
        # Every dense_every-th block + all of 12-20 (divergent region) + final
        anchor_blocks = sorted(set(
            list(range(0, n_blocks, args.dense_every))
            + list(range(12, min(21, n_blocks)))
            + [n_blocks - 1]
        ))
    elif args.loss_mode == "all":
        anchor_blocks = list(range(n_blocks))
    else:  # sparse (legacy)
        anchor_blocks = [n_blocks // 3, (2 * n_blocks) // 3, n_blocks - 1]
    log(f"\n  Feature anchor blocks ({len(anchor_blocks)}): {anchor_blocks}")
    log(f"  Logit distillation weight: {args.logit_weight}")

    # ── Discover samples ──────────────────────────────────────────────────────
    data_root = Path(args.data_root).expanduser().resolve()
    samples   = discover_samples(data_root)
    log(f"\nDiscovered {len(samples)} samples from {data_root}")
    if not samples:
        raise RuntimeError("No samples found — check --data-root")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(mlp_params, lr=args.lr, weight_decay=1e-2)
    scaler    = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # ── Feature hooks ─────────────────────────────────────────────────────────
    teacher_hooks = BlockFeatureCapture(teacher)
    student_hooks = BlockFeatureCapture(student)

    # Keep student in eval mode throughout — forward_grounding checks self.training
    # to decide whether to run the matcher (which requires find_target != None).
    # Gradient flow is controlled by requires_grad on mlp_params, not training mode.
    student.eval()

    # ── Training loop ─────────────────────────────────────────────────────────
    log(f"\n{'Step':>6}  {'LR':>8}  {'Loss':>10}  {'Loss/block':>12}")
    best_loss  = float("inf")
    loss_accum = 0.0
    log_every  = 10

    for step in range(args.steps):
        lr = cosine_lr_with_warmup(optimizer, step, args.warmup_steps, args.steps, args.lr)

        img_path, prompts = rng.choice(samples)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log(f"  [WARN] skip {img_path}: {e}")
            continue

        # Use student preprocessor (same transform as teacher — both use SAM3 defaults)
        img_tensor = image_to_tensor(student_proc, image, device)

        amp_ctx = autocast("cuda", dtype=torch.bfloat16,
                           enabled=(args.amp and device.type == "cuda"))

        # ── Teacher forward (no grad) ─────────────────────────────────────────
        teacher_hooks.clear()
        with torch.no_grad():
            teacher.backbone.forward_image(img_tensor)
        teacher_feats = {k: v.detach() for k, v in teacher_hooks.features.items()}

        # ── Teacher logit forward (no grad) ───────────────────────────────────
        teacher_logits = None
        if args.logit_weight > 0:
            with torch.no_grad(), amp_ctx:
                t_bb = dict(teacher.backbone.forward_image(img_tensor))
                t_text = teacher.backbone.forward_text(prompts, device=str(device))
                t_bb.update(t_text)
                find_input = FindStage(
                    img_ids=torch.zeros(len(prompts), dtype=torch.long, device=device),
                    text_ids=torch.arange(len(prompts), dtype=torch.long, device=device),
                    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                    input_points=None, input_points_mask=None,
                )
                geo = teacher._get_dummy_prompt(num_prompts=len(prompts))
                t_out = teacher.forward_grounding(
                    backbone_out=t_bb, find_input=find_input,
                    find_target=None, geometric_prompt=geo,
                )
            t_probs = t_out["pred_logits"].sigmoid().squeeze(-1).float()  # [P, Q]
            t_pres  = t_out.get("presence_logit_dec", None)
            if t_pres is not None:
                teacher_logits = (t_probs * t_pres.sigmoid().float()).detach()
            else:
                teacher_logits = t_probs.detach()

        # ── Student forward (grads through MLP) ───────────────────────────────
        student_hooks.clear()
        with amp_ctx:
            student.backbone.forward_image(img_tensor)
            student_feats = student_hooks.features

            # Feature distillation loss over anchor blocks
            feat_loss = sum(
                F.mse_loss(student_feats[i].float(), teacher_feats[i].float())
                for i in anchor_blocks
                if i in student_feats and i in teacher_feats
            )

            # Logit distillation loss
            logit_loss = torch.tensor(0.0, device=device)
            if args.logit_weight > 0 and teacher_logits is not None:
                s_bb = dict(student.backbone.forward_image(img_tensor))
                s_text = student.backbone.forward_text(prompts, device=str(device))
                s_bb.update(s_text)
                find_input_s = FindStage(
                    img_ids=torch.zeros(len(prompts), dtype=torch.long, device=device),
                    text_ids=torch.arange(len(prompts), dtype=torch.long, device=device),
                    input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                    input_points=None, input_points_mask=None,
                )
                geo_s = student._get_dummy_prompt(num_prompts=len(prompts))
                s_out = student.forward_grounding(
                    backbone_out=s_bb, find_input=find_input_s,
                    find_target=None, geometric_prompt=geo_s,
                )
                s_probs = s_out["pred_logits"].sigmoid().squeeze(-1).float()  # [P, Q]
                s_pres  = s_out.get("presence_logit_dec", None)
                if s_pres is not None:
                    s_combined = s_probs * s_pres.sigmoid().float()
                else:
                    s_combined = s_probs
                logit_loss = F.mse_loss(s_combined, teacher_logits)

            loss = args.feature_weight * feat_loss + args.logit_weight * logit_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(mlp_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val    = loss.item()
        loss_accum += loss_val

        if (step + 1) % log_every == 0:
            avg       = loss_accum / log_every
            per_block = avg / n_student_blocks
            logit_str = f"  logit={logit_loss.item():.6f}" if args.logit_weight > 0 else ""
            log(f"{step+1:>6}  {lr:>8.2e}  {avg:>10.6f}  {per_block:>12.6f}{logit_str}")
            loss_accum = 0.0

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            state     = {"detector." + k: v for k, v in student.state_dict().items()}
            ckpt_path = output_dir / f"student_step{step+1:04d}.pt"
            torch.save(state, ckpt_path)
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(state, output_dir / "student_best.pt")
                log(f"  → step {step+1}: new best loss={best_loss:.6f}")
            else:
                log(f"  → step {step+1}: checkpoint saved")

    # ── Final save ────────────────────────────────────────────────────────────
    final_state = {"detector." + k: v for k, v in student.state_dict().items()}
    torch.save(final_state, output_dir / "student_final.pt")
    shutil.copy(args.student_meta, output_dir / "mlp_pruned_meta.json")

    log(f"\nDone.")
    log(f"  Best  : {output_dir / 'student_best.pt'}  (loss={best_loss:.6f})")
    log(f"  Final : {output_dir / 'student_final.pt'}")
    log(f"  Meta  : {output_dir / 'mlp_pruned_meta.json'}")

    teacher_hooks.remove()
    student_hooks.remove()


if __name__ == "__main__":
    main()





