#!/usr/bin/env python3
"""
Benchmark original SAM3 vs any MLP/block-pruned SAM3 checkpoint.

The pruned variant is loaded via build_slim_sam3_image_model (supports both
block-pruned and MLP-channel-pruned checkpoints). The language backbone is
grafted from the original sam3.pt so both variants run identical text-prompted
inference.

Measures per variant:
  - VRAM allocated after model load
  - Peak VRAM during inference (activations included)
  - Activation overhead = peak - weights
  - Mean / median / std / min / max latency over N timed runs

Usage (on GPU node):
  python bench_slim_sam3.py \\
      --ckpt-path    /path/to/sam3.pt \\
      --pruned-ckpt  finetune_full32_mlp_1408/student_best.pt \\
      --pruned-meta  finetune_full32_mlp_1408/mlp_pruned_meta.json \\
      --mlp-hidden-dim 1408 \\
      --prompt "window" \\
      --n-warmup 5 --n-runs 50
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast

# ── import SAM3 modules ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so load_slim_sam3 is importable
import sam3
from sam3 import build_sam3_image_model
from sam3.model.data_misc import FindStage, interpolate
from sam3.model.sam3_image_processor import Sam3Processor
from load_slim_sam3 import build_slim_sam3_image_model

# ── defaults ──────────────────────────────────────────────────────────────────
KEEP_24 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31]


@contextmanager
def _no_bypass():
    """No-op context (original full model, nothing bypassed)."""
    yield


# ── single forward pass (shared by both variants) ─────────────────────────────
def run_one(model, processor, image, prompt, device):
    """Full encode + grounding + top-mask decode."""
    with torch.inference_mode():
        state = processor.set_image_batch([image])
        H = int(state["original_heights"][0])
        W = int(state["original_widths"][0])
        backbone_out = dict(state["backbone_out"])
        text_out = model.backbone.forward_text([prompt], device=str(device))
        backbone_out.update(text_out)

        find_input = FindStage(
            img_ids=torch.zeros(1, dtype=torch.long, device=device),
            text_ids=torch.zeros(1, dtype=torch.long, device=device),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        geo = model._get_dummy_prompt(num_prompts=1)

        if device.type == "cuda":
            with autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geo,
                )
        else:
            out = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geo,
            )

        # Decode top mask (counted in timing)
        probs = out["pred_logits"].sigmoid().squeeze(-1)
        presence = out.get("presence_logit_dec", None)
        if presence is not None:
            probs = probs * presence.sigmoid()
        qidx = int(probs[0].argmax().item())
        if "pred_masks" in out:
            ms = out["pred_masks"][0, qidx].view(
                1, 1, out["pred_masks"].shape[-2], out["pred_masks"].shape[-1]
            )
            _ = (interpolate(ms, (H, W), mode="bilinear", align_corners=False)
                 .sigmoid()[0, 0] > 0.5).cpu().numpy()


def _mb(b: int) -> str:
    return f"{b / 1024**2:.1f} MB"


def _bench(label, model, processor, image, prompt, device, n_warmup, n_runs):
    """Warm-up + timed benchmark. Returns stats dict."""
    is_cuda = device.type == "cuda"
    print(f"\n  --- {label} ---")

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        mem_weights = torch.cuda.memory_allocated(device)
        mem_reserved_weights = torch.cuda.memory_reserved(device)

    for i in range(n_warmup):
        run_one(model, processor, image, prompt, device)
        if is_cuda:
            torch.cuda.synchronize(device)
        print(f"    warm-up {i+1}/{n_warmup}")

    if is_cuda:
        torch.cuda.synchronize(device)
        mem_peak = torch.cuda.max_memory_allocated(device)
        mem_reserved_peak = torch.cuda.memory_reserved(device)

    latencies = []
    for i in range(n_runs):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        run_one(model, processor, image, prompt, device)
        if is_cuda:
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)
        print(f"    run {i+1:2d}/{n_runs}: {latencies[-1]:.1f} ms")

    lat = np.array(latencies)
    result = {
        "label": label,
        "mean_ms":   float(lat.mean()),
        "median_ms": float(np.median(lat)),
        "std_ms":    float(lat.std()),
        "min_ms":    float(lat.min()),
        "max_ms":    float(lat.max()),
        "fps":       float(1000 / lat.mean()),
    }
    if is_cuda:
        result.update({
            "vram_weights_mb":            mem_weights / 1024**2,
            "vram_reserved_weights_mb":   mem_reserved_weights / 1024**2,
            "vram_peak_mb":               mem_peak / 1024**2,
            "vram_reserved_peak_mb":      mem_reserved_peak / 1024**2,
            "vram_activation_overhead_mb": (mem_peak - mem_weights) / 1024**2,
        })
    return result


def load_original(ckpt_path, device, bpe_path):
    print(f"  Loading original SAM3 (32 blocks) from {ckpt_path} ...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(device),
        eval_mode=True,
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    ).eval().to(device)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    depth = len(list(model.backbone.vision_backbone.trunk.blocks))
    print(f"  Loaded — trunk depth: {depth} blocks")
    return model, processor


def load_pruned(pruned_ckpt_path, meta_json_path, mlp_hidden_dim, lang_backbone, device):
    """
    Load any MLP/block-pruned SAM3 checkpoint and optionally graft the
    language backbone from the original model.
    """
    print(f"  Loading pruned SAM3 from {pruned_ckpt_path} ...")
    proc = build_slim_sam3_image_model(
        slim_ckpt=str(pruned_ckpt_path),
        meta_json=str(meta_json_path) if meta_json_path and Path(meta_json_path).exists() else None,
        mlp_hidden_dim=mlp_hidden_dim,
        device="cpu",   # load to CPU first so we can graft lang backbone
        eval_mode=True,
    )
    model = proc.model

    if lang_backbone is not None:
        print("  Grafting language backbone from original model ...")
        model.backbone.language_backbone = lang_backbone

    model = model.eval().to(device)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    depth = len(model.backbone.vision_backbone.trunk.blocks)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trunk_params = sum(p.numel() for p in model.backbone.vision_backbone.trunk.parameters()) / 1e6
    print(f"  Loaded — depth={depth} blocks, trunk={trunk_params:.1f}M, total={total_params:.1f}M")
    return model, processor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path",       required=True,
                   help="Path to original sam3.pt (baseline)")
    p.add_argument("--pruned-ckpt",     required=True,
                   help="Path to pruned checkpoint (student_best.pt or any slim ckpt)")
    p.add_argument("--pruned-meta",     required=True,
                   help="Path to mlp_pruned_meta.json / slim_encoder_meta.json")
    p.add_argument("--mlp-hidden-dim",  type=int, default=None,
                   help="MLP hidden dim of pruned model (e.g. 3072); None = default 4736")
    p.add_argument("--no-text-encoder", action="store_true",
                   help="Skip grafting language backbone (if pruned ckpt has its own)")
    p.add_argument("--image-path",      default=None,
                   help="Test image. Omit to use a synthetic 640x480 RGB image.")
    p.add_argument("--prompt",    default="window")
    p.add_argument("--n-runs",    type=int, default=20)
    p.add_argument("--n-warmup",  type=int, default=3)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available on this node. "
            "Run bench_slim_sam3.py on a GPU node (e.g. via srun/interact with --gres=gpu:...)."
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    # Read pruned model meta to report block count
    pruned_meta_path = Path(args.pruned_meta).expanduser().resolve()
    with open(pruned_meta_path) as f:
        pruned_meta = json.load(f)
    keep_blocks   = pruned_meta.get("keep_blocks", list(range(32)))
    n_keep        = len(keep_blocks)
    mlp_hidden    = args.mlp_hidden_dim or pruned_meta.get("pruned_hidden_dim", 4736)
    orig_hidden   = pruned_meta.get("original_hidden_dim", 4736)

    sam3_root = Path(sam3.__file__).resolve().parent.parent
    bpe_path  = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    print(f"Device        : {device}")
    print(f"Prompt        : '{args.prompt}'")
    print(f"Warm-up/runs  : {args.n_warmup} / {args.n_runs}")
    print(f"Pruned blocks : {n_keep}/32   MLP hidden: {mlp_hidden} (orig {orig_hidden})")

    # ── Prepare image ─────────────────────────────────────────────────────────
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        print(f"\nImage    : {args.image_path}  ({image.width}x{image.height})")
    else:
        rng = np.random.default_rng(42)
        arr = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        print("\nImage    : synthetic 640x480 RGB")

    # ── [1] Load & benchmark original ─────────────────────────────────────────
    print("\n[1/2] Loading original SAM3 ...")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device)

    orig_model, orig_proc = load_original(Path(args.ckpt_path).expanduser().resolve(), device, bpe_path)

    if is_cuda:
        torch.cuda.synchronize(device)
        mem_orig_loaded = torch.cuda.memory_allocated(device)
        print(f"  VRAM after load  : {_mb(mem_orig_loaded)}  (delta: {_mb(mem_orig_loaded - mem_before)})")

    orig_n_params = sum(p.numel() for p in orig_model.parameters()) / 1e6
    orig_trunk_params = sum(p.numel() for p in orig_model.backbone.vision_backbone.trunk.parameters()) / 1e6
    print(f"  Total params: {orig_n_params:.1f}M  |  Trunk: {orig_trunk_params:.1f}M")

    print(f"\n  Benchmarking original ...")
    orig = _bench("Original SAM3 (32 blocks)", orig_model, orig_proc, image, args.prompt, device, args.n_warmup, args.n_runs)

    # Extract lang backbone before freeing original model
    lang_backbone = None if args.no_text_encoder else orig_model.backbone.language_backbone

    del orig_proc
    del orig_model
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    # ── [2] Load & benchmark pruned model ────────────────────────────────────
    print(f"\n[2/2] Loading pruned SAM3 ({n_keep} blocks, MLP hidden={mlp_hidden}) ...")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        mem_before_pruned = torch.cuda.memory_allocated(device)

    pruned_model, pruned_proc = load_pruned(
        pruned_ckpt_path=Path(args.pruned_ckpt).expanduser().resolve(),
        meta_json_path=pruned_meta_path,
        mlp_hidden_dim=args.mlp_hidden_dim,
        lang_backbone=lang_backbone,
        device=device,
    )

    if is_cuda:
        torch.cuda.synchronize(device)
        mem_pruned_loaded = torch.cuda.memory_allocated(device)
        print(f"  VRAM after load  : {_mb(mem_pruned_loaded)}  (delta: {_mb(mem_pruned_loaded - mem_before_pruned)})")

    pruned_label = f"Pruned SAM3 ({n_keep}blk, MLP={mlp_hidden})"
    pruned_result = _bench(pruned_label, pruned_model, pruned_proc, image, args.prompt, device, args.n_warmup, args.n_runs)

    del pruned_proc, pruned_model
    if is_cuda:
        torch.cuda.empty_cache()

    # ── Side-by-side comparison ───────────────────────────────────────────────
    def row(name, o_val, p_val, fmt=".1f", unit=""):
        o_s = f"{o_val:{fmt}}{unit}"
        p_s = f"{p_val:{fmt}}{unit}"
        d_s = f"{(p_val - o_val) / o_val * 100:+.1f}%" if o_val != 0 else "—"
        print(f"  {name:<30} {o_s:>11}  {p_s:>11}  {d_s:>9}")

    col_pruned = f"Pruned({n_keep}blk,{mlp_hidden})"
    print(f"\n{'='*68}")
    print(f"  BENCHMARK COMPARISON  (N={args.n_runs} runs, prompt='{args.prompt}')")
    print(f"{'='*68}")
    print(f"  {'Metric':<30} {'Original':>11}  {col_pruned:>14}  {'Delta':>9}")
    print(f"  {'-'*66}")
    print(f"  Latency:")
    row("  Mean",       orig["mean_ms"],   pruned_result["mean_ms"],   fmt=".1f", unit=" ms")
    row("  Median",     orig["median_ms"], pruned_result["median_ms"], fmt=".1f", unit=" ms")
    row("  Std",        orig["std_ms"],    pruned_result["std_ms"],    fmt=".1f", unit=" ms")
    row("  Min",        orig["min_ms"],    pruned_result["min_ms"],    fmt=".1f", unit=" ms")
    row("  Max",        orig["max_ms"],    pruned_result["max_ms"],    fmt=".1f", unit=" ms")
    row("  Throughput", orig["fps"],       pruned_result["fps"],       fmt=".2f", unit=" fps")
    if is_cuda:
        print(f"  VRAM:")
        row("  Weights (allocated)",     orig["vram_weights_mb"],             pruned_result["vram_weights_mb"],             fmt=".1f", unit=" MB")
        row("  Peak (allocated)",        orig["vram_peak_mb"],                pruned_result["vram_peak_mb"],                fmt=".1f", unit=" MB")
        row("  Peak (reserved)",         orig["vram_reserved_peak_mb"],       pruned_result["vram_reserved_peak_mb"],       fmt=".1f", unit=" MB")
        row("  Activation overhead",     orig["vram_activation_overhead_mb"], pruned_result["vram_activation_overhead_mb"], fmt=".1f", unit=" MB")
    print(f"{'='*68}")
    speedup = orig["mean_ms"] / pruned_result["mean_ms"]
    print(f"  Speedup   : {speedup:.3f}x  ({'faster' if speedup > 1 else 'slower'})")
    if is_cuda:
        vram_delta = orig["vram_peak_mb"] - pruned_result["vram_peak_mb"]
        n_dropped  = 32 - n_keep
        print(f"  VRAM saved: {vram_delta:+.1f} MB  (peak allocated, {n_dropped} fewer blocks, MLP {orig_hidden}→{mlp_hidden})")
    print(f"{'='*68}")
    print(f"  Note: both variants use the same text encoder (language backbone).")
    print(f"        Diff reflects ViT trunk blocks + MLP channel count.")


if __name__ == "__main__":
    main()



# python bench_slim_sam3.py \
#   --ckpt-path /ocean/projects/cis220039p/mdt2/hguo7/sam3/sam3.pt \
#   --pruned-ckpt /ocean/projects/cis220039p/mdt2/hguo7/sam3/pruning_algorithms/sam3_full32_mlp_pruned_2816_adaptive_merged/sam3_mlp_pruned_adaptive_avg2816of4736.pt \
#   --pruned-meta /ocean/projects/cis220039p/mdt2/hguo7/sam3/pruning_algorithms/sam3_full32_mlp_pruned_2816_adaptive_merged/mlp_pruned_meta.json \
#   --mlp-hidden-dim 2816 \
#   --prompt "window" \
#   --n-warmup 5 \
#   --n-runs 50 \
#   --device cuda