#!/usr/bin/env python3
"""
glass_frame_ring.py

For each RGB image in a folder:
  1. Run SAM3 to detect glass/window masks + NMS (same logic as test_sam_efficiency.py)
  2. Run DA2 (vits) to get metric depth
  3. For each SAM detection, compute an adaptive ring radius =
       max(--ring-min, int(0.10 * sqrt(mask_pixel_area)))
     Build per-mask ring = dilate(mask, radius) - mask, then union all rings.
  4. Detect depth jumps > --jump-thresh metres in the DA2 depth map;
     dilate those jump pixels by --jump-dilation (default 20 px).
  5. Remove dilated-depth-jump pixels from the frame ring
     (occlusion handling: wall/object edges in front of the window are not frames).
  6. Save per-image outputs:
       <stem>_ring.png      — RGB overlay: green=glass, red=frame ring
       <stem>_depthjump.png — DA2 depth map with jump pixels highlighted in yellow

Usage (full SAM3 model):
  conda run -n sam3 python glass_frame_ring.py \\
      --batch-dir <HOME>/GlassGuard/facade_glass \\
      --ckpt-path <HOME>/GlassGuard/sam3/sam3.pt \\
      --cached-text-features <HOME>/GlassGuard/sam3/prompt_features/window_glass.pt \\
      --prompt window glass --bf16

Usage (student model):
  conda run -n sam3 python glass_frame_ring.py \\
      --batch-dir <HOME>/GlassGuard/facade_glass \\
      --student \\
      --ckpt-path <HOME>/GlassGuard/sam3/checkpoints/slim_3072/student_best.pt \\
      --meta-json <HOME>/GlassGuard/sam3/checkpoints/slim_3072/mlp_pruned_meta.json \\
      --mlp-hidden-dim 3072 \\
      --cached-text-features <HOME>/GlassGuard/sam3/prompt_features/window_glass.pt \\
      --prompt window glass --bf16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast

# ── SAM3 imports ──────────────────────────────────────────────────────────────
_SAM3_DIR = Path(__file__).parent / "sam3"
sys.path.insert(0, str(_SAM3_DIR))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage, interpolate

# ── DA2 imports ───────────────────────────────────────────────────────────────
_DA2_ROOT_DEFAULT = Path(__file__).parent.parent / "Depth-Anything-V2"

_DA2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SAM3 glass mask + DA2 depth-jump frame-ring overlay"
    )

    # Input / output
    p.add_argument("--batch-dir",    required=True,
                   help="Folder containing rgb_*.jpg / rgb_*.png images.")
    p.add_argument("--out-dir",      default=None,
                   help="Where to save overlay PNGs (default: <batch-dir>/frame_ring_out/).")
    p.add_argument("--batch-max-frames", type=int, default=0,
                   help="Stop after N frames (0 = all).")

    # SAM3 model
    p.add_argument("--ckpt-path",    default="sam3/sam3.pt",
                   help="Path to SAM3 checkpoint (.pt).")
    p.add_argument("--student",      action="store_true",
                   help="Use slim/student SAM3 model.")
    p.add_argument("--meta-json",    default=None)
    p.add_argument("--mlp-hidden-dim", type=int, default=None)

    # SAM3 inference
    p.add_argument("--cached-text-features",
                   default="sam3/prompt_features/window_glass.pt",
                   help="Pre-computed text feature cache (.pt).")
    p.add_argument("--prompt",       nargs="+", default=["window", "glass"])
    p.add_argument("--all-prompts",  action="store_true",
                   help="Use all prompts stored in the cache file.")
    p.add_argument("--conf-th",      type=float, default=0.3)
    p.add_argument("--nms-iou",      type=float, default=0.5,
                   help="IoU threshold for cross-prompt NMS (0 = disabled).")

    # DA2
    p.add_argument("--da2-root",     default=str(_DA2_ROOT_DEFAULT),
                   help="Root of the Depth-Anything-V2 repository.")
    p.add_argument("--da2-encoder",  default="vits", choices=list(_DA2_MODEL_CONFIGS))
    p.add_argument("--da2-max-depth", type=float, default=10.0)
    p.add_argument("--da2-input-size", type=int, default=518)

    # Ring / depth-jump params
    p.add_argument("--ring-dilation", type=int, default=20,
                   help="Fixed dilation radius (pixels) for the frame ring (default 20).")
    p.add_argument("--jump-thresh",  type=float, default=0.25,
                   help="Sobel |\u2207z| threshold (metres) to classify a pixel as a depth jump. "
                        "Matches depth_jump_panel.py default of 0.25.")
    p.add_argument("--jump-dilation", type=int, default=20,
                   help="Dilation radius (pixels) applied to depth-jump pixels (default 20).")

    # Precision / device
    p.add_argument("--bf16",  action="store_true")
    p.add_argument("--int8",  action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with test_sam_efficiency.py
# ─────────────────────────────────────────────────────────────────────────────

def cast_nested_to(obj, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: cast_nested_to(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cast_nested_to(v, dtype) for v in obj)
    return obj


def cast_nested_to_bf16(obj):
    return cast_nested_to(obj, torch.bfloat16)


def convert_model_floating_to_bf16(model):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param is not None and param.is_floating_point():
                param.data = param.data.to(torch.bfloat16)
        for name, buf in module.named_buffers(recurse=False):
            if buf is not None and buf.is_floating_point():
                buf.data = buf.data.to(torch.bfloat16)
    return model


def add_linear_input_cast_hooks(model):
    hooks = []

    def _make_hook(module):
        def _hook(mod, args):
            x = args[0]
            if x.is_floating_point() and x.dtype != mod.weight.dtype:
                return (x.to(mod.weight.dtype),) + args[1:]
            return args
        return _hook

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_pre_hook(_make_hook(module)))
    return hooks


# ─────────────────────────────────────────────────────────────────────────────
# Morphology helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_disk_kernel(radius: int) -> np.ndarray:
    d = 2 * radius + 1
    k = np.zeros((d, d), np.uint8)
    cv2.circle(k, (radius, radius), radius, 1, -1)
    return k


def dilate_mask(mask_bool: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)


def _resize_bool(m: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(m.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Depth-jump detection
# ─────────────────────────────────────────────────────────────────────────────

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Jet-colorized depth map as a BGR uint8 image."""
    valid = depth > 0.01
    out   = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    d_min = float(depth[valid].min())
    d_max = float(depth[valid].max())
    norm  = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - d_min) / max(d_max - d_min, 1e-6)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def jump_mag_to_vis(jump_mag: np.ndarray, thresh: float) -> np.ndarray:
    """
    Hot-colormap visualisation of Sobel depth-jump magnitude,
    identical to depth_jump_panel.py's jump_to_vis().
    Sub-threshold pixels are black; above-threshold pixels use the hot
    colormap scaled to the 99th percentile of the jump magnitudes.
    """
    vis = jump_mag.copy()
    vis[vis < thresh] = 0.0
    above = vis > 0
    if above.any():
        p99 = float(np.percentile(vis[above], 99))
        vis = np.clip(vis / max(p99, 1e-6), 0.0, 1.0)
    # hot colormap: black → red → yellow → white
    bgr = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    bgr[~above] = 0
    return bgr


def depth_jump_mask(depth: np.ndarray, thresh: float):
    """
    Detect depth discontinuities using Sobel gradient magnitude,
    identical to depth_jump_panel.py's da2_jump_map().

    Returns
    -------
    jump_bool : (H, W) bool  — pixels where |∇z| > thresh
    jump_mag  : (H, W) float32 — raw Sobel magnitude (for visualisation)
    """
    d  = depth.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)          # (H, W) float32
    return mag > thresh, mag


# ─────────────────────────────────────────────────────────────────────────────
# Timing / VRAM helper
# ─────────────────────────────────────────────────────────────────────────────

def _vram_gb(device):
    if device.type != "cuda":
        return 0.0, 0.0
    return (torch.cuda.memory_allocated(device) / 1024**3,
            torch.cuda.memory_reserved(device)  / 1024**3)


class StepTimer:
    """Lightweight step-level profiler. Prints elapsed + VRAM delta per step."""
    def __init__(self, device):
        self.device  = device
        self.t_start = time.time()
        self._t      = self.t_start

    def tick(self, label: str):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.time()
        alloc, resv = _vram_gb(self.device)
        print(f"    [{label:30s}]  {now - self._t:6.3f}s   "
              f"vram alloc={alloc:.2f}GB  resv={resv:.2f}GB")
        self._t = now


# ─────────────────────────────────────────────────────────────────────────────
# NMS (box IoU + mask containment, identical logic to test_sam_efficiency.py)
# ─────────────────────────────────────────────────────────────────────────────

MAX_UNIQUE_MASKS = 30

def splat_deduplicate(masks_np, boxes, scores):
    """
    Splat-based deduplication: paint masks largest-first onto a (H,W) int16 label
    canvas.  Each mask gets a label id; as larger masks are splatted they overwrite
    pixels of smaller masks already on the canvas.  Any label whose remaining pixel
    count drops to zero is considered fully absorbed — it is removed.

    Cap: once MAX_UNIQUE_MASKS distinct labels are active on the canvas, a new mask
    that has NO overlap with any existing painted pixel is rejected (it would only
    add a new unique region).  A new mask that overlaps existing pixels is always
    accepted (it merges into / replaces existing regions, not adding net new ones).

    Returns:
        kept_idx  – list of original mask indices to keep (order: largest-first)
        label_map – (H,W) int16 array, value = index into kept_idx list, -1 = empty
    """
    if not masks_np:
        return [], None

    areas = np.array([int(m.sum()) for m in masks_np], dtype=np.int64)
    order = np.argsort(areas)[::-1].tolist()   # largest first

    H, W = masks_np[0].shape
    label_map       = np.full((H, W), -1, dtype=np.int16)
    label_remaining = {}   # label_id -> remaining pixel count
    kept_orig       = []   # original mask index per label_id
    removed         = set()  # label_ids fully absorbed by a later/larger mask

    for i in order:
        px = int(areas[i])
        if px == 0:
            continue

        mask = masks_np[i]

        # How many pixels of this mask land on already-painted canvas?
        overwritten = label_map[mask]   # (px,) int16
        valid = overwritten[overwritten >= 0]

        # Count currently-active unique labels (not yet removed)
        active_count = len(kept_orig) - len(removed)

        # If cap reached and this mask adds no new pixels, skip it
        if active_count >= MAX_UNIQUE_MASKS and valid.size == 0:
            continue

        if valid.size > 0:
            lids, cnts = np.unique(valid, return_counts=True)
            for lid, cnt in zip(lids.tolist(), cnts.tolist()):
                label_remaining[lid] -= cnt
                if label_remaining[lid] <= 0:
                    removed.add(lid)

        new_label = len(kept_orig)
        label_map[mask] = new_label
        label_remaining[new_label] = px
        kept_orig.append(i)

    kept_idx = [kept_orig[lab] for lab in range(len(kept_orig)) if lab not in removed]
    return kept_idx, label_map


# ─────────────────────────────────────────────────────────────────────────────
# SAM3 model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sam3(args, device_str: str):
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path  = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    if args.student:
        slim_mod = __import__("load_slim_sam3", fromlist=["build_slim_sam3_image_model"])
        model_or_proc = slim_mod.build_slim_sam3_image_model(
            slim_ckpt=args.ckpt_path,
            meta_json=args.meta_json,
            mlp_hidden_dim=args.mlp_hidden_dim,
            device=device_str,
            eval_mode=True,
        )
        model = model_or_proc.model if hasattr(model_or_proc, "model") else model_or_proc
    else:
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device_str,
            eval_mode=True,
            checkpoint_path=args.ckpt_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# DA2 model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_da2(args, device):
    da2_root = Path(args.da2_root)
    sys.path.insert(0, str(da2_root / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg  = {**_DA2_MODEL_CONFIGS[args.da2_encoder], "max_depth": args.da2_max_depth}
    ckpt = str(da2_root / "metric_depth" / "checkpoints" /
               f"depth_anything_v2_metric_hypersim_{args.da2_encoder}.pth")
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame SAM3 inference → list of (mask_np H×W bool, score, box)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def sam3_encode_image(processor, pil_img, device, args):
    """Run ONLY the image backbone (the expensive forward) and return the encoded state. Reuse this ONE
    state to decode MULTIPLE prompt sets (e.g. glass + floor) without paying the backbone twice."""
    # inference_mode() disables autograd so the backbone forward doesn't retain activations for backward.
    if args.bf16 and device.type == "cuda":
        with torch.inference_mode(), autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image_batch([pil_img])
    elif args.int8 and device.type == "cuda":
        with torch.inference_mode(), autocast("cuda", dtype=torch.float16):
            state = processor.set_image_batch([pil_img])
    else:
        with torch.inference_mode():
            state = processor.set_image_batch([pil_img])
    # VRAM: the backbone output is left in fp32 by set_image_batch, so each per-prompt decode used to
    # make a FULL fp16/bf16 COPY of it (backbone held twice + copied once per prompt = the 2-3GB spike).
    # Cast it ONCE here and store it back; the fp32 originals free immediately, and decode's cast_nested_to
    # then becomes a no-op for these tensors (Tensor.to(dtype) returns self when the dtype already matches),
    # so no duplicate is ever built -- glass/floor/doorway prompts all reuse this single cast backbone.
    if device.type == "cuda" and "backbone_out" in state:
        if args.bf16:
            state["backbone_out"] = cast_nested_to_bf16(state["backbone_out"])
        elif args.int8:
            state["backbone_out"] = cast_nested_to(state["backbone_out"], torch.float16)
    return state


def sam3_decode_prompts(model, state, per_prompt_feats, prompts, device, args):
    """Decode one or more prompt sets from an ALREADY-encoded image `state` (from sam3_encode_image) --
    the cheap grounding-head part. Returns (all_masks, all_scores, all_boxes, H_enc, W_enc). Does NOT free
    `state`, so the caller can reuse it for another prompt set."""
    H_enc = int(state["original_heights"][0])
    W_enc = int(state["original_widths"][0])
    backbone_base = dict(state["backbone_out"])

    all_masks  = []
    all_scores = []
    all_boxes  = []

    for text_feats in per_prompt_feats:
        bb = dict(backbone_base)
        bb.update(text_feats)

        find_input = FindStage(
            img_ids=torch.zeros(1, dtype=torch.long, device=device),
            text_ids=torch.zeros(1, dtype=torch.long, device=device),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        geo_prompt = model._get_dummy_prompt(num_prompts=1)

        if args.bf16 and device.type == "cuda":
            bb_in   = cast_nested_to_bf16(bb)
            amp_ctx = autocast("cuda", dtype=torch.bfloat16)
        elif args.int8 and device.type == "cuda":
            bb_in   = cast_nested_to(bb, torch.float16)
            amp_ctx = autocast("cuda", dtype=torch.float16)
        else:
            bb_in   = bb
            amp_ctx = torch.no_grad()

        # inference_mode() is ESSENTIAL: autocast alone does NOT disable autograd, so without it the
        # bf16/int8 grounding forward builds a full backward graph and retains ~3x the activations
        # (measured: 2.9GB peak vs 0.95GB with inference_mode). The fp32 branch used no_grad already;
        # this makes ALL precisions grad-free.
        with torch.inference_mode(), amp_ctx:
            outputs = model.forward_grounding(
                backbone_out=bb_in,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geo_prompt,
            )

        pred_logits  = outputs["pred_logits"]           # (1, Q, 1)
        presence_log = outputs.get("presence_logit_dec", None)
        probs = pred_logits.sigmoid().squeeze(-1)        # (1, Q)
        if presence_log is not None:
            probs = probs * presence_log.sigmoid()

        pred_boxes = outputs["pred_boxes"]               # (1, Q, 4)
        keep_mask  = probs > args.conf_th
        kept_p, kept_q = torch.where(keep_mask)

        if kept_p.numel() > 0 and "pred_masks" in outputs:
            kept_m = outputs["pred_masks"][kept_p, kept_q][:, None, :, :]
            masks_up = interpolate(
                kept_m, (H_enc, W_enc), mode="bilinear", align_corners=False,
            ).sigmoid() > 0.5  # (K, 1, H, W)

            boxes_cxcywh = pred_boxes[kept_p, kept_q].float().cpu()
            cx, cy, bw, bh = boxes_cxcywh.unbind(1)
            x1 = ((cx - bw / 2) * W_enc).clamp(0, W_enc)
            y1 = ((cy - bh / 2) * H_enc).clamp(0, H_enc)
            x2 = ((cx + bw / 2) * W_enc).clamp(0, W_enc)
            y2 = ((cy + bh / 2) * H_enc).clamp(0, H_enc)
            boxes_px = torch.stack([x1, y1, x2, y2], dim=1).tolist()
            scores_k = probs[kept_p, kept_q].float().cpu().tolist()

            for k in range(kept_p.numel()):
                m_np = masks_up[k, 0].cpu().numpy()  # (H, W) bool
                all_masks.append(m_np)
                all_scores.append(scores_k[k])
                all_boxes.append(boxes_px[k])

        del outputs, probs, keep_mask, bb_in, bb

    del backbone_base

    return all_masks, all_scores, all_boxes, H_enc, W_enc


def sam3_infer_frame(model, processor, pil_img, per_prompt_feats,
                     prompts, device, args):
    """Encode the image then decode the prompt set(s). Returns (all_masks, all_scores, all_boxes,
    H_enc, W_enc). For multiple prompt sets on the SAME image, call sam3_encode_image once then
    sam3_decode_prompts per set to avoid re-encoding."""
    state = sam3_encode_image(processor, pil_img, device, args)
    out = sam3_decode_prompts(model, state, per_prompt_feats, prompts, device, args)
    del state
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    batch_dir = Path(args.batch_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else batch_dir / "frame_ring_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ── Collect images ────────────────────────────────────────────────────────
    rgb_files = sorted(list(batch_dir.glob("rgb_*.jpg")) +
                       list(batch_dir.glob("rgb_*.png")))
    if not rgb_files:
        rgb_files = sorted(list(batch_dir.glob("*.jpg")) +
                           list(batch_dir.glob("*.png")))
    if args.batch_max_frames > 0:
        rgb_files = rgb_files[:args.batch_max_frames]
    if not rgb_files:
        print(f"No images found in {batch_dir}"); return
    print(f"Found {len(rgb_files)} images in {batch_dir}")

    # ── Build SAM3 ────────────────────────────────────────────────────────────
    print("Building SAM3 model...")
    build_device = "cpu" if device.type == "cuda" and (args.bf16 or args.int8) else str(device)
    sam_model = build_sam3(args, build_device)

    _bf16_hooks = []
    if device.type == "cuda" and args.bf16:
        sam_model = convert_model_floating_to_bf16(sam_model)
        sam_model = sam_model.to(device)
        torch.cuda.empty_cache()
        _bf16_hooks = add_linear_input_cast_hooks(sam_model)
    else:
        sam_model = sam_model.to(device)

    processor = Sam3Processor(sam_model, confidence_threshold=args.conf_th)

    # ── Load cached text features ─────────────────────────────────────────────
    cached_path = Path(args.cached_text_features).expanduser().resolve()
    if not cached_path.exists():
        raise FileNotFoundError(f"Cached text features not found: {cached_path}")

    cached = torch.load(str(cached_path), map_location="cpu", weights_only=False)
    cached_prompts = cached.get("prompts", [])

    prompts = list(cached_prompts) if args.all_prompts else list(args.prompt)
    try:
        row_indices = [cached_prompts.index(p) for p in prompts]
    except ValueError as e:
        raise ValueError(
            f"Prompt not found in cache: {e}. "
            f"Cached: {cached_prompts}, requested: {prompts}"
        ) from e

    print(f"SAM3 prompts: {prompts}  (rows {row_indices} from cache)")

    per_prompt_feats = []
    for ri in row_indices:
        idx1 = torch.tensor([ri], dtype=torch.long)
        tf = {}
        for key in ("language_features", "language_embeds"):
            if key in cached:
                tf[key] = cached[key][:, idx1, :].to(device)
        if "language_mask" in cached:
            tf["language_mask"] = cached["language_mask"][idx1:idx1+1, :].to(device)  # keep 2D [1, N_tokens]
        for key, val in cached.items():
            if key not in ("prompts",) and key not in tf and isinstance(val, torch.Tensor):
                tf[key] = val.to(device)
        per_prompt_feats.append(tf)

    # ── Build DA2 ─────────────────────────────────────────────────────────────
    print(f"Building DA2 ({args.da2_encoder})...")
    da2_model = build_da2(args, str(device))
    print(f"  ready on {device}")

    # ── Depth-jump dilation kernel and ring dilation kernel (both fixed 20 px) ──
    jump_kernel = make_disk_kernel(args.jump_dilation)
    ring_kernel = make_disk_kernel(args.ring_dilation)

    # ── Per-frame loop ────────────────────────────────────────────────────────
    t_total = time.time()

    for i, rgb_path in enumerate(rgb_files):
        t0   = time.time()
        stem = rgb_path.stem                          # e.g. "rgb_000042"
        print(f"[{i+1:4d}/{len(rgb_files)}]  {rgb_path.name}", end="  ", flush=True)

        # Load image
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            print("SKIP (unreadable)"); continue
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        H_img, W_img = bgr.shape[:2]

        t_infer_start = time.time()   # inference clock (excludes DA2 + I/O)

        # ── SAM3 image encode + grounding ─────────────────────────────────────
        with torch.inference_mode():
            all_masks, all_scores, all_boxes, H_enc, W_enc = sam3_infer_frame(
                sam_model, processor, pil_img, per_prompt_feats,
                prompts, device, args,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t_after_sam3 = time.time()
        print(f"    [SAM3                          ]  {t_after_sam3 - t_infer_start:6.3f}s", flush=True)

        # ── Splat-deduplication across all prompts ───────────────────────────
        keep_idx, label_map = splat_deduplicate(all_masks, all_boxes, all_scores)
        kept_masks  = [all_masks[k] for k in keep_idx]
        composite_enc = (label_map >= 0) if label_map is not None else np.zeros((H_enc, W_enc), dtype=bool)
        t_after_splat = time.time()
        print(f"    [Splat dedup                   ]  {t_after_splat - t_after_sam3:6.3f}s", flush=True)

        # ── DA2 depth on original image (excluded from reported inference time) ─
        with torch.no_grad():
            da2_raw = da2_model.infer_image(bgr, args.da2_input_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t_after_da2 = time.time()
        print(f"    [DA2 {args.da2_encoder} (excluded)          ]  {t_after_da2 - t_after_splat:6.3f}s", flush=True)

        da2_depth = cv2.resize(da2_raw.astype(np.float32),
                               (W_enc, H_enc), interpolation=cv2.INTER_LINEAR)

        # ── Depth-jump mask (Sobel) + dilation ───────────────────────────────
        jump, jump_mag   = depth_jump_mask(da2_depth, args.jump_thresh)
        jump_dilated     = dilate_mask(jump, jump_kernel)
        t_after_jump = time.time()
        print(f"    [Depth-jump Sobel + dilate     ]  {t_after_jump - t_after_da2:6.3f}s", flush=True)

        # ── Frame ring: dilate composite, strip jumps ─────────────────────────
        ring       = dilate_mask(composite_enc, ring_kernel) & ~composite_enc
        ring_clean = ring & ~jump_dilated
        t_after_ring = time.time()
        print(f"    [Ring build + cancel           ]  {t_after_ring - t_after_jump:6.3f}s", flush=True)

        # inference time = SAM3 + splat + Sobel/ring  (DA2 excluded)
        t_infer = (t_after_sam3 - t_infer_start) + (t_after_splat - t_after_sam3) + \
                  (t_after_jump - t_after_da2)    + (t_after_ring  - t_after_jump)

        # ── Resize to original resolution ────────────────────────────────────
        composite_full   = _resize_bool(composite_enc, W_img, H_img)
        ring_clean_full  = _resize_bool(ring_clean,    W_img, H_img)
        jump_dilated_full = _resize_bool(jump_dilated, W_img, H_img)

        # ── Build ring overlay ────────────────────────────────────────────────
        overlay = bgr.copy().astype(np.float32)
        if composite_full.any():
            overlay[composite_full] = (
                0.5 * overlay[composite_full] +
                0.5 * np.array([0, 200, 0], dtype=np.float32)
            )
        if ring_clean_full.any():
            overlay[ring_clean_full] = (
                0.5 * overlay[ring_clean_full] +
                0.5 * np.array([0, 0, 220], dtype=np.float32)
            )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # ── Depth-jump visualisation: dilated zone (orange) + hot magnitude ───
        # Start with original image, tint dilated-jump zone orange, then overlay
        # the hot Sobel magnitude on top so both are visible in one file.
        mag_full = cv2.resize(jump_mag, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
        dj_heat  = jump_mag_to_vis(mag_full, args.jump_thresh)
        dj_vis   = bgr.astype(np.float32)
        if jump_dilated_full.any():                          # orange tint for dilated zone
            dj_vis[jump_dilated_full] = (
                0.5 * dj_vis[jump_dilated_full] +
                0.5 * np.array([0, 140, 255], dtype=np.float32)   # BGR orange
            )
        active = dj_heat.sum(axis=2) > 10                   # hot map on top
        dj_vis[active] = (
            0.80 * dj_heat[active].astype(np.float32) +
            0.20 * dj_vis[active]
        )

        # ── Save outputs ──────────────────────────────────────────────────────
        t_save_start = time.time()
        cv2.imwrite(str(out_dir / f"{stem}_ring.png"),      overlay)
        cv2.imwrite(str(out_dir / f"{stem}_depthjump.png"), np.clip(dj_vis, 0, 255).astype(np.uint8))
        t_save_end = time.time()
        print(f"    [Save (excluded)               ]  {t_save_end - t_save_start:6.3f}s", flush=True)

        n_det   = len(kept_masks)
        n_glass = int(composite_full.sum())
        n_ring  = int(ring_clean_full.sum())
        alloc_f, resv_f = _vram_gb(device)
        print(f"  → infer={t_infer:.3f}s  dets={n_det}  glass_px={n_glass}  "
              f"ring_px={n_ring}  vram={alloc_f:.2f}/{resv_f:.2f}GB")

    total = time.time() - t_total
    avg   = total / max(len(rgb_files), 1)
    print(f"\n[DONE]  {len(rgb_files)} frames in {total/60:.1f} min  "
          f"({avg:.2f}s/frame wall avg)  →  {out_dir}/")


if __name__ == "__main__":
    main()


# ─── Example commands ─────────────────────────────────────────────────────────
#
# Full SAM3 model (sam3.pt):
#   conda run -n sam3 python glass_frame_ring.py \
#       --batch-dir <HOME>/GlassGuard/facade_glass \
#       --ckpt-path <HOME>/GlassGuard/sam3/sam3.pt \
#       --cached-text-features <HOME>/GlassGuard/sam3/prompt_features/window_glass.pt \
#       --prompt window glass --bf16
#       --ring-min 8 --jump-thresh 0.5 --jump-dilation 20
#
# Student model (slim_3072):
#   conda run -n sam3 python glass_frame_ring.py \
#       --batch-dir <HOME>/GlassGuard/facade_glass \
#       --student \
#       --ckpt-path <HOME>/GlassGuard/sam3/checkpoints/slim_2816/student_final.pt \
#       --meta-json  <HOME>/GlassGuard/sam3/checkpoints/slim_2816/mlp_pruned_meta.json \
#       --mlp-hidden-dim 2816 \
#       --cached-text-features <HOME>/GlassGuard/sam3/prompt_features/window_glass.pt \
#       --prompt glass --bf16 --jump-dilation 20 --jump-thresh 0.5 
