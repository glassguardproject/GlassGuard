#!/usr/bin/env python3
"""
debug_combined_frame.py

Combined: SAM3 ring + NormalHead per-mask normals + whole-scene PCA normals.
Depth source: cloud_XXXXXX.ply  (no DA2 for depth; DA2 ViT backbone used only
for NormalHead RGB feature extraction).

Pipeline:
  1. Load cloud_XXXXXX.ply  → camera-space xyz; project to 2-D depth image
  2. SAM3  → glass masks (splat-replace NMS)
  3. Ring per mask  (dilate – mask – composite)
  4. NormalHead per mask  → predicted surface normal (camera frame)
  5. PCA normals for the whole scene cloud
  6. Save:
       <out-dir>/combined_overlay_frame<N>.jpg
             2D: orange jump tint | per-mask fill | red ring | white normal arrow
       <out-dir>/ring_cloud_frame<N>.ply
             scene cloud (RGB) + per-mask ring points + yellow scene-PCA normal arrows
       <out-dir>/normal_cloud_frame<N>.ply
             scene cloud (RGB) + per-mask NormalHead arrow dots (per-mask colour)

Usage:
  conda run -n sam3 python <HOME>/GlassGuard/debug_combined_frame.py \
    --habitat-dir <HOME>/360_camera/habitat_output \
    --student \
    --ckpt-path   <HOME>/GlassGuard/sam3/checkpoints/slim_2816/student_final.pt \
    --meta-json   <HOME>/GlassGuard/sam3/checkpoints/slim_2816/mlp_pruned_meta.json \
    --cached-text-features <HOME>/GlassGuard/sam3/prompt_features/window_glass.pt \
    --normal-head-ckpt <HOME>/GlassGuard/normal_head_ep0060.pth \
    --prompt glass window --bf16 \
    --out-dir <HOME>/GlassGuard/guard_simple_out/debug_combined \
    --frame-id 100 
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import time

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# ─── Path setup ───────────────────────────────────────────────────────────────
GLASSGUARD_DIR = Path(__file__).resolve().parent
DA2_METRIC_DIR   = Path("<HOME>/Depth-Anything-V2/metric_depth")

for _p in (str(GLASSGUARD_DIR), str(DA2_METRIC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SAM3_PKG_DIR = str(GLASSGUARD_DIR / "sam3")
if _SAM3_PKG_DIR not in sys.path:
    sys.path.insert(0, _SAM3_PKG_DIR)

# ─── Import helpers ───────────────────────────────────────────────────────────
_gfr = importlib.import_module("glass_frame_ring")
_gkd = importlib.import_module("glassguard_deterministic")
_tnh = importlib.import_module("train_normal_head")

build_sam3                     = _gfr.build_sam3
build_da2                      = _gfr.build_da2
sam3_infer_frame               = _gfr.sam3_infer_frame
make_disk_kernel               = _gfr.make_disk_kernel
depth_jump_mask                = _gfr.depth_jump_mask
convert_model_floating_to_bf16 = _gfr.convert_model_floating_to_bf16
add_linear_input_cast_hooks    = _gfr.add_linear_input_cast_hooks

write_ply_xyzrgb_ascii = _gkd.write_ply_xyzrgb_ascii
load_ply_xyzrgb        = _gkd.load_ply_xyzrgb

NormalHead = _tnh.NormalHead

from sam3.model.sam3_image_processor import Sam3Processor

# ─── ImageNet normalisation stats (for ViT input) ────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PATCH_SIZE    = 14

# ─── Colour palettes ──────────────────────────────────────────────────────────
MASK_COLORS_BGR: List[Tuple[int, int, int]] = [
    (  0, 220, 255),
    (255,  80,   0),
    (  0, 255,  80),
    (255,   0, 220),
    (  0, 140, 255),
    (220, 255,   0),
    (255,   0, 100),
    (  0, 255, 220),
    (255, 180,   0),
    (100,   0, 255),
    (180, 255,   0),
    (  0, 100, 255),
]
YELLOW_RGB = np.array([255, 255, 0], dtype=np.uint8)   # scene-PCA normal arrows


# ─── Splat-replace canvas ────────────────────────────────────────────────────

def splat_replace_canvas(
    masks: List[np.ndarray],
    bgr_full: np.ndarray,
    frame_id: int,
    out_dir: str,
    overlap_thresh: float = 0.8,
    save_panels: bool = True,
) -> Tuple[List[int], np.ndarray]:
    """Paint masks onto a canvas one at a time (SAM3 output order).

    For each new mask:
      - Compute intersection / area_new against every canvas mask.
      - If any single canvas mask already covers >= overlap_thresh of the new
        mask's area → skip (don't add to canvas).  Save a before/after panel.
      - Otherwise → splat: add to canvas unconditionally, claiming that region.

    Panels are saved to <out_dir>/mask_splat/frame<N>_<count>_skip.jpg

    Returns (canvas_idx_list, composite_bool_mask).
    """
    if not masks:
        return [], np.zeros((1, 1), dtype=bool)

    H, W  = masks[0].shape
    areas = [int(m.sum()) for m in masks]
    valid = [(i, a) for i, a in enumerate(areas) if a > 0]
    if not valid:
        return [], np.zeros((H, W), dtype=bool)

    H_img, W_img = bgr_full.shape[:2]
    if save_panels:
        splat_dir = os.path.join(out_dir, "mask_splat")
        os.makedirs(splat_dir, exist_ok=True)
    debug_count  = 0

    def _render_canvas(canvas_idx: List[int]) -> np.ndarray:
        vis = bgr_full.copy().astype(np.float32)
        for idx in canvas_idx:
            mf = cv2.resize(masks[idx].astype(np.uint8), (W_img, H_img),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            c  = np.array(MASK_COLORS_BGR[idx % len(MASK_COLORS_BGR)], np.float32)
            vis[mf] = vis[mf] * 0.55 + c * 0.45
        return vis.clip(0, 255).astype(np.uint8)

    def _label(img: np.ndarray, txt: str) -> np.ndarray:
        img = img.copy()
        cv2.putText(img, txt, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (  0,   0,   0), 3)
        cv2.putText(img, txt, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return img

    canvas: List[int] = []

    for new_idx, new_area in valid:
        new_mask = masks[new_idx]

        # Check if any canvas mask already covers >= overlap_thresh of new mask
        blocking_idx = None
        for c_idx in canvas:
            inter = int((new_mask & masks[c_idx]).sum())
            if inter == 0:
                continue
            if inter / new_area >= overlap_thresh:
                blocking_idx = c_idx
                break

        if blocking_idx is not None:
            # Skip — canvas unchanged
            action    = "skip"
            lbl_after = (f"SKIP mask#{new_idx}(a={new_area}): "
                         f"{int((new_mask & masks[blocking_idx]).sum()) * 100 // new_area}% "
                         f"inside mask#{blocking_idx}(a={areas[blocking_idx]})")
        else:
            # Splat, then evict old masks >= overlap_thresh covered by new mask
            canvas.append(new_idx)
            evicted = []
            for c_idx in list(canvas[:-1]):
                inter = int((new_mask & masks[c_idx]).sum())
                if inter > 0 and inter / areas[c_idx] >= overlap_thresh:
                    canvas.remove(c_idx)
                    evicted.append(c_idx)
            if evicted:
                action    = "splat_evict"
                lbl_after = (f"SPLAT mask#{new_idx}(a={new_area}) "
                             f"evicts {evicted}")
            else:
                action    = "splat"
                lbl_after = f"SPLAT mask#{new_idx}(a={new_area}) canvas={len(canvas)} masks"

        if save_panels:
            before_vis = _render_canvas(canvas + [new_idx])
            after_vis  = _render_canvas(canvas)
            lbl_before = f"new mask#{new_idx}(a={new_area}) raw splat"
            panel = np.concatenate([
                _label(before_vis, lbl_before),
                _label(after_vis,  lbl_after),
            ], axis=1)
            panel_path = os.path.join(
                splat_dir,
                f"frame{frame_id:06d}_{debug_count:03d}_{action}.jpg",
            )
            cv2.imwrite(panel_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"[SPLAT] {action.upper():<12s} mask#{new_idx} → {panel_path}")
        debug_count += 1

    composite = np.zeros((H, W), dtype=bool)
    for idx in canvas:
        composite |= masks[idx]
    return canvas, composite


# ─── Text feature loader ──────────────────────────────────────────────────────

# Prompt axis for each cached text-feature key. sam3_infer_frame always feeds
# text_ids=0, and the model reads language_features[:, text_ids] / language_mask[text_ids]
# (sam3/model/sam3_image.py), so EVERY prompt-axis tensor must be sliced down to the
# chosen prompt — otherwise language_features stays full and column 0 (the first cached
# prompt) is used for every prompt.
_PROMPT_AXIS = {
    "language_features": 1,   # (seq, prompt, 256)
    "language_embeds":   1,   # (seq, prompt, 1024)
    "language_mask":     0,   # (prompt, seq)
    "text_features":     0,   # (prompt, ...)
}


def load_per_prompt_feats(args: argparse.Namespace, device: torch.device) -> List[Dict]:
    cached         = torch.load(args.cached_text_features, map_location="cpu", weights_only=False)
    cached_prompts = [p.lower() for p in cached.get("prompts", [])]
    n_prompts      = len(cached_prompts)
    per_prompt_feats: List[Dict] = []
    for prompt in args.prompt:
        pl  = prompt.lower()
        idx = cached_prompts.index(pl) if pl in cached_prompts else 0
        tf: Dict = {}
        for key, val in cached.items():
            if key == "prompts" or not isinstance(val, torch.Tensor):
                continue
            axis = _PROMPT_AXIS.get(key, None)
            if axis is not None and val.ndim > axis and val.shape[axis] == n_prompts:
                sl = [slice(None)] * val.ndim
                sl[axis] = slice(idx, idx + 1)
                tf[key] = val[tuple(sl)].to(device)
            else:
                tf[key] = val.to(device)
        per_prompt_feats.append(tf)
    return per_prompt_feats


# ─── NormalHead loader ────────────────────────────────────────────────────────

def load_normal_head(ckpt_path: str, device: str):
    from depth_anything_v2.dpt import DepthAnythingV2

    head_ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = head_ckpt.get("args", {})
    da2_ckpt   = saved_args.get(
        "da2_ckpt",
        str(DA2_METRIC_DIR.parent / "checkpoints" / "depth_anything_v2_vits.pth"),
    )
    if not os.path.isfile(da2_ckpt):
        raise FileNotFoundError(f"DA2 ViT-S backbone not found: {da2_ckpt}")

    da2 = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    da2.load_state_dict(torch.load(da2_ckpt, map_location="cpu", weights_only=False))
    backbone = da2.pretrained
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone = backbone.to(device).eval()

    head = NormalHead(in_dim=backbone.embed_dim, hidden=256).to(device)
    head.load_state_dict(head_ckpt["head_state_dict"])
    head.eval()

    ep  = head_ckpt.get("epoch", "?")
    ang = head_ckpt.get("val_ang_deg", None)
    print(f"[NormalHead] epoch={ep}  val_ang={f'{ang:.1f}°' if ang else '?'}")
    return backbone, head


@torch.no_grad()
def predict_normal_cam(
    backbone, head,
    bgr: np.ndarray,
    mask_bool: np.ndarray,   # (H, W) bool, full resolution
    input_size: int,
    device: str,
) -> np.ndarray:
    """Returns unit (3,) normal in camera optical frame (x=right, y=down, z=fwd)."""
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_r = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img_f = (rgb_r.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)

    ph = pw = input_size // _PATCH_SIZE
    mask_u8 = cv2.resize(mask_bool.astype(np.uint8), (pw, ph),
                         interpolation=cv2.INTER_NEAREST)
    mask_w  = torch.from_numpy(mask_u8.flatten().astype(np.float32)).unsqueeze(0).to(device)
    if mask_w.sum() < 1.0:
        mask_w = torch.ones_like(mask_w)

    feats = backbone.get_intermediate_layers(img_t, n=1, return_class_token=False)
    return head(feats[0], mask_w).squeeze(0).cpu().numpy()


# ─── Scene PCA normals (vectorised) ──────────────────────────────────────────

def estimate_scene_normals(
    pc_xyz: np.ndarray,   # (N, 3) full scene cloud, camera space
    stride: int = 20,
    k: int      = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA surface-normal estimation for every stride-th point in pc_xyz.
    Neighbours are queried from the full cloud so normals are accurate.

    Returns
    -------
    anchors : (M, 3) float32 — sampled 3-D positions
    normals : (M, 3) float32 — unit normals, oriented towards camera origin
    """
    from scipy.spatial import KDTree

    sample_idx = np.arange(0, len(pc_xyz), stride)
    anchors    = pc_xyz[sample_idx]                    # (M, 3)

    tree       = KDTree(pc_xyz)
    _, nn_idx  = tree.query(anchors, k=max(k, 4))      # (M, k)

    neighbors  = pc_xyz[nn_idx]                        # (M, k, 3)
    centered   = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov        = centered.transpose(0, 2, 1) @ centered  # (M, 3, 3)

    _, eigvecs = np.linalg.eigh(cov)    # ascending eigenvalue order
    normals    = eigvecs[:, :, 0].copy()  # smallest eigenvalue → surface normal

    # Orient towards camera (flip if pointing away from origin)
    to_cam = -anchors
    normals[(normals * to_cam).sum(axis=1) < 0] *= -1

    return anchors.astype(np.float32), normals.astype(np.float32)


# ─── Arrow helper ─────────────────────────────────────────────────────────────

def make_arrow_dots(
    origin: np.ndarray,   # (3,)
    normal: np.ndarray,   # (3,) unit
    length: float = 0.4,
    n_dots: int   = 60,
) -> np.ndarray:
    """(n_dots, 3) points evenly spaced from origin along normal."""
    t = np.linspace(0.0, length, n_dots, dtype=np.float32)
    return (origin[None] + t[:, None] * normal[None]).astype(np.float32)


# ─── 2-D combined overlay ────────────────────────────────────────────────────

def make_combined_overlay(
    bgr: np.ndarray,
    masks_full:    List[np.ndarray],   # (H, W) bool
    rings_full:    List[np.ndarray],   # (H, W) bool
    normals_cam:   List[np.ndarray],   # (3,) unit each
    jump_dil_full: np.ndarray,         # (H, W) bool
) -> np.ndarray:
    """
    Layers bottom → top:
      1. Orange tint   — depth-jump zone
      2. Per-mask fill — glass region
      3. Red overlay   — ring border
      4. White arrow   — NormalHead surface normal direction (per mask)
    """
    H, W = bgr.shape[:2]
    out  = bgr.copy().astype(np.float32)
    RED  = np.array([0,   0, 220], np.float32)   # BGR
    ORG  = np.array([0, 140, 255], np.float32)   # BGR orange

    # 1. Jump zone
    if jump_dil_full.any():
        out[jump_dil_full] = out[jump_dil_full] * 0.65 + ORG * 0.35

    # 2. Mask fills
    for i, mask in enumerate(masks_full):
        if mask.any():
            c = np.array(MASK_COLORS_BGR[i % len(MASK_COLORS_BGR)], np.float32)
            out[mask] = out[mask] * 0.55 + c * 0.45

    # 3. Rings
    for ring in rings_full:
        if ring.any():
            out[ring] = out[ring] * 0.30 + RED * 0.70

    out = out.clip(0, 255).astype(np.uint8)

    # 4. Normal arrows (black outline + white fill)
    arrow_len_px = max(60, min(H, W) // 7)
    for mask, n_cam in zip(masks_full, normals_cam):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        cx_m = int(xs.mean())
        cy_m = int(ys.mean())
        nx2d, ny2d = float(n_cam[0]), float(n_cam[1])
        mag2d = max(1e-6, np.hypot(nx2d, ny2d))
        ex = int(np.clip(cx_m + nx2d / mag2d * arrow_len_px, 0, W - 1))
        ey = int(np.clip(cy_m + ny2d / mag2d * arrow_len_px, 0, H - 1))
        cv2.arrowedLine(out, (cx_m, cy_m), (ex, ey), (  0,   0,   0), 5, tipLength=0.25)
        cv2.arrowedLine(out, (cx_m, cy_m), (ex, ey), (255, 255, 255), 2, tipLength=0.25)

    return out


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Combined: SAM3 ring + NormalHead arrows + scene PCA normals",
    )
    ap.add_argument("--habitat-dir",  required=True)
    ap.add_argument("--out-dir",      default="debug_combined_out")
    ap.add_argument("--frame-id",     type=int, default=0)

    # SAM3
    ap.add_argument("--student",        action="store_true")
    ap.add_argument("--ckpt-path",      default="")
    ap.add_argument("--meta-json",      default="")
    ap.add_argument("--mlp-hidden-dim", type=int, default=0)
    ap.add_argument("--cached-text-features", required=True)
    ap.add_argument("--prompt",         nargs="+", default=["glass", "window"])
    ap.add_argument("--bf16",           action="store_true")
    ap.add_argument("--int8",           action="store_true")
    ap.add_argument("--conf-th",        type=float, default=0.3)

    # Ring
    ap.add_argument("--ring-dilation",  type=int,   default=10)

    # DA2 (dense depth for jump detection)
    ap.add_argument("--da2-encoder",    default="vits")
    ap.add_argument("--da2-root",       default=str(DA2_METRIC_DIR.parent))
    ap.add_argument("--da2-input-size", type=int,   default=518)
    ap.add_argument("--da2-max-depth",  type=float, default=20.0)

    # NormalHead
    ap.add_argument("--normal-head-ckpt",   required=True)
    ap.add_argument("--normal-input-size",  type=int,   default=518)
    ap.add_argument("--da2-backbone-ckpt",  default="",
                    help="Override DA2 ViT-S backbone path (optional)")

    # Per-mask NormalHead arrow
    ap.add_argument("--arrow-length",   type=float, default=0.5,
                    help="Length of per-mask NormalHead arrow in metres")
    ap.add_argument("--arrow-dots",     type=int,   default=80,
                    help="Dots per NormalHead arrow")

    # Whole-scene PCA normal arrows
    ap.add_argument("--scene-normal-stride", type=int,   default=20,
                    help="Sample every N-th scene point as a PCA normal anchor")
    ap.add_argument("--scene-normal-k",      type=int,   default=20,
                    help="KNN neighbourhood size for PCA estimation")
    ap.add_argument("--scene-normal-length", type=float, default=0.15,
                    help="Length of yellow scene-normal arrows in metres")
    ap.add_argument("--scene-normal-dots",   type=int,   default=4,
                    help="Dots per yellow scene-normal arrow")

    # Depth source (PLY)
    ap.add_argument("--depth-min",      type=float, default=0.1)
    ap.add_argument("--depth-max",      type=float, default=20.0)

    # Depth-jump
    ap.add_argument("--jump-thresh",    type=float, default=0.5,
                    help="Sobel depth-jump threshold in metres")
    ap.add_argument("--jump-dilation",  type=int,   default=10,
                    help="Dilation radius (px) for jump mask")

    # Camera intrinsics
    ap.add_argument("--fx",             type=float, default=388.191)
    ap.add_argument("--fy",             type=float, default=422.048)
    ap.add_argument("--cx",             type=float, default=320.0)
    ap.add_argument("--cy",             type=float, default=240.0)
    ap.add_argument("--fx-scale-pct",   type=float, default=70.0,
                    help="habitatScaleXPercent from C++ (scales fx)")
    ap.add_argument("--fy-scale-pct",   type=float, default=40.0,
                    help="habitatScaleYPercent from C++ (scales fy)")

    # Viz
    ap.add_argument("--viz-stride",     type=int,   default=4,
                    help="Subsample stride for scene cloud in PLY output")

    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")

    timings: Dict[str, float] = {}
    frame_id = args.frame_id

    # ── RGB ──────────────────────────────────────────────────────────────────
    rgb_path = None
    for pat in (
        f"rgb_{frame_id:06d}.png", f"rgb_{frame_id:06d}.jpg",
        f"frame_{frame_id:06d}.png", f"frame_{frame_id:06d}.jpg",
        f"color_{frame_id:06d}.png", f"align_{frame_id:06d}.png",
    ):
        p = os.path.join(args.habitat_dir, pat)
        if os.path.isfile(p):
            rgb_path = p
            break
    if rgb_path is None:
        raise FileNotFoundError(f"No RGB for frame {frame_id:06d} in {args.habitat_dir}")
    print(f"[INFO] RGB: {rgb_path}")

    bgr            = cv2.imread(rgb_path)
    H_orig, W_orig = bgr.shape[:2]
    pil_img        = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    eff_fx = args.fx * args.fx_scale_pct / 100.0
    eff_fy = args.fy * args.fy_scale_pct / 100.0
    eff_cx = args.cx
    eff_cy = args.cy
    print(f"[INFO] Effective intrinsics: fx={eff_fx:.3f} fy={eff_fy:.3f} "
          f"cx={eff_cx:.1f} cy={eff_cy:.1f}")

    # ── Load PLY cloud ────────────────────────────────────────────────────────
    ply_path_in = None
    for pat in (f"cloud_{frame_id:06d}.ply", f"frame_{frame_id:06d}.ply",
                f"pointcloud_{frame_id:06d}.ply"):
        p = os.path.join(args.habitat_dir, pat)
        if os.path.isfile(p):
            ply_path_in = p
            break
    if ply_path_in is None:
        raise FileNotFoundError(f"No PLY for frame {frame_id:06d} in {args.habitat_dir}")
    print(f"[INFO] PLY: {ply_path_in}")

    _t = time.perf_counter()
    ply_xyz_raw, _ = load_ply_xyzrgb(ply_path_in)
    # PLY is saved in Z-up viewer frame: (x=right, y=forward/depth, z=up)
    # Restore camera frame (x=right, y=down, z=forward) for projection math.
    ply_xyz = np.column_stack([
        ply_xyz_raw[:, 0],   # x_cam = ply.x
        -ply_xyz_raw[:, 2],  # y_cam = -ply.z  (viewer-up → camera-down)
        ply_xyz_raw[:, 1],   # z_cam =  ply.y  (viewer-forward = camera-depth)
    ])

    # ── Project PLY → 2-D depth image ────────────────────────────────────────
    z_ply  = ply_xyz[:, 2]
    pos_z  = z_ply > 0
    u_f    = ply_xyz[pos_z, 0] / z_ply[pos_z] * eff_fx + eff_cx
    v_f    = ply_xyz[pos_z, 1] / z_ply[pos_z] * eff_fy + eff_cy
    z_f    = z_ply[pos_z]
    in_frm = (u_f >= 0) & (u_f < W_orig) & (v_f >= 0) & (v_f < H_orig)
    ui     = np.round(u_f[in_frm]).astype(np.int32).clip(0, W_orig - 1)
    vi     = np.round(v_f[in_frm]).astype(np.int32).clip(0, H_orig - 1)
    zi     = z_f[in_frm]

    depth_full = np.zeros((H_orig, W_orig), dtype=np.float32)
    order = np.argsort(zi)[::-1]          # far → near; near overwrites
    depth_full[vi[order], ui[order]] = zi[order]

    # ── Scene cloud: PLY xyz + RGB sampled from image ─────────────────────────
    rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pc_xyz  = ply_xyz[pos_z][in_frm]              # (M, 3) camera-space
    pc_rgb  = rgb_img[vi, ui].astype(np.uint8)    # (M, 3) sampled colours
    timings["PLY load + depth project"] = time.perf_counter() - _t
    print(f"[INFO] PLY → depth: {len(zi)} pts, range={zi.min():.2f}–{zi.max():.2f} m")
    print(f"[INFO] Scene cloud: {pc_xyz.shape[0]} pts")

    # ── DA2 dense depth (for depth-jump detection) ────────────────────────────
    _t = time.perf_counter()
    da2_model = build_da2(args, device)
    da2_model.eval()
    with torch.no_grad():
        _depth_raw = da2_model.infer_image(bgr, args.da2_input_size)
    depth_da2 = cv2.resize(_depth_raw.astype(np.float32), (W_orig, H_orig),
                           interpolation=cv2.INTER_LINEAR)
    del da2_model
    timings["DA2 depth (jump)"] = time.perf_counter() - _t
    print(f"  DA2 depth: {depth_da2.min():.2f}–{depth_da2.max():.2f} m")

    # ── Depth-jump mask (on dense DA2 depth) ──────────────────────────────
    _t = time.perf_counter()
    jump_bool, _  = depth_jump_mask(depth_da2, args.jump_thresh)
    jump_kernel   = make_disk_kernel(args.jump_dilation)
    jump_dil_full = cv2.dilate(jump_bool.astype(np.uint8),
                               jump_kernel, iterations=1).astype(bool)
    timings["Depth-jump mask"] = time.perf_counter() - _t
    print(f"  jump px: {jump_bool.sum()}  dilated: {jump_dil_full.sum()}")

    # ── SAM3 ─────────────────────────────────────────────────────────────────
    print("Building SAM3...")
    _t = time.perf_counter()
    build_dev = "cpu" if use_cuda and (args.bf16 or args.int8) else str(device)
    sam_model = build_sam3(args, build_dev)
    if use_cuda and args.bf16:
        sam_model = convert_model_floating_to_bf16(sam_model)
        sam_model = sam_model.to(device)
        torch.cuda.empty_cache()
        add_linear_input_cast_hooks(sam_model)
    else:
        sam_model = sam_model.to(device)
    sam_model.eval()
    timings["SAM3 build"] = time.perf_counter() - _t

    processor        = Sam3Processor(sam_model, confidence_threshold=args.conf_th)
    per_prompt_feats = load_per_prompt_feats(args, device)
    print("SAM3 ready. Running inference...")

    _t = time.perf_counter()
    all_masks, all_scores, all_boxes, H_enc, W_enc = sam3_infer_frame(
        sam_model, processor, pil_img, per_prompt_feats,
        args.prompt, device, args,
    )
    print(f"  raw detections: {len(all_scores)}")

    if not all_masks:
        print("[DONE] No detections.")
        return

    keep_idx, composite_enc = splat_replace_canvas(
        all_masks, bgr, frame_id, args.out_dir,
    )
    kept_masks  = [all_masks[k]  for k in keep_idx]
    kept_scores = [all_scores[k] for k in keep_idx]
    timings["SAM3 inference + splat"] = time.perf_counter() - _t
    print(f"  kept after splat: {len(kept_masks)}")

    # ── Mask-check panel (raw detections vs final canvas) ─────────────────────
    def _mask_panel(masks_enc: List[np.ndarray], label: str) -> np.ndarray:
        panel = bgr.copy().astype(np.float32)
        for i, m in enumerate(masks_enc):
            mf = cv2.resize(m.astype(np.uint8), (W_orig, H_orig),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            c = np.array(MASK_COLORS_BGR[i % len(MASK_COLORS_BGR)], np.float32)
            panel[mf] = panel[mf] * 0.55 + c * 0.45
        panel = panel.clip(0, 255).astype(np.uint8)
        txt = f"{label}  ({len(masks_enc)})"
        cv2.putText(panel, txt, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (  0,   0,   0), 3)
        cv2.putText(panel, txt, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
        return panel

    check_panel = np.concatenate(
        [_mask_panel(all_masks,  "raw detections"),
         _mask_panel(kept_masks, "final canvas")],
        axis=1,
    )
    check_dir  = os.path.join(args.out_dir, "maskcheck")
    os.makedirs(check_dir, exist_ok=True)
    check_path = os.path.join(check_dir, f"maskcheck_frame{frame_id:06d}.jpg")
    cv2.imwrite(check_path, check_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[SAVE] Mask-check panel    → {check_path}")

    # ── Rings ─────────────────────────────────────────────────────────────────
    _t = time.perf_counter()
    ring_kernel    = make_disk_kernel(args.ring_dilation)
    masks_full:    List[np.ndarray] = []
    rings_full:    List[np.ndarray] = []
    ring_pts_list: List[np.ndarray] = []

    for mi, mask_enc in enumerate(kept_masks):
        mask_dil_enc = cv2.dilate(mask_enc.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
        ring_enc     = mask_dil_enc & ~composite_enc   # no jump subtraction

        mask_full = cv2.resize(mask_enc.astype(np.uint8), (W_orig, H_orig),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        ring_full = cv2.resize(ring_enc.astype(np.uint8), (W_orig, H_orig),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        masks_full.append(mask_full)
        rings_full.append(ring_full)

        ys_r, xs_r = np.nonzero(ring_full)
        z_r = depth_full[ys_r, xs_r]
        good = (z_r > args.depth_min) & (z_r < args.depth_max)
        ys_r, xs_r, z_r = ys_r[good], xs_r[good], z_r[good]
        ring_pts = np.stack([(xs_r - eff_cx) / eff_fx * z_r,
                             (ys_r - eff_cy) / eff_fy * z_r,
                             z_r], axis=1).astype(np.float32)
        ring_pts_list.append(ring_pts)
        print(f"  mask {mi:02d} (s={kept_scores[mi]:.2f})  "
              f"ring_px={int(ring_full.sum())}  ring_3d={ring_pts.shape[0]}")
    timings["Ring computation"] = time.perf_counter() - _t

    # ── NormalHead ───────────────────────────────────────────────────────────
    print(f"\nLoading NormalHead from {args.normal_head_ckpt} ...")
    _t = time.perf_counter()
    nh_ckpt = args.normal_head_ckpt
    if args.da2_backbone_ckpt:
        ckpt_data = torch.load(nh_ckpt, map_location="cpu", weights_only=False)
        ckpt_data.setdefault("args", {})["da2_ckpt"] = args.da2_backbone_ckpt
        nh_ckpt = nh_ckpt + ".patched.pt"
        torch.save(ckpt_data, nh_ckpt)
    nh_backbone, nh_head = load_normal_head(nh_ckpt, str(device))
    timings["NormalHead build"] = time.perf_counter() - _t

    _t = time.perf_counter()
    normals_cam: List[np.ndarray] = []
    for mi, mask_full in enumerate(masks_full):
        n_cam = predict_normal_cam(
            nh_backbone, nh_head, bgr, mask_full,
            args.normal_input_size, str(device),
        )
        normals_cam.append(n_cam)
        print(f"  mask {mi:02d} n_cam=[{n_cam[0]:+.3f}, {n_cam[1]:+.3f}, {n_cam[2]:+.3f}]")
    timings["NormalHead inference"] = time.perf_counter() - _t

    # ── Scene PCA normals ─────────────────────────────────────────────────────
    print(f"\nEstimating scene PCA normals (stride={args.scene_normal_stride}, "
          f"k={args.scene_normal_k}) ...")
    _t = time.perf_counter()
    scene_anchors, scene_normals = estimate_scene_normals(
        pc_xyz,
        stride=args.scene_normal_stride,
        k=args.scene_normal_k,
    )
    timings["Scene PCA normals"] = time.perf_counter() - _t
    print(f"  {len(scene_anchors)} anchor points → normals computed")

    # ── Timing summary (before any saves) ────────────────────────────────────
    t_total = sum(timings.values())
    print("\n── Inference Timing ────────────────────────────────────")
    for step, dt in timings.items():
        print(f"  {step:<35s}: {dt:6.3f} s")
    print(f"  {'─'*43}")
    print(f"  {'TOTAL (excl. saves)':<35s}: {t_total:6.3f} s")
    print("────────────────────────────────────────────────────────")

    stride = max(1, args.viz_stride)

    # ── 2-D depth-jump overlay ───────────────────────────────────────────────────
    jump_vis = bgr.copy().astype(np.float32)
    _ORG = np.array([0, 140, 255], np.float32)   # BGR orange
    if jump_dil_full.any():
        jump_vis[jump_dil_full] = jump_vis[jump_dil_full] * 0.55 + _ORG * 0.45
    jump_vis = jump_vis.clip(0, 255).astype(np.uint8)
    jump_path = os.path.join(args.out_dir, f"depth_jump_frame{frame_id:06d}.jpg")
    cv2.imwrite(jump_path, jump_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n[SAVE] Depth-jump overlay   → {jump_path}")

    # ── 2-D lidar projection ─────────────────────────────────────────────────────
    # Per-pixel ring membership (-1 = not a ring pixel)
    ring_label = np.full((H_orig, W_orig), -1, dtype=np.int32)
    for mi, ring_full in enumerate(rings_full):
        ring_label[ring_full] = mi

    # `order` = far→near indices into ui/vi/zi (already computed above)
    u_ord = ui[order]
    v_ord = vi[order]
    nat_bgr  = bgr[v_ord, u_ord].copy()   # natural image colours (BGR)
    lbl_ord  = ring_label[v_ord, u_ord]
    for mi in range(len(rings_full)):
        hit = lbl_ord == mi
        if hit.any():
            nat_bgr[hit] = MASK_COLORS_BGR[mi % len(MASK_COLORS_BGR)]

    lidar_vis = bgr.copy()
    lidar_vis[v_ord, u_ord] = nat_bgr
    lidar_path = os.path.join(args.out_dir, f"lidar_proj_frame{frame_id:06d}.jpg")
    cv2.imwrite(lidar_path, lidar_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[SAVE] Lidar projection     → {lidar_path}")

    # ── PLY: ring cloud (scene + per-mask ring points, no normal arrows) ────────
    ring_parts_xyz: List[np.ndarray] = [pc_xyz[::stride]]
    ring_parts_rgb: List[np.ndarray] = [pc_rgb[::stride]]

    for mi, ring_pts in enumerate(ring_pts_list):
        if ring_pts.shape[0] == 0:
            continue
        c_bgr = MASK_COLORS_BGR[mi % len(MASK_COLORS_BGR)]
        c_rgb = np.array([c_bgr[2], c_bgr[1], c_bgr[0]], dtype=np.uint8)
        ring_parts_xyz.append(ring_pts)
        ring_parts_rgb.append(np.tile(c_rgb, (ring_pts.shape[0], 1)))

    ring_xyz = np.concatenate(ring_parts_xyz, axis=0)
    ring_rgb = np.concatenate(ring_parts_rgb, axis=0)
    ring_ply = os.path.join(args.out_dir, f"ring_cloud_frame{frame_id:06d}.ply")
    write_ply_xyzrgb_ascii(ring_ply, ring_xyz, ring_rgb)
    print(f"[SAVE] Ring cloud           → {ring_ply}  ({ring_xyz.shape[0]} pts)")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
