#!/usr/bin/env python3
"""
Train a lightweight MLP head to predict surface normals from RGB + mask input,
using a frozen DA-V2 ViT-S patch encoder.

Architecture:
  frozen DA-V2 ViT-S backbone  (DINOv2 ViT-S, embed_dim=384, patch_size=14)
    ↓  get_intermediate_layers (last block)  →  (B, N_patches, 384)
  mask-weighted average pool  →  (B, 384)
    ↓  LayerNorm → Linear(384→256) → GELU → Linear(256→128) → GELU → Linear(128→3)
  L2-normalize  →  unit normal vector (B, 3)

Ground truth: normals stored in {stem}_da2_normals_merged.json produced by
              sam3_normal_pipeline_dav2.py with --in-place-update.

Loss: 1 – cosine_similarity  (angular loss, range [0, 2])

Usage:
    python train_normal_head_dav2.py \
        --data-dirs \
            /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/train_train_val \
            /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/campus_walk2_train_val \
            /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/street_walk_day1_train_val \
        --da2-ckpt /ocean/projects/cis220039p/mdt2/hguo7/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
        --output-dir /tmp/normal_head_out \
        --epochs 60 \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ── Repo path ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train surface-normal head on top of frozen DA-V2 ViT-S"
    )
    ap.add_argument(
        "--data-dirs",
        nargs="+",
        default=[
            "/ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/train_train_val",
            "/ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/campus_walk2_train_val",
            "/ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/campus_wallk1_train_val",
            "/ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/street_walk_day1_train_val",
            "/ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/street_walk_day2_train_val",
        ],
        help="One or more roots of DA2-annotated datasets (each containing per-image subfolders).",
    )
    ap.add_argument(
        "--da2-ckpt",
        default="checkpoints/depth_anything_v2_vits.pth",
        help="Path to the DA-V2 ViT-S checkpoint (.pth).",
    )
    ap.add_argument(
        "--output-dir",
        default="/ocean/projects/cis220039p/mdt2/hguo7/Depth-Anything-V2/normal_head_out",
        help="Directory for saved checkpoints and logs.",
    )
    ap.add_argument("--input-size", type=int, default=518,
                    help="Image resize (height and width) fed to the ViT encoder.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1,
                    help="Fraction of data to hold out for validation.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-every", type=int, default=10,
                    help="Save a checkpoint every N epochs (in addition to best).")
    ap.add_argument("--min-mask-px", type=int, default=50,
                    help="Skip mask samples with fewer valid pixels than this.")
    ap.add_argument("--augment", action="store_true", default=True,
                    help="Apply color-jitter and horizontal-flip augmentation.")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="Save visual normal overlays every N epochs.")
    ap.add_argument("--eval-vis-n", type=int, default=10,
                    help="Number of unique images to visualize per visual-eval run.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in --output-dir.")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

class NormalSample:
    """Lightweight container so the Dataset is easy to index."""
    __slots__ = ("image_path", "mask_path", "normal_xyz")

    def __init__(self, image_path: Path, mask_path: Path, normal_xyz: np.ndarray):
        self.image_path = image_path
        self.mask_path = mask_path
        self.normal_xyz = normal_xyz.astype(np.float32)


def _collect_samples(data_dirs: List[Path], min_mask_px: int) -> List[NormalSample]:
    """Scan all data_dirs for subfolders with DA2 normals JSON and collect samples."""
    samples: List[NormalSample] = []
    skipped_no_json = 0
    skipped_bad_normal = 0
    skipped_small_mask = 0
    scanned = 0

    # Count total subfolders first for progress display
    all_subs: List[Tuple[Path, str]] = []
    for data_dir in data_dirs:
        for sub in sorted(data_dir.iterdir()):
            if sub.is_dir():
                all_subs.append((sub, sub.name))
    total_subs = len(all_subs)
    print(f"[Dataset] Scanning {total_subs} subfolders across {len(data_dirs)} dir(s)...")

    for data_dir in data_dirs:
        for sub in sorted(data_dir.iterdir()):
            if not sub.is_dir():
                continue
            scanned += 1
            if scanned % 500 == 0 or scanned == total_subs:
                print(f"  prepared {len(samples):>6} samples | scanned {scanned:>6} / {total_subs} subfolders")
            stem = sub.name
            json_path = sub / f"{stem}_da2_normals_merged.json"
            # Use original RGB image (not white overlay) as model input
            rgb_image = sub / f"{stem}.png"

            if not json_path.exists() or not rgb_image.exists():
                skipped_no_json += 1
                continue

            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                skipped_no_json += 1
                continue

            for entry in data.get("normals", []):
                n_xyz = entry.get("normal_xyz")
                mask_file = entry.get("mask_file")
                if n_xyz is None or mask_file is None:
                    skipped_bad_normal += 1
                    continue

                normal = np.asarray(n_xyz, dtype=np.float32)
                nn_len = float(np.linalg.norm(normal))
                if nn_len < 1e-6:
                    skipped_bad_normal += 1
                    continue
                normal = normal / nn_len

                mask_path = sub / mask_file
                if not mask_path.exists():
                    skipped_bad_normal += 1
                    continue

                # Quick pixel-count check (cheap, avoids loading at training time).
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is None or int((mask_img > 0).sum()) < min_mask_px:
                    skipped_small_mask += 1
                    continue

                samples.append(NormalSample(rgb_image, mask_path, normal))

    print(
        f"[Dataset] {len(samples)} samples collected from {len(data_dirs)} dir(s). "
        f"Skipped: {skipped_no_json} (no JSON/RGB), "
        f"{skipped_bad_normal} (bad normal), "
        f"{skipped_small_mask} (tiny mask)."
    )
    return samples


class NormalDataset(Dataset):
    """
    Returns (img_tensor, mask_weights, normal_gt) per sample.

    img_tensor   : (3, H, W) float32, ImageNet-normalized
    mask_weights : (N_patches,) float32, mask projected to patch grid (0/1)
    normal_gt    : (3,) float32, unit normal vector
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        samples: List[NormalSample],
        input_size: int = 518,
        patch_size: int = 14,
        augment: bool = True,
    ):
        self.samples = samples
        self.input_size = input_size
        self.patch_size = patch_size
        self.augment = augment
        self.patch_h = input_size // patch_size
        self.patch_w = input_size // patch_size
        self.N_patches = self.patch_h * self.patch_w

        # Colour-jitter applied to BGR uint8 before converting to tensor.
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),   # (H,W,3) uint8 → (3,H,W) float32 [0,1]
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # ── Load image (RGB uint8) ────────────────────────────────────
        bgr = cv2.imread(str(s.image_path))
        if bgr is None:
            raise RuntimeError(f"Cannot read {s.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── Load mask (binary uint8) ──────────────────────────────────
        mask_gray = cv2.imread(str(s.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f"Cannot read {s.mask_path}")
        mask_bin = (mask_gray > 0).astype(np.uint8)  # 0/1

        # ── Resize both to input_size × input_size ───────────────────
        rgb_resized = cv2.resize(rgb, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask_bin, (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_NEAREST)

        normal = s.normal_xyz.copy()

        # ── Augmentation ─────────────────────────────────────────────
        if self.augment and random.random() < 0.5:
            # Horizontal flip: x-component of normal flips sign
            rgb_resized = np.fliplr(rgb_resized).copy()
            mask_resized = np.fliplr(mask_resized).copy()
            normal = normal * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
            normal = normal / max(float(np.linalg.norm(normal)), 1e-9)

        # Colour jitter (applied to PIL to reuse torchvision)
        if self.augment:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(rgb_resized)
            pil_img = self.color_jitter(pil_img)
            rgb_resized = np.asarray(pil_img).copy()

        # ── To tensor ────────────────────────────────────────────────
        img_tensor = self.to_tensor(rgb_resized)  # (3, H, W)

        # ── Mask → patch weights ─────────────────────────────────────
        mask_patch = cv2.resize(mask_resized, (self.patch_w, self.patch_h),
                                 interpolation=cv2.INTER_NEAREST)
        mask_weights = torch.from_numpy(mask_patch.flatten().astype(np.float32))

        # Fallback: if the mask collapses to zero patches, use uniform weights
        if mask_weights.sum() < 1.0:
            mask_weights = torch.ones(self.N_patches, dtype=torch.float32)

        normal_gt = torch.from_numpy(normal)
        return img_tensor, mask_weights, normal_gt


# ═══════════════════════════════════════════════════════════════════════════
#  Visual evaluation
# ═══════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prepare_sample_tensors(
    image_path: Path,
    mask_path: Path,
    input_size: int,
    patch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load and resize image + mask, return:
      img_t      : (1, 3, H, W) float32 tensor on device
      mask_w     : (1, N_patches) float32 tensor on device
      rgb_orig   : (H_orig, W_orig, 3) uint8 for visualization
      mask_orig  : (H_orig, W_orig) bool at original resolution
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Cannot read {image_path}")
    rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise RuntimeError(f"Cannot read {mask_path}")
    mask_orig = mask_gray > 0

    h_orig, w_orig = rgb_orig.shape[:2]
    rgb_r = cv2.resize(rgb_orig, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    mask_r = cv2.resize(mask_gray, (input_size, input_size), interpolation=cv2.INTER_NEAREST) > 0

    img_f = rgb_r.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)

    ph = input_size // patch_size
    pw = input_size // patch_size
    mask_patch = cv2.resize(mask_r.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)
    mask_w = torch.from_numpy(mask_patch.flatten().astype(np.float32)).unsqueeze(0).to(device)
    if mask_w.sum() < 1.0:
        mask_w = torch.ones_like(mask_w)

    return img_t, mask_w, rgb_orig, mask_orig


def _draw_arrow_2d(
    vis: np.ndarray,
    mask: np.ndarray,
    normal: np.ndarray,
    color: Tuple[int, int, int],
    arrow_px: int = 60,
) -> None:
    """
    Draw a 2D arrow on vis representing the normal direction.
    Direction is derived from (nx, -ny) of the 3D unit normal
    (x right, y down in image convention).
    Draws in-place.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return
    cx, cy = int(np.median(xs)), int(np.median(ys))

    # Project 3D normal to 2D image plane: (nx, -ny)
    nx, ny = float(normal[0]), float(normal[1])
    length = max(1e-9, (nx ** 2 + ny ** 2) ** 0.5)
    dx = nx / length * arrow_px
    dy = -ny / length * arrow_px  # flip y (image y points down)

    tip = (int(cx + dx), int(cy + dy))
    cv2.arrowedLine(vis, (cx, cy), tip, color, thickness=3,
                    tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), 5, color, -1, lineType=cv2.LINE_AA)


_VIS_COLORS = [
    (255,  60,  60),  # red
    ( 60, 200,  60),  # green
    ( 60, 150, 255),  # blue
    (255, 200,  60),  # yellow
    (220,  60, 220),  # magenta
    ( 60, 220, 220),  # cyan
    (255, 140,  40),  # orange
    (160, 120, 255),  # purple
]


@torch.no_grad()
def visual_eval(
    model: "NormalEstimator",
    val_samples: List[NormalSample],
    epoch: int,
    out_dir: Path,
    device: str,
    input_size: int,
    patch_size: int,
    n_images: int = 10,
    rng_seed: int = 0,
) -> None:
    """
    Pick n_images unique images from val_samples, run the model on each of
    their masks independently, and save an overlay PNG per image showing:
      - coloured mask tint + contour per mask
      - RED arrow  = predicted normal (model output)
      - GREEN arrow = GT normal (from DA2 JSON)
      - angular error text per mask
    """
    model.eval()
    vis_dir = out_dir / "vis_eval" / f"ep{epoch:04d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Group val_samples by image_path so all masks of one image go together
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for s in val_samples:
        groups[str(s.image_path)].append(s)

    img_paths = list(groups.keys())
    rng = np.random.default_rng(rng_seed + epoch)
    chosen = rng.choice(len(img_paths),
                        size=min(n_images, len(img_paths)),
                        replace=False)

    for ci, idx in enumerate(chosen):
        img_key = img_paths[int(idx)]
        group = groups[img_key]  # list of NormalSample sharing this image

        # Load original RGB once for drawing
        bgr = cv2.imread(img_key)
        if bgr is None:
            continue
        vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        h_orig, w_orig = vis.shape[:2]

        mask_losses: List[str] = []

        for mi, s in enumerate(group):
            color = _VIS_COLORS[mi % len(_VIS_COLORS)]

            # Prepare tensors (uses original resolution image, resized inside)
            try:
                img_t, mask_w, _rgb, mask_orig = _prepare_sample_tensors(
                    s.image_path, s.mask_path, input_size, patch_size, device
                )
            except RuntimeError:
                continue

            # Forward pass — this is the exact same computation as training
            feats = model.backbone.get_intermediate_layers(
                img_t, n=1, return_class_token=False
            )
            pred_normal = model.head(feats[0], mask_w).squeeze(0).cpu().numpy()  # (3,)
            gt_normal   = s.normal_xyz  # (3,) already unit

            cos = float(np.dot(pred_normal, gt_normal))
            ang_err = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            mask_losses.append(f"M{mi} {ang_err:.1f}°")

            # ── Draw mask tint + contour ──────────────────────────────
            mask_draw = cv2.resize(
                mask_orig.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            tint = np.zeros_like(vis)
            tint[mask_draw] = color
            vis = cv2.addWeighted(vis, 1.0, tint, 0.25, 0)

            cnts, _ = cv2.findContours(
                (mask_draw.astype(np.uint8) * 255),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, cnts, -1, color, 2)

            # ── GT arrow (bright green, solid) ───────────────────────
            _draw_arrow_2d(vis, mask_draw, gt_normal, (0, 230, 80), arrow_px=70)

            # ── Pred arrow (red, solid) ───────────────────────────────
            _draw_arrow_2d(vis, mask_draw, pred_normal, (255, 50, 50), arrow_px=70)

            # ── Angular error label at mask centroid ─────────────────
            ys, xs = np.where(mask_draw)
            if ys.size > 0:
                lx, ly = int(np.median(xs)), int(np.median(ys))
                label = f"{ang_err:.1f}d"
                cv2.putText(vis, label, (lx + 8, ly - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, label, (lx + 8, ly - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Legend strip at top ───────────────────────────────────────
        legend = "  ".join(mask_losses) if mask_losses else "(no masks)"
        cv2.putText(vis, f"ep{epoch} | " + legend, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"ep{epoch} | " + legend, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Legend key: green = GT, red = pred
        cv2.putText(vis, "GREEN=GT  RED=pred", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, "GREEN=GT  RED=pred", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1, cv2.LINE_AA)

        stem = Path(img_key).stem
        out_path = vis_dir / f"{ci:02d}_{stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"[VisEval] ep{epoch}: saved {len(chosen)} image(s) → {vis_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  Encoder  (frozen DA-V2 ViT-S)
# ═══════════════════════════════════════════════════════════════════════════

def load_da2_backbone(ckpt_path: str, device: str) -> nn.Module:
    """
    Load DepthAnythingV2 (ViT-S) from checkpoint, discard the DPT depth head,
    return the DINOv2 backbone frozen.
    """
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    importlib.invalidate_caches()

    dpt_module = importlib.import_module("depth_anything_v2.dpt")
    DepthAnythingV2 = dpt_module.DepthAnythingV2

    model_cfg = dict(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    da2 = DepthAnythingV2(**model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    da2.load_state_dict(state)
    backbone = da2.pretrained  # DINOv2 ViT-S

    # Freeze all backbone parameters.
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone = backbone.to(device).eval()

    embed_dim = backbone.embed_dim   # 384 for ViT-S
    patch_size = backbone.patch_size  # 14
    print(
        f"[Encoder] DA-V2 ViT-S loaded from {ckpt_path} | "
        f"embed_dim={embed_dim}, patch_size={patch_size} — FROZEN"
    )
    return backbone


# ═══════════════════════════════════════════════════════════════════════════
#  Normal head
# ═══════════════════════════════════════════════════════════════════════════

class NormalHead(nn.Module):
    """
    Lightweight MLP head:
      masked-average-pool(patch_tokens, mask_weights)
        → LayerNorm → Linear(384→256) → GELU
        → Linear(256→128) → GELU → Linear(128→3)
        → L2-normalize
    """

    def __init__(self, in_dim: int = 384, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,   # (B, N, D)
        mask_weights: torch.Tensor,   # (B, N)  float, ≥0
    ) -> torch.Tensor:                # (B, 3) unit normals
        # Masked weighted average pool
        w = mask_weights.unsqueeze(-1).clamp(min=0.0)  # (B, N, 1)
        w_sum = w.sum(dim=1).clamp(min=1e-6)           # (B, 1)
        pooled = (patch_tokens * w).sum(dim=1) / w_sum  # (B, D)

        pooled = self.norm(pooled)
        out = self.mlp(pooled)                          # (B, 3)
        return F.normalize(out, dim=-1)


class NormalEstimator(nn.Module):
    """
    Full model: frozen encoder + trainable head.
    Only NormalHead parameters are updated during training.
    """

    def __init__(self, backbone: nn.Module, head: NormalHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        images: torch.Tensor,       # (B, 3, H, W)
        mask_weights: torch.Tensor, # (B, N_patches)
    ) -> torch.Tensor:              # (B, 3)
        with torch.no_grad():
            # Last-layer patch tokens: tuple of length 1, each (B, N, D)
            feats = self.backbone.get_intermediate_layers(
                images, n=1, return_class_token=False
            )
        patch_tokens = feats[0]  # (B, N, D)
        return self.head(patch_tokens, mask_weights)


# ═══════════════════════════════════════════════════════════════════════════
#  Loss & metrics
# ═══════════════════════════════════════════════════════════════════════════

def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 – cosine_similarity, averaged over batch. Range [0, 2]."""
    cos = F.cosine_similarity(pred, target, dim=-1)  # (B,)
    return (1.0 - cos).mean()


@torch.no_grad()
def mean_angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean angular error in degrees."""
    cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1.0, 1.0)
    return float(torch.acos(cos).mean() * (180.0 / torch.pi))


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: NormalEstimator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.head.train()
    total_loss = 0.0
    total_ang = 0.0
    n_batches = 0

    for imgs, masks, normals in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        normals = normals.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(imgs, masks)
        loss = angular_loss(pred, normals)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_ang += mean_angular_error_deg(pred, normals)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_ang / max(n_batches, 1)


@torch.no_grad()
def val_epoch(
    model: NormalEstimator,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n_batches = 0

    for imgs, masks, normals in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        normals = normals.to(device, non_blocking=True)

        pred = model(imgs, masks)
        loss = angular_loss(pred, normals)

        total_loss += float(loss)
        total_ang += mean_angular_error_deg(pred, normals)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_ang / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable — falling back to CPU.")
        device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    data_dirs = [Path(d).resolve() for d in args.data_dirs]
    for d in data_dirs:
        if not d.exists():
            raise FileNotFoundError(f"data-dir not found: {d}")
    print(f"[Data] Scanning {len(data_dirs)} directory(ies):")
    for d in data_dirs:
        print(f"       {d}")

    all_samples = _collect_samples(data_dirs, args.min_mask_px)
    if len(all_samples) == 0:
        raise RuntimeError(
            "No samples found. Run sam3_normal_pipeline_dav2.py with "
            "--in-place-update first to generate the DA2 normals JSON files."
        )

    n_val = max(1, int(len(all_samples) * args.val_split))
    n_train = len(all_samples) - n_val

    train_set, val_set = random_split(
        range(len(all_samples)),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Wrap indices into sample lists
    train_samples = [all_samples[i] for i in train_set.indices]
    val_samples   = [all_samples[i] for i in val_set.indices]

    patch_size = 14  # DINOv2 ViT-S fixed patch size
    train_ds = NormalDataset(train_samples, args.input_size, patch_size, augment=args.augment)
    val_ds   = NormalDataset(val_samples,   args.input_size, patch_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(
        f"[Data] train={len(train_ds)}, val={len(val_ds)} | "
        f"input_size={args.input_size}, patch_grid={args.input_size // patch_size}²"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    backbone = load_da2_backbone(args.da2_ckpt, device)
    head = NormalHead(in_dim=backbone.embed_dim, hidden=256).to(device)
    model = NormalEstimator(backbone, head)

    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[Model] Trainable head parameters: {n_params:,}")

    # ── Optimiser & scheduler (built after resume so last_epoch can be set) ──
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    resume_epoch = 0
    if args.resume:
        ckpt_candidates = sorted(out_dir.glob("normal_head_ep*.pth")) + \
                          ([out_dir / "best_normal_head.pth"]
                           if (out_dir / "best_normal_head.pth").exists() else [])
        latest_ckpt = None
        latest_epoch = 0
        for cp in ckpt_candidates:
            try:
                meta = torch.load(cp, map_location="cpu", weights_only=False)
                ep = int(meta.get("epoch", 0))
                if ep > latest_epoch:
                    latest_epoch = ep
                    latest_ckpt = cp
            except Exception:
                continue
        if latest_ckpt is None:
            print("[Resume] No checkpoint found — starting from scratch.")
        else:
            ckpt_data = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            head.load_state_dict(ckpt_data["head_state_dict"])
            if "optimizer_state_dict" in ckpt_data:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            # scheduler needs initial_lr on param groups when last_epoch > -1
            for pg in optimizer.param_groups:
                pg.setdefault("initial_lr", args.lr)
            resume_epoch = latest_epoch
            start_epoch = latest_epoch + 1
            best_val_loss = float(ckpt_data.get("val_loss", "inf"))
            print(f"[Resume] Loaded {latest_ckpt.name} | "
                  f"epoch={latest_epoch}, val_loss={best_val_loss:.4f} "
                  f"→ continuing from epoch {start_epoch}")

    # Build scheduler with last_epoch so LR is correct without calling step() in a loop
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=resume_epoch
    )

    # ── Training loop ─────────────────────────────────────────────────────
    log_path = out_dir / "train_log.csv"
    log_mode = "a" if (args.resume and log_path.exists()) else "w"
    with open(log_path, log_mode) as lf:
        if log_mode == "w":
            lf.write("epoch,train_loss,train_ang_deg,val_loss,val_ang_deg,lr\n")

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAng°':>10}  "
          f"{'ValLoss':>9}  {'ValAng°':>9}  {'LR':>9}")
    print("-" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_ang = train_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_ang = val_epoch(model, val_loader, device)
        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ang:>10.2f}  "
            f"{vl_loss:>9.4f}  {vl_ang:>9.2f}  {lr:>9.2e}"
        )

        with open(log_path, "a") as lf:
            lf.write(f"{epoch},{tr_loss:.6f},{tr_ang:.4f},{vl_loss:.6f},{vl_ang:.4f},{lr:.2e}\n")

        # Visual evaluation + best checkpoint at eval epochs and final epoch
        is_eval_epoch = (epoch % args.eval_every == 0 or epoch == args.epochs)
        if is_eval_epoch:
            visual_eval(
                model=model,
                val_samples=val_samples,
                epoch=epoch,
                out_dir=out_dir,
                device=device,
                input_size=args.input_size,
                patch_size=patch_size,
                n_images=args.eval_vis_n,
                rng_seed=args.seed,
            )
            # Save periodic checkpoint (always, with optimizer state for resume)
            p_ckpt = out_dir / f"normal_head_ep{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": vl_loss,
                    "val_ang_deg": vl_ang,
                },
                p_ckpt,
            )
            # Track best and save only at eval checkpoints
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                ckpt_path = out_dir / "best_normal_head.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "head_state_dict": head.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": vl_loss,
                        "val_ang_deg": vl_ang,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"          ↳ best checkpoint saved (val_loss={vl_loss:.4f})")
        else:
            # Still track best value silently without saving
            if vl_loss < best_val_loss:
                best_val_loss = vl_loss

    print(f"\n[DONE] Best val_loss={best_val_loss:.4f} | checkpoints → {out_dir.resolve()}")


if __name__ == "__main__":
    main()


# ─── Example run ──────────────────────────────────────────────────────────
# python /ocean/projects/cis220039p/mdt2/hguo7/Depth-Anything-V2/train_normal_head_dav2.py \
#     --data-dirs \
#         /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/train_train_val \
#         /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/campus_walk2_train_val \
#         /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/campus_wallk1_train_val \
#         /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/street_walk_day1_train_val \
#         /ocean/projects/cis220039p/mdt2/hguo7/GlassGuard_trainer/full_runs/street_walk_day2_train_val \
#     --da2-ckpt /ocean/projects/cis220039p/mdt2/hguo7/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
#     --output-dir /ocean/projects/cis220039p/mdt2/hguo7/Depth-Anything-V2/normal_head_out \
#     --epochs 80 \
#     --batch-size 32 \
#     --lr 1e-3 \
#     --num-workers 4 \
#     --input-size 518
