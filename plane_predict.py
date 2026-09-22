#!/usr/bin/env python3
"""Train and run a mask-conditioned plane predictor from RGB + SAM mask.

This script is designed for the dataset layout produced by
`build_glass_dataset.py`:
- one sample folder contains `sample.json`
- RGB path from `sample.json["files"]["rgb"]`
- SAM masks in `sample.json["files"]["sam_masks_dir"]`
- per-mask GT filled depth in `sample.json["files"]["depth_gt_filled_by_mask_dir"]`
- intrinsics K in `pose.json`

Model:
- Input: cropped ROI from RGB plus a mask channel (4-channel tensor)
- Output: plane normal (always), optional plane offset d, optional extents (w, h)

Notes:
- Normal regression is the primary reliable target from RGB+mask.
- Offset/size can be enabled, but are more ambiguous from monocular input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import models
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torchvision is required for this script. Please install torchvision in your environment."
    ) from e


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, dict) else {}


def _configure_torch_cache(cache_dir: Optional[str]) -> Optional[str]:
    """Point torch/torchvision cache to a writable path to avoid permission errors."""
    cdir = str(cache_dir).strip() if cache_dir is not None else ""
    if len(cdir) == 0:
        cdir = os.path.join(tempfile.gettempdir(), "torch_cache")
    try:
        os.makedirs(cdir, exist_ok=True)
    except Exception:
        return None
    if not os.access(cdir, os.W_OK):
        return None

    os.environ["TORCH_HOME"] = cdir
    torch.hub.set_dir(cdir)
    return cdir


def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    a = a[:3]
    n = float(np.linalg.norm(a))
    if n <= eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (a / n).astype(np.float32)


def canonicalize_plane_orientation(
    n: np.ndarray,
    d: Optional[float] = None,
    center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """Enforce a single normal direction convention.

    Preferred orientation is toward camera center (origin), i.e. n dot center <= 0.
    If center is unavailable or ambiguous, use z <= 0 as a fallback convention.
    """
    nn = normalize_vec(n)
    dd = None if d is None else float(d)

    flip = False
    if center is not None:
        c = np.asarray(center, dtype=np.float32).reshape(-1)
        if c.size >= 3:
            c = c[:3]
            dc = float(np.dot(nn, c))
            if abs(dc) > 1e-8:
                flip = dc > 0.0
            else:
                if abs(float(nn[2])) > 1e-8:
                    flip = float(nn[2]) > 0.0
                else:
                    k = int(np.argmax(np.abs(nn)))
                    flip = float(nn[k]) < 0.0
    else:
        if abs(float(nn[2])) > 1e-8:
            flip = float(nn[2]) > 0.0
        else:
            k = int(np.argmax(np.abs(nn)))
            flip = float(nn[k]) < 0.0

    if flip:
        nn = -nn
        if dd is not None:
            dd = -dd
    return nn.astype(np.float32), dd


def load_rgb(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return arr


def load_mask(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def backproject_mask_points(depth_m: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > 1e-6)
    if int(np.sum(valid)) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ys = ys[valid].astype(np.float32)
    xs = xs[valid].astype(np.float32)
    z = z[valid]

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    x = (xs - cx) * z / max(1e-8, fx)
    y = (ys - cy) * z / max(1e-8, fy)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def fit_plane_from_points(points_c: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, Tuple[float, float]]:
    """Fit plane in camera frame and estimate in-plane extents.

    Returns:
    - n: unit normal (camera frame), oriented to face camera
    - d: plane offset in n.x + d = 0
    - center: centroid in camera frame
    - (w, h): two in-plane extents in meters
    """
    p = np.asarray(points_c, dtype=np.float32).reshape(-1, 3)
    if p.shape[0] < 16:
        raise ValueError("Not enough points to fit plane")

    c = p.mean(axis=0).astype(np.float32)
    q = p - c[None, :]
    _, _, vh = np.linalg.svd(q, full_matrices=False)

    n = normalize_vec(vh[-1])
    # Face the camera center at origin: for visible surfaces, n dot c should be <= 0.
    if float(np.dot(n, c)) > 0.0:
        n = -n

    d = -float(np.dot(n, c))

    # Plane basis from first two principal directions.
    e0 = normalize_vec(vh[0])
    e1 = normalize_vec(vh[1])
    u = q @ e0.reshape(3, 1)
    v = q @ e1.reshape(3, 1)
    w = float(np.percentile(u, 98.0) - np.percentile(u, 2.0))
    h = float(np.percentile(v, 98.0) - np.percentile(v, 2.0))
    w = max(1e-3, w)
    h = max(1e-3, h)

    return n.astype(np.float32), float(d), c.astype(np.float32), (w, h)


def crop_with_mask(rgb: np.ndarray, mask: np.ndarray, out_size: int, pad_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        x0, y0, x1, y1 = 0, 0, w - 1, h - 1
    else:
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())

    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    padx = int(round(bw * float(max(0.0, pad_frac))))
    pady = int(round(bh * float(max(0.0, pad_frac))))

    xa = max(0, x0 - padx)
    xb = min(w, x1 + padx + 1)
    ya = max(0, y0 - pady)
    yb = min(h, y1 + pady + 1)

    rgb_crop = rgb[ya:yb, xa:xb, :]
    mask_crop = mask[ya:yb, xa:xb]

    rgb_out = np.asarray(
        Image.fromarray(rgb_crop).resize((out_size, out_size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    mask_out = np.asarray(
        Image.fromarray(mask_crop).resize((out_size, out_size), resample=Image.NEAREST),
        dtype=np.uint8,
    )
    mask_out = (mask_out > 0).astype(np.uint8)
    return rgb_out, mask_out


def resize_full_frame(rgb: np.ndarray, mask: np.ndarray, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resize full-frame RGB and mask to the model input size.

    This keeps global scene context and avoids ROI cropping around a mask.
    """
    rgb_out = np.asarray(
        Image.fromarray(rgb).resize((out_size, out_size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    mask_out = np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((out_size, out_size), resample=Image.NEAREST),
        dtype=np.uint8,
    )
    return rgb_out, (mask_out > 0).astype(np.uint8)


def rgb_mask_to_tensor(rgb_u8: np.ndarray, mask_u8: np.ndarray) -> torch.Tensor:
    rgb = rgb_u8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean[None, None, :]) / std[None, None, :]
    m = mask_u8.astype(np.float32)[..., None]
    x = np.concatenate([rgb, m], axis=2)
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x.astype(np.float32))


def angular_error_deg(pred_n: torch.Tensor, gt_n: torch.Tensor) -> torch.Tensor:
    cosv = torch.sum(F.normalize(pred_n, dim=1) * F.normalize(gt_n, dim=1), dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosv))


def angular_loss_directional(pred_n: torch.Tensor, gt_n: torch.Tensor) -> torch.Tensor:
    cosv = torch.sum(F.normalize(pred_n, dim=1) * F.normalize(gt_n, dim=1), dim=1)
    return (1.0 - cosv).mean()


def angular_error_deg_np(pred_n: np.ndarray, gt_n: np.ndarray) -> float:
    a = normalize_vec(pred_n)
    b = normalize_vec(gt_n)
    cosv = float(np.clip(float(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


@dataclass
class Entry:
    sample_name: str
    rgb_path: str
    mask_path: str
    gt_depth_path: str
    gt_plane_path: Optional[str]
    gt_plane_inline: Optional[Dict[str, object]]
    k: np.ndarray


class GlassPlaneDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        splits: Sequence[str],
        image_size: int = 224,
        crop_pad_frac: float = 0.2,
        jitter_brightness: float = 0.1,
        jitter_contrast: float = 0.1,
    ):
        self.image_size = int(image_size)
        self.crop_pad_frac = float(crop_pad_frac)
        self.jitter_brightness = float(max(0.0, jitter_brightness))
        self.jitter_contrast = float(max(0.0, jitter_contrast))

        self.entries: List[Entry] = []
        for split in splits:
            split_dir = os.path.join(dataset_root, split)
            if not os.path.isdir(split_dir):
                continue
            for name in sorted(os.listdir(split_dir)):
                sd = os.path.join(split_dir, name)
                if not os.path.isdir(sd):
                    continue
                sample_json = os.path.join(sd, "sample.json")
                if not os.path.exists(sample_json):
                    continue
                meta = read_json(sample_json)
                files = meta.get("files", {}) if isinstance(meta.get("files", {}), dict) else {}

                rgb_rel = str(files.get("rgb", "")).strip()
                sam_dir_rel = str(files.get("sam_masks_dir", "sam_masks")).strip()
                gt_dir_rel = str(files.get("depth_gt_filled_by_mask_dir", "depth_gt_filled_by_mask")).strip()
                pose_rel = str(files.get("pose", "pose.json")).strip() or "pose.json"
                plane_dir_rel = ""
                for kplane in [
                    "plane_gt_by_mask_dir",
                    "plane_params_by_mask_dir",
                    "plane_by_mask_dir",
                    "plane_params_dir",
                    "planes_dir",
                    "plane_dir",
                ]:
                    v = str(files.get(kplane, "")).strip()
                    if len(v) > 0:
                        plane_dir_rel = v
                        break

                rgb_path = os.path.join(sd, rgb_rel)
                sam_dir = os.path.join(sd, sam_dir_rel)
                gt_dir = os.path.join(sd, gt_dir_rel)
                pose_path = os.path.join(sd, pose_rel)
                plane_dir = os.path.join(sd, plane_dir_rel) if len(plane_dir_rel) > 0 else ""

                inline_planes = {}
                for km in ["planes_by_mask", "plane_gt_by_mask", "plane_params_by_mask", "per_mask_planes"]:
                    vv = meta.get(km, None)
                    if isinstance(vv, dict):
                        inline_planes = vv
                        break

                if not (os.path.exists(rgb_path) and os.path.isdir(sam_dir) and os.path.isdir(gt_dir) and os.path.exists(pose_path)):
                    continue

                pose = read_json(pose_path)
                k = np.asarray(pose.get("K", []), dtype=np.float32)
                if k.shape != (3, 3):
                    continue

                for mname in sorted(os.listdir(sam_dir)):
                    if not mname.lower().endswith(".png"):
                        continue
                    stem = os.path.splitext(mname)[0]
                    mask_path = os.path.join(sam_dir, mname)
                    gt_path = os.path.join(gt_dir, f"{stem}.npy")
                    if not os.path.exists(gt_path):
                        continue

                    plane_inline = inline_planes.get(stem, None)
                    plane_path = None
                    cands: List[str] = []
                    if isinstance(plane_dir, str) and len(plane_dir) > 0:
                        cands.append(os.path.join(plane_dir, f"{stem}.json"))
                    cands.extend(
                        [
                            os.path.join(sd, "plane_gt_by_mask", f"{stem}.json"),
                            os.path.join(sd, "plane_params_by_mask", f"{stem}.json"),
                            os.path.join(sd, "plane_by_mask", f"{stem}.json"),
                            os.path.join(sd, "planes_by_mask", f"{stem}.json"),
                            os.path.join(gt_dir, f"{stem}.json"),
                            os.path.join(sam_dir, f"{stem}.json"),
                        ]
                    )
                    for cp in cands:
                        if os.path.exists(cp):
                            plane_path = cp
                            break

                    self.entries.append(
                        Entry(
                            sample_name=name,
                            rgb_path=rgb_path,
                            mask_path=mask_path,
                            gt_depth_path=gt_path,
                            gt_plane_path=plane_path,
                            gt_plane_inline=plane_inline if isinstance(plane_inline, dict) else None,
                            k=k.copy(),
                        )
                    )

        if len(self.entries) == 0:
            raise RuntimeError("No training entries found. Check dataset_root and split layout.")

    def __len__(self) -> int:
        return len(self.entries)

    def _jitter_rgb(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32)
        if self.jitter_brightness > 0:
            b = 1.0 + random.uniform(-self.jitter_brightness, self.jitter_brightness)
            x = x * b
        if self.jitter_contrast > 0:
            c = 1.0 + random.uniform(-self.jitter_contrast, self.jitter_contrast)
            mean = x.mean(axis=(0, 1), keepdims=True)
            x = (x - mean) * c + mean
        return np.clip(x, 0, 255).astype(np.uint8)

    @staticmethod
    def _parse_plane_payload(d: Dict[str, object]) -> Optional[Tuple[np.ndarray, float, np.ndarray, Tuple[float, float]]]:
        normal = None
        offset = None
        center = None
        size = None

        for kn in ["normal_cam", "normal", "plane_normal", "n"]:
            v = d.get(kn, None)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
                normal = np.asarray(v[:3], dtype=np.float32)
                break

        if normal is None:
            pv = d.get("plane", None)
            if isinstance(pv, (list, tuple, np.ndarray)) and len(pv) >= 4:
                arr = np.asarray(pv, dtype=np.float32).reshape(-1)
                normal = arr[:3]
                offset = float(arr[3])

        for kd in ["offset_d", "offset", "d", "plane_offset", "plane_d"]:
            if kd in d:
                try:
                    offset = float(d[kd])
                    break
                except Exception:
                    pass

        for kc in ["center_cam", "center", "plane_center", "centroid"]:
            v = d.get(kc, None)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
                center = np.asarray(v[:3], dtype=np.float32)
                break

        for ks in ["size_wh_m", "size", "wh", "plane_size", "extent_wh_m"]:
            v = d.get(ks, None)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                size = (float(v[0]), float(v[1]))
                break

        if normal is None:
            return None

        if center is None:
            center = np.zeros((3,), dtype=np.float32)
        if offset is None:
            offset = -float(np.dot(normalize_vec(normal), center))
        if size is None:
            size = (1.0, 1.0)

        normal, offset = canonicalize_plane_orientation(normal, float(offset), center)
        return normal.astype(np.float32), float(offset), center.astype(np.float32), (max(1e-3, float(size[0])), max(1e-3, float(size[1])))

    @staticmethod
    def _load_plane_from_entry(e: Entry) -> Optional[Tuple[np.ndarray, float, np.ndarray, Tuple[float, float]]]:
        if isinstance(e.gt_plane_inline, dict):
            pp = GlassPlaneDataset._parse_plane_payload(e.gt_plane_inline)
            if pp is not None:
                return pp

        if isinstance(e.gt_plane_path, str) and len(e.gt_plane_path) > 0 and os.path.exists(e.gt_plane_path):
            jd = read_json(e.gt_plane_path)
            pp = GlassPlaneDataset._parse_plane_payload(jd)
            if pp is not None:
                return pp
        return None

    @staticmethod
    def build_target_from_entry(e: Entry) -> Optional[Dict[str, np.ndarray]]:
        rgb = load_rgb(e.rgb_path)
        mask = load_mask(e.mask_path)
        pgt = GlassPlaneDataset._load_plane_from_entry(e)
        if pgt is not None:
            n, d, center, (sw, sh) = pgt
            return {
                "rgb": rgb,
                "mask": mask.astype(np.uint8),
                "normal": n.astype(np.float32),
                "offset": np.array([float(d)], dtype=np.float32),
                "size": np.array([float(sw), float(sh)], dtype=np.float32),
                "center": center.astype(np.float32),
            }

        gt_depth = np.asarray(np.load(e.gt_depth_path), dtype=np.float32)

        if gt_depth.ndim != 2 or rgb.shape[:2] != gt_depth.shape[:2] or mask.shape != gt_depth.shape:
            return None

        gt_mask = (mask > 0) & np.isfinite(gt_depth) & (gt_depth > 1e-6)
        pts = backproject_mask_points(gt_depth, e.k, gt_mask.astype(np.uint8))
        if pts.shape[0] < 16:
            return None

        try:
            n, d, center, (sw, sh) = fit_plane_from_points(pts)
        except Exception:
            return None

        n, d = canonicalize_plane_orientation(n, d, center)

        return {
            "rgb": rgb,
            "mask": mask.astype(np.uint8),
            "normal": n.astype(np.float32),
            "offset": np.array([float(d)], dtype=np.float32),
            "size": np.array([float(sw), float(sh)], dtype=np.float32),
            "center": center.astype(np.float32),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Retry a few times to avoid occasional degenerate masks/depth.
        for _ in range(8):
            e = self.entries[index]
            t = self.build_target_from_entry(e)
            if t is None:
                index = random.randrange(0, len(self.entries))
                continue

            rgb = t["rgb"]
            mask = t["mask"]
            n = t["normal"]
            d = float(t["offset"][0])
            sw = float(t["size"][0])
            sh = float(t["size"][1])
            center = t["center"]

            rgb_aug = self._jitter_rgb(rgb)
            rgb_full, mask_full = resize_full_frame(rgb_aug, mask.astype(np.uint8), out_size=self.image_size)
            x = rgb_mask_to_tensor(rgb_full, mask_full)

            return {
                "x": x,
                "normal": torch.from_numpy(n.astype(np.float32)),
                "offset": torch.tensor([float(d)], dtype=torch.float32),
                "size": torch.tensor([float(sw), float(sh)], dtype=torch.float32),
                "center": torch.from_numpy(center.astype(np.float32)),
            }

        raise RuntimeError("Failed to sample a valid training item after retries")


class PlaneParamNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        predict_offset: bool = False,
        predict_size: bool = False,
    ):
        super().__init__()
        self.predict_offset = bool(predict_offset)
        self.predict_size = bool(predict_size)

        if backbone == "vit_b16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            build_fn = models.vit_b_16
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            build_fn = models.resnet34
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            build_fn = models.resnet18

        try:
            net = build_fn(weights=weights)
        except Exception as e:
            if pretrained:
                print(
                    f"[WARN] failed to load pretrained {backbone} weights ({type(e).__name__}: {e}); "
                    "falling back to random init."
                )
                net = build_fn(weights=None)
            else:
                raise

        if backbone == "vit_b16":
            old_proj = net.conv_proj
            net.conv_proj = nn.Conv2d(
                4,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=(old_proj.bias is not None),
            )
            with torch.no_grad():
                net.conv_proj.weight[:, :3, :, :] = old_proj.weight
                net.conv_proj.weight[:, 3:4, :, :] = old_proj.weight[:, :1, :, :]
                if old_proj.bias is not None:
                    net.conv_proj.bias.copy_(old_proj.bias)
            feat_dim = int(getattr(net, "hidden_dim", 768))
            net.heads = nn.Identity()
        else:
            old_conv = net.conv1
            net.conv1 = nn.Conv2d(
                4,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                net.conv1.weight[:, :3, :, :] = old_conv.weight
                net.conv1.weight[:, 3:4, :, :] = old_conv.weight[:, :1, :, :]
            feat_dim = int(net.fc.in_features)
            net.fc = nn.Identity()
        self.backbone = net

        self.head_normal = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 3),
        )

        if self.predict_offset:
            self.head_offset = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 2, 1),
            )
        else:
            self.head_offset = None

        if self.predict_size:
            self.head_size = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 2, 2),
            )
        else:
            self.head_size = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        n = F.normalize(self.head_normal(f), dim=1)
        out: Dict[str, torch.Tensor] = {"normal": n}
        if self.head_offset is not None:
            out["offset"] = self.head_offset(f)
        if self.head_size is not None:
            # Predict positive extents in meters.
            out["size"] = F.softplus(self.head_size(f)) + 1e-4
        return out


def _load_model_from_checkpoint(checkpoint: str, device: torch.device) -> Tuple[PlaneParamNet, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = PlaneParamNet(
        backbone=str(cfg.get("backbone", "resnet18")),
        pretrained=False,
        predict_offset=bool(cfg.get("predict_offset", False)),
        predict_size=bool(cfg.get("predict_size", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cfg


def run_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        try:
            dev_idx = int(torch.cuda.current_device())
            dev_name = torch.cuda.get_device_name(dev_idx)
            print(f"[device] using cuda:{dev_idx} ({dev_name})")
        except Exception:
            print("[device] using CUDA")
    else:
        print("[device] CUDA not available, training on CPU")

    cache_hint = _configure_torch_cache(getattr(args, "torch_cache_dir", ""))
    if bool(args.pretrained_backbone):
        if cache_hint is not None:
            print(f"[cache] TORCH_HOME={cache_hint}")
        else:
            print("[cache][WARN] could not configure writable torch cache; pretrained download may fail")

    train_ds = GlassPlaneDataset(
        dataset_root=args.dataset_root,
        splits=["train"],
        image_size=args.image_size,
        crop_pad_frac=args.crop_pad_frac,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
    )

    val_split = "val" if os.path.isdir(os.path.join(args.dataset_root, "val")) else "train"
    val_ds = GlassPlaneDataset(
        dataset_root=args.dataset_root,
        splits=[val_split],
        image_size=args.image_size,
        crop_pad_frac=args.crop_pad_frac,
        jitter_brightness=0.0,
        jitter_contrast=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=cuda_ok,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=cuda_ok,
        drop_last=False,
    )

    model = PlaneParamNet(
        backbone=args.backbone,
        pretrained=bool(args.pretrained_backbone),
        predict_offset=bool(args.predict_offset),
        predict_size=bool(args.predict_size),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_ang_sum = 0.0
        n_train = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            gt_n = batch["normal"].to(device, non_blocking=True)
            gt_d = batch["offset"].to(device, non_blocking=True)
            gt_s = batch["size"].to(device, non_blocking=True)

            pred = model(x)
            loss_n = angular_loss_directional(pred["normal"], gt_n)
            loss = args.w_normal * loss_n

            if args.predict_offset:
                loss_d = F.smooth_l1_loss(pred["offset"], gt_d)
                loss = loss + args.w_offset * loss_d

            if args.predict_size:
                # log-space is stabler for size regression.
                ps = torch.log(pred["size"].clamp_min(1e-4))
                gs = torch.log(gt_s.clamp_min(1e-4))
                loss_s = F.smooth_l1_loss(ps, gs)
                loss = loss + args.w_size * loss_s

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            with torch.no_grad():
                ang = angular_error_deg(pred["normal"], gt_n).mean().item()
            train_loss_sum += float(loss.item()) * x.shape[0]
            train_ang_sum += float(ang) * x.shape[0]
            n_train += int(x.shape[0])

        model.eval()
        val_loss_sum = 0.0
        val_ang_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                gt_n = batch["normal"].to(device, non_blocking=True)
                gt_d = batch["offset"].to(device, non_blocking=True)
                gt_s = batch["size"].to(device, non_blocking=True)

                pred = model(x)
                loss_n = angular_loss_directional(pred["normal"], gt_n)
                loss = args.w_normal * loss_n

                if args.predict_offset:
                    loss_d = F.smooth_l1_loss(pred["offset"], gt_d)
                    loss = loss + args.w_offset * loss_d

                if args.predict_size:
                    ps = torch.log(pred["size"].clamp_min(1e-4))
                    gs = torch.log(gt_s.clamp_min(1e-4))
                    loss_s = F.smooth_l1_loss(ps, gs)
                    loss = loss + args.w_size * loss_s

                ang = angular_error_deg(pred["normal"], gt_n).mean().item()
                val_loss_sum += float(loss.item()) * x.shape[0]
                val_ang_sum += float(ang) * x.shape[0]
                n_val += int(x.shape[0])

        train_loss = train_loss_sum / max(1, n_train)
        train_ang = train_ang_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        val_ang = val_ang_sum / max(1, n_val)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.5f} train_ang={train_ang:.3f}deg "
            f"val_loss={val_loss:.5f} val_ang={val_ang:.3f}deg"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "args": vars(args),
            "val_loss": float(val_loss),
            "val_ang_deg": float(val_ang),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

        if val_loss < best_val:
            best_val = float(val_loss)
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))


@torch.no_grad()
def run_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model_from_checkpoint(args.checkpoint, device)

    image_size = int(cfg.get("image_size", 224))
    rgb = load_rgb(args.rgb)
    mask = load_mask(args.mask)
    if rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and mask must have the same spatial size")

    rgb_full, mask_full = resize_full_frame(rgb, mask, out_size=image_size)
    x = rgb_mask_to_tensor(rgb_full, mask_full).unsqueeze(0).to(device)

    pred = model(x)
    n = pred["normal"][0].detach().cpu().numpy().astype(np.float32)
    n, _ = canonicalize_plane_orientation(n, None, None)

    out: Dict[str, object] = {
        "normal_cam": [float(n[0]), float(n[1]), float(n[2])],
    }

    d_val = None
    if "offset" in pred:
        d_val = float(pred["offset"][0, 0].detach().cpu().item())
        out["offset_d"] = d_val

    if "size" in pred:
        s = pred["size"][0].detach().cpu().numpy().astype(np.float32)
        out["size_wh_m"] = [float(s[0]), float(s[1])]

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

    if args.render_depth_npy:
        if d_val is None:
            raise ValueError("Checkpoint does not predict offset. Re-train with --predict_offset for depth rendering.")
        if args.intrinsics_json is None:
            raise ValueError("--intrinsics_json is required to render depth")

        k = np.asarray(read_json(args.intrinsics_json).get("K", []), dtype=np.float32)
        if k.shape != (3, 3):
            raise ValueError("intrinsics_json must contain key 'K' with shape 3x3")

        h, w = mask.shape
        depth = render_plane_depth(h, w, k, n, d_val, max_depth_m=args.max_depth_m)
        np.save(args.render_depth_npy, depth.astype(np.float32))

        if args.render_depth_png:
            vis = depth_to_vis(depth)
            Image.fromarray(vis).save(args.render_depth_png)


def _draw_normal_arrow(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    normal: np.ndarray,
    color: Tuple[int, int, int],
    arrow_len: float,
    width: int,
) -> Tuple[float, float]:
    v = np.asarray(normal[:2], dtype=np.float32)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-8:
        v = np.array([0.0, -1.0], dtype=np.float32)
        nv = 1.0
    v = v / nv

    ex = float(cx + arrow_len * float(v[0]))
    ey = float(cy + arrow_len * float(v[1]))
    draw.line([(cx, cy), (ex, ey)], fill=color, width=width)

    # Arrow head.
    theta = math.atan2(float(v[1]), float(v[0]))
    a = 0.45
    hl = max(8.0, 0.18 * float(arrow_len))
    p1 = (ex - hl * math.cos(theta - a), ey - hl * math.sin(theta - a))
    p2 = (ex - hl * math.cos(theta + a), ey - hl * math.sin(theta + a))
    draw.line([(ex, ey), p1], fill=color, width=width)
    draw.line([(ex, ey), p2], fill=color, width=width)
    return ex, ey


def _write_ply_ascii_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    p = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    c = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    n = int(min(p.shape[0], c.shape[0]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = p[i]
            r, g, b = c[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _make_arrow_3d_points(
    start: np.ndarray,
    direction: np.ndarray,
    length_m: float,
    color: Tuple[int, int, int],
    shaft_samples: int = 60,
    head_samples: int = 20,
    head_ratio: float = 0.22,
    head_angle_deg: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(start, dtype=np.float32).reshape(3)
    d = normalize_vec(np.asarray(direction, dtype=np.float32).reshape(3))
    L = float(max(1e-4, length_m))

    end = s + d * L
    t = np.linspace(0.0, 1.0, int(max(2, shaft_samples)), dtype=np.float32)
    shaft = s[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]

    # Build a stable perpendicular basis for arrow head branches.
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(d, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(d, up).astype(np.float32)
    un = float(np.linalg.norm(u))
    if un <= 1e-8:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        u = u / un

    head_len = L * float(max(0.05, min(0.6, head_ratio)))
    theta = math.radians(float(head_angle_deg))
    b1 = -d * math.cos(theta) + u * math.sin(theta)
    b2 = -d * math.cos(theta) - u * math.sin(theta)
    p1 = end + b1 * head_len
    p2 = end + b2 * head_len

    th = np.linspace(0.0, 1.0, int(max(2, head_samples)), dtype=np.float32)
    h1 = end[None, :] * (1.0 - th[:, None]) + p1[None, :] * th[:, None]
    h2 = end[None, :] * (1.0 - th[:, None]) + p2[None, :] * th[:, None]

    pts = np.concatenate([shaft, h1, h2], axis=0).astype(np.float32)
    col = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 3), (pts.shape[0], 1))
    return pts, col


def draw_normal_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    pred_n: np.ndarray,
    gt_n: Optional[np.ndarray] = None,
    angle_deg: Optional[float] = None,
    arrow_len_px: float = 110.0,
) -> np.ndarray:
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    ys, xs = np.where(m)
    if ys.size > 0:
        cx = float(xs.mean())
        cy = float(ys.mean())
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 255, 0), width=2)
    else:
        cx = float(w * 0.5)
        cy = float(h * 0.5)

    _draw_normal_arrow(draw, cx, cy, np.asarray(pred_n, dtype=np.float32), (255, 60, 60), arrow_len_px, 4)
    if gt_n is not None:
        _draw_normal_arrow(draw, cx, cy, np.asarray(gt_n, dtype=np.float32), (50, 220, 80), arrow_len_px, 4)

    txt1 = f"pred n=({float(pred_n[0]):+.3f},{float(pred_n[1]):+.3f},{float(pred_n[2]):+.3f})"
    draw.text((12, 12), txt1, fill=(255, 80, 80))
    y2 = 28
    if gt_n is not None:
        txt2 = f"gt   n=({float(gt_n[0]):+.3f},{float(gt_n[1]):+.3f},{float(gt_n[2]):+.3f})"
        draw.text((12, y2), txt2, fill=(80, 255, 120))
        y2 += 16
    if angle_deg is not None:
        draw.text((12, y2), f"ang_err={float(angle_deg):.2f} deg", fill=(255, 255, 255))

    return np.asarray(img, dtype=np.uint8)


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    model, cfg = _load_model_from_checkpoint(args.checkpoint, device)
    image_size = int(cfg.get("image_size", 224))
    split = str(args.split).strip().lower()
    if not os.path.isdir(os.path.join(args.dataset_root, split)):
        split = "train"

    ds = GlassPlaneDataset(
        dataset_root=args.dataset_root,
        splits=[split],
        image_size=image_size,
        crop_pad_frac=float(cfg.get("crop_pad_frac", 0.2)),
        jitter_brightness=0.0,
        jitter_contrast=0.0,
    )

    n_avail = len(ds.entries)
    n_pick = min(max(1, int(args.num_samples)), n_avail)
    picks = random.sample(range(n_avail), k=n_pick)

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "normal_overlays")
    os.makedirs(vis_dir, exist_ok=True)
    ply_dir = os.path.join(args.out_dir, "normal_ply")
    os.makedirs(ply_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    angs: List[float] = []
    kept = 0
    for idx_out, idx in enumerate(picks):
        e = ds.entries[idx]
        t = ds.build_target_from_entry(e)
        if t is None:
            continue

        rgb = t["rgb"]
        mask = t["mask"]
        gt_n = t["normal"]

        rgb_full, mask_full = resize_full_frame(rgb, mask, out_size=image_size)
        x = rgb_mask_to_tensor(rgb_full, mask_full).unsqueeze(0).to(device)
        pred = model(x)
        pred_n = pred["normal"][0].detach().cpu().numpy().astype(np.float32)
        pred_n, _ = canonicalize_plane_orientation(pred_n, None, None)

        ang = angular_error_deg_np(pred_n, gt_n)
        angs.append(float(ang))

        stem = os.path.splitext(os.path.basename(e.mask_path))[0]
        out_name = f"{idx_out:03d}_{e.sample_name}_{stem}.png"
        vis = draw_normal_overlay(
            rgb=rgb,
            mask=mask,
            pred_n=pred_n,
            gt_n=gt_n,
            angle_deg=ang,
            arrow_len_px=float(args.arrow_len_px),
        )
        Image.fromarray(vis).save(os.path.join(vis_dir, out_name))

        # 3D PLY visualization in camera frame: GT plane points + GT/pred arrows.
        gt_depth = np.asarray(np.load(e.gt_depth_path), dtype=np.float32)
        gt_mask_3d = (mask > 0) & np.isfinite(gt_depth) & (gt_depth > 1e-6)
        plane_pts = backproject_mask_points(gt_depth, e.k, gt_mask_3d.astype(np.uint8))
        if plane_pts.shape[0] > int(args.ply_plane_max_points):
            sel = np.random.choice(plane_pts.shape[0], size=int(args.ply_plane_max_points), replace=False)
            plane_pts = plane_pts[sel]

        center = np.asarray(t.get("center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)
        if plane_pts.shape[0] > 0:
            center = plane_pts.mean(axis=0).astype(np.float32)

        gt_arrow_pts, gt_arrow_col = _make_arrow_3d_points(
            start=center,
            direction=gt_n,
            length_m=float(args.arrow_len_m),
            color=(50, 220, 80),
        )
        pred_arrow_pts, pred_arrow_col = _make_arrow_3d_points(
            start=center,
            direction=pred_n,
            length_m=float(args.arrow_len_m),
            color=(255, 60, 60),
        )

        if plane_pts.shape[0] > 0:
            plane_col = np.tile(np.array([[80, 180, 255]], dtype=np.uint8), (plane_pts.shape[0], 1))
            xyz = np.concatenate([plane_pts.astype(np.float32), gt_arrow_pts, pred_arrow_pts], axis=0)
            col = np.concatenate([plane_col, gt_arrow_col, pred_arrow_col], axis=0)
        else:
            xyz = np.concatenate([gt_arrow_pts, pred_arrow_pts], axis=0)
            col = np.concatenate([gt_arrow_col, pred_arrow_col], axis=0)

        ply_name = out_name.replace(".png", ".ply")
        _write_ply_ascii_xyzrgb(os.path.join(ply_dir, ply_name), xyz, col)

        rows.append(
            {
                "sample_name": e.sample_name,
                "mask_name": os.path.basename(e.mask_path),
                "pred_normal_cam": [float(pred_n[0]), float(pred_n[1]), float(pred_n[2])],
                "gt_normal_cam": [float(gt_n[0]), float(gt_n[1]), float(gt_n[2])],
                "angular_error_deg": float(ang),
                "overlay_png": os.path.join("normal_overlays", out_name),
                "normal_ply": os.path.join("normal_ply", ply_name),
            }
        )
        kept += 1

    if len(angs) == 0:
        raise RuntimeError("No valid eval samples found after filtering")

    arr = np.asarray(angs, dtype=np.float32)
    summary = {
        "split": split,
        "requested_num_samples": int(args.num_samples),
        "evaluated_num_samples": int(kept),
        "mean_ang_deg": float(arr.mean()),
        "median_ang_deg": float(np.median(arr)),
        "p90_ang_deg": float(np.percentile(arr, 90.0)),
        "acc_at_5deg": float(np.mean(arr <= 5.0)),
        "acc_at_10deg": float(np.mean(arr <= 10.0)),
        "acc_at_15deg": float(np.mean(arr <= 15.0)),
        "overlay_dir": "normal_overlays",
        "normal_ply_dir": "normal_ply",
        "samples": rows,
    }

    with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def render_plane_depth(h: int, w: int, k: np.ndarray, n: np.ndarray, d: float, max_depth_m: float) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    rays = np.stack([(xx - cx) / max(1e-8, fx), (yy - cy) / max(1e-8, fy), np.ones_like(xx)], axis=-1)
    nn = normalize_vec(np.asarray(n, dtype=np.float32))
    denom = rays @ nn.reshape(3, 1)
    denom = denom[..., 0]

    z = np.zeros((h, w), dtype=np.float32)
    valid = np.abs(denom) > 1e-8
    t = np.zeros((h, w), dtype=np.float32)
    t[valid] = -float(d) / denom[valid]
    valid &= np.isfinite(t) & (t > 1e-6) & (t <= float(max(1e-3, max_depth_m)))
    z[valid] = t[valid]
    return z


def depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 1e-6)
    out = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    lo = float(np.percentile(d[valid], 2.0))
    hi = float(np.percentile(d[valid], 98.0))
    if hi <= lo + 1e-8:
        hi = lo + 1.0
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    g = (t * 255.0).astype(np.uint8)
    out[..., 0] = g
    out[..., 1] = g
    out[..., 2] = g
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train/predict mask-conditioned plane parameters from RGB")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train plane predictor")
    tr.add_argument("--dataset_root", type=str, required=True, help="Root folder with train/val sample dirs")
    tr.add_argument("--out_dir", type=str, default="runs_plane_rgb_mask")
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch_size", type=int, default=24)
    tr.add_argument("--num_workers", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--image_size", type=int, default=224)
    tr.add_argument("--crop_pad_frac", type=float, default=0.2)
    tr.add_argument("--jitter_brightness", type=float, default=0.12)
    tr.add_argument("--jitter_contrast", type=float, default=0.12)
    tr.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "vit_b16"])
    tr.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument("--predict_offset", action=argparse.BooleanOptionalAction, default=False)
    tr.add_argument("--predict_size", action=argparse.BooleanOptionalAction, default=False)
    tr.add_argument("--w_normal", type=float, default=1.0)
    tr.add_argument("--w_offset", type=float, default=0.2)
    tr.add_argument("--w_size", type=float, default=0.1)
    tr.add_argument(
        "--torch_cache_dir",
        type=str,
        default="",
        help="Writable cache directory for torchvision pretrained weights (defaults to /tmp/torch_cache)",
    )

    pr = sub.add_parser("predict", help="Predict plane parameters for one RGB+mask")
    pr.add_argument("--checkpoint", type=str, required=True)
    pr.add_argument("--rgb", type=str, required=True)
    pr.add_argument("--mask", type=str, required=True)
    pr.add_argument("--output_json", type=str, default="")
    pr.add_argument("--intrinsics_json", type=str, default=None, help="JSON file with key K (3x3), optional")
    pr.add_argument("--render_depth_npy", type=str, default="", help="Optional output .npy plane depth")
    pr.add_argument("--render_depth_png", type=str, default="", help="Optional output visualization .png")
    pr.add_argument("--max_depth_m", type=float, default=15.0)

    ev = sub.add_parser("eval", help="Evaluate on random samples and save normal overlays")
    ev.add_argument("--checkpoint", type=str, required=True)
    ev.add_argument("--dataset_root", type=str, required=True)
    ev.add_argument("--split", type=str, default="val", choices=["train", "val", "vis"])
    ev.add_argument("--num_samples", type=int, default=20)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--arrow_len_px", type=float, default=110.0)
    ev.add_argument("--arrow_len_m", type=float, default=0.6)
    ev.add_argument("--ply_plane_max_points", type=int, default=12000)
    ev.add_argument("--out_dir", type=str, default="eval_plane_rgb_mask")

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "predict":
        if len(str(args.output_json).strip()) == 0:
            args.output_json = None
        if len(str(args.render_depth_npy).strip()) == 0:
            args.render_depth_npy = None
        if len(str(args.render_depth_png).strip()) == 0:
            args.render_depth_png = None
        run_predict(args)
    elif args.cmd == "eval":
        run_eval(args)
    else:  # pragma: no cover
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
# python train_plane_from_rgb_mask.py train --dataset_root glass_dataset_out --out_dir runs_plane_rgb_mask --epochs 20 --batch_size 24
# python train_plane_from_rgb_mask.py eval \
#   --checkpoint runs_plane_rgb_mask/best.pt \
#   --dataset_root glass_dataset_out \
#   --split val \
#   --num_samples 20 \
#   --seed 24 \
#   --out_dir eval_plane_rgb_mask_20_new