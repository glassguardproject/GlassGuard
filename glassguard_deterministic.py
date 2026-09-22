#!/usr/bin/env python3
"""Deterministic plane fitting from scene PLY + SAM masks.

Workflow:
1) Load world scene cloud from --scene_ply.
2) Pick one SAM mask folder from --sam_mask_root (random but seeded),
   or use --mask_folder if provided.
3) For each mask in that folder:
   - project scene cloud into that frame using poses.csv + intrinsics
   - keep points inside mask (front-most points per pixel)
   - run N random normal hypotheses (default 5)
   - fit bounded plane for each try and save visualization PLY

Visualization PLY colors:
- scene (subsampled): gray
- masked scene points: green
- fitted plane patch: blue
- normal arrow points: red
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from plane_predict import (
        _load_model_from_checkpoint,
        canonicalize_plane_orientation,
        resize_full_frame,
        rgb_mask_to_tensor,
    )
except Exception:  # pragma: no cover
    _load_model_from_checkpoint = None
    canonicalize_plane_orientation = None
    resize_full_frame = None
    rgb_mask_to_tensor = None


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def normalize_vec(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    a = a[:3]
    n = float(np.linalg.norm(a))
    if n <= eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (a / n).astype(np.float32)


def angular_error_deg_unsigned(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize_vec(a)
    bb = normalize_vec(b)
    cosv = float(np.clip(abs(float(np.dot(aa, bb))), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def rotate_vec_about_axis_deg(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotate vector around an arbitrary axis by deg using Rodrigues formula."""
    vv = normalize_vec(v)
    aa = normalize_vec(axis)
    th = float(np.deg2rad(float(deg)))
    c = float(np.cos(th))
    s = float(np.sin(th))
    term1 = vv * c
    term2 = np.cross(aa, vv).astype(np.float32) * s
    term3 = aa * float(np.dot(aa, vv)) * (1.0 - c)
    return normalize_vec(term1 + term2 + term3)


def normal_match_tier(
    cand_normal: np.ndarray,
    pred_normal: Optional[np.ndarray],
    left_right_axis: np.ndarray,
    direct_tol_deg: float,
    right_angle_tol_deg: float = 10.0,
) -> int:
    """Return match priority tier: 2=direct, 1=left/right 90 fallback, 0=no match."""
    if pred_normal is None:
        return 0

    a_direct = angular_error_deg_unsigned(cand_normal, pred_normal)
    if float(a_direct) <= float(direct_tol_deg):
        return 2

    pred_lr_plus = rotate_vec_about_axis_deg(pred_normal, left_right_axis, +90.0)
    pred_lr_minus = rotate_vec_about_axis_deg(pred_normal, left_right_axis, -90.0)
    a_lr_plus = angular_error_deg_unsigned(cand_normal, pred_lr_plus)
    a_lr_minus = angular_error_deg_unsigned(cand_normal, pred_lr_minus)
    if min(float(a_lr_plus), float(a_lr_minus)) <= float(right_angle_tol_deg):
        return 1
    return 0


def uniform_seed_indices(num_points: int, max_seeds: int) -> np.ndarray:
    n = int(max(0, num_points))
    m = int(max(1, max_seeds))
    if n <= m:
        return np.arange(n, dtype=np.int64)
    step = float(n - 1) / float(m - 1)
    idx = [int(round(i * step)) for i in range(m)]
    return np.asarray(sorted(set(idx)), dtype=np.int64)


class PlaneNormalPredictor:
    def __init__(self, checkpoint: str):
        if torch is None:
            raise RuntimeError("torch is required for --normal_checkpoint, but torch import failed")
        if _load_model_from_checkpoint is None or resize_full_frame is None or rgb_mask_to_tensor is None:
            raise RuntimeError("plane_predict.py helpers are unavailable; cannot use --normal_checkpoint")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = _load_model_from_checkpoint(checkpoint, self.device)
        self.image_size = int(self.cfg.get("image_size", 224))

    @torch.no_grad()
    def predict_normal_cam(self, rgb_u8: np.ndarray, mask01: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        mask = (np.asarray(mask01, dtype=np.uint8) > 0).astype(np.uint8)
        rgb_full, mask_full = resize_full_frame(rgb, mask, out_size=self.image_size)
        x = rgb_mask_to_tensor(rgb_full, mask_full).unsqueeze(0).to(self.device)
        pred = self.model(x)
        n = pred["normal"][0].detach().cpu().numpy().astype(np.float32)
        if canonicalize_plane_orientation is not None:
            n, _ = canonicalize_plane_orientation(n, None, None)
        return normalize_vec(n)


def choose_guided_try(
    per_try: List[Dict[str, object]],
    rng: np.random.Generator,
    target_coverage_percent: float,
    max_angle_deg: float,
) -> Optional[Dict[str, object]]:
    ok_tries = [r for r in per_try if r.get("status") == "ok"]
    if len(ok_tries) == 0:
        return None

    eligible = []
    for r in ok_tries:
        cov = float(r.get("coverage_percent", 0.0))
        tier = int(r.get("normal_match_tier", 0))
        ang = float(r.get("pred_angle_deg", 180.0))
        if cov >= float(target_coverage_percent) and (tier > 0 or ang <= float(max_angle_deg)):
            eligible.append(r)

    if len(eligible) > 0:
        # Prioritize direct matches over 90-degree fallback matches.
        direct = [r for r in eligible if int(r.get("normal_match_tier", 0)) == 2]
        pool = direct if len(direct) > 0 else eligible
        pick = pool[int(rng.integers(0, len(pool)))]
        pick["selected"] = True
        pick["selection_reason"] = "guided_random_direct" if len(direct) > 0 else "guided_random_right_angle"
        return pick

    # Fallback: prefer higher coverage, then smaller angular error.
    pick = max(
        ok_tries,
        key=lambda r: (
            int(r.get("normal_match_tier", 0)),
            float(r.get("coverage_percent", 0.0)),
            -float(r.get("pred_angle_deg", 180.0)),
        ),
    )
    pick["selected"] = True
    pick["selection_reason"] = "fallback_best_coverage"
    return pick


def write_ply_xyzrgb_ascii(path: str, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    if xyz.shape[0] != rgb_u8.shape[0]:
        raise ValueError("xyz and rgb must have same number of points")
    n = int(xyz.shape[0])
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
        for p, c in zip(xyz, rgb_u8):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _quat_xyzw_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.eye(3, dtype=np.float32)
    q = q / qn
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def load_poses_csv(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            t = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32)
            R = _quat_xyzw_to_R(float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]))
            poses[frame] = (t, R)
    return poses


def get_R_depth_to_pose(name: str) -> np.ndarray:
    if name == "identity":
        return np.eye(3, dtype=np.float32)
    if name == "ros_optical_to_link":
        return np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    if name == "ros_link_to_optical":
        return np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
    raise ValueError(f"Unknown depth_to_pose: {name}")


def parse_frame_id_from_folder(folder_name: str) -> int:
    patterns = [
        r"rgb_(\d+)",
        r"masks_(\d+)",
        r"depth_(\d+)",
        r"(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, folder_name)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not parse frame id from folder name: {folder_name}")


def strip_trailing_score_from_mask_name(mask_name: str) -> str:
    """Remove trailing numeric score from SAM mask stem, e.g. *_0.746 -> *."""
    m = re.match(r"^(.*)_([0-9]+(?:\.[0-9]+)?)$", mask_name)
    if m:
        return m.group(1)
    return mask_name


def list_mask_folders(mask_root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(mask_root):
        return out
    for n in sorted(os.listdir(mask_root)):
        p = os.path.join(mask_root, n)
        if os.path.isdir(p):
            try:
                has_mask = any(
                    fn.startswith("mask_") and fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
                    for fn in os.listdir(p)
                )
            except OSError:
                has_mask = False
            if has_mask:
                out.append(p)
    return out


def list_masks(mask_folder: str) -> List[str]:
    out: List[str] = []
    for n in sorted(os.listdir(mask_folder)):
        p = os.path.join(mask_folder, n)
        if not os.path.isfile(p):
            continue
        if not n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        if n.startswith("mask_"):
            out.append(p)
    return out


def load_mask01(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def dilate_mask(mask01: np.ndarray, dilation_px: int) -> np.ndarray:
    d = int(max(0, dilation_px))
    m = (np.asarray(mask01, dtype=np.uint8) > 0).astype(np.uint8)
    if d <= 0:
        return m
    ksz = 2 * d + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    md = cv2.dilate(m, kernel, iterations=1)
    return (md > 0).astype(np.uint8)


def save_rgb_mask_overlay_png(
    rgb_bgr: np.ndarray,
    mask01: np.ndarray,
    out_path: str,
    alpha: float = 0.45,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Save a reference image with binary mask overlaid on RGB frame."""
    if rgb_bgr is None or rgb_bgr.size == 0:
        return

    h, w = rgb_bgr.shape[:2]
    if mask01.shape[0] != h or mask01.shape[1] != w:
        mask01 = cv2.resize(mask01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask01 > 0
    overlay = rgb_bgr.copy()
    color = np.array(color_bgr, dtype=np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    overlay[mask_bool] = (
        (1.0 - a) * overlay[mask_bool].astype(np.float32) + a * color[None, :]
    ).astype(np.uint8)
    cv2.imwrite(out_path, overlay)


def save_pred_normal_overlay_png(
    rgb_bgr: np.ndarray,
    mask01: np.ndarray,
    pred_normal_cam: np.ndarray,
    out_path: str,
    arrow_len_px: float = 100.0,
) -> None:
    """Overlay mask box and predicted normal arrow on RGB frame."""
    if rgb_bgr is None or rgb_bgr.size == 0:
        return

    img = rgb_bgr.copy()
    h, w = img.shape[:2]
    m = np.asarray(mask01, dtype=np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    mb = m > 0

    ys, xs = np.nonzero(mb)
    if xs.size > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    else:
        cx = float(w * 0.5)
        cy = float(h * 0.5)

    v2 = np.asarray(pred_normal_cam[:2], dtype=np.float32)
    n2 = float(np.linalg.norm(v2))
    if n2 <= 1e-8:
        v2 = np.array([0.0, -1.0], dtype=np.float32)
        n2 = 1.0
    v2 = v2 / n2

    ex = float(cx + float(arrow_len_px) * float(v2[0]))
    ey = float(cy + float(arrow_len_px) * float(v2[1]))
    cv2.arrowedLine(
        img,
        (int(round(cx)), int(round(cy))),
        (int(round(ex)), int(round(ey))),
        (0, 0, 255),
        3,
        tipLength=0.18,
    )

    txt = f"pred n=({float(pred_normal_cam[0]):+.3f},{float(pred_normal_cam[1]):+.3f},{float(pred_normal_cam[2]):+.3f})"
    cv2.putText(img, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 30, 240), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def save_rgb_mask_pred_overlay_png(
    rgb_bgr: np.ndarray,
    mask01: np.ndarray,
    pred_normal_cam: Optional[np.ndarray],
    out_path: str,
    alpha: float = 0.45,
    mask_color_bgr: Tuple[int, int, int] = (0, 255, 0),
    arrow_len_px: float = 100.0,
) -> None:
    """Save one overlay image containing mask tint + bbox + predicted normal arrow."""
    if rgb_bgr is None or rgb_bgr.size == 0:
        return

    img = rgb_bgr.copy()
    h, w = img.shape[:2]
    m = np.asarray(mask01, dtype=np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    mb = m > 0

    # 1) Mask tint.
    a = float(np.clip(alpha, 0.0, 1.0))
    color = np.array(mask_color_bgr, dtype=np.float32)
    img[mb] = ((1.0 - a) * img[mb].astype(np.float32) + a * color[None, :]).astype(np.uint8)

    # 2) Bounding rectangle.
    ys, xs = np.nonzero(mb)
    if xs.size > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx = float(xs.mean())
        cy = float(ys.mean())
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
    else:
        cx = float(w * 0.5)
        cy = float(h * 0.5)

    # 3) Predicted normal arrow.
    if pred_normal_cam is not None:
        v2 = np.asarray(pred_normal_cam[:2], dtype=np.float32)
        n2 = float(np.linalg.norm(v2))
        if n2 <= 1e-8:
            v2 = np.array([0.0, -1.0], dtype=np.float32)
        else:
            v2 = v2 / n2

        ex = float(cx + float(arrow_len_px) * float(v2[0]))
        ey = float(cy + float(arrow_len_px) * float(v2[1]))
        cv2.arrowedLine(
            img,
            (int(round(cx)), int(round(cy))),
            (int(round(ex)), int(round(ey))),
            (180, 60, 255),
            3,
            tipLength=0.18,
        )
        txt = (
            f"pred n=({float(pred_normal_cam[0]):+.3f},"
            f"{float(pred_normal_cam[1]):+.3f},"
            f"{float(pred_normal_cam[2]):+.3f})"
        )
        cv2.putText(img, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 60, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


def save_normal_tries_ply(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    mask_points_world: np.ndarray,
    all_normals_arrows_world: np.ndarray,
    per_try: List[Dict[str, object]],
    selected_try: Optional[Dict[str, object]],
    selected_color: Tuple[int, int, int],
    out_ply: str,
) -> None:
    """Save one PLY with scene, masked points, and all sampled/selected normals.

    Colors:
    - scene cloud: original RGB (or gray fallback from caller)
    - masked points: green
    - all sampled try arrows: red
    - selected try arrow: green
    """
    xyz_parts: List[np.ndarray] = []
    rgb_parts: List[np.ndarray] = []
    closest_rec: Optional[Dict[str, object]] = None
    closest_ang = float("inf")

    sx = np.asarray(scene_xyz, dtype=np.float32).reshape(-1, 3)
    sc = np.asarray(scene_rgb, dtype=np.uint8).reshape(-1, 3)
    if sx.shape[0] > 0 and sc.shape[0] == sx.shape[0]:
        xyz_parts.append(sx)
        rgb_parts.append(sc)

    mp = np.asarray(mask_points_world, dtype=np.float32).reshape(-1, 3)
    if mp.shape[0] > 0:
        mcol = np.tile(np.array([[60, 255, 60]], dtype=np.uint8), (mp.shape[0], 1))
        xyz_parts.append(mp)
        rgb_parts.append(mcol)

    an = np.asarray(all_normals_arrows_world, dtype=np.float32).reshape(-1, 3)
    if an.shape[0] > 0:
        # All pre-sampled normals before random hypothesis sampling.
        acol = np.tile(np.array([[255, 120, 60]], dtype=np.uint8), (an.shape[0], 1))
        xyz_parts.append(an)
        rgb_parts.append(acol)

    for rec in per_try:
        if rec.get("status") != "ok":
            continue

        pa = rec.get("pred_angle_deg", None)
        if pa is not None:
            ang = float(pa)
            if ang < closest_ang:
                closest_ang = ang
                closest_rec = rec

        arr = np.asarray(rec.get("arrow_world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue

        is_selected = bool(selected_try is rec)
        if is_selected:
            col = np.tile(np.array([[selected_color[0], selected_color[1], selected_color[2]]], dtype=np.uint8), (arr.shape[0], 1))
        else:
            col = np.tile(np.array([[255, 60, 60]], dtype=np.uint8), (arr.shape[0], 1))
        xyz_parts.append(arr)
        rgb_parts.append(col)

    # Add an emphasized arrow for the normal closest to predictor: purple and longer.
    if closest_rec is not None:
        c = np.asarray(closest_rec.get("center_fit", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        n = np.asarray(closest_rec.get("normal_fit", [0.0, 0.0, 1.0]), dtype=np.float32).reshape(3)
        long_arrow = build_normal_arrow(center=c, normal=n, length=0.9, npts=120)
        long_col = np.tile(np.array([[200, 60, 255]], dtype=np.uint8), (long_arrow.shape[0], 1))
        xyz_parts.append(long_arrow)
        rgb_parts.append(long_col)

    if len(xyz_parts) == 0:
        return

    xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32)
    rgb = np.concatenate(rgb_parts, axis=0).astype(np.uint8)
    write_ply_xyzrgb_ascii(out_ply, xyz, rgb)


def project_world_points_to_image(
    points_world: np.ndarray,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_depth_to_pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_near: float,
    z_far: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to image pixels. Returns (u, v, z_optical)."""
    if points_world.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    xyz_pose = (R_world.T @ (points_world - t_world[None, :]).T).T
    xyz_opt = (R_depth_to_pose.T @ xyz_pose.T).T
    x = xyz_opt[:, 0]
    y = xyz_opt[:, 1]
    z = xyz_opt[:, 2]

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > max(1e-6, float(z_near)))
    if float(z_far) > 0:
        valid &= z <= float(z_far)
    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    xv, yv, zv = x[valid], y[valid], z[valid]
    u = np.rint(float(fx) * (xv / zv) + float(cx)).astype(np.int32)
    v = np.rint(float(fy) * (yv / zv) + float(cy)).astype(np.int32)
    return u, v, zv.astype(np.float32)


def save_plane_projection_overlay_png(
    rgb_bgr: np.ndarray,
    plane_points_world: np.ndarray,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_depth_to_pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_near: float,
    z_far: float,
    out_path: str,
) -> None:
    """Overlay projected 3D plane points on RGB using depth-based colors."""
    if rgb_bgr is None or rgb_bgr.size == 0 or plane_points_world.shape[0] == 0:
        return

    h, w = rgb_bgr.shape[:2]
    u, v, z = project_world_points_to_image(
        points_world=plane_points_world,
        t_world=t_world,
        R_world=R_world,
        R_depth_to_pose=R_depth_to_pose,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        z_near=z_near,
        z_far=z_far,
    )
    if u.shape[0] == 0:
        return

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        return
    u = u[in_img]
    v = v[in_img]
    z = z[in_img]

    # Near = warm colors, far = cool colors.
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    denom = max(1e-6, z_max - z_min)
    zn = np.clip((z - z_min) / denom, 0.0, 1.0)
    cmap_idx = np.clip((255.0 * (1.0 - zn)).astype(np.uint8), 0, 255)
    colors = cv2.applyColorMap(cmap_idx.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)

    out = rgb_bgr.copy()
    for px, py, c in zip(u, v, colors):
        cv2.circle(out, (int(px), int(py)), 1, (int(c[0]), int(c[1]), int(c[2])), -1)

    cv2.imwrite(out_path, out)


def load_ply_xyzrgb(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read PLY vertex xyz (+ optional rgb). Supports ascii and binary_little_endian."""
    type_map = {
        "char": np.int8,
        "uchar": np.uint8,
        "int8": np.int8,
        "uint8": np.uint8,
        "short": np.int16,
        "ushort": np.uint16,
        "int16": np.int16,
        "uint16": np.uint16,
        "int": np.int32,
        "uint": np.uint32,
        "int32": np.int32,
        "uint32": np.uint32,
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "float64": np.float64,
    }

    with open(path, "rb") as f:
        if f.readline().decode("ascii", errors="strict").strip() != "ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = ""
        vertex_count = 0
        in_vertex = False
        props: List[Tuple[str, str]] = []

        while True:
            line_raw = f.readline()
            if not line_raw:
                raise ValueError(f"{path}: unexpected EOF in header")
            line = line_raw.decode("ascii", errors="strict").strip()
            if line.startswith("format "):
                parts = line.split()
                fmt = parts[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = (len(parts) >= 3 and parts[1] == "vertex")
                if in_vertex:
                    vertex_count = int(parts[2])
                    props = []
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[2], parts[1]))
            elif line == "end_header":
                break

        names = [n for n, _ in props]
        if "x" not in names or "y" not in names or "z" not in names:
            raise ValueError(f"{path}: vertex props missing x/y/z")

        if fmt == "ascii":
            rows: List[List[float]] = []
            ncols = len(props)
            for _ in range(vertex_count):
                s = f.readline().decode("ascii", errors="ignore").strip()
                if not s:
                    continue
                vals = s.split()
                if len(vals) < ncols:
                    continue
                rows.append([float(vals[i]) for i in range(ncols)])
            if len(rows) == 0:
                return np.zeros((0, 3), np.float32), None
            arr = np.asarray(rows, dtype=np.float32)
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            xyz = arr[:, [ix, iy, iz]].astype(np.float32)
            rgb = None
            if all(c in names for c in ("red", "green", "blue")):
                ir, ig, ib = names.index("red"), names.index("green"), names.index("blue")
                rgb = np.clip(arr[:, [ir, ig, ib]], 0, 255).astype(np.uint8)
            return xyz, rgb

        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported format {fmt}")

        dt = np.dtype([(n, type_map[t]) for n, t in props])
        data = np.fromfile(f, dtype=dt, count=vertex_count)
        if data.size == 0:
            return np.zeros((0, 3), np.float32), None

        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        rgb = None
        if all(c in data.dtype.names for c in ("red", "green", "blue")):
            rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
        return xyz, rgb


def fit_plane_svd(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = xyz.mean(axis=0)
    x0 = xyz - c[None, :]
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    n = vh[-1]
    n = n / max(1e-9, float(np.linalg.norm(n)))
    return c.astype(np.float32), n.astype(np.float32)


def estimate_local_normal(points: np.ndarray, anchor_idx: int, k_neighbors: int) -> np.ndarray:
    p0 = points[anchor_idx]
    d2 = np.sum((points - p0[None, :]) ** 2, axis=1)
    k = min(max(8, int(k_neighbors)), points.shape[0])
    idx = np.argpartition(d2, k - 1)[:k]
    _, n = fit_plane_svd(points[idx])
    return n


def project_scene_to_mask_points(
    xyz_world: np.ndarray,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_depth_to_pose: np.ndarray,
    mask01: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_near: float,
    z_far: float,
    occlusion_tol: float,
    mask_dilation_px: int = 0,
) -> np.ndarray:
    # world -> pose
    xyz_pose = (R_world.T @ (xyz_world - t_world[None, :]).T).T
    # pose -> optical (inverse of depth_to_pose)
    xyz_opt = (R_depth_to_pose.T @ xyz_pose.T).T

    x = xyz_opt[:, 0]
    y = xyz_opt[:, 1]
    z = xyz_opt[:, 2]

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > max(1e-6, float(z_near)))
    if float(z_far) > 0:
        valid &= z <= float(z_far)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32)

    xv, yv, zv = x[valid], y[valid], z[valid]
    world_valid = xyz_world[valid]

    u = np.rint(float(fx) * (xv / zv) + float(cx)).astype(np.int32)
    v = np.rint(float(fy) * (yv / zv) + float(cy)).astype(np.int32)

    h, w = mask01.shape
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        return np.zeros((0, 3), np.float32)

    u = u[in_img]
    v = v[in_img]
    z = zv[in_img]
    world_keep = world_valid[in_img]

    mask_use = dilate_mask(mask01, dilation_px=int(mask_dilation_px))
    in_mask = mask_use[v, u] > 0
    if not np.any(in_mask):
        return np.zeros((0, 3), np.float32)

    u = u[in_mask]
    v = v[in_mask]
    z = z[in_mask]
    world_keep = world_keep[in_mask]

    # Keep front-most points per pixel (and near-front within tolerance).
    pix = v.astype(np.int64) * int(w) + u.astype(np.int64)
    order = np.argsort(pix)
    pix_s = pix[order]
    z_s = z[order]
    world_s = world_keep[order]

    unique_pix, start = np.unique(pix_s, return_index=True)
    minz = z_s[start]

    keep = np.zeros_like(z_s, dtype=bool)
    for i in range(len(unique_pix)):
        s0 = start[i]
        s1 = start[i + 1] if i + 1 < len(start) else len(pix_s)
        zmin = minz[i]
        keep[s0:s1] = z_s[s0:s1] <= (zmin + float(occlusion_tol))

    pts = world_s[keep]
    return pts.astype(np.float32)


def build_plane_patch(
    inliers: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    patch_step: float,
) -> np.ndarray:
    n = normal / max(1e-9, float(np.linalg.norm(normal)))
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e1 = np.cross(n, ref)
    e1 = e1 / max(1e-9, float(np.linalg.norm(e1)))
    e2 = np.cross(n, e1)
    e2 = e2 / max(1e-9, float(np.linalg.norm(e2)))

    rel = inliers - center[None, :]
    uu = rel @ e1
    vv = rel @ e2

    umin, umax = float(uu.min()), float(uu.max())
    vmin, vmax = float(vv.min()), float(vv.max())

    step = max(1e-3, float(patch_step))
    ug = np.arange(umin, umax + step, step, dtype=np.float32)
    vg = np.arange(vmin, vmax + step, step, dtype=np.float32)
    U, V = np.meshgrid(ug, vg)

    patch = center[None, None, :] + U[..., None] * e1[None, None, :] + V[..., None] * e2[None, None, :]
    return patch.reshape(-1, 3).astype(np.float32)


def build_plane_patch_from_mask_pixels(
    center: np.ndarray,
    normal: np.ndarray,
    mask01: np.ndarray,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_depth_to_pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """Create plane points by intersecting per-mask camera rays with fitted plane.

    This guarantees the 2D support of the plane patch is exactly bounded by SAM mask pixels.
    """
    ys, xs = np.nonzero(mask01 > 0)
    if xs.size == 0:
        return np.zeros((0, 3), np.float32)

    # Optical-frame ray directions for each mask pixel.
    x = (xs.astype(np.float32) - float(cx)) / float(fx)
    y = (ys.astype(np.float32) - float(cy)) / float(fy)
    dirs_opt = np.stack([x, y, np.ones_like(x, dtype=np.float32)], axis=1)

    # optical -> pose -> world direction
    dirs_pose = (R_depth_to_pose @ dirs_opt.T).T
    dirs_world = (R_world @ dirs_pose.T).T

    n = normal / max(1e-9, float(np.linalg.norm(normal)))
    num = float(np.dot(n, center - t_world))
    den = dirs_world @ n
    valid = np.abs(den) > 1e-8
    if not np.any(valid):
        return np.zeros((0, 3), np.float32)

    s = np.zeros_like(den, dtype=np.float32)
    s[valid] = (num / den[valid]).astype(np.float32)
    valid &= s > 0.0
    if not np.any(valid):
        return np.zeros((0, 3), np.float32)

    pts_world = t_world[None, :] + s[:, None] * dirs_world
    pts_world = pts_world[valid]
    if pts_world.shape[0] == 0:
        return np.zeros((0, 3), np.float32)

    # Enforce depth range in optical frame.
    xyz_pose = (R_world.T @ (pts_world - t_world[None, :]).T).T
    xyz_opt = (R_depth_to_pose.T @ xyz_pose.T).T
    z = xyz_opt[:, 2]
    keep = np.isfinite(z) & (z > max(1e-6, float(z_near)))
    if float(z_far) > 0:
        keep &= z <= float(z_far)
    pts_world = pts_world[keep]
    if pts_world.shape[0] == 0:
        return np.zeros((0, 3), np.float32)

    return pts_world.astype(np.float32)


def clip_world_points_to_mask_by_projection(
    points_world: np.ndarray,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_depth_to_pose: np.ndarray,
    mask01: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """Keep world points whose projection lands inside the SAM mask."""
    if points_world.shape[0] == 0:
        return points_world.astype(np.float32)

    # world -> pose -> optical
    xyz_pose = (R_world.T @ (points_world - t_world[None, :]).T).T
    xyz_opt = (R_depth_to_pose.T @ xyz_pose.T).T

    x = xyz_opt[:, 0]
    y = xyz_opt[:, 1]
    z = xyz_opt[:, 2]

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > max(1e-6, float(z_near)))
    if float(z_far) > 0:
        valid &= z <= float(z_far)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32)

    pts = points_world[valid]
    xv, yv, zv = x[valid], y[valid], z[valid]
    u = np.rint(float(fx) * (xv / zv) + float(cx)).astype(np.int32)
    v = np.rint(float(fy) * (yv / zv) + float(cy)).astype(np.int32)

    h, w = mask01.shape
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_img):
        return np.zeros((0, 3), np.float32)

    pts = pts[in_img]
    u = u[in_img]
    v = v[in_img]
    in_mask = mask01[v, u] > 0
    if not np.any(in_mask):
        return np.zeros((0, 3), np.float32)

    return pts[in_mask].astype(np.float32)


def build_normal_arrow(center: np.ndarray, normal: np.ndarray, length: float, npts: int = 64) -> np.ndarray:
    n = normal / max(1e-9, float(np.linalg.norm(normal)))
    ts = np.linspace(0.0, float(length), int(max(8, npts)), dtype=np.float32)
    return (center[None, :] + ts[:, None] * n[None, :]).astype(np.float32)


def voxel_downsample_indices(points: np.ndarray, voxel_size: float, max_points: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if p.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    v = float(max(1e-4, voxel_size))
    ijk = np.floor(p / v).astype(np.int64)
    seen: Dict[Tuple[int, int, int], int] = {}
    for i in range(ijk.shape[0]):
        key = (int(ijk[i, 0]), int(ijk[i, 1]), int(ijk[i, 2]))
        if key not in seen:
            seen[key] = i
    idx = np.array(list(seen.values()), dtype=np.int64)
    if idx.size > int(max_points):
        # Uniformly thin if still too many seeds.
        step = max(1, idx.size // int(max_points))
        idx = idx[::step][: int(max_points)]
    return idx


def plane_rect_coverage_hits(
    points_world: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    rect_source_points: np.ndarray,
    dist_tol: float,
    rect_margin: float,
) -> np.ndarray:
    """Return boolean hit mask: near plane and inside plane-aligned rectangle."""
    if points_world.shape[0] == 0 or rect_source_points.shape[0] == 0:
        return np.zeros((points_world.shape[0],), dtype=bool)

    n = normal / max(1e-9, float(np.linalg.norm(normal)))
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e1 = np.cross(n, ref)
    e1 = e1 / max(1e-9, float(np.linalg.norm(e1)))
    e2 = np.cross(n, e1)
    e2 = e2 / max(1e-9, float(np.linalg.norm(e2)))

    src_rel = rect_source_points - center[None, :]
    src_u = src_rel @ e1
    src_v = src_rel @ e2
    margin = max(0.0, float(rect_margin))
    umin = float(src_u.min()) - margin
    umax = float(src_u.max()) + margin
    vmin = float(src_v.min()) - margin
    vmax = float(src_v.max()) + margin

    rel = points_world - center[None, :]
    dist = np.abs(rel @ n)
    u = rel @ e1
    v = rel @ e2

    near_plane = dist <= max(1e-6, float(dist_tol))
    inside_rect = (u >= umin) & (u <= umax) & (v >= vmin) & (v <= vmax)
    return near_plane & inside_rect


def run_one_mask(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    mask01: np.ndarray,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_depth_to_pose: np.ndarray,
    pred_normal_world: Optional[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    mask_points = project_scene_to_mask_points(
        xyz_world=scene_xyz,
        t_world=t_world,
        R_world=R_world,
        R_depth_to_pose=R_depth_to_pose,
        mask01=mask01,
        fx=float(args.fx),
        fy=float(args.fy),
        cx=float(args.cx),
        cy=float(args.cy),
        z_near=float(args.depth_min),
        z_far=float(args.depth_max),
        occlusion_tol=float(args.occlusion_tol),
        mask_dilation_px=int(args.mask_dilation_px),
    )

    if mask_points.shape[0] < int(args.min_mask_points):
        return [{"status": "skip", "reason": f"too_few_mask_points:{mask_points.shape[0]}"}]

    # Remove local mask regions around upward/downward (vertical) local normals.
    vertical_removed_points = 0
    vertical_seed_count = 0
    if bool(args.reject_vertical_normals):
        pre_n = int(mask_points.shape[0])
        seed_pre = uniform_seed_indices(pre_n, int(args.sampled_normals_max))
        if seed_pre.size > 0:
            up_axis_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            vertical_cos = float(np.cos(np.deg2rad(float(args.vertical_normal_max_tilt_deg))))
            radius2 = float(max(0.0, float(args.vertical_unmask_radius_m))) ** 2
            keep = np.ones((pre_n,), dtype=bool)
            for li in seed_pre:
                ni = estimate_local_normal(mask_points, anchor_idx=int(li), k_neighbors=int(args.normal_k))
                if abs(float(np.dot(ni, up_axis_world))) >= vertical_cos:
                    vertical_seed_count += 1
                    if radius2 > 0.0:
                        c = mask_points[int(li)]
                        d2 = np.sum((mask_points - c[None, :]) ** 2, axis=1)
                        keep &= d2 > radius2
            if vertical_seed_count > 0 and np.any(keep):
                mask_points = mask_points[keep]
                vertical_removed_points = pre_n - int(mask_points.shape[0])

    if mask_points.shape[0] < int(args.min_mask_points):
        return [
            {
                "status": "skip",
                "reason": f"too_few_mask_points_after_vertical_filter:{mask_points.shape[0]}",
                "vertical_seed_count": int(vertical_seed_count),
                "vertical_removed_points": int(vertical_removed_points),
            }
        ]

    results: List[Dict[str, object]] = []
    npts = mask_points.shape[0]

    # Match debug script behavior: uniform seed sampling directly from mask points.
    seed_idx = uniform_seed_indices(npts, int(args.sampled_normals_max))
    if seed_idx.size == 0:
        return [{"status": "skip", "reason": "no_seed_points_after_sampling"}]

    all_normals: List[np.ndarray] = []
    kept_seed_local_idx: List[int] = []
    all_arrows: List[np.ndarray] = []
    pre_len = float(args.all_normals_vis_len)
    up_axis_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    vertical_cos = float(np.cos(np.deg2rad(float(args.vertical_normal_max_tilt_deg))))
    for pi in seed_idx:
        i = int(pi)
        ni = estimate_local_normal(mask_points, anchor_idx=i, k_neighbors=int(args.normal_k))
        if bool(args.reject_vertical_normals) and abs(float(np.dot(ni, up_axis_world))) >= vertical_cos:
            continue
        if pred_normal_world is not None and float(np.dot(ni, pred_normal_world)) < 0:
            ni = -ni
        ni_n = normalize_vec(ni)
        all_normals.append(ni_n)
        kept_seed_local_idx.append(i)
        all_arrows.append(build_normal_arrow(center=mask_points[i], normal=ni_n, length=pre_len, npts=12))

    if len(all_normals) == 0:
        return [
            {
                "status": "skip",
                "reason": "all_seed_normals_vertical_or_invalid",
                "vertical_seed_count": int(vertical_seed_count),
                "vertical_removed_points": int(vertical_removed_points),
            }
        ]

    all_normals_np = np.asarray(all_normals, dtype=np.float32)
    seed_idx = np.asarray(kept_seed_local_idx, dtype=np.int64)

    all_normals_arrows = (
        np.concatenate(all_arrows, axis=0).astype(np.float32)
        if len(all_arrows) > 0
        else np.zeros((0, 3), dtype=np.float32)
    )

    num_seed = int(seed_idx.size)
    num_sample = int(min(max(1, int(args.sampled_normals_max)), num_seed))
    if pred_normal_world is not None:
        lr_axis_world = normalize_vec((R_world @ (R_depth_to_pose @ np.array([0.0, 1.0, 0.0], dtype=np.float32).reshape(3, 1))).reshape(3))
        all_tier = np.array(
            [
                normal_match_tier(
                    cand_normal=all_normals_np[i],
                    pred_normal=pred_normal_world,
                    left_right_axis=lr_axis_world,
                    direct_tol_deg=float(args.normal_similarity_deg),
                    right_angle_tol_deg=float(args.right_angle_tol_deg),
                )
                for i in range(num_seed)
            ],
            dtype=np.int32,
        )
        pref_direct = np.array(
            [i for i in range(num_seed) if int(all_tier[i]) == 2],
            dtype=np.int64,
        )
        pref_right = np.array(
            [i for i in range(num_seed) if int(all_tier[i]) == 1],
            dtype=np.int64,
        )
        pref_set = set([int(i) for i in np.concatenate([pref_direct, pref_right], axis=0).tolist()]) if (pref_direct.size + pref_right.size) > 0 else set()
        other = np.array([i for i in range(num_seed) if i not in pref_set], dtype=np.int64)

        pick_parts: List[np.ndarray] = []
        if pref_direct.size > 0:
            k1 = int(min(num_sample, pref_direct.size))
            pick_parts.append(rng.choice(pref_direct, size=k1, replace=False).astype(np.int64))
        picked = int(np.sum([p.size for p in pick_parts]))
        if picked < num_sample and pref_right.size > 0:
            k_mid = int(min(num_sample - picked, pref_right.size))
            pick_parts.append(rng.choice(pref_right, size=k_mid, replace=False).astype(np.int64))
            picked = int(np.sum([p.size for p in pick_parts]))
        if picked < num_sample and other.size > 0:
            k2 = int(min(num_sample - picked, other.size))
            pick_parts.append(rng.choice(other, size=k2, replace=False).astype(np.int64))
        if len(pick_parts) == 0:
            sampled_seed_idx = rng.choice(num_seed, size=num_sample, replace=False)
        else:
            sampled_seed_idx = np.concatenate(pick_parts, axis=0)
    else:
        sampled_seed_idx = rng.choice(num_seed, size=num_sample, replace=False)

    sampled_seed_pool = np.asarray(sampled_seed_idx, dtype=np.int64).reshape(-1)
    max_attempts = int(sampled_seed_pool.size)
    t_start = time.perf_counter()
    attempts_total = 0
    keep_try_records: List[Dict[str, object]] = []
    success_found = False

    while attempts_total < max_attempts:
        sidx = int(sampled_seed_pool[attempts_total])

        anchor_idx = int(seed_idx[sidx])
        n0 = all_normals_np[sidx]
        attempts_total += 1

        # Initial plane through anchor point.
        p0 = mask_points[anchor_idx]
        d0 = -float(np.dot(n0, p0))
        dist = mask_points @ n0 + d0
        inlier = np.abs(dist) <= float(args.plane_dist_thresh)

        if int(inlier.sum()) < int(args.min_inliers):
            rec_skip = {
                "status": "skip",
                "reason": f"too_few_inliers:{int(inlier.sum())}",
                "normal_seed": n0.tolist(),
                "anchor_index": anchor_idx,
                "attempt_index": int(attempts_total),
                "elapsed_sec": float(time.perf_counter() - t_start),
            }
            if len(keep_try_records) < int(args.max_visualized_tries):
                keep_try_records.append(rec_skip)
            continue

        inliers = mask_points[inlier]
        c_fit, n_fit = fit_plane_svd(inliers)
        coverage_dist_tol = float(args.coverage_dist_tol)
        if coverage_dist_tol <= 0:
            coverage_dist_tol = max(1e-6, 1.5 * float(args.plane_dist_thresh))

        hits = plane_rect_coverage_hits(
            points_world=mask_points,
            center=c_fit,
            normal=n_fit,
            rect_source_points=inliers,
            dist_tol=coverage_dist_tol,
            rect_margin=float(args.coverage_rect_margin),
        )
        num_hit = int(hits.sum())
        coverage_ratio = float(num_hit) / float(mask_points.shape[0])
        coverage_percent = 100.0 * coverage_ratio

        # Orient normal towards predicted normal when available.
        if pred_normal_world is not None:
            if float(np.dot(n_fit, pred_normal_world)) < 0:
                n_fit = -n_fit
        elif float(np.dot(n_fit, n0)) < 0:
            n_fit = -n_fit

        pred_angle_deg = None
        lr_axis_world = normalize_vec((R_world @ (R_depth_to_pose @ np.array([0.0, 1.0, 0.0], dtype=np.float32).reshape(3, 1))).reshape(3))
        if pred_normal_world is not None:
            pred_angle_deg = angular_error_deg_unsigned(n_fit, pred_normal_world)
        normal_tier = normal_match_tier(
            cand_normal=n_fit,
            pred_normal=pred_normal_world,
            left_right_axis=lr_axis_world,
            direct_tol_deg=float(args.normal_similarity_deg),
            right_angle_tol_deg=float(args.right_angle_tol_deg),
        )
        normal_ok = bool(normal_tier > 0)
        coverage_ok = bool(coverage_percent >= float(args.target_coverage_percent))

        patch = build_plane_patch_from_mask_pixels(
            center=c_fit,
            normal=n_fit,
            mask01=mask01,
            t_world=t_world,
            R_world=R_world,
            R_depth_to_pose=R_depth_to_pose,
            fx=float(args.fx),
            fy=float(args.fy),
            cx=float(args.cx),
            cy=float(args.cy),
            z_near=float(args.depth_min),
            z_far=float(args.depth_max),
        )
        if patch.shape[0] == 0:
            rec_skip = {
                "status": "skip",
                "reason": "plane_patch_from_mask_empty",
                "normal_seed": n0.tolist(),
                "anchor_index": anchor_idx,
                "attempt_index": int(attempts_total),
                "elapsed_sec": float(time.perf_counter() - t_start),
            }
            if len(keep_try_records) < int(args.max_visualized_tries):
                keep_try_records.append(rec_skip)
            continue
        arrow = build_normal_arrow(center=c_fit, normal=n_fit, length=float(args.normal_vis_len), npts=64)

        # Visualization scene base (subsample by stride).
        stride = max(1, int(args.viz_scene_stride))
        scene_base_xyz = scene_xyz[::stride]
        scene_base_rgb = scene_rgb[::stride]

        rgb_mask = np.tile(np.array([[60, 255, 60]], dtype=np.uint8), (mask_points.shape[0], 1))
        rgb_plane = np.tile(np.array([[60, 60, 255]], dtype=np.uint8), (patch.shape[0], 1))
        rgb_arrow = np.tile(np.array([[255, 60, 60]], dtype=np.uint8), (arrow.shape[0], 1))

        xyz_out = np.concatenate([scene_base_xyz, mask_points, patch, arrow], axis=0)
        rgb_out = np.concatenate([scene_base_rgb, rgb_mask, rgb_plane, rgb_arrow], axis=0)

        rec_ok = {
            "status": "ok",
            "sample_try_index": int(min(attempts_total, sampled_seed_pool.size)),
            "attempt_index": int(attempts_total),
                "anchor_index": anchor_idx,
                "num_mask_points": int(mask_points.shape[0]),
                "num_inliers": int(inliers.shape[0]),
                "num_coverage_hits": num_hit,
                "num_patch_points": int(patch.shape[0]),
                "coverage_ratio": coverage_ratio,
                "coverage_percent": coverage_percent,
                "pred_angle_deg": pred_angle_deg,
                "normal_ok": normal_ok,
                "normal_match_tier": int(normal_tier),
                "coverage_ok": coverage_ok,
                "normal_seed": n0.tolist(),
                "normal_fit": n_fit.tolist(),
                "center_fit": c_fit.tolist(),
                "mask_points_world": mask_points.copy(),
                "all_normals_arrows_world": all_normals_arrows,
                "patch_world": patch,
                "arrow_world": arrow,
                "xyz_out": xyz_out,
                "rgb_out": rgb_out,
                "vertical_seed_count": int(vertical_seed_count),
                "vertical_removed_points": int(vertical_removed_points),
                "elapsed_sec": float(time.perf_counter() - t_start),
        }
        if len(keep_try_records) < int(args.max_visualized_tries):
            keep_try_records.append(rec_ok)
        elif normal_ok and coverage_ok:
            keep_try_records[-1] = rec_ok

        if bool(args.search_until_success):
            if normal_ok and coverage_ok:
                success_found = True
                break

        # Safety valve only on elapsed-time, not on number of tries.
        if float(args.search_timeout_sec) > 0 and (time.perf_counter() - t_start) >= float(args.search_timeout_sec):
            break

    if bool(args.search_until_success) and (not success_found):
        rec_end = {
            "status": "skip",
            "reason": "no_success_after_exhaustion",
            "attempt_index": int(attempts_total),
            "elapsed_sec": float(time.perf_counter() - t_start),
        }
        if len(keep_try_records) < int(args.max_visualized_tries):
            keep_try_records.append(rec_end)
        elif len(keep_try_records) > 0:
            keep_try_records[-1] = rec_end

    results.extend(keep_try_records)
    return results


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic plane-fit tries from scene PLY + SAM masks")
    ap.add_argument("--scene_ply", default="", help="Optional prebuilt world scene PLY")
    ap.add_argument(
        "--habitat_dir",
        default="",
        help="Habitat output dir containing cloud_*.ply; used when --scene_ply is not set",
    )
    ap.add_argument("--sam_mask_root", required=True, help="Folder containing SAM mask subfolders")
    ap.add_argument("--mask_folder", default="", help="Optional explicit mask folder; if empty, pick one randomly")
    ap.add_argument("--poses_csv", required=True, help="poses.csv path")
    ap.add_argument("--out_dir", default="guard_simple_out/deterministic_plane_fits")

    ap.add_argument("--fx", type=float, default=388.1910413097385)
    ap.add_argument("--fy", type=float, default=422.0475153598262)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--depth_min", type=float, default=0.2)
    ap.add_argument("--depth_max", type=float, default=10.0)
    ap.add_argument(
        "--depth_to_pose",
        choices=["identity", "ros_optical_to_link", "ros_link_to_optical"],
        default="ros_optical_to_link",
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_tries", type=int, default=5)
    ap.add_argument("--normal_k", type=int, default=128, help="Neighbors used for local normal estimate")
    ap.add_argument("--plane_dist_thresh", type=float, default=0.04)
    ap.add_argument("--min_inliers", type=int, default=200)
    ap.add_argument("--min_mask_points", type=int, default=400)
    ap.add_argument("--patch_step", type=float, default=0.03)
    ap.add_argument("--normal_vis_len", type=float, default=0.4)
    ap.add_argument("--all_normals_vis_len", type=float, default=0.10)
    ap.add_argument("--sampled_normals_max", type=int, default=40, help="Uniformly sample up to this many local normals per mask")
    ap.add_argument("--mask_dilation_px", type=int, default=15, help="Dilate SAM mask by this many pixels before selecting cloud points")
    ap.add_argument("--normal_seed_voxel_m", type=float, default=0.06, help="Voxel size for robust normal seeds (meters)")
    ap.add_argument("--normal_seed_max_points", type=int, default=120, help="Maximum normal seed points after voxel thinning")
    ap.add_argument("--max_visualized_tries", type=int, default=24, help="Max tries kept in output/PLY for readability")
    ap.add_argument("--search_until_success", action=argparse.BooleanOptionalAction, default=True, help="Keep trying until normal+coverage success")
    ap.add_argument("--search_timeout_sec", type=float, default=0.0, help="Optional timeout per mask search (0 disables)")
    ap.add_argument("--occlusion_tol", type=float, default=0.05)
    ap.add_argument("--viz_scene_stride", type=int, default=20)
    ap.add_argument(
        "--coverage_dist_tol",
        type=float,
        default=-1.0,
        help="Distance tolerance (meters) for coverage hit test; <=0 uses 1.5 * --plane_dist_thresh",
    )
    ap.add_argument(
        "--coverage_rect_margin",
        type=float,
        default=0.03,
        help="Extra margin (meters) added to fitted plane rectangle for tolerant coverage counting",
    )
    ap.add_argument(
        "--run_all_folders",
        action="store_true",
        help="Process every mask folder/frame and write one merged visualization PLY.",
    )
    ap.add_argument(
        "--target_coverage_percent",
        type=float,
        default=60.0,
        help="Target coverage threshold used for reporting success in merged mode.",
    )
    ap.add_argument(
        "--merged_out_ply",
        default="",
        help="Output path for merged PLY in --run_all_folders mode. Default: <out_dir>/merged_all_frames_masks_planes.ply",
    )
    ap.add_argument(
        "--normal_checkpoint",
        default="",
        help="Optional plane_predict checkpoint (best.pt). If set, use predicted normal to guide random normal selection.",
    )
    ap.add_argument(
        "--normal_similarity_deg",
        type=float,
        default=30.0,
        help="Max angular difference (deg) from predicted normal for guided random selection.",
    )
    ap.add_argument(
        "--right_angle_tol_deg",
        type=float,
        default=10.0,
        help="Tolerance (deg) for left/right 90-degree match fallback.",
    )
    ap.add_argument(
        "--reject_vertical_normals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject up/down (vertical) local normals and remove nearby mask points.",
    )
    ap.add_argument(
        "--vertical_normal_max_tilt_deg",
        type=float,
        default=20.0,
        help="Treat local normal as vertical when tilt from world up/down is within this angle.",
    )
    ap.add_argument(
        "--vertical_unmask_radius_m",
        type=float,
        default=0.12,
        help="Unmask radius (meters) around rejected vertical-normal seeds.",
    )
    ap.add_argument(
        "--save_per_frame_details",
        action="store_true",
        help="In --run_all_folders mode, save per-frame folders with per-mask overlays and normals-tries PLY.",
    )
    ap.add_argument(
        "--per_frame_dir",
        default="",
        help="Optional base folder for detailed per-frame outputs. Default: <out_dir>/per_frame_details",
    )
    return ap.parse_args()


def resolve_poses_csv_path(poses_csv: str, habitat_dir: str) -> str:
    p = str(poses_csv).strip()
    if not p:
        raise RuntimeError("--poses_csv is empty")
    if os.path.isfile(p):
        return os.path.abspath(p)

    # If user passed a relative name like "poses.csv", try habitat_dir.
    if habitat_dir:
        candidate = os.path.join(habitat_dir, p)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(f"poses.csv not found: {p}")


def _load_frame_scene_world(
    habitat_dir: str,
    frame_id: int,
    t_world: np.ndarray,
    R_world: np.ndarray,
    R_dp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str]:
    cloud_path = os.path.join(habitat_dir, f"cloud_{frame_id:06d}.ply")
    if not os.path.isfile(cloud_path):
        raise FileNotFoundError(f"Frame cloud file not found: {cloud_path}")
    xyz_cam, rgb_cam = load_ply_xyzrgb(cloud_path)
    if xyz_cam.shape[0] == 0:
        raise RuntimeError(f"cloud has no points: {cloud_path}")
    xyz_pose = (R_dp @ xyz_cam.T).T
    scene_xyz = (R_world @ xyz_pose.T).T + t_world[None, :]
    if rgb_cam is None:
        scene_rgb = np.tile(np.array([[170, 170, 170]], dtype=np.uint8), (scene_xyz.shape[0], 1))
    else:
        scene_rgb = rgb_cam
    return scene_xyz.astype(np.float32), scene_rgb.astype(np.uint8), os.path.abspath(cloud_path)


def main() -> None:
    args = get_args()
    ensure_dir(args.out_dir)

    rng = np.random.default_rng(int(args.seed))

    poses_csv_path = resolve_poses_csv_path(args.poses_csv, args.habitat_dir)
    print(f"[LOAD] poses: {poses_csv_path}")
    poses = load_poses_csv(poses_csv_path)
    R_dp = get_R_depth_to_pose(str(args.depth_to_pose))

    normal_predictor: Optional[PlaneNormalPredictor] = None
    if str(args.normal_checkpoint).strip():
        normal_predictor = PlaneNormalPredictor(str(args.normal_checkpoint).strip())
        print(f"[LOAD] normal predictor: {os.path.abspath(str(args.normal_checkpoint).strip())}")

    if args.run_all_folders:
        if not args.habitat_dir:
            raise RuntimeError("--run_all_folders requires --habitat_dir")

        folders = list_mask_folders(args.sam_mask_root)
        valid_folders: List[str] = []
        for fd in folders:
            try:
                fid = parse_frame_id_from_folder(os.path.basename(fd))
            except ValueError:
                continue
            if fid in poses:
                valid_folders.append(fd)
        if len(valid_folders) == 0:
            raise RuntimeError("No valid mask folders found for provided poses.")

        merged_xyz_parts: List[np.ndarray] = []
        merged_rgb_parts: List[np.ndarray] = []
        total_masks = 0
        kept_masks = 0

        print(f"[RUN] merged mode over {len(valid_folders)} folders")
        for fi, folder in enumerate(valid_folders):
            frame_id = parse_frame_id_from_folder(os.path.basename(folder))
            t_world, R_world = poses[frame_id]
            scene_xyz, scene_rgb, scene_source = _load_frame_scene_world(
                habitat_dir=args.habitat_dir,
                frame_id=frame_id,
                t_world=t_world,
                R_world=R_world,
                R_dp=R_dp,
            )

            stride = max(1, int(args.viz_scene_stride))
            scene_base_xyz = scene_xyz[::stride]
            scene_base_rgb = scene_rgb[::stride]
            merged_xyz_parts.append(scene_base_xyz)
            merged_rgb_parts.append(scene_base_rgb)

            rgb_bgr = None
            frame_dir = ""
            if args.save_per_frame_details:
                rgb_path = os.path.join(args.habitat_dir, f"rgb_{frame_id:06d}.png")
                if os.path.isfile(rgb_path):
                    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                per_frame_root = (
                    str(args.per_frame_dir).strip()
                    if str(args.per_frame_dir).strip()
                    else os.path.join(args.out_dir, "per_frame_details")
                )
                frame_dir = os.path.join(per_frame_root, f"rgb_{frame_id:06d}")
                ensure_dir(frame_dir)

            mask_paths = list_masks(folder)
            total_masks += len(mask_paths)
            print(f"[FOLDER] {fi+1}/{len(valid_folders)} frame={frame_id} masks={len(mask_paths)} source={scene_source}")

            for mp in mask_paths:
                mask01 = load_mask01(mp)
                mask_name = os.path.splitext(os.path.basename(mp))[0]
                mask_label = strip_trailing_score_from_mask_name(mask_name)

                pred_normal_world = None
                pred_n_cam = None
                if normal_predictor is not None:
                    rgb_path = os.path.join(args.habitat_dir, f"rgb_{frame_id:06d}.png")
                    if os.path.isfile(rgb_path):
                        rgb_for_pred = rgb_bgr
                        if rgb_for_pred is None:
                            rgb_for_pred = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                        if rgb_for_pred is not None and rgb_for_pred.size > 0:
                            rgb_u8 = cv2.cvtColor(rgb_for_pred, cv2.COLOR_BGR2RGB)
                            pred_n_cam = normal_predictor.predict_normal_cam(rgb_u8, mask01)
                            pred_normal_world = normalize_vec((R_world @ (R_dp @ pred_n_cam.reshape(3, 1))).reshape(3))

                per_try = run_one_mask(
                    scene_xyz=scene_xyz,
                    scene_rgb=scene_rgb,
                    mask01=mask01,
                    t_world=t_world,
                    R_world=R_world,
                    R_depth_to_pose=R_dp,
                    pred_normal_world=pred_normal_world,
                    args=args,
                    rng=rng,
                )

                best = choose_guided_try(
                    per_try=per_try,
                    rng=rng,
                    target_coverage_percent=float(args.target_coverage_percent),
                    max_angle_deg=float(args.normal_similarity_deg),
                )
                if best is None:
                    print(f"[MASK] {mask_label} best=skip")
                    continue

                cov = float(best["coverage_percent"])
                ang = float(best.get("pred_angle_deg", 180.0))
                hit_target = cov >= float(args.target_coverage_percent)
                kept_masks += 1

                rgb_mask = np.tile(np.array([[60, 255, 60]], dtype=np.uint8), (best["mask_points_world"].shape[0], 1))
                rgb_plane = np.tile(np.array([[60, 60, 255]], dtype=np.uint8), (best["patch_world"].shape[0], 1))
                rgb_arrow = np.tile(np.array([[255, 60, 60]], dtype=np.uint8), (best["arrow_world"].shape[0], 1))

                merged_xyz_parts.append(best["mask_points_world"])
                merged_rgb_parts.append(rgb_mask)
                merged_xyz_parts.append(best["patch_world"])
                merged_rgb_parts.append(rgb_plane)
                merged_xyz_parts.append(best["arrow_world"])
                merged_rgb_parts.append(rgb_arrow)

                if args.save_per_frame_details and len(frame_dir) > 0 and rgb_bgr is not None and rgb_bgr.size > 0:
                    region_dir = os.path.join(frame_dir, mask_label)
                    ensure_dir(region_dir)

                    combined_overlay_png = os.path.join(region_dir, "rgb_mask_pred_overlay.png")
                    save_rgb_mask_pred_overlay_png(
                        rgb_bgr=rgb_bgr,
                        mask01=mask01,
                        pred_normal_cam=pred_n_cam,
                        out_path=combined_overlay_png,
                        alpha=0.45,
                        arrow_len_px=100.0,
                    )

                    sel_normal_ok = bool(best.get("normal_ok", False))
                    sel_cov_ok = bool(best.get("coverage_ok", False))
                    sel_tag = f"{'yes' if sel_normal_ok else 'no'}-{'yes' if sel_cov_ok else 'no'}"
                    tries_ply = os.path.join(region_dir, f"everything_selected_{sel_tag}.ply")
                    save_normal_tries_ply(
                        scene_xyz=scene_xyz,
                        scene_rgb=scene_rgb,
                        mask_points_world=np.asarray(best.get("mask_points_world", np.zeros((0, 3), dtype=np.float32))),
                        all_normals_arrows_world=np.asarray(
                            best.get("all_normals_arrows_world", np.zeros((0, 3), dtype=np.float32))
                        ),
                        per_try=per_try,
                        selected_try=best,
                        selected_color=(
                            (60, 255, 60)
                            if (
                                bool(best.get("normal_ok", False))
                                and str(best.get("selection_reason", "")).startswith("guided_random")
                            )
                            else (255, 220, 60)
                        ),
                        out_ply=tries_ply,
                    )

                print(
                    f"[MASK] {mask_label} best_coverage={cov:.1f}% "
                    f"pred_ang={ang:.1f}deg target60={str(hit_target).lower()} "
                    f"reason={best.get('selection_reason', 'n/a')} "
                    f"attempts={int(best.get('attempt_index', -1))} elapsed={float(best.get('elapsed_sec', 0.0)):.2f}s"
                )

        if len(merged_xyz_parts) == 0:
            raise RuntimeError("No visualization points collected in merged mode.")

        merged_xyz = np.concatenate(merged_xyz_parts, axis=0)
        merged_rgb = np.concatenate(merged_rgb_parts, axis=0)
        out_ply = args.merged_out_ply.strip() if str(args.merged_out_ply).strip() else os.path.join(
            args.out_dir, "merged_all_frames_masks_planes.ply"
        )
        write_ply_xyzrgb_ascii(out_ply, merged_xyz.astype(np.float32), merged_rgb.astype(np.uint8))
        print(f"[DONE] merged_ply={os.path.abspath(out_ply)} points={merged_xyz.shape[0]}")
        print(f"[DONE] masks_kept={kept_masks}/{total_masks}")
        return

    if args.mask_folder:
        chosen_folder = args.mask_folder
    else:
        folders = list_mask_folders(args.sam_mask_root)
        if len(folders) == 0:
            raise RuntimeError(f"No mask folders found in {args.sam_mask_root}")

        valid_folders: List[str] = []
        for fd in folders:
            try:
                fid = parse_frame_id_from_folder(os.path.basename(fd))
            except ValueError:
                continue
            if fid in poses:
                valid_folders.append(fd)

        if len(valid_folders) == 0:
            raise RuntimeError(
                f"No mask folders in {args.sam_mask_root} have frame ids present in poses.csv"
            )

        chosen_folder = valid_folders[int(rng.integers(0, len(valid_folders)))]

    frame_id = parse_frame_id_from_folder(os.path.basename(chosen_folder))
    if frame_id not in poses:
        raise RuntimeError(f"Frame {frame_id} from folder {chosen_folder} not found in poses.csv")

    t_world, R_world = poses[frame_id]

    rgb_ref_path = ""
    rgb_ref_bgr: Optional[np.ndarray] = None
    if args.habitat_dir:
        candidate = os.path.join(args.habitat_dir, f"rgb_{frame_id:06d}.png")
        if os.path.isfile(candidate):
            rgb_ref_bgr = cv2.imread(candidate, cv2.IMREAD_COLOR)
            if rgb_ref_bgr is not None:
                rgb_ref_path = os.path.abspath(candidate)

    scene_source = ""
    if args.scene_ply:
        scene_xyz, scene_rgb = load_ply_xyzrgb(args.scene_ply)
        scene_source = os.path.abspath(args.scene_ply)
    else:
        if not args.habitat_dir:
            raise RuntimeError("Provide --scene_ply or --habitat_dir")
        cloud_path = os.path.join(args.habitat_dir, f"cloud_{frame_id:06d}.ply")
        if not os.path.isfile(cloud_path):
            raise FileNotFoundError(f"Frame cloud file not found: {cloud_path}")

        # cloud_XXXXXX.ply is camera-frame; convert to world once for unified processing.
        xyz_cam, rgb_cam = load_ply_xyzrgb(cloud_path)
        if xyz_cam.shape[0] == 0:
            raise RuntimeError(f"cloud has no points: {cloud_path}")
        xyz_pose = (R_dp @ xyz_cam.T).T
        scene_xyz = (R_world @ xyz_pose.T).T + t_world[None, :]
        scene_rgb = rgb_cam
        scene_source = os.path.abspath(cloud_path)

    if scene_xyz.shape[0] == 0:
        raise RuntimeError("scene source has no points")
    if scene_rgb is None:
        scene_rgb = np.tile(np.array([[170, 170, 170]], dtype=np.uint8), (scene_xyz.shape[0], 1))

    mask_paths = list_masks(chosen_folder)
    if len(mask_paths) == 0:
        raise RuntimeError(f"No mask_*.png found in {chosen_folder}")

    out_base = os.path.join(args.out_dir, os.path.basename(chosen_folder))
    ensure_dir(out_base)

    summary: Dict[str, object] = {
        "scene_source": scene_source,
        "rgb_ref_path": rgb_ref_path,
        "chosen_mask_folder": os.path.abspath(chosen_folder),
        "frame_id": int(frame_id),
        "num_masks": len(mask_paths),
        "num_tries": int(args.num_tries),
        "seed": int(args.seed),
        "results": [],
    }

    print(f"[LOAD] scene source: {scene_source}")
    print(f"[LOAD] scene points: {scene_xyz.shape[0]}")
    print(f"[LOAD] chosen folder: {chosen_folder}")
    print(f"[LOAD] masks in folder: {len(mask_paths)}")

    for mi, mp in enumerate(mask_paths):
        mask01 = load_mask01(mp)
        mask_name = os.path.splitext(os.path.basename(mp))[0]
        mask_label = strip_trailing_score_from_mask_name(mask_name)

        pred_normal_world = None
        pred_normal_cam = None
        if normal_predictor is not None and rgb_ref_bgr is not None and rgb_ref_bgr.size > 0:
            rgb_u8 = cv2.cvtColor(rgb_ref_bgr, cv2.COLOR_BGR2RGB)
            pred_normal_cam = normal_predictor.predict_normal_cam(rgb_u8, mask01)
            pred_normal_world = normalize_vec((R_world @ (R_dp @ pred_normal_cam.reshape(3, 1))).reshape(3))

        overlay_png = os.path.join(out_base, f"{mask_label}_overlay.png")
        if rgb_ref_bgr is not None:
            save_rgb_mask_overlay_png(rgb_ref_bgr, mask01, overlay_png)

        per_try = run_one_mask(
            scene_xyz=scene_xyz,
            scene_rgb=scene_rgb,
            mask01=mask01,
            t_world=t_world,
            R_world=R_world,
            R_depth_to_pose=R_dp,
            pred_normal_world=pred_normal_world,
            args=args,
            rng=rng,
        )

        selected = choose_guided_try(
            per_try=per_try,
            rng=rng,
            target_coverage_percent=float(args.target_coverage_percent),
            max_angle_deg=float(args.normal_similarity_deg),
        )

        mask_record = {
            "mask_path": os.path.abspath(mp),
            "overlay_png": os.path.abspath(overlay_png) if rgb_ref_bgr is not None else "",
            "pred_normal_cam": pred_normal_cam.tolist() if pred_normal_cam is not None else None,
            "pred_normal_world": pred_normal_world.tolist() if pred_normal_world is not None else None,
            "selected_try": -1,
            "tries": [],
        }

        for ti, rec in enumerate(per_try):
            if rec.get("status") == "ok":
                coverage_percent = float(rec["coverage_percent"])
                is_selected = bool(selected is rec)
                pred_ang = rec.get("pred_angle_deg", None)
                normal_ok = bool(rec.get("normal_ok", False))
                coverage_ok = bool(rec.get("coverage_ok", False))
                yn_tag = f"{'yes' if normal_ok else 'no'}-{'yes' if coverage_ok else 'no'}"
                out_ply = os.path.join(
                    out_base,
                    f"{mask_label}_{yn_tag}_coverage_{coverage_percent:.1f}_ang_{float(pred_ang) if pred_ang is not None else 999.0:.1f}_try_{ti+1:02d}{'_selected' if is_selected else ''}.ply",
                )
                plane_on_rgb_png = os.path.join(
                    out_base,
                    f"{mask_label}_{yn_tag}_coverage_{coverage_percent:.1f}_ang_{float(pred_ang) if pred_ang is not None else 999.0:.1f}_try_{ti+1:02d}{'_selected' if is_selected else ''}_plane2d.png",
                )
                write_ply_xyzrgb_ascii(out_ply, rec["xyz_out"], rec["rgb_out"])
                if rgb_ref_bgr is not None:
                    save_plane_projection_overlay_png(
                        rgb_bgr=rgb_ref_bgr,
                        plane_points_world=rec["patch_world"],
                        t_world=t_world,
                        R_world=R_world,
                        R_depth_to_pose=R_dp,
                        fx=float(args.fx),
                        fy=float(args.fy),
                        cx=float(args.cx),
                        cy=float(args.cy),
                        z_near=float(args.depth_min),
                        z_far=float(args.depth_max),
                        out_path=plane_on_rgb_png,
                    )
                mask_record["tries"].append(
                    {
                        "try": ti + 1,
                        "selected": is_selected,
                        "selection_reason": rec.get("selection_reason", ""),
                        "status": "ok",
                        "out_ply": os.path.abspath(out_ply),
                        "plane_on_rgb_png": os.path.abspath(plane_on_rgb_png) if rgb_ref_bgr is not None else "",
                        "num_mask_points": int(rec["num_mask_points"]),
                        "num_inliers": int(rec["num_inliers"]),
                        "num_coverage_hits": int(rec["num_coverage_hits"]),
                        "num_patch_points": int(rec["num_patch_points"]),
                        "coverage_percent": coverage_percent,
                        "normal_ok": normal_ok,
                        "coverage_ok": coverage_ok,
                        "pred_angle_deg": float(pred_ang) if pred_ang is not None else None,
                        "attempt_index": int(rec.get("attempt_index", -1)),
                        "elapsed_sec": float(rec.get("elapsed_sec", 0.0)),
                        "normal_seed": rec["normal_seed"],
                        "normal_fit": rec["normal_fit"],
                        "center_fit": rec["center_fit"],
                    }
                )
                if is_selected:
                    mask_record["selected_try"] = int(ti + 1)
                print(
                    f"[TRY] {mask_label} try={ti+1:02d} "
                    f"coverage={coverage_percent:.1f}% "
                    f"pred_ang={(float(pred_ang) if pred_ang is not None else -1.0):.1f}deg "
                    f"attempt={int(rec.get('attempt_index', -1))} elapsed={float(rec.get('elapsed_sec', 0.0)):.2f}s "
                    f"({int(rec['num_coverage_hits'])}/{int(rec['num_mask_points'])}), "
                    f"patch_pts={int(rec['num_patch_points'])}"
                )
            else:
                mask_record["tries"].append(
                    {
                        "try": ti + 1,
                        "status": "skip",
                        "reason": rec.get("reason", "unknown"),
                        "normal_seed": rec.get("normal_seed", None),
                    }
                )

        summary["results"].append(mask_record)
        print(f"[MASK] {mi+1}/{len(mask_paths)} {os.path.basename(mp)} done")

    summary_path = os.path.join(out_base, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OUT] {out_base}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()


# single command (whole-scene + per-region subfolders + combined overlay + everything PLY):
# <HOME>/miniconda3/envs/sam3/bin/python <HOME>/GlassGuard/glassguard_deterministic.py --habitat_dir <HOME>/360_camera/habitat_output --sam_mask_root <HOME>/360_camera/habitat_output/sam_mask --poses_csv <HOME>/360_camera/habitat_output/poses.csv --out_dir <HOME>/GlassGuard/guard_simple_out/deterministic_plane_fits_v3 --run_all_folders --normal_checkpoint <HOME>/GlassGuard/best.pt --normal_similarity_deg 30 --right_angle_tol_deg 10 --sampled_normals_max 40 --reject_vertical_normals --vertical_normal_max_tilt_deg 20 --vertical_unmask_radius_m 0.12 --target_coverage_percent 60 --save_per_frame_details --per_frame_dir <HOME>/GlassGuard/guard_simple_out/deterministic_plane_fits_v3/per_frame_details --merged_out_ply <HOME>/GlassGuard/guard_simple_out/deterministic_plane_fits_v3/merged_all_frames_masks_planes.ply
