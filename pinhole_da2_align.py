"""Pinhole DA2 <-> lidar alignment for the live glass-killer node.

Purpose
-------
The 360 lidar cloud is projected into an equirectangular pano and used to build the range image
the glass pipeline needs. If the lidar<->camera extrinsic has a small rotational/latency error the
projection smears. This module corrects it PER FRAME using the "habitat" pinhole DA2 method
(ported from batch_combined.py):

  1. Reproject the equirect RGB into a fixed FRONT-facing PINHOLE image (we own the intrinsics).
  2. Run DepthAnything-V2 on that pinhole RGB  -> dense metric depth.
  3. Project the LAST SINGLE lidar scan into the same pinhole  -> sparse lidar depth.
     (The single scan, NOT the 5s stack, because the stack stacks points seen THROUGH glass from
      several viewpoints -> a false "see-through" surface.)
  4. Solve a 6-DOF rigid pose (Powell) that maximises the rank-normalised Pearson correlation
     between the lidar-projected depth grid and the DA2 depth grid.

The returned pose transforms lidar points into the camera/DA2 frame; the node applies it to the
whole cloud (an extrinsic correction is global) before the equirect projection.

Coordinate frame: camera frame everywhere -- +X right, +Y down, +Z forward. The front pinhole
looks down +Z, i.e. the pano centre (u = W/2). This matches bsp._project_pano_xyz / the node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:                                                  # scipy is already a dep (cKDTree in bsp)
    from scipy.optimize import minimize as _scipy_minimize
except Exception:                                     # pragma: no cover
    _scipy_minimize = None


# ─── Pinhole camera config ────────────────────────────────────────────────────
@dataclass
class PinholeCfg:
    """Front-facing pinhole we synthesise from the pano. We choose the intrinsics, so RGB reprojection,
    lidar projection and DA2 all share them. Default = 90 deg horizontal FOV, 640x480."""
    out_w: int = 640
    out_h: int = 480
    fx: float = 320.0        # 320 / tan(45deg) => 90 deg horizontal FOV
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0
    yaw_deg: float = 0.0     # view direction within the 360 (0 = front / pano centre)
    pitch_deg: float = 0.0
    pano_pixel_offset: float = 1.0
    cell_size: int = 8       # Powell alignment grid cell (px)
    max_pts: int = 10_000    # lidar pts subsampled per solve
    min_score: float = 0.90  # below this the alignment is untrusted (caller may skip applying it)


# ─── Equirect -> pinhole reprojection ────────────────────────────────────────
def build_equirect_to_pinhole_remap(pano_w: int, pano_h: int, cfg: PinholeCfg
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute cv2.remap maps that sample the equirect pano for every pinhole pixel. Depends only on
    (pano size, cfg) -> build once and cache. Returns (map_x, map_y) float32 of shape (out_h, out_w)."""
    W, H = int(pano_w), int(pano_h)
    uu, vv = np.meshgrid(np.arange(cfg.out_w, dtype=np.float64), np.arange(cfg.out_h, dtype=np.float64))
    x = (uu - cfg.cx) / cfg.fx
    y = (vv - cfg.cy) / cfg.fy
    z = np.ones_like(x)
    # optional yaw (about +Y / down) and pitch (about +X / right) of the view direction
    if cfg.yaw_deg or cfg.pitch_deg:
        ya = np.radians(cfg.yaw_deg); pa = np.radians(cfg.pitch_deg)
        Ry = np.array([[np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(pa), -np.sin(pa)], [0, np.sin(pa), np.cos(pa)]])
        R = Ry @ Rx
        xyz = np.stack([x, y, z], -1) @ R.T
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    hori = np.sqrt(x * x + z * z)
    off = float(cfg.pano_pixel_offset)
    map_x = (W / (2.0 * np.pi) * np.arctan2(x, z) + W / 2.0 + off).astype(np.float32)
    map_y = (W / (2.0 * np.pi) * np.arctan2(y, hori + 1e-6) + H / 2.0 + off).astype(np.float32)
    return map_x, map_y


def equirect_to_pinhole_rgb(pano_bgr: np.ndarray, remap: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    import cv2
    map_x, map_y = remap
    return cv2.remap(pano_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)   # WRAP: the pano is 360, seam is continuous in u


# ─── Cloud -> pinhole sparse depth ───────────────────────────────────────────
def project_cloud_to_pinhole_depth(xyz_cam: np.ndarray, cfg: PinholeCfg) -> np.ndarray:
    """Project camera-frame points into the pinhole; keep the nearest-Z point per pixel. -> (out_h,out_w)
    float32 sparse depth (0 = empty)."""
    P = np.asarray(xyz_cam, np.float32).reshape(-1, 3)
    depth = np.zeros((cfg.out_h, cfg.out_w), np.float32)
    if P.shape[0] == 0:
        return depth
    z = P[:, 2]
    ok = z > 0.05
    if not np.any(ok):
        return depth
    P = P[ok]; z = z[ok]
    u = P[:, 0] / z * cfg.fx + cfg.cx
    v = P[:, 1] / z * cfg.fy + cfg.cy
    ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
    inb = (ui >= 0) & (ui < cfg.out_w) & (vi >= 0) & (vi < cfg.out_h)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    if ui.size == 0:
        return depth
    pid = vi * cfg.out_w + ui
    order = np.lexsort((z, pid))                      # nearest-Z winner per pixel
    _, first = np.unique(pid[order], return_index=True)
    w = order[first]
    depth[vi[w], ui[w]] = z[w]
    return depth


# ─── Powell alignment (ported verbatim from batch_combined.py) ────────────────
def _rank_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    return ranks / float(len(x) - 1) if len(x) > 1 else np.zeros_like(x)


def _corr_manual(a, b):
    a = a.astype(np.float32, copy=False) - a.mean()
    b = b.astype(np.float32, copy=False) - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-6
    return float(np.dot(a, b)) / den


def _build_grid_fast(depth_img, cell_size):
    H, W = depth_img.shape
    rows, cols = H // cell_size, W // cell_size
    d = depth_img[:rows * cell_size, :cols * cell_size]
    d = d.reshape(rows, cell_size, cols, cell_size).transpose(0, 2, 1, 3)
    d = d.reshape(rows, cols, cell_size * cell_size).astype(np.float32)
    vm = d > 0.01
    vc = vm.sum(axis=2)
    gv = vc > 0
    gd = np.where(gv, np.where(vm, d, 0.0).sum(axis=2) / np.maximum(vc, 1), 0.0)
    return gd, gv


def _build_lidar_grid_direct(pts, fx, fy, cx, cy, rows_g, cols_g, cell_size):
    Z = pts[:, 2]; ok = Z > 0.01
    pts, Z = pts[ok], Z[ok]
    gc_i = ((fx * pts[:, 0] / Z + cx) / cell_size).astype(np.int32)
    gr_i = ((fy * pts[:, 1] / Z + cy) / cell_size).astype(np.int32)
    valid = (gc_i >= 0) & (gc_i < cols_g) & (gr_i >= 0) & (gr_i < rows_g)
    gc_i, gr_i, Z = gc_i[valid], gr_i[valid], Z[valid]
    n = rows_g * cols_g
    cid = gr_i * cols_g + gc_i
    sums = np.bincount(cid, weights=Z.astype(np.float32), minlength=n).astype(np.float32)
    counts = np.bincount(cid, minlength=n).astype(np.float32)
    gv = counts > 0
    gd = np.where(gv, sums / np.maximum(counts, 1.0), 0.0).reshape(rows_g, cols_g)
    return gd, gv.reshape(rows_g, cols_g)


def _unproject_depth(dm, fx, fy, cx, cy):
    rows, cols = np.where(dm > 0.01)
    Z = dm[rows, cols].astype(np.float32)
    X = ((cols - cx) * Z / fx).astype(np.float32)
    Y = ((rows - cy) * Z / fy).astype(np.float32)
    return np.stack([X, Y, Z], axis=1)


def transform_pts(pts, rx_deg, ry_deg, rz_deg, tx, ty, tz):
    """Rigid transform of camera-frame points by (Rz Ry Rx) + t. Public: the node applies the solved
    pose to the full cloud with this."""
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], np.float32)
    return pts @ (Rz @ Ry @ Rx).T + np.array([tx, ty, tz], np.float32)


def solve_lidar_pose(lidar_depth, da2g, dv, fx, fy, cx, cy, cell_size, H, W,
                     max_pts=10_000, rng_seed=42):
    """6-DOF Powell solve: transform the lidar points so their projected depth-grid best rank-correlates
    with the DA2 depth-grid. Returns {rx,ry,rz(deg), tx,ty,tz(m), score in [-1,1], n_iter}."""
    if _scipy_minimize is None:
        return {"rx": 0.0, "ry": 0.0, "rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0,
                "score": 0.0, "n_iter": 0}
    rows_g = H // cell_size
    cols_g = W // cell_size
    pts0 = _unproject_depth(lidar_depth, fx, fy, cx, cy)
    if len(pts0) == 0:
        return {"rx": 0.0, "ry": 0.0, "rz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0,
                "score": 0.0, "n_iter": 0}
    if len(pts0) > max_pts:
        rng = np.random.default_rng(rng_seed)
        pts0 = pts0[rng.choice(len(pts0), max_pts, replace=False)]
    base_valid = dv & np.isfinite(da2g) & (da2g > 0)
    da2_rank_grid = np.zeros(da2g.shape, np.float32)
    da2_rank_grid[base_valid] = _rank_normalize(da2g[base_valid].astype(np.float32))
    _TS = np.float32(0.05)   # 1 param unit == 5 cm

    def _neg(p):
        p = np.asarray(p, dtype=np.float32)
        pts = transform_pts(pts0, float(p[0]), float(p[1]), float(p[2]),
                            float(p[3]) * _TS, float(p[4]) * _TS, float(p[5]) * _TS)
        lg, lv = _build_lidar_grid_direct(pts, fx, fy, cx, cy, rows_g, cols_g, cell_size)
        mask = lv & base_valid
        if mask.sum() < 8:
            return 0.0
        score = _corr_manual(_rank_normalize(lg[mask]), da2_rank_grid[mask])
        return -(0.5 * (score + 1.0))

    res = _scipy_minimize(_neg, np.zeros(6), method="Powell",
                          options={"maxiter": 300, "ftol": 1e-4, "xtol": 1e-3})
    p = res.x
    return {"rx": float(p[0]), "ry": float(p[1]), "rz": float(p[2]),
            "tx": float(p[3]) * float(_TS), "ty": float(p[4]) * float(_TS),
            "tz": float(p[5]) * float(_TS),
            "score": float(-res.fun), "n_iter": int(res.nfev)}


# ─── Top-level orchestrator ──────────────────────────────────────────────────
def align_scan_to_da2(pano_bgr: np.ndarray,
                      scan_xyz_cam: np.ndarray,
                      da2_infer: Callable[[np.ndarray], np.ndarray],
                      cfg: PinholeCfg,
                      remap: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
    """Full pinhole DA2<->lidar alignment for one frame.

    pano_bgr     : equirectangular BGR (H_pano, W_pano, 3)
    scan_xyz_cam : LAST single lidar scan in CAMERA frame (N,3)  (+X right,+Y down,+Z fwd)
    da2_infer    : callable(bgr_pinhole) -> metric depth (out_h, out_w) float32
    remap        : optional prebuilt (map_x,map_y); built here if None.

    Returns dict: pose(dict), score, rgb_pinhole, da2_depth, lidar_depth, remap.
    """
    ph, pw = pano_bgr.shape[:2]
    if remap is None:
        remap = build_equirect_to_pinhole_remap(pw, ph, cfg)
    rgb_pin = equirect_to_pinhole_rgb(pano_bgr, remap)
    da2 = np.asarray(da2_infer(rgb_pin), np.float32)
    lidar_depth = project_cloud_to_pinhole_depth(scan_xyz_cam, cfg)
    da2g, dv = _build_grid_fast(da2, cfg.cell_size)
    pose = solve_lidar_pose(lidar_depth, da2g, dv, cfg.fx, cfg.fy, cfg.cx, cfg.cy,
                            cfg.cell_size, cfg.out_h, cfg.out_w, max_pts=cfg.max_pts)
    return {"pose": pose, "score": pose["score"], "rgb_pinhole": rgb_pin,
            "da2_depth": da2, "lidar_depth": lidar_depth, "remap": remap}
