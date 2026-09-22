#!/usr/bin/env python3
"""Occupancy-grid evaluation of per-frame glass-plane predictions against
annotated GT planes (gts/gt_planes.json), in the shared world frame.

For each resolution (0.2 / 0.5 / 1 / 2 m):
  * GT glass voxels  = voxelized GT plane rectangles
  * obstacle voxels  = voxelized union of all frames' lidar clouds
  * per frame, ray-trace from the robot position (360 rig -> full azimuth FOV)
    to every GT glass voxel: it is DIRECTLY VISIBLE if no obstacle voxel blocks
    the line of sight, evaluated at range caps 10 / 8 / 5 / 2 / 1 m.
  * coverage  = visible glass voxels that contain predicted glass points,
                accumulated over frames (sum covered / sum visible)
  * spill     = predicted voxels that are neither GT glass nor real obstacle:
                  - AIR    spill: free-space voxels
                  - GROUND spill: voxels at floor level (even if lidar-occupied --
                    a phantom wall on traversable floor blocks the robot)
                predicted voxels on non-ground obstacles are neutral ("fine").

Outputs a coverage table (resolution x distance) + accumulated spill table,
printed and saved to <root>/gts/occupancy_eval_<pred-suffix>.json.

Usage:
  conda run -n da2 python eval_occupancy.py \
      --root <BASELINES>/MonoGlass3D/monoglass_output --pred-suffix pred_glass
  conda run -n da2 python eval_occupancy.py \
      --root <BASELINES>/MonoGlass3D/monoglass_output \
      --pred-suffix glassrecon_glass --pred-dir <BASELINES>/GlassRecon/output
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).resolve().parent))
from annotate_gt import discover_frames, viewer_to_world, R_STATIC, T_STATIC

RESOLUTIONS = [0.2, 0.5, 1.0, 2.0]
DISTANCES = [1.0, 2.0, 5.0, 8.0, 10.0, 13.0]
SPILL_HORIZON = 10.0   # spill charged only while the robot is within this range (2026-09-14: was 13)
OFF = 1 << 20  # voxel-key packing offset


def keys_of(pts, res):
    g = np.floor(np.asarray(pts, np.float64) / res).astype(np.int64)
    return (g[:, 0] + OFF) | ((g[:, 1] + OFF) << 21) | ((g[:, 2] + OFF) << 42)


def centers_of(keys, res):
    x = (keys & ((1 << 21) - 1)) - OFF
    y = ((keys >> 21) & ((1 << 21) - 1)) - OFF
    z = ((keys >> 42) & ((1 << 21) - 1)) - OFF
    return (np.stack([x, y, z], axis=1).astype(np.float64) + 0.5) * res


def member(sorted_arr, q):
    if len(sorted_arr) == 0:
        return np.zeros(len(q), bool)
    i = np.searchsorted(sorted_arr, q)
    i = np.minimum(i, len(sorted_arr) - 1)
    return sorted_arr[i] == q


def sample_gt_quads(gt_planes, step=0.05):
    """-> (pts, plane_idx): dense samples of every GT quad + which plane each came from."""
    pts, pidx = [], []
    for j, rec in enumerate(gt_planes):
        tl, tr, br, bl = [np.asarray(c, np.float64) for c in rec["corners_world"]]
        n_u = max(2, int(np.linalg.norm(tr - tl) / step))
        n_v = max(2, int(np.linalg.norm(bl - tl) / step))
        u = np.linspace(0, 1, n_u)[:, None, None]
        v = np.linspace(0, 1, n_v)[None, :, None]
        grid = (tl * (1 - u) + tr * u) * (1 - v) + (bl * (1 - u) + br * u) * v
        g = grid.reshape(-1, 3)
        pts.append(g)
        pidx.append(np.full(len(g), j, np.int32))
    if not pts:
        return np.empty((0, 3)), np.empty(0, np.int32)
    return np.concatenate(pts), np.concatenate(pidx)


def plane_active(rec, fid):
    """Dynamic planes are GT only inside their [frame_start, frame_end) window."""
    if not rec.get("dynamic"):
        return True
    fs = rec.get("frame_start")
    fe = rec.get("frame_end")
    if fs is None:
        return True
    return fid >= fs and (fe is None or fid < fe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("GG_BASELINES", os.path.expanduser("~/baselines")) + "/MonoGlass3D/monoglass_output")
    ap.add_argument("--pred-suffix", default="pred_glass",
                    help="NNNNNN_<suffix>.ply (pred_glass=MonoGlass3D, glassrecon_glass=GlassRecon)")
    ap.add_argument("--pred-world-frame", action="store_true", default=False,
                    help="Prediction PLYs are already in WORLD frame (per-frame global tracker export); "
                         "skip viewer_to_world.")
    ap.add_argument("--pred-dir", default=None,
                    help="Flat dir with prediction PLYs (default: inside each frame dir)")
    ap.add_argument("--ground-band", type=float, default=0.25,
                    help="Voxel centers below ground_z + this are 'ground' (m).")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Evaluate every Nth frame (use for dense realTestSaver captures).")
    ap.add_argument("--latency-s", type=float, default=0.0,
                    help="RUNTIME-CADENCE gate: simulate the live node's drop-to-latest "
                         "schedule at this per-frame latency (s). Frames the node would "
                         "have skipped while busy are evaluated as producing NO output. "
                         "0 = off (method keeps up with the capture rate).")
    ap.add_argument("--frame-dt", type=float, default=0.667,
                    help="capture keyframe spacing (s) for the cadence simulation")
    ap.add_argument("--pred-accumulate", action="store_true", default=False,
                    help="CURRENT-map coverage: treat the published map as the RUNNING UNION "
                         "of per-frame predictions (baselines never delete). Off = each "
                         "frame's pred file IS the current map (GK per-frame global maps).")
    ap.add_argument("--expel-win", type=int, default=0,
                    help="Spill self-correction window (inference frames): wrong voxels the method "
                         "stops predicting within this window are NOT counted as spill. 0 = off "
                         "(every wrong voxel counts). Use 5 for methods with spill-removal logic.")
    ap.add_argument("--scene-voxel", type=float, default=0.0,
                    help="Voxel-downsample each cloud on load (m). Keeps RAM sane on dense captures; "
                         "0.05 recommended for realTestSaver data. 0 = off.")
    ap.add_argument("--cam-fov-cfg", default=None, metavar="JSON",
                    help="camera_config.json of a PINHOLE method: GT voxels count as eligible "
                         "(bands AND the Ever denominator) only while inside the pinhole frustum. "
                         "Use for GK-pinhole / MonoGlass3D / GlassRecon so glass the camera never "
                         "saw is not demanded of it. Omit for 360 methods (full-FOV).")
    ap.add_argument("--obstacle-as-covered", action="store_true",
                    help="CURRENT/band/retention only: a GT voxel the LiDAR itself registers as "
                         "an obstacle (>= obstacle-min-pts non-ground returns in the voxel over "
                         "the run) counts as SATISFIED -- protected by LiDAR, so never a miss. "
                         "Demand set is unchanged (full denominators).")
    ap.add_argument("--obstacle-min-pts", type=int, default=5,
                    help="Returns required inside a voxel to call it obstacle-registered (default 5; "
                         "guards against single stray points).")
    ap.add_argument("--demand-empty-only", action="store_true",
                    help="CURRENT/band/retention only: demand only GT voxels that contain NO "
                         "non-ground LiDAR returns themselves (an obstacle-occupied voxel is "
                         "visible to the planner without glass detection). Support eligibility "
                         "(structure NEARBY) still applies. Main-table Ever/bands unchanged.")
    ap.add_argument("--dump-band", default="0,2",
                    help="Which band the dump covers, as 'lo,hi' (default 0,2).")
    ap.add_argument("--dump-fids", default="",
                    help="Comma-separated frame indices to dump (default: all frames with demand).")
    ap.add_argument("--dump-band-dir", default="",
                    help="Debug: write per-frame PLYs of the 0-2m band's demanded GT voxels "
                         "(green=covered, red=missed) + nearby pred points, plus summary.txt.")
    ap.add_argument("--cov-dilate", type=int, default=0,
                    help="CURRENT/retention only: dilate pred voxels by N cells (26-nbhd) for the "
                         "coverage test, i.e. a pred within ~N*res of the GT voxel counts. "
                         "0 = exact cell (legacy). Main-table Ever/bands unchanged.")
    ap.add_argument("--max-height-m", type=float, default=0.0,
                    help="Cap GT + pred voxels to <= ground_z + this (m); 0 = no cap. e.g. 4 = "
                         "robot navigation band, so cross-floor glass above it is ignored.")
    args = ap.parse_args()

    root = Path(args.root)
    gts_dir = root / "gts"
    gt = json.loads((gts_dir / "gt_planes.json").read_text())["planes"]
    print(f"{len(gt)} GT planes")

    frames = discover_frames(root)[::max(1, args.frame_stride)]
    bad_fp = gts_dir / "bad_frames.json"
    if bad_fp.exists():
        bad = set(json.loads(bad_fp.read_text()).get("bad_frames", []))
        frames = [f for f in frames if f["fid"] not in bad]
        print(f"Skipping {len(bad)} bad-aligned frames")
    print(f"{len(frames)} frames (stride {args.frame_stride})")
    pred_dir = Path(args.pred_dir) if args.pred_dir else None

    fov_cam = None
    if args.cam_fov_cfg:
        _c = json.loads(Path(args.cam_fov_cfg).read_text())
        _ci = _c["camera_internal"]
        fov_cam = (float(_ci["fx"]), float(_ci["fy"]), float(_ci["cx"]), float(_ci["cy"]),
                   int(_c["width"]), int(_c["height"]))
        print(f"FOV gating ON: pinhole {fov_cam[4]}x{fov_cam[5]} fx={fov_cam[0]:.0f}")

    def in_pinhole_fov(pts_w, pose):
        """world points -> bool: inside the pinhole frustum at this frame's pose."""
        R_body, T = pose
        V = ((pts_w - T) @ R_body - T_STATIC) @ R_STATIC          # world -> viewer
        cx_, cy_, cz_ = V[:, 0], -V[:, 2], V[:, 1]                # viewer -> camera (x, -z, y)
        fx, fy, cxp, cyp, Wp, Hp = fov_cam
        z = np.where(cz_ > 0.05, cz_, 1.0)
        u = fx * cx_ / z + cxp
        v = fy * cy_ / z + cyp
        return (cz_ > 0.05) & (u >= 0) & (u < Wp) & (v >= 0) & (v < Hp)

    # ── load all clouds + predictions once (world frame) ──
    # WORLD-frame recordings (frame_convention.txt in the scene root): clouds are already
    # registered scans in map coordinates -> no viewer_to_world.
    _fc = root / "frame_convention.txt"
    world_clouds = _fc.exists() and "world" in _fc.read_text()
    if world_clouds:
        print("scene clouds are WORLD-frame (frame_convention.txt): no alignment applied")
    clouds_w, preds_w, robots = [], [], []
    for f in frames:
        pc = np.asarray(o3d.io.read_point_cloud(str(f["cloud"])).points)
        if args.scene_voxel > 0 and len(pc):
            key = np.floor(pc / args.scene_voxel).astype(np.int64)
            _, ui = np.unique(key, axis=0, return_index=True)
            pc = pc[ui]
        clouds_w.append((pc if world_clouds else viewer_to_world(pc, *f["pose"])).astype(np.float32))
        robots.append(viewer_to_world(np.zeros((1, 3)), *f["pose"])[0])
        base = pred_dir if pred_dir else f["cloud"].parent
        pf = base / f"{f['fid']:06d}_{args.pred_suffix}.ply"
        if pf.exists():
            pp = np.asarray(o3d.io.read_point_cloud(str(pf)).points)
            # RAM guard: dedupe predictions on the scene voxel grid. EXACT for every
            # evaluation resolution (aligned grids: a 0.1m cell lies wholly inside each
            # coarser cell), but caps memory on dense per-frame global-map preds.
            if args.scene_voxel > 0 and len(pp):
                _k = np.floor(pp / args.scene_voxel).astype(np.int64)
                _, _ui = np.unique(_k, axis=0, return_index=True)
                pp = pp[_ui].astype(np.float32)
            if args.pred_world_frame:
                preds_w.append(pp.astype(np.float64) if len(pp) else np.empty((0, 3)))
            else:
                preds_w.append(viewer_to_world(pp, *f["pose"]) if len(pp) else np.empty((0, 3)))
        else:
            preds_w.append(np.empty((0, 3)))
    all_scene = np.concatenate(clouds_w)
    ground_z = np.percentile(all_scene[:, 2], 5)
    print(f"scene pts={len(all_scene):,}  ground_z={ground_z:.2f}m")

    gt_pts, gt_pidx = sample_gt_quads(gt)
    if args.max_height_m > 0:
        _zc = ground_z + args.max_height_m
        _gm = gt_pts[:, 2] <= _zc
        gt_pts, gt_pidx = gt_pts[_gm], gt_pidx[_gm]
        for _i in range(len(preds_w)):
            if len(preds_w[_i]):
                preds_w[_i] = preds_w[_i][preds_w[_i][:, 2] <= _zc]
        print(f"height cap: GT+pred <= ground_z+{args.max_height_m:.1f}m ({_zc:.2f}m); "
              f"{len(gt_pts):,} GT sample pts kept")
    dyn_planes = [j for j, rec in enumerate(gt) if rec.get("dynamic")]
    if dyn_planes:
        for j in dyn_planes:
            print(f"  dynamic GT plane #{j}: active frames "
                  f"[{gt[j].get('frame_start')}, {gt[j].get('frame_end')})")

    # ── fine grid for the light trace (occlusion is physical, not eval-grid) ──
    FINE = 0.2
    fine_glass = np.sort(np.unique(keys_of(gt_pts, FINE)))
    fine_centers = centers_of(fine_glass, FINE)
    fine_obst = np.sort(np.unique(keys_of(all_scene, FINE)))
    fine_block = fine_obst[~member(fine_glass, fine_obst)]
    # floor does not block line of sight
    fb_z = centers_of(fine_block, FINE)[:, 2]
    fine_block = np.sort(fine_block[fb_z >= ground_z + args.ground_band])
    print(f"fine trace grid {FINE}m: glass={len(fine_glass)} blockers={len(fine_block)}")

    # CUMULATIVE-BY-APPROACH thresholds: column T = "by the time the robot was T meters
    # away, was this voxel already detected?" -> detected in ANY frame with robot distance
    # in [T, 10]. A far (8-10m) detection therefore counts for every closer column too.
    BANDS = [10.0, 8.0, 5.0, 2.0, 1.0, 0.3]  # thresholds T (max range = max(DISTANCES))
    DIL_OFFS = np.array([dx + (dy << 21) + (dz << 42)
                         for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                        np.int64)  # 26-nbhd key offsets for --cov-dilate

    # per-resolution structures + fine->coarse glass mapping
    per_res = {}
    for res in RESOLUTIONS:
        glass_sorted = np.sort(np.unique(keys_of(gt_pts, res)))
        n_g = len(glass_sorted)
        # voxels of a STATIC plane are always GT; each dynamic plane's voxels switch
        # on/off with its frame window (a static-overlapped voxel stays always-GT).
        static_mask = np.zeros(n_g, bool)
        static_pt = ~np.isin(gt_pidx, dyn_planes)
        if static_pt.any():
            static_mask[np.searchsorted(glass_sorted, np.unique(keys_of(gt_pts[static_pt], res)))] = True
        dyn_vox_idx = {j: np.searchsorted(glass_sorted,
                                          np.unique(keys_of(gt_pts[gt_pidx == j], res)))
                       for j in dyn_planes}
        # fine->coarse glass index; a fine voxel whose CENTER rounds to a coarse cell not in
        # glass_sorted (grid-edge effect) is marked invalid and dropped from the vis mapping.
        _mf_key = keys_of(fine_centers, res)
        _mf_idx = np.searchsorted(glass_sorted, _mf_key)
        _mf_clamp = np.minimum(_mf_idx, n_g - 1)
        _mf_valid = (_mf_idx < n_g) & (glass_sorted[_mf_clamp] == _mf_key)
        per_res[res] = {
            "glass_sorted": glass_sorted,
            "static_mask": static_mask,
            "dyn_vox_idx": dyn_vox_idx,
            "glass_centers": centers_of(glass_sorted, res),
            "obst_sorted": np.sort(np.unique(keys_of(all_scene, res))),
            "map_fine": np.where(_mf_valid, _mf_clamp, 0),
            "map_fine_valid": _mf_valid,
            # per-GT-voxel lifetime records
            "support": None,   # observed structure within 1 voxel of the GT voxel (filled below)
            "eligible": {b: np.zeros(n_g, bool) for b in BANDS},  # visible from that band >=1 frame
            "detected": {b: np.zeros(n_g, bool) for b in BANDS},  # covered while visible in that band
            "ever": np.zeros(n_g, bool),                          # covered in ANY frame (no gating)
            "fov_ever": np.zeros(n_g, bool),                      # in the pinhole frustum >=1 frame
            # spill bookkeeping: {voxel_key: [first_seen, last_seen]} in INFERENCE-frame index.
            # A wrong voxel EXPELLED by the algorithm within EXPEL_WIN inference frames
            # (last-first < EXPEL_WIN) is self-corrected and NOT counted as spill.
            "spill_air": {}, "spill_ground": {}, "neutral": {},
            "acc_air_ticks": [], "acc_gnd_ticks": [],   # per-frame standing spill (accumulate mode)
            # RETENTION: once a GT voxel is correctly covered by the CURRENT map, how often does it
            # STAY covered while the pane is still present? Separates eviction/transience cost from
            # discovery lag (which also suppresses current-coverage for every method).
            "first_cov": np.full(n_g, -1, np.int64),
            "ret_num": 0, "ret_den": 0,
            "gap_run": np.zeros(n_g, np.int64),   # current absence-run length per voxel
            "gap_lens": [], "gap_lens_unrec": [], # closed gaps: recovered vs robot-left
            "prev_d": None,                        # last frame's per-voxel robot distance
            "gap_appr": np.zeros(n_g, bool),       # open gap started while APPROACHING?
            "gapsA": [], "gapsA_un": [],           # gaps opened while approaching: recovered / not
            "gapsR": [], "gapsR_un": [],           # gaps opened while receding:    recovered / not
            "ret_num_appr": 0, "ret_den_appr": 0,  # retention restricted to approaching frames
            # CURRENT-map coverage (time-aligned with concurrent spill): a GT voxel is
            # ELIGIBLE at frame k once it has been visible (and in-frustum when gated) in
            # ANY frame <= k -- previously observed panes REMAIN eligible outside the
            # current FOV, so deletions lose credit (persistence is tested).
            "seen": np.zeros(n_g, bool),
            "cur_num": 0, "cur_den": 0,
            "cur_bands": {bb: [0, 0] for bb in ((0, 2), (2, 5), (5, 8), (8, 13))},
            # per-VOXEL band demand/coverage counters (for protection heat maps)
            "vox_band": {bb: [np.zeros(n_g, np.int64), np.zeros(n_g, np.int64)]
                         for bb in ((0, 2), (2, 5))},
            "acc_pred": set(),                             # --pred-accumulate running union
        }
        print(f"[res {res}m] glass_voxels={n_g}  "
              f"obstacle_voxels={len(per_res[res]['obst_sorted'])}")

    step = FINE * 0.8
    s = 0.6 + np.arange(int((max(DISTANCES) - 0.6) / step) + 1) * step  # (S,)

    # ── runtime-cadence gate: drop-to-latest schedule over uniform keyframes ──
    if args.latency_s > 0:
        _dt, _L = float(args.frame_dt), float(args.latency_s)
        _keep = np.zeros(len(frames), bool)
        _c, _last = 0.0, -1
        while True:
            _i = int(_c // _dt)                     # newest frame arrived by time _c
            if _i <= _last:
                _i = _last + 1                      # nothing new yet: wait for the next
                _c = _i * _dt
            if _i >= len(frames):
                break
            _keep[_i] = True
            _last = _i
            _c = max(_c, _i * _dt) + _L             # busy until inference completes
        for _i in range(len(frames)):
            if not _keep[_i]:
                preds_w[_i] = np.empty((0, 3))
        print(f"runtime cadence: latency {_L:.2f}s over dt {_dt:.3f}s -> "
              f"{int(_keep.sum())}/{len(frames)} frames processed")

    # STRUCTURE-SUPPORT eligibility: a GT glass voxel counts only if the union scene cloud
    # registered ANY surface within one voxel of it -- glass itself returns nothing, but its
    # frames/walls do; zero structure near a pane means the sensor never swept that area
    # (e.g. beyond the run's sensing range), so no method is charged for it.
    for res in RESOLUTIONS:
        R = per_res[res]
        gs = R["glass_sorted"]; ob = R["obst_sorted"]
        sup = np.zeros(len(gs), bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = gs + dx + (dy << 21) + (dz << 42)
                    sup |= member(ob, nb)
        R["support"] = sup
        # self-occupied: the GT voxel ITSELF holds non-ground LiDAR returns (frame/mullion/partial
        # glass return) -> planner sees it without glass detection (--demand-empty-only).
        _ag = all_scene[all_scene[:, 2] > ground_z + args.ground_band]
        R["self_occ"] = member(np.sort(np.unique(keys_of(_ag, res))), gs)
        # obstacle-REGISTERED (for --obstacle-as-covered): >= obstacle-min-pts returns in the
        # voxel, so a lone stray point does not qualify as LiDAR protection.
        _k = keys_of(_ag, res)
        _uk, _cnt = np.unique(_k, return_counts=True)
        R["obst_reg"] = member(np.sort(_uk[_cnt >= args.obstacle_min_pts]), gs)
        if res == 1.0 and args.obstacle_as_covered:
            print(f"[obstacle-credit] {int(R['obst_reg'].sum())}/{len(gs)} GT voxels are "
                  f"LiDAR-registered obstacles (>= {args.obstacle_min_pts} pts) -> count as satisfied")
        if res == 1.0:
            print(f"[support] {int(sup.sum())}/{len(gs)} GT voxels (1.0m) have nearby observed structure")
            if args.demand_empty_only:
                print(f"[empty-only] {int(R['self_occ'].sum())}/{len(gs)} GT voxels are obstacle-occupied "
                      f"-> excluded from CURRENT/retention demand")

    EXPEL_WIN = int(args.expel_win)  # 0 = off: last-first >= 0 always true -> all wrong voxels count
    fids = [f["fid"] for f in frames]
    inf_idx = -1    # counts only frames where this method produced a prediction
    for fi, (fid, T, pred) in enumerate(zip(fids, robots, preds_w)):
        if len(pred):
            inf_idx += 1
        # fine-grid light trace from the robot to every fine glass voxel
        fd = np.linalg.norm(fine_centers - T, axis=1)
        ci = np.where((fd > 0.3) & (fd <= max(DISTANCES)))[0]
        vis_fine = np.zeros(len(fine_glass), bool)
        if len(ci):
            dirs = (fine_centers[ci] - T) / fd[ci, None]
            valid = s[None, :] < (fd[ci] - 0.75 * FINE)[:, None]
            P = T[None, None, :] + dirs[:, None, :] * s[None, :, None]
            k = keys_of(P.reshape(-1, 3), FINE).reshape(len(ci), -1)
            blocked = member(fine_block, k.reshape(-1)).reshape(len(ci), -1) & valid
            vis_fine[ci] = ~blocked.any(axis=1)
        vis_fine[fd <= 0.6] = fd[fd <= 0.6] > 0.3  # too close to march: count as visible

        for res in RESOLUTIONS:
            R = per_res[res]
            glass_sorted = R["glass_sorted"]
            # coarse glass voxel visible if ANY of its fine children is visible
            vis = np.zeros(len(glass_sorted), bool)
            _mfv = R["map_fine_valid"]
            np.maximum.at(vis, R["map_fine"][_mfv], vis_fine[_mfv])
            d = np.linalg.norm(R["glass_centers"] - T, axis=1)

            pred_keys = np.empty(0, np.int64)
            pred_keys_near = np.empty(0, np.int64)   # <= max(DISTANCES) of robot: SPILL judgment only
            if len(pred):
                # COVERAGE uses ALL pred points (no robot-distance crop): the bands already gate by
                # the GT voxel's own distance, so cropping preds to max(DISTANCES) only silently
                # strangled EVER (far/high panes the map held but never approached counted missed).
                pred_keys = np.unique(keys_of(pred, res))
                # SPILL keeps the crop: a wrong voxel is only charged while the robot is within
                # range of it (matches the node's SPILL_DIST_MAX concept; far phantoms aren't
                # navigation hazards until approached).
                pd = np.linalg.norm(pred - T, axis=1)
                pred_keys_near = np.unique(keys_of(pred[pd <= SPILL_HORIZON], res))
            cov_glass = member(np.sort(pred_keys), glass_sorted) if len(pred_keys) else \
                np.zeros(len(glass_sorted), bool)

            # per-frame GT activity: static always on, dynamic only inside its window
            active = R["static_mask"]
            if R["dyn_vox_idx"]:
                active = active.copy()
                for j, vidx in R["dyn_vox_idx"].items():
                    if plane_active(gt[j], fid):
                        active[vidx] = True

            # per-GT-voxel record: cumulative "known by the time robot was T away" + ever
            fovm = None
            if fov_cam is not None:
                fovm = in_pinhole_fov(R["glass_centers"], frames[fi]["pose"])
                # denominator: only GT the camera could ACTUALLY have seen -- in the frustum AND
                # unoccluded (ray-trace vis) in some frame. Frustum-only marked ~every voxel over a
                # long walk (the cone sweeps everything), silently turning the gate into a no-op.
                R["fov_ever"] |= fovm & vis & active & R["support"]
            for b in BANDS:
                in_band = vis & (d >= b) & (d <= max(DISTANCES)) & active & R["support"]
                if fovm is not None:
                    in_band = in_band & fovm
                R["eligible"][b] |= in_band
                R["detected"][b] |= in_band & cov_glass
            R["ever"] |= cov_glass & active

            # ── CURRENT-map coverage (unfiltered by current robot distance) ──
            if len(pred):
                pk_all = np.unique(keys_of(pred, res))
                if args.pred_accumulate:
                    R["acc_pred"].update(int(x) for x in pk_all)
                    pk_cur = np.fromiter(R["acc_pred"], np.int64)
                    pk_cur.sort()
                else:
                    pk_cur = np.sort(pk_all)
                cov_cur = member(pk_cur, glass_sorted)
                if args.cov_dilate > 0:
                    pk_d = pk_cur
                    for _ in range(args.cov_dilate):
                        pk_d = np.unique((pk_d[:, None] + DIL_OFFS[None, :]).ravel())
                    cov_cur = member(pk_d, glass_sorted)
                if args.obstacle_as_covered:
                    cov_cur = cov_cur | R["obst_reg"]   # LiDAR-registered voxel = satisfied
                if args.demand_empty_only:
                    empt = ~R["self_occ"]
                else:
                    empt = None
                seen_now = vis & active & R["support"]
                if fovm is not None:
                    seen_now = seen_now & fovm
                R["seen"] |= seen_now
                # ACTIONABLE demand: the current map is only asked about voxels the robot is
                # NEAR right now (<= 10 m, the placement range) among those ever eligible --
                # a pane 40 m behind the robot or beyond placement range is not a live demand.
                near_now = d <= 10.0
                # CURRENT demand requires the voxel to be answerable THIS frame: line-of-sight
                # visible now, and (pinhole) inside the frustum now -- same rule as retention.
                E = R["seen"] & active & near_now & vis
                if fovm is not None:
                    E = E & fovm
                if empt is not None:
                    E = E & empt
                R["cur_num"] += int((E & cov_cur).sum())
                R["cur_den"] += int(E.sum())
                # RETENTION demand: after first correct coverage, a voxel is demanded only while
                # the map could actually restore it -- robot within 10 m, pane line-of-sight
                # VISIBLE this frame (occluded panes need no restoring), and (pinhole methods)
                # currently inside the frustum. Occlusion/out-of-view FREEZES a gap, it does not
                # close it: if the pane re-emerges still missing, the same gap continues.
                dem = (R["first_cov"] >= 0) & (R["first_cov"] < fi) & active & near_now & vis
                if empt is not None:
                    dem = dem & empt
                if fovm is not None:
                    dem = dem & fovm
                R["ret_num"] += int((dem & cov_cur).sum())
                R["ret_den"] += int(dem.sum())
                # APPROACH split: is the robot currently closing on this voxel?
                appr = (d < R["prev_d"] - 0.02) if R["prev_d"] is not None else np.zeros(len(d), bool)
                R["ret_num_appr"] += int((dem & appr & cov_cur).sum())
                R["ret_den_appr"] += int((dem & appr).sum())
                newly = (R["first_cov"] < 0) & cov_cur & R["seen"] & active
                R["first_cov"][newly] = fi
                recov = dem & cov_cur & (R["gap_run"] > 0)   # demanded, covered again -> gap RECOVERED
                if recov.any():
                    ga = recov & R["gap_appr"]; gr = recov & ~R["gap_appr"]
                    R["gapsA"].extend(R["gap_run"][ga].tolist())
                    R["gapsR"].extend(R["gap_run"][gr].tolist())
                    R["gap_lens"].extend(R["gap_run"][recov].tolist())
                    R["gap_run"][recov] = 0
                opening = dem & ~cov_cur & (R["gap_run"] == 0)   # gap opens this frame
                R["gap_appr"][opening] = appr[opening]
                R["gap_run"][dem & ~cov_cur] += 1            # demanded & absent -> extend
                # (not demanded -> frozen: no change)
                for (lo, hi), acc in R["cur_bands"].items():
                    m = E & (d >= lo) & (d < hi)
                    acc[0] += int((m & cov_cur).sum())
                    acc[1] += int(m.sum())
                    if (lo, hi) in R["vox_band"]:
                        vb = R["vox_band"][(lo, hi)]
                        vb[0][m] += 1                    # demanded
                        vb[1][m & cov_cur] += 1          # covered while demanded
                    # per-frame band dump: EXACTLY the demanded voxels the metric counted
                    _dlo, _dhi = (int(x) for x in args.dump_band.split(","))
                    _fids_ok = (not args.dump_fids) or (fi in {int(x) for x in args.dump_fids.split(",")})
                    if (args.dump_band_dir and res == 1.0 and (lo, hi) == (_dlo, _dhi)
                            and m.any() and _fids_ok):
                        import os as _os
                        _os.makedirs(args.dump_band_dir, exist_ok=True)
                        ctr = R["glass_centers"][m]
                        covm = cov_cur[m]
                        with open(f"{args.dump_band_dir}/f{fi:06d}_gt{_dlo}{_dhi}.ply", "w") as _f:
                            _f.write("ply\nformat ascii 1.0\n"
                                     f"element vertex {len(ctr)}\n"
                                     "property float x\nproperty float y\nproperty float z\n"
                                     "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                                     "end_header\n")
                            for p, cv in zip(ctr, covm):
                                col = "0 255 0" if cv else "255 0 0"
                                _f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {col}\n")
                        _pv = centers_of(pk_cur, res)
                        _pv = _pv[np.linalg.norm(_pv - robots[fi][None, :], axis=1) <= 12.0]
                        with open(f"{args.dump_band_dir}/f{fi:06d}_predvox.ply", "w") as _f:
                            _f.write("ply\nformat ascii 1.0\n"
                                     f"element vertex {len(_pv)}\n"
                                     "property float x\nproperty float y\nproperty float z\nend_header\n")
                            for p in _pv:
                                _f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n")
                        with open(f"{args.dump_band_dir}/summary.txt", "a") as _f:
                            _f.write(f"fid {fi:6d} robot {robots[fi][0]:.1f},{robots[fi][1]:.1f}"
                                     f" dem02 {int(m.sum())} covered {int(covm.sum())}\n")

            # spill classification of predicted voxels (first/last inference-frame tracking)
            # NEAR set only: wrong voxels are charged while the robot is in range (see above)
            if len(pred_keys_near):
                # a hit on an INACTIVE dynamic plane is NOT glass -> falls through to spill
                if len(glass_sorted):
                    ii = np.minimum(np.searchsorted(glass_sorted, pred_keys_near), len(glass_sorted) - 1)
                    is_glass = (glass_sorted[ii] == pred_keys_near) & active[ii]
                else:
                    is_glass = np.zeros(len(pred_keys_near), bool)
                is_obst = member(R["obst_sorted"], pred_keys_near)
                pc_z = centers_of(pred_keys_near, res)[:, 2]
                is_ground = pc_z < ground_z + args.ground_band
                k_inf = inf_idx  # this method's inference-frame counter

                def _mark(dd, keys):
                    for kk in keys:
                        kk = int(kk)
                        e = dd.get(kk)
                        if e is None:
                            dd[kk] = [k_inf, k_inf]
                        else:
                            e[1] = k_inf
                _mark(R["spill_ground"], pred_keys_near[~is_glass & is_ground])
                _mark(R["spill_air"], pred_keys_near[~is_glass & ~is_obst & ~is_ground])
                _mark(R["neutral"], pred_keys_near[~is_glass & is_obst & ~is_ground])
                # ACCUMULATED methods (baselines never delete): the planner sees every wrong voxel
                # from its FIRST detection until end-of-run, not just frames it is re-detected.
                # Track per-frame standing spill = accumulated wrong voxels within the 13m horizon.
                if args.pred_accumulate:
                    for dd, tick in ((R["spill_ground"], R["acc_gnd_ticks"]),
                                     (R["spill_air"], R["acc_air_ticks"])):
                        if dd:
                            kk = np.fromiter(dd.keys(), np.int64)
                            cen = centers_of(kk, res)
                            n_near = int((np.linalg.norm(cen - T, axis=1) <= max(DISTANCES)).sum())
                            tick.append(n_near)
            R["prev_d"] = d.copy()

    # ── spill placement accuracy: distance of each spill voxel to the nearest GT glass ──
    try:
        from scipy.spatial import cKDTree
        gt_kd = cKDTree(gt_pts)
    except Exception:
        gt_kd = None

    def spill_dists(key_set, res):
        if not key_set:
            return np.empty(0, np.float64)
        C = centers_of(np.array(sorted(key_set), np.int64), res)
        if gt_kd is not None:
            dd, _ = gt_kd.query(C, k=1)
            return dd
        return np.array([np.min(np.linalg.norm(gt_pts - c, axis=1)) for c in C])

    def dist_stats(dd):
        if len(dd) == 0:
            return {"n": 0}
        return {"n": int(len(dd)),
                "median_m": float(np.median(dd)), "mean_m": float(np.mean(dd)),
                "p90_m": float(np.percentile(dd, 90)),
                "frac_le_0.5m": float((dd <= 0.5).mean()),
                "frac_0.5_1m": float(((dd > 0.5) & (dd <= 1.0)).mean()),
                "frac_1_2m": float(((dd > 1.0) & (dd <= 2.0)).mean()),
                "frac_gt_2m": float((dd > 2.0).mean())}

    def persistent(dd):
        """Spill voxels NOT expelled: survived >= EXPEL_WIN inference frames."""
        return {k for k, (f0, f1) in dd.items() if (f1 - f0) >= EXPEL_WIN}

    def temporal_spill(dd, n_frames):
        """TEMPORAL spill from the per-voxel [first,last] spans (presence assumed contiguous):
        mean = avg # of wrong voxels present per inference frame; peak = max concurrent."""
        if not dd or n_frames <= 0:
            return {"mean_per_frame": 0.0, "peak_concurrent": 0}
        ev = {}
        for f0, f1 in dd.values():
            ev[f0] = ev.get(f0, 0) + 1
            ev[f1 + 1] = ev.get(f1 + 1, 0) - 1
        cur = peak = 0
        total = 0
        last = None
        for f in sorted(ev):
            if last is not None:
                total += cur * (f - last)
            cur += ev[f]
            peak = max(peak, cur)
            last = f
        return {"mean_per_frame": float(total) / max(1, n_frames), "peak_concurrent": int(peak)}

    # flush gaps still open at end of run -> unrecovered
    for res in RESOLUTIONS:
        _R = per_res[res]
        _open = _R["gap_run"] > 0
        if _open.any():
            _R["gap_lens_unrec"].extend(_R["gap_run"][_open].tolist())
            _R["gapsA_un"].extend(_R["gap_run"][_open & _R["gap_appr"]].tolist())
            _R["gapsR_un"].extend(_R["gap_run"][_open & ~_R["gap_appr"]].tolist())
            _R["gap_run"][_open] = 0

    results = {}
    for res in RESOLUTIONS:
        R = per_res[res]
        n_g = len(R["glass_sorted"])
        n_air_seen, n_gnd_seen = len(R["spill_air"]), len(R["spill_ground"])
        _raw_air = dict(R["spill_air"]); _raw_gnd = dict(R["spill_ground"])
        R["spill_air"] = persistent(R["spill_air"])
        R["spill_ground"] = persistent(R["spill_ground"])
        R["neutral"] = persistent(R["neutral"])
        air_d = spill_dists(R["spill_air"], res)
        gnd_d = spill_dists(R["spill_ground"], res)
        results[res] = {
            "glass_voxels": n_g,
            "band_coverage": {f"by_{b:g}m": {
                "eligible_voxels": int(R["eligible"][b].sum()),
                "detected_voxels": int(R["detected"][b].sum()),
                "coverage": (float(R["detected"][b].sum()) / float(R["eligible"][b].sum())
                             if R["eligible"][b].any() else float("nan")),
            } for b in BANDS},
            "ever_detected_voxels": int(R["ever"].sum()),
            # SHARED-DENOMINATOR scheme: every band over the SAME denominator (voxels ever
            # eligible = visible [and in-frustum when FOV-gated] at some qualifying distance),
            # so columns are monotone by construction. ever_shared keeps occluded-time credit:
            # a voxel the map covers in any frame counts, as long as it was ever eligible.
            "shared": {
                "eligible_voxels": int(R["eligible"][min(BANDS)].sum()),
                "bands": {f"by_{b:g}m": int(R["detected"][b].sum()) for b in BANDS},
                "ever_detected_voxels": int((R["ever"] & R["eligible"][min(BANDS)]).sum()),
            },
            # FOV-gated runs: Ever = detected-in-FOV / GT-ever-in-FOV (glass the camera never
            # saw is not demanded); ungated runs keep the original all-GT denominator.
            "ever_detected_frac": (float((R["ever"] & R["eligible"][min(BANDS)]).sum())
                                   / max(1, int(R["eligible"][min(BANDS)].sum()))),
            "fov_gated": fov_cam is not None,
            "fov_glass_voxels": int(R["fov_ever"].sum()) if fov_cam is not None else n_g,
            "current_coverage": {
                "overall": (float(R["cur_num"]) / R["cur_den"]) if R["cur_den"] else float("nan"),
                "num": int(R["cur_num"]), "den": int(R["cur_den"]),
                "by_band": {f"{lo}-{hi}m": {
                    "coverage": (float(a[0]) / a[1]) if a[1] else float("nan"),
                    "num": int(a[0]), "den": int(a[1])}
                    for (lo, hi), a in R["cur_bands"].items()},
                "accumulated_pred": bool(args.pred_accumulate),
            },
            "retention": {"num": int(R["ret_num"]), "den": int(R["ret_den"]),
                          "frac": (float(R["ret_num"]) / R["ret_den"]) if R["ret_den"] else None},
            "gaps": {"n_recovered": len(R["gap_lens"]), "n_unrecovered": len(R["gap_lens_unrec"]),
                     "median_frames": (float(np.median(R["gap_lens"])) if R["gap_lens"] else None),
                     "p90_frames": (float(np.percentile(R["gap_lens"], 90)) if R["gap_lens"] else None),
                     "approaching": {"n_rec": len(R["gapsA"]), "n_unrec": len(R["gapsA_un"]),
                                     "median_frames": (float(np.median(R["gapsA"])) if R["gapsA"] else None)},
                     "receding": {"n_rec": len(R["gapsR"]), "n_unrec": len(R["gapsR_un"]),
                                  "median_frames": (float(np.median(R["gapsR"])) if R["gapsR"] else None)}},
            "retention_approaching": {"num": int(R["ret_num_appr"]), "den": int(R["ret_den_appr"]),
                                      "frac": (float(R["ret_num_appr"]) / R["ret_den_appr"]) if R["ret_den_appr"] else None},
            "spill_air_voxels": len(R["spill_air"]),
            # accumulate mode: the planner sees every wrong voxel from FIRST detection until the
            # end of the run (baselines never delete) -- per-frame standing spill comes from the
            # accumulated-within-13m tick counts, not the re-detection spans.
            "spill_air_temporal": ({"mean_per_frame": float(np.sum(R["acc_air_ticks"])) / max(1, inf_idx + 1),
                                    "peak_concurrent": int(max(R["acc_air_ticks"])) if R["acc_air_ticks"] else 0}
                                   if args.pred_accumulate else
                                   temporal_spill({k: v for k, v in _raw_air.items()}, inf_idx + 1)),
            "spill_ground_temporal": ({"mean_per_frame": float(np.sum(R["acc_gnd_ticks"])) / max(1, inf_idx + 1),
                                       "peak_concurrent": int(max(R["acc_gnd_ticks"])) if R["acc_gnd_ticks"] else 0}
                                      if args.pred_accumulate else
                                      temporal_spill({k: v for k, v in _raw_gnd.items()}, inf_idx + 1)),
            "spill_air_expelled": n_air_seen - len(R["spill_air"]),
            "spill_ground_voxels": len(R["spill_ground"]),
            "spill_ground_expelled": n_gnd_seen - len(R["spill_ground"]),
            "neutral_obstacle_voxels": len(R["neutral"]),
            "spill_air_m3": len(R["spill_air"]) * res ** 3,
            "spill_ground_m3": len(R["spill_ground"]) * res ** 3,
            "spill_air_dist_to_gt": dist_stats(air_d),
            "spill_ground_dist_to_gt": dist_stats(gnd_d),
        }

    # ── tables ──
    name = args.pred_suffix
    print("\n" + "=" * 86)
    print(f"PER-GT-VOXEL GLASS DETECTION — prediction: {name}")
    print("(column 'by Tm' = voxel already detected at ANY distance >= T when robot reached T;")
    print(" far detections count for all closer columns. EVER = any frame, no gating.)")
    print("=" * 86)
    hdr = "res \\ col   ".join(f"{f'by {b:g}m':>10}" for b in BANDS) + f"{'EVER':>10}"
    print(hdr)
    for res in RESOLUTIONS:
        R = per_res[res]
        row = f"{res:>6.1f} m   "
        for b in BANDS:
            el = int(R["eligible"][b].sum())
            c = results[res]["band_coverage"][f"by_{b:g}m"]["coverage"]
            row += f"{c * 100:>9.1f}%" if el else f"{'n/a':>10}"
        row += f"{results[res]['ever_detected_frac'] * 100:>9.1f}%"
        print(row)
    print("\n(eligible GT voxels per band:)")
    for res in RESOLUTIONS:
        R = per_res[res]
        row = f"{res:>6.1f} m   "
        for b in BANDS:
            row += f"{int(R['eligible'][b].sum()):>10}"
        row += f"{results[res]['glass_voxels']:>10}"
        print(row)

    print(f"\nSPILL — PERSISTENT only (wrong voxels expelled by the method within {EXPEL_WIN} "
          f"inference frames are NOT counted) + distance to nearest GT glass")
    print(f"{'res':>6} {'air vox':>8} {'expelled':>9} {'m^3':>7} {'med':>6} {'p90':>6} {'<=0.5m':>7} {'>2m':>6} "
          f"| {'gnd vox':>8} {'m^3':>7} {'med':>6} | {'on-obst':>8}")
    for res in RESOLUTIONS:
        r = results[res]
        ad = r["spill_air_dist_to_gt"]; gd = r["spill_ground_dist_to_gt"]
        adm = f"{ad.get('median_m', 0):.2f}" if ad["n"] else "  -"
        adp = f"{ad.get('p90_m', 0):.2f}" if ad["n"] else "  -"
        adf = f"{ad.get('frac_le_0.5m', 0) * 100:.0f}%" if ad["n"] else "  -"
        adf2 = f"{ad.get('frac_gt_2m', 0) * 100:.0f}%" if ad["n"] else "  -"
        gdm = f"{gd.get('median_m', 0):.2f}" if gd["n"] else "  -"
        print(f"{res:>5.1f}m {r['spill_air_voxels']:>8} {r['spill_air_expelled']:>9} {r['spill_air_m3']:>7.2f} {adm:>6} {adp:>6} "
              f"{adf:>7} {adf2:>6} | {r['spill_ground_voxels']:>8} {r['spill_ground_m3']:>7.2f} {gdm:>6} "
              f"| {r['neutral_obstacle_voxels']:>8}")

    # per-voxel protection stats at 1.0m (heat maps): centers + dem/cov per band
    if 1.0 in per_res:
        _R = per_res[1.0]
        np.savez(gts_dir / f"voxel_protection_{name}.npz",
                 centers=_R["glass_centers"],
                 dem02=_R["vox_band"][(0, 2)][0], cov02=_R["vox_band"][(0, 2)][1],
                 dem25=_R["vox_band"][(2, 5)][0], cov25=_R["vox_band"][(2, 5)][1])
    out = gts_dir / f"occupancy_eval_{name}.json"
    out.write_text(json.dumps(
        {"pred_suffix": name, "n_frames": len(frames), "n_gt_planes": len(gt),
         "ground_z": ground_z, "mode": "per_gt_voxel_bands",
         "results": {str(k): v for k, v in results.items()}}, indent=1, default=float))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
