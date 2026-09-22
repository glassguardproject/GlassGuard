#!/usr/bin/env python3
"""
glass_killer_ros_node.py

End-to-end ROS 2 wrapper around the batch_bigmask_4ray_randomopt.py clench
vertical-rectangle plane-fitting pipeline (decision=BIGMASK_VERTICAL_SEED_OVERLAP).
Instead of reading cloud_*.ply + rgb_*.png from disk and writing PLYs, this node
runs the algorithm live on ROS messages and PUBLISHES the result (no disk I/O):

  subscribes:
    /glass_killer/cloud   (sensor_msgs/PointCloud2)  -- the camera-frame cloud
                                                         published by glassKillerNode
    /habitat/rgb          (sensor_msgs/Image, bgr8)  -- the 360 panorama RGB
  publishes:
    /glass_killer/planes  (sensor_msgs/PointCloud2, xyz+rgb)
                           -- the fitted window planes + normal lines, in the
                              SAME frame/stamp as /glass_killer/cloud so they
                              overlay it directly in RViz (no PLY saved).

It always processes the most recent (cloud, image) pair and drops anything that
arrives while a frame is being computed, so a ~0.7 s/frame pipeline never backs
up -- it just runs at whatever rate it can (~1.4 Hz) and skips the rest.

Run it in an environment that has BOTH ROS 2 (rclpy) and the sam3/torch stack:

  source /opt/ros/jazzy/setup.bash
  source <ROS_WS>/install/setup.bash
  conda run -n sam3 --no-capture-output python ./glass_killer_ros_node.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import queue
import threading
from collections import deque
from types import SimpleNamespace

# Must be set BEFORE torch initializes CUDA: expandable segments trim the reserved-pool overhead and
# avoid fragmentation OOM (see the VRAM analysis). Harmless if the caller already set it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import cv2
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import StaticTransformBroadcaster


# Fixed sensor->camera_link mount transform baked into glassKillerNode's cloud:
# the body-frame camera rotation + (x,z,-y) viewer swap collapse to a -90 deg yaw.
_R_STATIC = np.array([[0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)   # Rz(-90 deg)
_T_STATIC = np.array([-0.12, -0.075, 0.255], dtype=np.float64)  # camX, camY, camZ


def _R_to_quat(R: np.ndarray):
    """Rotation matrix -> (qx, qy, qz, qw), Shepperd's method (numerically safe)."""
    R = np.asarray(R, np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2.0
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return (q[0], q[1], q[2], q[3])


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)

# Reuse the newest batch pipeline (clench vertical-rectangle algorithm,
# decision=BIGMASK_VERTICAL_SEED_OVERLAP): importing it sets up sys.path, loads the
# glass_frame_ring / glass_killer_deterministic / debug_combined_frame helpers, and
# exposes every function we need. main() only runs under __main__, so import is
# side-effect-safe here. This node runs the algorithm live and PUBLISHES results;
# it never writes any PNG/PLY to disk.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent: bsp, pda
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # this folder: transport
import batch_bigmask_4ray_randomopt as bsp
import pinhole_da2_align as pda
import pipeline_transport as tx
from std_msgs.msg import String


# --------------------------------------------------------------------------- #
# Config: defaults mirror the working command at the bottom of the batch file.
# Override any of these with ROS params of the same name (dashes -> underscores).
# --------------------------------------------------------------------------- #
def build_args(node: "GlassKillerNode") -> SimpleNamespace:
    g = lambda name, default: node.declare_parameter(name, default).value

    # Defaults: INT8 SAM3, 2816 student (per request).
    ckpt = g("ckpt_path", "./slim_sam3/checkpoints/student_final.pt")
    meta = g("meta_json", "./slim_sam3/checkpoints/mlp_pruned_meta.json")
    feats = g("cached_text_features", "./slim_sam3/prompt_features/window_glass.pt")
    precision = str(g("precision", "bf16")).lower()   # bf16 (default) | int8 | fp32

    # Synthesize argv and let the batch's own parser fill EVERY default/type,
    # so we never drift from the script's argument contract.
    argv = [
        "--habitat-dir", "/unused",
        "--cached-text-features", feats,
        "--prompt", *list(g("prompt", ["glass", "window"])),
        "--max-samples", "1",
    ]
    if g("student", True):
        argv.append("--student")
    if ckpt:
        argv += ["--ckpt-path", ckpt]
    if meta:
        argv += ["--meta-json", meta]
    if precision == "int8":
        argv.append("--int8")
    elif precision == "bf16":
        argv.append("--bf16")
    argv += ["--device", g("device", "cuda")]
    # SAM detection confidence. Raised to 0.4: at 0.2 some low-quality masks slipped through and
    # produced bad planes. Override with -p conf_th:=<value> (batch replays should pass the same).
    argv += ["--conf-th", str(float(g("conf_th", 0.4)))]
    # Min seed-coverage a plane must hold to be placed (fraction). 0.0 -> no minimum, so the pipeline
    # just keeps the best-coverage candidate. Batch default is 0.20.
    argv += ["--clench-min-coverage-frac", str(float(g("min_cov", 0.20)))]
    argv += ["--grounding-cell-size", str(g("grounding_cell_size", 10))]
    # Plane-fit depth search ceiling (batch default 15). Must cover PLACE_DIST_MAX: a 20m
    # range run needs -p opt_depth_max_m:=20 or far planes can't fit at all.
    argv += ["--opt-depth-max-m", str(float(g("opt_depth_max_m", 15.0)))]
    # projection range crop: points beyond this are dropped before anything. Must be >= the
    # provider maxRange for a long-range (RANGE_M) run, else far cloud points never reach the node.
    argv += ["--depth-max", str(float(g("depth_max", 20.0)))]
    # 360-style support quad for pinhole corners (solid edge orients, mask sizes). Batch default
    # is ON; pass the explicit on/off flag so -p quad_support:=false can actually disable it.
    argv += ["--pinhole-quad-support" if bool(g("quad_support", True)) else "--no-pinhole-quad-support"]
    # Widen solved quads over mask parts dropped by largest-component cleaning (default OFF: knob).
    if bool(g("extend_full_mask", False)):
        argv += ["--extend-full-mask", "--extend-mask-gap-px", str(int(g("extend_mask_gap_px", 40)))]
    # Speed knobs (live-tunable): horizontal-bar candidates per mask (0 = OFF, drops the ~2D-RANSAC
    # bar cost) and the number of random pillar pairs tried per mask.
    argv += ["--clench-h-max-pillars", str(int(g("clench_h_max_pillars", 8)))]
    argv += ["--clench-random-tries", str(int(g("clench_random_tries", 15)))]
    # Building A: occlusion check OFF by default (matches the batch --no-occ-check runs). Parallelogram
    # check now OFF by default too (evaluating placement without it). Re-enable with par_check:=true.
    if not bool(g("occ_check", False)):
        argv.append("--no-occ-check")
    if not bool(g("par_check", False)):
        argv.append("--no-par-check")
    # Param-check variant. DEFAULT (classic): the ORIGINAL gate -- raw 4-corner-ray touch quad,
    # opposite-side parallelogram test, irregular-quad detection + the corner-move fallback
    # (repairs a single bad corner), bbox retry opt-in. diag_robust_quad:=true switches to the
    # DEPTH-AWARE robust-diagonal gate (trusted-diagonal rectangle + off-corner dh/dw residuals,
    # 1-of-2 outlier tolerance, strict 2/2 under robust_min_az_deg for sliver masks).
    if bool(g("diag_robust_quad", False)):
        argv.append("--clench-diag-robust-quad")
    argv += ["--clench-robust-min-az-deg", str(float(g("robust_min_az_deg", 8.0)))]
    # CURVE-REPAIR corner fallback (the new irregularity rule, validated on the 60-frame panels):
    # score the two horizontal through-corner great circles vs the mask boundary; one failed ->
    # solve the parallel curve + shoot the good side's columns onto it; both failed -> keep raw
    # corners. Replaces the corner-move fallback. Disable with curve_repair:=false.
    if bool(g("curve_repair", True)):
        argv.append("--corner-curve-repair")
    # 360 ANGLE-PAR (opt-in): tau/length-adaptive angle gate (great-circle d3 vs plane facing) instead
    # of the classic dv/dh side-length par test. Rescues oblique-but-correctly-facing planes. Needs
    # curve_repair. tau reuses pinhole_par_tau_m (default 0.5 = half nav voxel).
    if bool(g("par_360_angle", False)):
        argv.append("--par-360-angle")
        argv += ["--pinhole-par-tau-m", str(float(g("par_360_tau_m", 0.5)))]
    # Angle-gate disarm threshold (support-edge px span under which d3 is deemed ill-conditioned
    # -> dv/dh fallback). 0 = never disarm: a trusted support line ALWAYS judges via the angle
    # test (the mask sizes the quad; dv/dh only bldgA untrusted masks with no support line).
    argv += ["--pinhole-dir-trust-min-span-px", str(float(g("par_min_span_px", 140.0)))]
    # CONDITIONING floor for the trusted support-edge direction (both camera models): rules out
    # the camera-height/short-span degeneracy where a straight line carries no yaw information.
    argv += ["--dir-trust-min-cond", str(float(g("dir_trust_min_cond", 0.02)))]
    # SYMMETRIC seed band (in+out around the mask boundary) -- captures frame/rail returns
    # projecting just inside the mask. Launcher knob SEED_BAND_SYM.
    if bool(g("seed_band_symmetric", False)):
        argv.append("--seed-band-symmetric")
    # SMALL-MASK FALLBACK: a failed big mask retries with its largest owned smalls.
    # Launcher knob SMALL_FALLBACK.
    if bool(g("small_fallback", False)):
        argv.append("--clench-small-fallback")
    # HORIZONTAL-BAR recovery: the edge-span restriction + tight corner-ray direction gate were
    # discarding silhouette-supported horizontal bars. h_edge_curve=false uses ALL edge columns;
    # h_ray_hit_tol_px loosens the direction gate (14 default was too tight for wide/oblique walls).
    if not bool(g("h_edge_curve", True)):
        argv.append("--no-clench-h-edge-curve")
    argv += ["--clench-h-ray-hit-tol-px", str(float(g("h_ray_hit_tol_px", 14.0)))]
    # Vertical-pillar COLUMN-NESS bldgA (validated on keysave frame 216 mask 5: 3 real mullions
    # fill 0.57-0.91 kept, 2 scattered-stray cells fill 0.18-0.21 rejected): occupied/spanned layer
    # fill ratio + horizontal seed extent per 1m cell. pillar_min_fill:=0 disables.
    argv += ["--clench-pillar-depth-gap-m", str(float(g("pillar_depth_gap_m", 0.2)))]
    argv += ["--clench-pillar-min-fill", str(float(g("pillar_min_fill", 0.5)))]
    argv += ["--clench-pillar-max-xz-ext-m", str(float(g("pillar_max_xz_ext_m", 99.0)))]   # extent gate OFF (fill carries it)
    # Horizontal-bar plane height: bar height now comes from the bar REP's depth (not the far
    # endpoint), and finalize caps it to max((1+oversize) x seed height, this floor).
    argv += ["--clench-h-clamp-height-floor-m", str(float(g("h_clamp_height_floor_m", 3.0)))]

    saved = sys.argv
    try:
        sys.argv = ["glass_killer_ros_node"] + argv
        args = bsp.parse_args()
    finally:
        sys.argv = saved
    # Live path: keep the per-track plane-merge diagnostic OFF (it builds a string per track per plane).
    args.track_debug = bool(g("track_debug", False))
    return args


def _parse_cloud_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract Nx3 float32 xyz from a PointCloud2 regardless of extra fields."""
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    dt = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [np.float32, np.float32, np.float32],
        "offsets": [off["x"], off["y"], off["z"]],
        "itemsize": msg.point_step,
    })
    raw = np.frombuffer(bytes(msg.data), dtype=dt)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def _parse_cloud_xyzi(msg: PointCloud2):
    """Extract (Nx3 xyz, N intensity) float32 from a PointCloud2 (e.g. /terrain_map, where
    intensity = height above the estimated ground)."""
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z", "intensity")):
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32)
    dt = np.dtype({
        "names": ["x", "y", "z", "intensity"],
        "formats": [np.float32, np.float32, np.float32, np.float32],
        "offsets": [off["x"], off["y"], off["z"], off["intensity"]],
        "itemsize": msg.point_step,
    })
    raw = np.frombuffer(bytes(msg.data), dtype=dt)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    inten = np.asarray(raw["intensity"], np.float32)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(inten)
    return xyz[finite], inten[finite]


def _make_xyzrgb_cloud(header: Header, xyz: np.ndarray, rgb_u8: np.ndarray) -> PointCloud2:
    """Pack xyz + rgb (uint8 Nx3, RGB order) into a PointCloud2 with an 'rgb' float field.
    viz_uniform_rgb recolors EVERY xyzrgb topic uniformly (video mode); "" restores colors."""
    rgb_u8 = _viz_recolor(rgb_u8)
    n = int(xyz.shape[0])
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = False
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    if n == 0:
        msg.data = b""
        return msg
    buf = np.zeros(n, dtype=np.dtype({
        "names": ["x", "y", "z", "rgb"],
        "formats": [np.float32, np.float32, np.float32, np.uint32],
        "offsets": [0, 4, 8, 12],
        "itemsize": 16,
    }))
    buf["x"] = xyz[:, 0]
    buf["y"] = xyz[:, 1]
    buf["z"] = xyz[:, 2]
    r = rgb_u8[:, 0].astype(np.uint32)
    g = rgb_u8[:, 1].astype(np.uint32)
    b = rgb_u8[:, 2].astype(np.uint32)
    buf["rgb"] = (r << 16) | (g << 8) | b
    msg.data = buf.tobytes()
    return msg


def _make_xyzi_cloud(header: Header, xyz: np.ndarray, intensity: float) -> PointCloud2:
    """Pack xyz + constant intensity into a PointCloud2 matching pcl::PointXYZI layout
    (the format localPlanner's addedObstaclesHandler deserialises)."""
    n = int(xyz.shape[0])
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = True
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    if n == 0:
        msg.data = b""
        return msg
    buf = np.zeros((n, 4), dtype=np.float32)
    buf[:, :3] = xyz
    buf[:, 3] = float(intensity)
    msg.data = buf.tobytes()
    return msg


# VIDEO/RViz uniform recolor for every glass topic (planes AND obstacles): "r,g,b" from the
# viz_uniform_rgb param; None -> original per-plane colors. Display-only -- planner semantics
# (intensity=200) untouched.
_VIZ_RGB = None


def _viz_recolor(rgb_u8: np.ndarray) -> np.ndarray:
    if _VIZ_RGB is None or rgb_u8.shape[0] == 0:
        return rgb_u8
    return np.broadcast_to(np.asarray(_VIZ_RGB, np.uint8), rgb_u8.shape)


def _make_xyzi_rgb_cloud(header: Header, xyz: np.ndarray, intensity: float) -> PointCloud2:
    """PointXYZI + an EXTRA display-only rgb field (planner matches fields by name; rviz can
    render RGB8). Used for /added_obstacles when viz_uniform_rgb is set."""
    n = int(xyz.shape[0])
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = True
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 20
    msg.row_step = msg.point_step * n
    if n == 0:
        msg.data = b""
        return msg
    r, g, b = (_VIZ_RGB if _VIZ_RGB is not None else (120, 190, 255))
    rgbf = np.frombuffer(np.uint32((int(r) << 16) | (int(g) << 8) | int(b)).tobytes(), np.float32)[0]
    buf = np.zeros((n, 5), dtype=np.float32)
    buf[:, :3] = xyz
    buf[:, 3] = float(intensity)
    buf[:, 4] = rgbf
    msg.data = buf.tobytes()
    return msg


class GlassKillerNode(Node):
    def __init__(self):
        super().__init__("glass_killer_plane_node")
        # PIPELINE role: "mono" = original single-process behavior; "perception" = run
        # detect+geometry, publish placed planes to /gkpipe/geom and SKIP the tracker;
        # "mapping" = subscribe /gkpipe/geom, run the tracker/evict + publish the map.
        self.role = str(self.declare_parameter("role", "mono").value)
        self._map_lock = threading.Lock()
        self._geom_latest = None

        self.args = build_args(self)
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        # uniform LIGHT-BLUE recolor of every glass topic (planes + obstacles) for videos;
        # viz_uniform_rgb:="" restores the per-plane debug colors
        global _VIZ_RGB
        _v = str(self.declare_parameter("viz_uniform_rgb", "120,190,255").value).strip()
        _VIZ_RGB = tuple(int(x) for x in _v.split(",")) if _v else None
        # viz_full is a PER-MASK colour demo, so the uniform recolor must not run: _viz_recolor
        # broadcasts one flat colour over EVERY xyzrgb topic and would overwrite the mask colours
        # on planes, seeds, lines and rays alike. Pass viz_uniform_rgb explicitly to override.
        self.viz_full = bool(self.declare_parameter("viz_full", False).value)
        if self.viz_full:
            _VIZ_RGB = None

        self._build_models()

        # latest-only buffers; processing always uses the freshest pair.
        self._lock = threading.Lock()
        self._latest_cloud = None
        self._latest_image = None
        self._img_buf = deque(maxlen=12)     # (stamp_sec, Image) -> match the image to the CLOUD stamp
        self._img_match_gap = 0.0
        self._prev_proc_stamp = 0.0      # cloud stamp of the previous PROCESSED frame (SYNC line)
        self._latest_last_scan = None   # newest SINGLE scan (from /glass_killer/last_scan) for DA2 align
        self._pinhole_remap = None      # cached equirect->pinhole cv2.remap maps (built once)
        self._busy = False
        # Watchdog timestamps (wall clock) so a background thread can tell whether the
        # node is stuck inside _process vs simply not receiving new clouds.
        self._t_last_cloud = 0.0
        self._t_last_frame = 0.0
        self._busy_since = 0.0
        self._max_cloud_points = int(self.declare_parameter("max_cloud_points", 500000).value)

        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST)
        self.sub_cloud = self.create_subscription(
            PointCloud2, "/glass_killer/cloud", self._on_cloud, sensor_qos)
        # Newest SINGLE scan (no 5s stacking -> no glass see-through) for the DA2 pinhole alignment.
        self.sub_last_scan = self.create_subscription(
            PointCloud2, "/glass_killer/last_scan", self._on_last_scan, sensor_qos)
        # FLOOR evidence comes from the SAM3 'floor' mask (see _process). OBSTACLE evidence (the orange
        # cells) comes from the driver's REGISTERED /terrain_map: it is published in the MAP frame with
        # intensity = height above ground, so its points are world-consistent (they do NOT move with the
        # camera). We take only the obstacle points (intensity >= thresh); floor stays SAM3.
        self.use_terrain_floor = bool(self.declare_parameter("use_terrain_floor", False).value)
        # terrain points at <= this height above ground count as FLOOR (only with use_terrain_floor)
        self.terrain_floor_thresh = float(self.declare_parameter("terrain_floor_thresh", 0.1).value)
        # lateral band (m) around every TRACKED plane segment cleared of terrain-floor evidence
        # (ground under/through the glass must not green-trim the plane standing on it)
        self.terrain_floor_plane_clear_m = float(self.declare_parameter("terrain_floor_plane_clear_m", 0.6).value)
        self.use_terrain_obstacle = bool(self.declare_parameter("use_terrain_obstacle", True).value)
        self.terrain_obstacle_thresh = float(self.declare_parameter("terrain_obstacle_thresh", 0.2).value)
        self._latest_terrain = None                    # (xyz map-frame, intensity)
        self.sub_terrain = self.create_subscription(
            PointCloud2, str(self.declare_parameter("terrain_topic", "/terrain_map").value),
            self._on_terrain, sensor_qos)
        self.sub_image = self.create_subscription(
            Image, "/habitat/rgb", self._on_image, sensor_qos)
        # Camera pose in map (republished by glassKillerNode) -> lets us emit
        # planes directly in the map frame, independent of the TF tree.
        self._latest_pose = None
        # Buffer recent poses WITH timestamps so a frame can be transformed by the pose from the
        # CLOUD's capture time, not "whatever pose is latest when inference finishes". Under jitter
        # (e.g. a co-running VLA slowing inference) the latest-pose approach places planes where the
        # robot moved TO; capture-time matching keeps world placement locked regardless of latency.
        self._pose_buf = deque(maxlen=200)             # (stamp_sec, R_body, T)
        # EXACT de-rotation poses keyed by cloud stamp (ns): one per cloud, from the provider.
        self._cloud_pose = {}
        self._cloud_pose_keys = deque()                # insertion order, bounded to 64 below
        self._pose_src = "buffer"                      # which path _pose_at took (SYNC log)
        self._viz_hold = None                          # list -> HOLD outgoing viz msgs for a burst
        self._viz_lidar_jpg = None                     # perception: rendered lidar pano, shipped
        self.sub_pose = self.create_subscription(
            PoseStamped, "/habitat/state_estimation", self._on_pose, sensor_qos)
        # SCAN-LOCKED poses: /state_estimation_at_scan is stamped exactly at each registered
        # scan, so capture-time matching finds a 0ms-offset pose even when odometry callbacks
        # starve during inference (a 0.3s-stale pose at turning rate = a slanted plane).
        try:
            from nav_msgs.msg import Odometry as _Odom
            self.sub_pose_at_scan = self.create_subscription(
                _Odom, "/state_estimation_at_scan", self._on_odom_at_scan, sensor_qos)
        except Exception:
            pass
        # The EXACT pose the provider de-rotated each cloud with, stamped identically to that
        # cloud. _pose_at prefers an exact-stamp hit here over the nearest-neighbour odom search:
        # it is the very pose the cloud was built with, so cam->world is identity by construction
        # (the odom search re-rotated every output by the image<->newest-scan gap, 0-100 ms/frame).
        # depth 16: still find the matching entry after being blocked in inference for a while.
        self.sub_cloud_pose = self.create_subscription(
            PoseStamped, "/glass_killer/cloud_pose", self._on_cloud_pose,
            QoSProfile(depth=16, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST))
        self.publish_frame = self.declare_parameter("publish_frame", "map").value
        # Debug mask-overlay images (big + small masks). OFF by default: rendering findContours +
        # alpha-blend per mask on the full pano and publishing it costs ~150-300ms/frame. Set
        # publish_overlays:=true to view /glass_killer/big_masks and /small_masks in RViz.
        self.publish_overlays = bool(self.declare_parameter("publish_overlays", False).value)
        # Keep the last N rounds of planes visible at once (so a window seen in 2
        # consecutive runs shows both, instead of each frame flashing on/off).
        self.stack_rounds = int(self.declare_parameter("stack_rounds", 2).value)
        self._plane_hist = deque(maxlen=max(1, self.stack_rounds))
        self._seed_hist = deque(maxlen=max(1, self.stack_rounds))
        self.pub_planes = self.create_publisher(PointCloud2, "/glass_killer/planes", 2)
        # Per-mask-colored outward seed points (the "mask color" dots).
        self.pub_seeds = self.create_publisher(PointCloud2, "/glass_killer/seeds", 2)
        # THIS-frame-only, world(map)-frame streams for the separate top-down map node (NOT stacked,
        # so the accumulator can add each round exactly once). Only computed when subscribed.
        self.pub_map_seeds = self.create_publisher(PointCloud2, "/glass_killer/map_seeds", 2)
        self.pub_map_jump = self.create_publisher(PointCloud2, "/glass_killer/map_jump", 2)
        # SAM3 big-mask + small-mask overlays as images (view in RViz Image display).
        self.pub_big_masks = self.create_publisher(Image, "/glass_killer/big_masks", 2)
        self.pub_small_masks = self.create_publisher(Image, "/glass_killer/small_masks", 2)

        # Bridge the camera-local plane frame into the existing map tree, so
        # planes render alongside /registered_scan with Fixed Frame = map.
        # Chain becomes: camera_link -> sensor_at_scan (this static TF)
        #                            -> map (live, published by the autonomy stack).
        self._publish_static_camera_tf()

        # Heavy work runs in the timer (fast callbacks just store latest).
        self.create_timer(0.05, self._tick)
        # Watchdog runs on its OWN thread, so it keeps logging even when the single
        # threaded executor is blocked inside a long/hung _process().
        threading.Thread(target=self._watchdog, daemon=True).start()

        # GPU KEEP-WARM: this is a LAPTOP GPU that downclocks hard (to ~500 MHz / <10 W) when idle
        # between frames, so the next SAM3 forward runs cold and ~3x slower. A tiny background matmul
        # keeps the clocks up (this is what DA2 implicitly did before it was disabled). gpu_keep_warm:=false
        # to turn off.
        self.gpu_keep_warm = bool(self.declare_parameter("gpu_keep_warm", False).value)  # DA2 warms the GPU now
        self._stop_keepwarm = False
        self._sam3_running = False          # True ONLY during the SAM3 forward (keep-warm pauses then)
        if self.gpu_keep_warm and self.device.type == "cuda":
            threading.Thread(target=self._gpu_keepwarm_worker, daemon=True).start()
            self.get_logger().info("GPU keep-warm ON (prevents laptop-GPU idle downclock; gpu_keep_warm:=false to disable)")

        # Debug-visual saving runs on a SEPARATE thread so it never slows the live
        # algorithm: every inference packages its data and enqueues it; the worker
        # writes the same per-frame files the batch process_frame produces.
        self.debug_save = bool(self.declare_parameter("debug_save", False).value)
        self.debug_save_dir = str(self.declare_parameter(
            "debug_save_dir", "./glass_killer_ros_debug").value)
        self._save_counter = 0
        # torch.cuda.empty_cache() every N inferences (0 = never, keep the pool warm). N=1 -> nvidia-smi
        # reads the true ~1 GB live size at a per-frame re-malloc cost; larger N trades less often.
        self.empty_cache_every = int(self.declare_parameter("empty_cache_every", 0).value)
        self._save_q = None
        if self.debug_save and self.debug_save_dir:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            # small queue + drop-on-full so a slow disk never backs up memory/latency.
            self._save_q = queue.Queue(maxsize=int(self.declare_parameter("debug_save_queue", 2).value))
            threading.Thread(target=self._debug_saver_worker, daemon=True).start()
            self.get_logger().info(f"debug visuals -> {self.debug_save_dir} (async saver thread)")

        # Per-frame top-down HORIZONTAL-pillar PNG. OFF by default: the node runs in "running mode" and saves
        # NOTHING to disk unless explicitly enabled (the batch is the dev/save path). save_topdown_horiz:=true
        # to turn it back on. (SPACE keysave is still available on demand.)
        self.save_topdown_horiz = bool(self.declare_parameter("save_topdown_horiz", False).value)
        self._hz_dir = str(self.declare_parameter(
            "topdown_horiz_dir", "./glass_killer_ros_topdown_horiz").value)
        self._hz_q = None
        if self.save_topdown_horiz and self._hz_dir:
            os.makedirs(self._hz_dir, exist_ok=True)
            self._hz_q = queue.Queue(maxsize=int(self.declare_parameter("topdown_horiz_queue", 2).value))
            threading.Thread(target=self._topdown_horiz_saver_worker, daemon=True).start()
            self.get_logger().info(f"top-down horizontal-pillar PNGs -> {self._hz_dir} (async, per frame)")

        # On-demand SPACE-key snapshot: keep the last 5 inferences in a rolling buffer and, when
        # SPACE is pressed, dump ALL their debug (ply/mask/plane + a txt log of every plane try:
        # occlusion, params, coverage). Recording the per-try info runs inside the algorithm but
        # is near-free (reuses the gate's occlusion); the disk write runs on its own thread.
        self.args.clench_debug_tries = True
        self._recent = deque(maxlen=5)
        self._keysave_dir = str(self.declare_parameter(
            "keysave_dir", "./glass_killer_ros_keysave").value)
        # keysave_full:=true (default) -> SPACE saves EVERYTHING (the original full debug set).
        # keysave_full:=false -> minimal snapshot: topdown (+root copies) + seed PLY + accumulated-floor
        # PLY only (plane-compete / seed-vs-floor debugging).
        self.keysave_full = bool(self.declare_parameter("keysave_full", True).value)
        self._snap_q = queue.Queue(maxsize=2)         # SPACE 5-frame full-debug snapshots
        self._snap_counter = 0
        threading.Thread(target=self._snapshot_saver_worker, daemon=True).start()
        threading.Thread(target=self._key_listener, daemon=True).start()

        # AUTOMATIC global-map PNG: from the first inference on, save the global floor/planes visual
        # (the seed-vs-floor tug map) once per inference to auto_map_dir on its own thread -- no key
        # press needed. Non-blocking (drops a frame if the writer falls behind).
        self.auto_map_dir = str(self.declare_parameter(
            "auto_map_dir", "./glass_killer_ros_globalmap").value)
        self.save_auto_map = bool(self.declare_parameter("save_auto_map", True).value)
        if self.viz_full:
            self.save_auto_map = False   # viz_full is a LIVE-ONLY demo: publish, never write to disk
        if self.save_auto_map and self.auto_map_dir:
            os.makedirs(self.auto_map_dir, exist_ok=True)
        self.auto_map_min_period = float(self.declare_parameter("auto_map_min_period", 1.5).value)
        self._last_auto_map_t = 0.0
        self._auto_q = queue.Queue(maxsize=2)
        threading.Thread(target=self._auto_map_worker, daemon=True).start()
        # LOCAL TUG VIEW: per-frame PNG of the seed-vs-floor tug count in a small window around the robot.
        self.save_local_tug = bool(self.declare_parameter("save_local_tug", False).value)
        self.local_tug_dir = str(self.declare_parameter(
            "local_tug_dir", "./glass_killer_ros_localtug").value)
        self.local_tug_half = int(self.declare_parameter("local_tug_half_cells", 50).value)  # +/- cells
        if self.save_local_tug and self.local_tug_dir:
            os.makedirs(self.local_tug_dir, exist_ok=True)

        # Live top-down cloud-grid map (same visual as the batch top_down_map PNG): rendered +
        # published as a ROS Image on its OWN thread at topdown_hz, reading only the latest
        # inference's data -- so it never slows the plane algorithm and writes nothing to disk.
        self.topdown_hz = float(self.declare_parameter("topdown_hz", 2.0).value)
        if self.viz_full:
            # The worker sleeps 1/topdown_hz between renders, so 2 Hz would cap the demo streams
            # at ~2 frames/s no matter how fast the pipeline runs. Poll fast instead: the loop
            # still skips when the sequence has not advanced, so this costs nothing when idle.
            self.topdown_hz = float(self.declare_parameter("viz_full_hz", 30.0).value)
        # OFF by default: rendering the top-down Image holds the GIL and slows the live inference. Set
        # publish_topdown:=true to view /glass_killer/topdown_map again.
        self.publish_topdown = bool(self.declare_parameter("publish_topdown", False).value)
        self.pub_topdown = self.create_publisher(Image, "/glass_killer/topdown_map", 1)
        # DEMO (viz_full:=true): the four --full-visual demo images, published LIVE instead of saved.
        # They render on the existing async panel thread, so inference is never blocked.
        if self.viz_full:                 # declared earlier (it also bldgA the uniform recolor)
            self.pub_viz_tug = self.create_publisher(Image, "/glass_killer/viz/floor_seed_tug", 1)
            self.pub_viz_compete = self.create_publisher(Image, "/glass_killer/viz/plane_compete", 1)
            self.pub_viz_pillars = self.create_publisher(Image, "/glass_killer/viz/pillars_topdown", 1)
            self.pub_viz_silhouette = self.create_publisher(Image, "/glass_killer/viz/silhouette_allmasks", 2)
            self.pub_viz_spill = self.create_publisher(Image, "/glass_killer/viz/multiview_spill", 2)
            self.pub_viz_rgb = self.create_publisher(Image, "/glass_killer/viz/rgb_pano", 2)
            self.pub_viz_lidar = self.create_publisher(Image, "/glass_killer/viz/lidar_pano", 2)
            # ground LINE of every tracked plane, drawn in its owning BIG-MASK colour
            self.pub_viz_lines = self.create_publisher(PointCloud2, "/glass_killer/viz/plane_lines", 2)
            # the 4 corner/derived reference rays per big mask, in that mask's colour
            self.pub_viz_rays = self.create_publisher(PointCloud2, "/glass_killer/viz/mask_rays", 2)
            # planes REMOVED by the global filters, coloured by which mechanism removed them
            self.pub_viz_evicted = self.create_publisher(PointCloud2, "/glass_killer/viz/evicted_planes", 2)
            # per-PANE vertical lines: each small mask owned by a big mask, intersected with that
            # big mask's ACCEPTED plane -> the pane's left/right extents ON the plane (mullions)
            self.pub_viz_pane_lines = self.create_publisher(PointCloud2, "/glass_killer/viz/pane_lines", 2)
            self.get_logger().info(
                f"[viz_full] role={self.role}: viz publishers up "
                f"(images: tug/compete/pillars/silhouette/spill/rgb/lidar; clouds: plane_lines, "
                f"mask_rays, evicted_planes, pane_lines) -- images+pane_lines+mask_rays are fed by "
                f"PERCEPTION, tug/compete/spill/plane_lines/evicted by MAPPING")
        # Cross-frame ACCUMULATED plane map (world/map frame). ON BY DEFAULT (enable_global_map:=false
        # to disable): /glass_killer/global_planes is the RViz wall cloud, and the SAME dense patches go
        # out on /added_obstacles so the local planner treats the glass as hard obstacles (that is the
        # point of Glass Killer -- glass is invisible to LiDAR). publish_added_obstacles:=false to run
        # visualization-only. The two-panel IMAGE is gated separately by publish_global_map_image.
        self.enable_global_map = bool(self.declare_parameter("enable_global_map", True).value)
        self.publish_global_map_image = bool(self.declare_parameter("publish_global_map_image", False).value)
        # ACCUMULATED-seed coverage: build a persistent world seed-count map and WEIGHT each seed's
        # contribution to plane coverage by how many times its cell has been seen (use_accum_seed_cover).
        self.use_accum_seed_cover = bool(self.declare_parameter("use_accum_seed_cover", False).value)
        # REPROJECTION-SPILL eviction (motion-based depth check): reproject each placed plane into the
        # current view and evict if it drifts off the SAM glass mask (spills onto non-glass) under motion.
        self.reproject_evict = bool(self.declare_parameter("reproject_evict", False).value)
        self.args.track_reproject_evict = self.reproject_evict
        # DEPTH-SWEEP eviction (multi-hypothesis plane sweep, README §8): ray-consistent depth
        # hypotheses raced vs the observed mask; evicts a >=1m-wrong plane after ~1m of motion at
        # 1% false-evict, and covers head-on approach via looming. Replaces reproject_evict.
        self.depth_sweep_evict = bool(self.declare_parameter("depth_sweep_evict", True).value)
        self.args.track_depth_sweep_evict = self.depth_sweep_evict
        # aggressiveness dials: bound = ln((1-b)/a) (4.55 = 1% false-evict, 2.9 = 5% ~35% faster);
        # cal_checks = bias burn-in before evidence counts (3 robust, 2 faster).
        # aggressive defaults chosen live (2026-08): bound 2.9 (alpha=5%) + 2-check burn-in halve
        # time-to-evict; a false-evicted plane re-places on its next detection.
        self.args.track_sweep_bound = float(self.declare_parameter("depth_sweep_bound", 2.9).value)
        self.args.track_sweep_cal_checks = int(self.declare_parameter("depth_sweep_cal_checks", 2).value)
        # A/B ablation switches (default true = production): robot-path and green-floor eviction
        self.args.track_path_evict = bool(self.declare_parameter("path_evict", True).value)
        self.args.track_floor_evict = bool(self.declare_parameter("floor_evict", True).value)
        self.args.track_merge = bool(self.declare_parameter("track_merge", True).value)   # A/B: global consolidation
        # sweep-evict snapshots (plane + downscaled scene ply) land next to the debug visuals
        self.args.out_dir = self.debug_save_dir
        self.args.track_spill_thresh = float(self.declare_parameter("spill_thresh", 0.20).value)
        # SPILL TUG-OF-WAR eviction (2026-08): a check is judged only when >= spill_cov_min of the
        # footprint is on the mask (else the count freezes). Each plane's OWN under-segmentation spill
        # floor is calibrated over spill_base_n checks; the tug then counts EXCESS over that floor
        # (rise +1 / fall -1, min 0) so correct planes (excess ~0) are spared and drifting ones evict.
        self.args.track_spill_persist = int(self.declare_parameter("spill_persist", 3).value)
        self.args.track_spill_cov_min = float(self.declare_parameter("spill_cov_min", 0.6).value)
        # judge spill only when >= this frac of the pane is IN VIEW (else the count freezes) -- a
        # correct pane leaving the view otherwise evicts itself as its visible sliver thins
        self.args.track_spill_vis_min = float(self.declare_parameter("spill_vis_min", 0.5).value)
        self.args.track_spill_tug_start = float(self.declare_parameter("spill_tug_start", 0.10).value)
        self.args.track_spill_base_n = int(self.declare_parameter("spill_base_n", 3).value)
        # radius (m, top-down) within which a SEED discards floor evidence (0 = no veto)
        self.args.clench_floor_seed_clear_m = float(self.declare_parameter("floor_seed_clear_m", 0.25).value)
        # candidate rejected when >= this frac of its width touches accumulated floor
        self.args.clench_floor_touch_reject_frac = float(self.declare_parameter("floor_reject_frac", 0.10).value)
        # shave this many pixels off the SAM3 floor-mask boundary before sampling LiDAR (0 = raw mask);
        # counters mask bleed onto the glass base / floor reflections becoming floor evidence
        self.args.floor_mask_erode_px = int(self.declare_parameter("floor_mask_erode_px", 3).value)
        # judged spill >= this frac -> DIRECT evict (before baseline calibration / tug)
        self.args.track_spill_hard_frac = float(self.declare_parameter("spill_hard_frac", 0.33).value)
        self.args.track_spill_baseline_min_m = float(self.declare_parameter("spill_baseline_min_m", 0.4).value)
        # A spill CHECK fires every this-many metres of robot MOTION since the plane's last check (paces
        # eviction by movement, not frame rate). spill_persist CONSECUTIVE over-threshold checks -> evict.
        self.args.track_spill_check_move_m = float(self.declare_parameter("spill_check_move_m", 0.25).value)
        self.args.track_spill_max = float(self.declare_parameter("spill_max", 0.50).value)   # (unused: hard-thresh mode)
        # Spill IoU GATE: only spill-test a plane whose reprojection overlaps the 2D mask by >= this IoU.
        self.args.track_spill_iou_min = float(self.declare_parameter("spill_iou_min", 0.50).value)
        self.args.track_spill_iou_gate = bool(self.declare_parameter("spill_iou_gate", False).value)
        # Shrink every fitted plane's top+bottom inward by this fraction of its height (0.05 = 5% each end).
        self.args.track_plane_height_shrink = float(self.declare_parameter("plane_height_shrink", 0.05).value)
        # Spill occlusion by OTHER planes (glass is LiDAR-invisible, so a plane behind another plane is
        # only occluded if we rasterize planes as occluders too).
        self.args.track_spill_plane_occ = bool(self.declare_parameter("spill_plane_occ", True).value)
        # Replace the raw silhouette corners (TL/TR/BL/BR) that seed the corner curves with the
        # occlusion-robust good-edge-segment endpoints (iterative great-circle).
        self.args.corner_seg_curve = bool(self.declare_parameter("corner_seg_curve", True).value)
        # RANGE GATES: skip the spill comparison beyond spill_dist_max; never place a plane beyond
        # place_dist_max (robot->plane distance).
        self.args.track_spill_dist_max_m = float(self.declare_parameter("spill_dist_max_m", 10.0).value)
        self.args.track_place_dist_max_m = float(self.declare_parameter("place_dist_max_m", 10.0).value)
        # MASK CLAMP at placement: trim every placed plane to the 2D glass mask, so ALL RViz plane
        # topics (they're rebuilt from the tracker) show the mask-bounded geometry. Needs the glass
        # mask each frame (built below whenever this OR reproject/panel is on).
        self.mask_clamp = bool(self.declare_parameter("mask_clamp", True).value)
        self.args.track_mask_clamp = self.mask_clamp
        # GLOBAL FLOOR CHECK on/off, and SEED dilation rings (0 = single grid cell, no spill). These
        # affect ONLY the global-map tug; per-frame plane fitting/seed coverage is unaffected.
        self.args.track_floor_check = bool(self.declare_parameter("floor_check", True).value)
        self.args.track_seed_dilate = int(self.declare_parameter("seed_dilate", 1).value)
        # Seed cell selection: grid-ring (True) = the ring of grid cells JUST OUTSIDE the mask (no pixel
        # dilate); False = classic pixel dilate(dilate_steps) & ~mask.
        self.args.seed_grid_ring = bool(self.declare_parameter("seed_grid_ring", False).value)
        self.args.seed_ring_cells = int(self.declare_parameter("seed_ring_cells", 1).value)  # ring thickness (cells)
        self.args.seed_cell_frac = float(self.declare_parameter("seed_cell_frac", 0.5).value)  # cell=mask if >= covered
        # A floor point under a seed within this vertical tolerance (m) is the glass's own base near the
        # ground -> keep that cell SEED (don't floor it), so a low glass (~0.5m base) still solves.
        self.args.track_seed_floor_vtol = float(self.declare_parameter("seed_floor_vtol", 0.5).value)
        # Save a per-frame REPROJECT panel: LEFT = top-down accumulated map (evicted planes marked red),
        # RIGHT = RGB + reprojected plane silhouettes vs SAM mask (high-spill planes marked red).
        self.save_reproject_panel = bool(self.declare_parameter("save_reproject_panel", False).value)
        self.reproject_panel_dir = str(self.declare_parameter(
            "reproject_panel_dir", "./glass_killer_ros_reproject").value)
        if self.save_reproject_panel:
            os.makedirs(self.reproject_panel_dir, exist_ok=True)
        self._plane_tracker = bsp._PlaneTracker(self.args)
        # SPILL DEBUG: per-evicted-plane check-history JSONs, written by a daemon thread
        _sdd = str(self.declare_parameter("spill_debug_dir", "").value)
        if _sdd:
            self._plane_tracker.spill_debug_dir = _sdd
            self.get_logger().info(f"[spill-debug] evicted-plane traces -> {_sdd}")
        self._evict_ghosts = []          # [(expiry_ts, patch_xyz)] BLACK ghosts of evicted planes
        self._evict_vis_ghosts = []      # viz_full: [(expiry, patch_xyz, rgb)] coloured BY MECHANISM
        # FINAL / OVERVIEW map: a SECOND, independent plane accumulator fed the same detections as the
        # current map, but with a LOOSER (smaller) spill-eviction radius -> distant planes are never
        # spill-pruned, so driving around builds up a stable overall plane map. Published on its own
        # topic (/glass_killer/final_global_planes); the current map stays on /glass_killer/global_planes.
        self.enable_final_map = bool(self.declare_parameter("enable_final_map", True).value)
        self._final_tracker = bsp._PlaneTracker(self.args)
        self._final_tracker.spill_dist_max = float(
            self.declare_parameter("final_spill_dist_max_m", 5.0).value)
        # Overview map uses a HIGHER spill threshold than the current map -> more tolerant, keeps more.
        self._final_tracker.spill_thresh = float(
            self.declare_parameter("final_spill_thresh", 0.35).value)
        self.pub_final_global_planes = self.create_publisher(
            PointCloud2, "/glass_killer/final_global_planes", 2)
        self.pub_global_map = self.create_publisher(Image, "/glass_killer/global_map", 1)
        self.global_plane_height_m = float(self.declare_parameter("global_plane_height_m", 2.0).value)
        # Metric sample pitch of the dense wall patches (both the RViz cloud and the obstacle cloud):
        # points every grid_m along AND up the wall, so plane size drives point count.
        self.global_plane_grid_m = float(self.declare_parameter("global_plane_grid_m", 0.05).value)
        # Pull the emitted OBSTACLE cloud in from each plane END by this much so glass doesn't bleed
        # into an adjacent doorway/opening once the local planner inflates it by the robot clearance.
        self.obstacle_end_inset_m = float(self.declare_parameter("obstacle_end_inset_m", 0.25).value)
        # planner feeds only: extend each wall's BOTTOM down by this much (capped at v0).
        # The tracked vertical range is the OBSERVED one -- when a railing/desk occludes the
        # pane's lower part, the injected wall hovers and a drone-band path dives under it.
        self.obstacle_extend_bottom_m = float(
            self.declare_parameter("obstacle_extend_bottom_m", 0.8).value)
        # scan-channel walls are voxel-deduped to this pitch before publishing: the union +
        # bottom extension at the visual 0.05m grid is 100k+ points at 5 Hz, enough DDS/ingest
        # load to lag the stack and starve SLAM (pose jumps). Planning grids are >=0.15m.
        self.obstacle_scan_voxel_m = float(
            self.declare_parameter("obstacle_scan_voxel_m", 0.15).value)
        # obstacle wall patches are generated at THIS pitch (planner feeds only; the 0.05m
        # global_plane_grid_m stays for rviz visuals). 0.05m obstacle patches were the bulk
        # of the global_map phase cost: pitch^2 scaling, both channels, every frame.
        self.obstacle_grid_m = float(self.declare_parameter("obstacle_grid_m", 0.15).value)
        # EVICTION MIRROR (feed-level): when a plane vanishes from the CURRENT tracker, its
        # segment suppresses overlapping FINAL-tracker wall points from the obstacle feeds for
        # this window. Real glass re-detects immediately (current walls publish via the union),
        # so suppression only starves feeds where detection actually stopped -- removed glass
        # stops blocking within ~a frame instead of waiting out final-map persistence.
        self.evict_suppress_s = float(self.declare_parameter("evict_suppress_s", 30.0).value)
        self.evict_suppress_lat_m = float(
            self.declare_parameter("evict_suppress_lat_m", 0.4).value)
        self._evict_suppress = []
        self._prev_cur_pids = {}
        self.pub_global_planes = self.create_publisher(PointCloud2, "/glass_killer/global_planes", 2)
        # OBSTACLE injection mode:
        #  "scan"  (default) -> publish glass wall points onto /registered_scan so the driver's
        #          terrain-analysis processes them like any LiDAR return: it does ground segmentation
        #          AND drops points above vehicleHeight (~1.5m). That is what makes DOORS work -- glass
        #          ABOVE a doorway (transom/clerestory) is filtered out instead of hard-blocking the
        #          opening at ground level, and low glass still becomes a proper obstacle in terrain_map.
        #  "added" -> the old direct /added_obstacles injection (forced intensity=200, NO height filter;
        #          reliable but blocks a door if glass continues above it).
        #  "both"  -> publish on both.
        self.obstacle_mode = str(self.declare_parameter("obstacle_mode", "scan").value).lower()
        self.obstacle_scan_topic = str(self.declare_parameter("obstacle_scan_topic", "/registered_scan").value)
        # Terrain keeps a rolling voxel grid, so re-publish the latest glass points at this rate to keep
        # them fresh between (slower) inferences. 0 -> only publish on each inference.
        self.obstacle_scan_hz = float(self.declare_parameter("obstacle_scan_hz", 5.0).value)
        self.publish_added_obstacles = bool(self.declare_parameter("publish_added_obstacles", True).value)
        # PLANNER FEED SOURCE: true (default) = /added_obstacles comes from the FINAL/overview
        # tracker (far walls persist: spill-evicts only within final_spill_dist) -- eviction
        # churn on the CURRENT map opened a 6m through-glass breach for the planner (bldgA_f5,
        # t=360s). false = the old current-map feed.
        self.obstacle_from_final = bool(self.declare_parameter("obstacle_from_final", True).value)
        self.pub_added_obstacles = self.create_publisher(PointCloud2, "/added_obstacles", 2)
        self.pub_scan_obstacles = self.create_publisher(PointCloud2, self.obstacle_scan_topic, 5)
        self._latest_wall_xyz = np.empty((0, 3), np.float32)   # world-frame glass points for scan republish
        self._pose_recv_mono = None
        if self.obstacle_scan_hz > 0.0:
            # DEDICATED THREAD, not an executor timer: SAM3 inference blocks the executor for the
            # whole frame (~1-2s), starving a timer past terrain-analysis's 2s decay -> the glass
            # stripe flashes. A thread keeps the republish cadence through inference.
            import threading as _th
            _t = _th.Thread(target=self._scan_obstacle_pump, daemon=True)
            _t.start()
        # RViz copy of the obstacle walls (same dense points as /added_obstacles, dimmed track color)
        # so the original colored-plane visual and the injected wall can be compared side by side.
        self.pub_obstacle_walls = self.create_publisher(PointCloud2, "/glass_killer/obstacle_walls", 2)
        # GT PLANES overlay (debug): densely sampled annotated GT rectangles, published on a
        # slow timer in the map frame -- WHITE points, so live coverage gaps are visible in rviz.
        self._gt_cloud_msg = None
        _gt_json = str(self.declare_parameter("gt_planes_json", "").value)
        if _gt_json and os.path.isfile(_gt_json):
            try:
                _gt = json.load(open(_gt_json)).get("planes", [])
                _pts = []
                for _rec in _gt:
                    _c = [np.asarray(x, np.float64) for x in _rec["corners_world"]]
                    tl, tr, br, bl = _c[0], _c[1], _c[2], _c[3]
                    nu = max(2, int(np.linalg.norm(tr - tl) / 0.12))
                    nv = max(2, int(np.linalg.norm(bl - tl) / 0.12))
                    uu = np.linspace(0, 1, nu)[:, None, None]
                    vv = np.linspace(0, 1, nv)[None, :, None]
                    _g = (tl * (1 - uu) + tr * uu) * (1 - vv) + (bl * (1 - uu) + br * uu) * vv
                    _pts.append(_g.reshape(-1, 3))
                if _pts:
                    _P = np.concatenate(_pts).astype(np.float32)
                    self._gt_xyz = _P                          # raw (recorded-frame) GT points
                    self._gt_rgb = np.full((len(_P), 3), 255, np.uint8)
                    self._gt_align_T = np.eye(4, dtype=np.float64)   # recorded->live-map correction
                    # ICP alignment is OPT-IN: the recorded and live SLAM frames agree to ~3cm on
                    # the same bag, while the ICP z-init (median glass-z vs median obstacle-z) can
                    # drag correctly-annotated GT onto the wrong structure. Raw GT by default.
                    self._gt_aligned = not bool(self.declare_parameter("gt_align_icp", False).value)
                    self._live_world_buf = []                 # accumulated live obstacle cloud for ICP
                    # SCENE-CLOUD anchor: a reference map (e.g. canonical_run_pinhole/scene_cloud.ply,
                    # the SAME frame gt_planes_canonical.json lives in). When set, alignment is
                    # structure-to-structure ICP (ref cloud -> live map) and GT rides the transform.
                    self._gt_ref_xyz = None
                    _ref = str(self.declare_parameter("gt_ref_cloud", "").value)
                    if _ref and os.path.isfile(_ref):
                        try:
                            import open3d as _o3
                            self._gt_ref_xyz = np.asarray(
                                _o3.io.read_point_cloud(_ref).points, np.float64)
                            self.get_logger().info(f"[gt-align] reference scene cloud loaded: "
                                                   f"{len(self._gt_ref_xyz)} pts ({_ref})")
                        except Exception as e:
                            self.get_logger().warn(f"[gt-align] ref cloud load failed: {e}")
                    _hdr = Header(); _hdr.frame_id = "map"
                    self._gt_cloud_msg = _make_xyzrgb_cloud(_hdr, _P, self._gt_rgb)
                    # TRANSIENT_LOCAL (latched): rviz receives the GT overlay the moment it
                    # subscribes, without waiting for the next timer tick.
                    _gt_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                         history=HistoryPolicy.KEEP_LAST,
                                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
                    self.pub_gt_planes = self.create_publisher(PointCloud2, "/glass_killer/gt_planes", _gt_qos)
                    self._pub_gt_planes()                 # publish IMMEDIATELY at startup
                    self.create_timer(2.0, self._pub_gt_planes)
                    self.get_logger().info(f"GT planes overlay: {len(_gt)} rects, {len(_P)} pts -> /glass_killer/gt_planes")
            except Exception as e:
                self.get_logger().warn(f"GT planes overlay failed to load ({_gt_json}): {e}")
        # ---- CANONICAL RUN: one recorded reference run per scene owns the scene position ----
        # record_run:=true -> while running, dump into canonical_dir:
        #   trajectory.txt   (t x y z of every processed frame -- the scene's POSITION ANCHOR)
        #   scene_cloud.ply  (world obstacle cloud, voxel-0.10 downsampled)
        #   planes_run.json  (EVER-solved planes + EVICTED planes with reasons)
        # GT is then annotated against THIS run. Every later run rigid-aligns its live
        # trajectory onto trajectory.txt (same physical path = same curve, only the SLAM
        # anchor differs) and draws the GT overlay through that transform -- deterministic,
        # no cloud ICP.
        self._canon_dir = str(self.declare_parameter("canonical_dir", "").value)
        self._record_run = bool(self.declare_parameter("record_run", False).value)
        self._rec_poses = []            # (t, x, y, z) of every processed frame (also the live traj)
        self._rec_cloud = []            # world-frame cloud chunks, voxel-deduped on save
        self._rec_ever = {}             # pid -> latest plane geometry
        self._rec_evicted = []          # planes that left the tracker (+reason, frame)
        self._rec_frame = 0
        self._canon_traj = None         # canonical trajectory (N,3) when anchoring a later run
        self._gt_traj_T = None          # solved live->canonical rigid transform
        self._input_q_canon = None
        self._canon_placed_now = False
        if self._canon_dir:
            if self._record_run:
                os.makedirs(self._canon_dir, exist_ok=True)
                # placed-input recorder (OPT-IN): the EXACT algorithm input (cloud/rgb/depth/pose)
                # of every frame where a plane was placed -> canonical_run/placed_input. With many
                # placements this approaches per-frame recording (PLY+PNG each time), so the
                # DEFAULT canonical record is just the light artifacts: trajectory.txt +
                # scene_cloud.ply + planes_run.json (ever+evicted) -- annotation/eval needs no more.
                if bool(self.declare_parameter("record_placed_input", False).value):
                    os.makedirs(os.path.join(self._canon_dir, "placed_input"), exist_ok=True)
                    self._input_q_canon = queue.Queue(maxsize=8)
                    threading.Thread(target=self._input_saver_worker,
                                     args=(self._input_q_canon,), daemon=True).start()
                self.get_logger().info(f"[canon] RECORDING canonical run -> {self._canon_dir} "
                                       f"({'+ placed-frame algorithm inputs' if self._input_q_canon is not None else 'light: trajectory + scene cloud + plane ledger'})")
            else:
                # ANCHOR trajectory: GT is annotated on the SCENE FRAMES (the annotator displays
                # frame clouds), so the anchor must be the frames' own pose trajectory -- NOT the
                # canonical run's (a different session, offset by its own SLAM anchor). Fall back
                # to canonical trajectory.txt only when no frame poses exist.
                _anchor, _src = None, ""
                _scene = os.path.dirname(os.path.dirname(_gt_json)) if _gt_json else ""
                try:
                    import glob as _glob
                    _pfs = sorted(_glob.glob(os.path.join(_scene, "pose_*.txt")))
                    if len(_pfs) >= 10:
                        _anchor = np.asarray([[float(x) for x in open(p).read().split()[:3]]
                                              for p in _pfs], np.float64)
                        _src = f"frame poses ({len(_anchor)})"
                except Exception as e:
                    self.get_logger().warn(f"[canon] frame-pose anchor failed: {e}")
                if _anchor is None:
                    _tf = os.path.join(self._canon_dir, "trajectory.txt")
                    if os.path.isfile(_tf):
                        try:
                            _anchor = np.loadtxt(_tf, skiprows=1, ndmin=2)[:, 1:4].astype(np.float64)
                            _src = f"canonical trajectory ({len(_anchor)})"
                        except Exception as e:
                            self.get_logger().warn(f"[canon] trajectory load failed: {e}")
                if _anchor is not None:
                    self._canon_traj = _anchor
                    self.get_logger().info(f"[canon] GT anchor = {_src} -- overlay will align to it")
        self._td_lock = threading.Lock()
        self._td_data = None
        self._viz_mask_rgb = {}          # viz_full: mask id -> RGB, shipped from perception
        self._td_seq = 0
        threading.Thread(target=self._topdown_worker, daemon=True).start()

        # Save the RAW algorithm INPUT for every inference (cloud_XXXXXX.ply + rgb_XXXXXX.png, the
        # exact format batch_bigmask_4ray_randomopt.py reads via --habitat-dir), so the whole run
        # can be replayed offline with the usual batch command. Written on its own thread.
        self.save_input = bool(self.declare_parameter("save_input", False).value)
        self.input_save_dir = str(self.declare_parameter(
            "input_save_dir", "./glass_killer_ros_input").value)
        self._input_q = None
        if self.save_input and self.input_save_dir:
            os.makedirs(self.input_save_dir, exist_ok=True)
            self._write_input_conventions(self.input_save_dir)   # ship the axis/frame/depth doc
            self._input_q = queue.Queue(maxsize=8)     # drop-on-full so disk never backs up latency
            threading.Thread(target=self._input_saver_worker, daemon=True).start()
            self.get_logger().info(f"algorithm input -> {self.input_save_dir} "
                                   f"(replay: python batch_bigmask_4ray_randomopt.py --habitat-dir {self.input_save_dir} ...)")

        self.get_logger().info("glass_killer_plane_node ready; waiting for /glass_killer/cloud + /habitat/rgb")

        # ---- PIPELINE wiring (created last, after trackers + output publishers exist) ----
        self._pub_geom = self.create_publisher(String, "/gkpipe/geom", 1)   # perception -> mapping
        if self.role == "mapping":
            self.create_subscription(String, "/gkpipe/geom", self._on_geom, 1)
            threading.Thread(target=self._mapping_worker, daemon=True).start()
            self.get_logger().info("[pipeline] MAPPING role: tracker fed from /gkpipe/geom")
        elif self.role == "perception":
            self.get_logger().info("[pipeline] PERCEPTION role: placed planes -> /gkpipe/geom")

    # ============================ PIPELINE: perception -> mapping ============================
    def _viz_pack_extra(self, big_idx, big_masks_full, small_idx, small_masks_full, horiz_by_mask):
        """viz_full: everything the mapping burst needs that the geom payload did not carry."""
        bk = [int(k) for k in (big_masks_full or {}).keys()]
        sk = [int(k) for k in (small_masks_full or {}).keys()]
        return {
            "big_idx": [int(x) for x in big_idx], "small_idx": [int(x) for x in small_idx],
            "big_keys": bk, "big_packed": tx.pack_masks([big_masks_full[k] for k in bk]),
            "small_keys": sk, "small_packed": tx.pack_masks([small_masks_full[k] for k in sk]),
            "horiz_by_mask": horiz_by_mask,
            "lidar_jpg": self._viz_lidar_jpg,
        }

    def _publish_geom(self, header, clench_by_mask, seed_records, mask_ray_records,
                      floor_gate_xyz, pc_xyz, H_orig, W_orig, big_masks_full, _viz_bgr=None,
                      _viz_extra=None):
        """Perception side: compute this frame's floor/obstacle/glass-mask evidence (exactly as
        the inline global_map block does) and ship the placed-plane payload to the mapping node."""
        pose_tuple = self._frame_pose
        floor_world = None
        if pose_tuple is not None and self._sam_floor_cam is not None and len(self._sam_floor_cam):
            floor_world = bsp._cam_to_world(np.asarray(self._sam_floor_cam, np.float64), pose_tuple)
        obst_world = None
        if self.use_terrain_obstacle and pose_tuple is not None:
            with self._lock:
                terr = self._latest_terrain
            if terr is not None:
                _txyz, _tint = terr
                obst_world = _txyz[_tint >= self.terrain_obstacle_thresh]
        glass_mask = None
        if self.reproject_evict or self.depth_sweep_evict or self.save_reproject_panel or self.mask_clamp:
            glass_mask = np.zeros((H_orig, W_orig), bool)
            for _mk in (big_masks_full or {}).values():
                glass_mask |= np.asarray(_mk, bool)
        payload = {
            "hstamp": (int(header.stamp.sec), int(header.stamp.nanosec)),
            "frame_id": header.frame_id,
            "pose": pose_tuple,
            "clench_by_mask": clench_by_mask,
            "seed_records": seed_records,
            "mask_ray_records": mask_ray_records,
            "floor_gate_xyz": floor_gate_xyz,
            "pc_xyz": pc_xyz,
            "floor_world": floor_world,
            "obst_world": obst_world,
            "glass_mask": tx.pack_masks([glass_mask])[0] if glass_mask is not None else None,
            "wh": (W_orig, H_orig),
            "pano_offset": float(self.args.pano_pixel_offset),
            # viz_full only: the pano, JPEG-compressed, so the MAPPING node (which owns the real
            # tracker) can draw the spill verdict on it. Off by default -- costs payload bytes.
            "bgr_jpg": (cv2.imencode(".jpg", _viz_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
                        if (getattr(self, "viz_full", False) and _viz_bgr is not None) else None),
            "fid": int(self._save_counter),
            "viz": _viz_extra,                     # viz_full only (None otherwise)
            # mask id -> the SAME colour the rays/seeds/silhouette use, so plane patches and
            # ground lines can match. Colours come from mask_ray_records (keyed by raw mask id);
            # they are indexed by POSITION in big_idx, not by the mask id, so they cannot be
            # recomputed downstream from the id alone.
            "mask_rgb": ({int(k): [int(c) for c in np.asarray(v["color_rgb"]).reshape(3)]
                          for k, v in (mask_ray_records or {}).items()}
                         if getattr(self, "viz_full", False) else None),
        }
        tx.write_payload(self._pub_geom, header, payload, tag="geom")

    def _on_geom(self, msg):
        sns, payload = tx.read_payload(msg, tag="geom")
        if payload is None:
            return
        with self._map_lock:
            self._geom_latest = payload             # drop-to-latest

    def _mapping_worker(self):
        while rclpy.ok():
            with self._map_lock:
                payload = self._geom_latest; self._geom_latest = None
            if payload is None:
                time.sleep(0.002); continue
            try:
                self._run_mapping(payload)
            except Exception as e:
                self.get_logger().warn(f"[mapping] run failed: {e}", throttle_duration_sec=5.0)

    def _run_mapping(self, payload):
        """Mapping side: reconstruct the frame locals from the payload and run the tracker
        update + eviction + publishes -- a faithful copy of the inline global_map core, minus the
        perception-only debug/recording bits (item snapshots, canonical, panels, local-tug)."""
        a = self.args
        hdr = Header()
        hdr.stamp.sec, hdr.stamp.nanosec = payload["hstamp"]
        hdr.frame_id = payload["frame_id"]
        pose_tuple = payload["pose"]
        bsp._WORLD_SAVE_POSE = pose_tuple
        if payload.get("mask_rgb"):
            self._viz_mask_rgb = {int(k): v for k, v in payload["mask_rgb"].items()}
        clench_by_mask = payload["clench_by_mask"]
        seed_records = payload["seed_records"]
        mask_ray_records = payload["mask_ray_records"]
        floor_gate_xyz = payload["floor_gate_xyz"]
        pc_xyz = payload["pc_xyz"]
        floor_world = payload["floor_world"]
        obst_world = payload["obst_world"]
        _glass_mask = tx.unpack_masks([payload["glass_mask"]])[0] if payload["glass_mask"] is not None else None
        W_orig, H_orig = payload["wh"]
        _pano_off = float(payload["pano_offset"])
        if not self.enable_global_map:
            return
        _pre_geo = {int(tp.pid): (np.asarray(tp.p0, np.float64).copy(),
                                  np.asarray(tp.p1, np.float64).copy(),
                                  tuple(int(c) for c in tp.color),
                                  float(tp.v0) if tp.v0 is not None else 0.0,
                                  float(tp.v1) if tp.v1 is not None else 2.0)
                    for tp in self._plane_tracker.planes}
        _final_box = {}
        def _run_final_tracker():
            if not self.enable_final_map:
                return
            try:
                self._final_tracker.update_from_frame(
                    clench_by_mask, seed_records, pose_tuple, mask_ray_records=mask_ray_records,
                    floor_cam=floor_gate_xyz, scene_cam=pc_xyz, floor_world=floor_world,
                    obst_world=obst_world, glass_mask=_glass_mask,
                    img_wh=(W_orig, H_orig), pano_offset=_pano_off)
                _final_box["tracks"] = [(tp.apex.copy(), tp.p0.copy(), tp.p1.copy(),
                            (int(tp.color[0]), int(tp.color[1]), int(tp.color[2])),
                            bool(tp.fixed), int(tp.mask)) for tp in self._final_tracker.planes]
                _final_box["vr"] = [(tp.v0, tp.v1) for tp in self._final_tracker.planes]
            except Exception as e:
                _final_box["err"] = e
        _final_thread = threading.Thread(target=_run_final_tracker, name="final_tracker", daemon=True)
        _final_thread.start()
        try:
            try:
                self._plane_tracker._dbg_bgr = bgr
            except NameError:
                pass
            tracker_logs = self._plane_tracker.update_from_frame(
                clench_by_mask, seed_records, pose_tuple, mask_ray_records=mask_ray_records,
                floor_cam=floor_gate_xyz, scene_cam=pc_xyz, floor_world=floor_world,
                obst_world=obst_world, glass_mask=_glass_mask,
                img_wh=(W_orig, H_orig), pano_offset=_pano_off)
            import re as _re
            def _whereis(_ln):
                _m = _re.search(r"PRUNE #(\d+)|#(\d+)", _ln)
                _pid = int(_m.group(1) or _m.group(2)) if _m else -1
                if _pid not in _pre_geo or pose_tuple is None:
                    return ""
                _q0, _q1, _col = _pre_geo[_pid][:3]
                _c = 0.5 * (_q0 + _q1)
                _Rb, _T = pose_tuple
                _v = _c[:2] - np.asarray(_T[:2], np.float64)
                _rng = float(np.linalg.norm(_v))
                _fwd = (_Rb @ np.array([1.0, 0.0, 0.0]))[:2]
                _ang = float(np.degrees(np.arctan2(_fwd[0]*_v[1]-_fwd[1]*_v[0], _fwd @ _v)))
                _side = "left" if _ang > 0 else "right"
                _name = bsp._rgb_name(_col[2], _col[1], _col[0])
                return f" [{_name}, {abs(_ang):.0f}deg {_side}, {_rng:.1f}m]"
            for _ln in (tracker_logs or []):
                if "PRUNE" in _ln:
                    self.get_logger().warn(f"[EVICT]{_whereis(_ln)} {_ln.strip()}")
        except Exception as e:
            self.get_logger().warn(f"[global-map] update failed: {e}", throttle_duration_sec=5.0)
        tracks = [(tp.apex.copy(), tp.p0.copy(), tp.p1.copy(),
                   (int(tp.color[0]), int(tp.color[1]), int(tp.color[2])),
                   bool(tp.fixed), int(tp.mask)) for tp in self._plane_tracker.planes]
        track_vr = [(tp.v0, tp.v1) for tp in self._plane_tracker.planes]
        _now_m = time.monotonic()
        _cur_pids = {int(tp.pid): (np.asarray(tp.p0[:2], np.float64),
                                   np.asarray(tp.p1[:2], np.float64))
                     for tp in self._plane_tracker.planes if hasattr(tp, "pid")}
        for _pid, (_sa, _sb) in self._prev_cur_pids.items():
            if _pid not in _cur_pids:
                self._evict_suppress.append((_sa, _sb, _now_m))
        self._prev_cur_pids = _cur_pids
        if getattr(self, "viz_full", False):
            # ONE BURST PER FRAME. Every visual for this frame is rendered first with its message
            # HELD, then the planes are published, then the held messages are flushed right behind
            # them -- so in RViz the masks, seeds, rays, top-downs and planes of frame k appear at
            # the same instant instead of the overlay leading the geometry by most of a frame.
            self._viz_hold = []
            try:
                self._render_viz_burst(payload, pose_tuple, hdr)
            except Exception as e:
                self.get_logger().warn(f"[viz_full] burst render failed: {e}", throttle_duration_sec=5.0)
            held, self._viz_hold = self._viz_hold, None
            self._publish_global_planes(hdr, tracks, pose_tuple, track_vr)   # planes (+lines, evicted)
            for _p, _m in held:
                _p.publish(_m)
        else:
            self._publish_global_planes(hdr, tracks, pose_tuple, track_vr)
        if self.enable_final_map:
            _final_thread.join()
            if "err" in _final_box:
                self.get_logger().warn(f"[final-map] update failed: {_final_box['err']}",
                                       throttle_duration_sec=5.0)
            elif "tracks" in _final_box:
                self._publish_final_planes(hdr, _final_box["tracks"], pose_tuple, _final_box["vr"])

    def _canonical_tick(self, pose_tuple, pc_xyz, tracker_logs, pre_geo):
        """CANONICAL-RUN bookkeeping, called once per processed frame: live trajectory buffer
        (serves both recording and GT anchoring), downsampled world cloud + ever/evicted plane
        ledger (record_run only), and the one-shot trajectory-anchored GT alignment."""
        if pose_tuple is not None:
            _T = np.asarray(pose_tuple[1], np.float64).reshape(3)
            _q = _R_to_quat(np.asarray(pose_tuple[0], np.float64))
            self._rec_poses.append((time.time(), float(_T[0]), float(_T[1]), float(_T[2]),
                                    float(_q[0]), float(_q[1]), float(_q[2]), float(_q[3])))
        self._rec_frame += 1
        # -- later-run GT anchoring: retry every 20 frames until it locks. DISABLED when the
        # scene-cloud ICP anchor is active (gt_ref_cloud) -- one aligner at a time.
        if (self._canon_traj is not None and self._gt_traj_T is None
                and getattr(self, "_gt_ref_xyz", None) is None
                and getattr(self, "_gt_cloud_msg", None) is not None
                and len(self._rec_poses) >= 40 and self._rec_frame % 20 == 0):
            self._align_gt_trajectory()
        if not self._record_run:
            return
        if pc_xyz is not None and len(pc_xyz) and pose_tuple is not None:
            P = np.asarray(pc_xyz, np.float32)
            if len(P) > 20000:
                P = P[np.random.default_rng(self._rec_frame).choice(len(P), 20000, replace=False)]
            self._rec_cloud.append(bsp._cam_to_world(P.astype(np.float64), pose_tuple).astype(np.float32))
        cur = {}
        for tp in self._plane_tracker.planes:
            cur[int(tp.pid)] = {
                "pid": int(tp.pid),
                "p0": [float(v) for v in np.asarray(tp.p0).reshape(3)],
                "p1": [float(v) for v in np.asarray(tp.p1).reshape(3)],
                "v0": float(tp.v0) if tp.v0 is not None else 0.0,
                "v1": float(tp.v1) if tp.v1 is not None else 2.0,
                "color": [int(c) for c in tp.color],
                "frame": int(self._rec_frame),
            }
        # frames where a NEW plane appeared get their algorithm input recorded (placed_input)
        self._canon_placed_now = bool(set(cur) - set(pre_geo or {}))
        # "frame" is the last-alive update; keep the FIRST-seen frame as the stable placement stamp
        for pid, e in cur.items():
            e["placed_frame"] = int(self._rec_ever.get(pid, {}).get("placed_frame", self._rec_frame))
        self._rec_ever.update(cur)                       # latest geometry wins while alive
        gone = set(pre_geo or {}) - set(cur)
        if gone:
            _logs = " | ".join(tracker_logs or [])
            for pid in gone:
                if pid in self._rec_ever and not any(e["pid"] == pid for e in self._rec_evicted):
                    import re as _re
                    e = dict(self._rec_ever[pid]); e["evict_frame"] = int(self._rec_frame)
                    m = _re.search(rf"[A-Z][A-Z-]+[^|]*#{pid}\b[^|]*", _logs)
                    e["reason"] = m.group(0).strip() if m else "evicted"
                    self._rec_evicted.append(e)
        if self._rec_frame % 20 == 0:
            self._save_canonical()

    def _save_canonical(self):
        """Write/refresh the canonical-run artifacts (called every 20 frames -- crash-safe)."""
        try:
            d = self._canon_dir
            if self._rec_poses:
                np.savetxt(os.path.join(d, "trajectory.txt"),
                           np.asarray(self._rec_poses, np.float64),
                           fmt="%.6f", header="t x y z qx qy qz qw", comments="")
            if self._rec_cloud:
                P = np.concatenate(self._rec_cloud)
                k = np.unique(np.floor(P / 0.10).astype(np.int64), axis=0)
                P = (k.astype(np.float64) + 0.5) * 0.10          # voxel centers, 0.10m
                self._rec_cloud = [P.astype(np.float32)]         # keep the deduped set only
                import open3d as o3d
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(P)
                o3d.io.write_point_cloud(os.path.join(d, "scene_cloud.ply"), pc)
            json.dump({"ever": list(self._rec_ever.values()), "evicted": self._rec_evicted},
                      open(os.path.join(d, "planes_run.json"), "w"), indent=1)
        except Exception as e:
            self.get_logger().warn(f"[canon] save failed: {e}")

    def _align_gt_trajectory(self):
        """Anchor GT to THIS run: rigid ICP of the live trajectory onto the CANONICAL run's
        trajectory (same physical path -> same 3D curve; only the SLAM anchor differs), then
        redraw the GT overlay (annotated in the canonical frame) through the inverse.
        Deterministic -- no clouds, no median-Z guessing."""
        live = np.asarray([p[1:] for p in self._rec_poses], np.float64)
        span = float(np.linalg.norm(live.max(0) - live.min(0)))
        if span < 4.0:                                   # need enough curve to be distinctive
            return
        try:
            import open3d as o3d
            # source = ANCHOR (frames) trajectory, target = live: every anchor pose has a true
            # correspondence on the live curve once the robot has traversed it; live poses beyond
            # the anchor span (e.g. camera-dead tail) then simply never enter the fit.
            src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(self._canon_traj)
            tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(live)
            T = np.eye(4)
            reg = None
            for th in (2.0, 0.5, 0.15):                  # coarse -> fine
                reg = o3d.pipelines.registration.registration_icp(
                    src, tgt, th, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                T = np.asarray(reg.transformation)
            if reg is None or reg.fitness < 0.6:
                self.get_logger().warn(f"[gt-traj] fitness {0.0 if reg is None else reg.fitness:.2f}"
                                       f" too low with {len(live)} live poses -- will retry")
                return
            Tinv = T                                     # anchor(frames) -> live, applies to GT directly
            P = (self._gt_xyz.astype(np.float64) @ Tinv[:3, :3].T) + Tinv[:3, 3]
            _hdr = Header(); _hdr.frame_id = "map"
            self._gt_cloud_msg = _make_xyzrgb_cloud(_hdr, P.astype(np.float32), self._gt_rgb)
            self._gt_traj_T = T
            self.get_logger().info(f"[gt-traj] GT anchored to canonical trajectory: "
                                   f"fit={reg.fitness:.2f} shift=({Tinv[0,3]:+.2f},"
                                   f"{Tinv[1,3]:+.2f},{Tinv[2,3]:+.2f})m")
        except Exception as e:
            self.get_logger().warn(f"[gt-traj] failed: {e}")

    def _align_gt_to_live(self):
        """ICP-register the recorded-frame GT points onto THIS session's live map, using the
        accumulated live obstacle cloud. BOUNDED: reject a solution that translates > 3 m or
        rotates > 20 deg (that means ICP snapped GT to the wrong wall) and keep identity."""
        buf = getattr(self, "_live_world_buf", [])
        n = sum(len(b) for b in buf)
        if n < 12000:
            return False
        live = np.concatenate(buf).astype(np.float64)
        _ref = getattr(self, "_gt_ref_xyz", None)
        src_pts = _ref if _ref is not None else self._gt_xyz.astype(np.float64)
        try:
            import open3d as o3d
            # STRUCTURE-to-structure when a reference scene cloud is set (the canonical run's
            # map, same frame as the pre-aligned GT): ICP the ref map onto the live map and
            # let GT ride the transform. Identity init: same bag => anchors agree to ~cm.
            _g = o3d.geometry.PointCloud(); _g.points = o3d.utility.Vector3dVector(src_pts)
            _l = o3d.geometry.PointCloud(); _l.points = o3d.utility.Vector3dVector(live)
            _l = _l.voxel_down_sample(0.15); _g = _g.voxel_down_sample(0.15)
            T = np.eye(4)
            reg = None
            for th in (2.0, 0.6, 0.2):                     # coarse -> fine
                reg = o3d.pipelines.registration.registration_icp(
                    _g, _l, th, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
                T = np.asarray(reg.transformation)
            trans = float(np.linalg.norm(T[:3, 3]))
            rot = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
            if trans > 3.0 or rot > 20.0 or reg.fitness < 0.3:
                self.get_logger().warn(f"[gt-align] rejected (trans={trans:.2f}m rot={rot:.0f}deg "
                                       f"fit={reg.fitness:.2f}) -- raw GT for now, will retry")
                self._live_world_buf = buf[-40:]           # keep accumulating; retry on next tick
                return False
            self._gt_align_T = T
            self.get_logger().info(f"[gt-align] {'scene-cloud' if _ref is not None else 'GT-point'} "
                                   f"ICP onto live map: trans={trans:.2f}m rot={rot:.1f}deg "
                                   f"fit={reg.fitness:.2f}")
        except Exception as e:
            self.get_logger().warn(f"[gt-align] failed: {e}")
            self._gt_align_T = np.eye(4)
        # rebuild the published cloud with the correction applied
        P = (self._gt_xyz.astype(np.float64) @ self._gt_align_T[:3, :3].T) + self._gt_align_T[:3, 3]
        _hdr = Header(); _hdr.frame_id = "map"
        self._gt_cloud_msg = _make_xyzrgb_cloud(_hdr, P.astype(np.float32), self._gt_rgb)
        self._gt_aligned = True
        self._live_world_buf = []
        return True

    def _pub_gt_planes(self):
        if self._gt_cloud_msg is None:
            return
        if not getattr(self, "_gt_aligned", True):
            # ICP is ~1s of CPU: run it on a WORKER THREAD (an executor-thread ICP every 2s
            # tick stalled cloud processing -> visibly slow seed publishing), and only retry
            # when the live buffer actually GREW 50% since the last failed attempt (same data
            # -> same rejection; retrying without new information just burns CPU).
            if not getattr(self, "_gt_align_busy", False):
                n = sum(len(b) for b in getattr(self, "_live_world_buf", []))
                last = int(getattr(self, "_gt_align_last_n", 0))
                if n >= 12000 and n >= int(1.5 * last):
                    self._gt_align_busy = True
                    self._gt_align_last_n = n

                    def _bg():
                        try:
                            self._align_gt_to_live()
                        finally:
                            self._gt_align_busy = False
                    threading.Thread(target=_bg, daemon=True).start()
        self._gt_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_gt_planes.publish(self._gt_cloud_msg)

    def _publish_static_camera_tf(self):
        a = self.args
        parent = self.declare_parameter("tf_parent_frame", "sensor_at_scan").value
        child = self.declare_parameter("tf_child_frame", "camera_link").value
        # Camera mount offset (from the launch camX/camY/camZ).
        tx = self.declare_parameter("tf_tx", -0.12).value
        ty = self.declare_parameter("tf_ty", -0.075).value
        tz = self.declare_parameter("tf_tz", 0.255).value
        # The body->viewer rotation collapses to a -90 deg yaw (qz,qw below).
        # Flip the sign of tf_qz if the planes appear mirrored about vertical.
        qz = self.declare_parameter("tf_qz", -0.70710678).value
        qw = self.declare_parameter("tf_qw", 0.70710678).value

        self._static_tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = float(tz)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self._static_tf.sendTransform(t)
        self.get_logger().info(f"static TF {child} -> {parent} published (mount offset + yaw)")

    def _build_models(self):
        a = self.args
        self.use_da2 = bool(self.declare_parameter("use_da2", False).value)   # OFF: no DA2 depth-jump filtering
        use_cuda = self.device.type == "cuda"
        self.get_logger().info(f"Building SAM3 + DA2 on {self.device} ...")
        t0 = time.perf_counter()
        build_dev = "cpu" if use_cuda and (a.bf16 or a.int8) else str(self.device)
        sam = bsp.build_sam3(a, build_dev)
        if use_cuda and a.int8:
            sam = bsp.convert_model_linear_to_int8(sam)
            sam = sam.to(self.device)
            torch.cuda.empty_cache()
        elif use_cuda and a.bf16:
            sam = bsp.convert_model_floating_to_bf16(sam)
            sam = sam.to(self.device)
            torch.cuda.empty_cache()
            bsp.add_linear_input_cast_hooks(sam)
        else:
            sam = sam.to(self.device)
        sam.eval()
        self.get_logger().info(
            f"SAM3 precision={'int8' if a.int8 else ('bf16' if a.bf16 else 'fp32')} ckpt={a.ckpt_path}")
        self.sam_model = sam
        self.processor = bsp.Sam3Processor(sam, confidence_threshold=a.conf_th)
        self.per_prompt_feats = bsp.load_per_prompt_feats(a, self.device)
        # FLOOR-touch gate: a second SAM3 prompt ("floor") each frame -> floor points -> accumulated in the
        # tracker -> passed to clench so floor-lying planes are rejected (matches the batch). OFF -> skipped.
        self.floor_gate = bool(self.declare_parameter("floor_gate", True).value)
        self._floor_prompt = list(self.declare_parameter("floor_prompt", ["floor"]).value)
        self._floor_feats = None
        if self.floor_gate:
            try:
                import copy as _copy
                _fa = _copy.copy(a); _fa.prompt = self._floor_prompt
                self._floor_feats = bsp.load_per_prompt_feats(_fa, self.device)
                self.get_logger().info(f"floor-touch gate ON  (floor prompt = {self._floor_prompt})")
            except Exception as e:
                self._floor_feats = None
                self.get_logger().warn(f"floor prompt features failed ({e}) -- floor gate disabled")
        # OPEN-DOORWAY gate: a mask scoring higher on "open doorway" than glass is an OPENING, not a plane
        # -> not solved, and its world cells are LOCKED in the tracker until doorway_cancel_frames consecutive
        # glass frames cancel it. Separate cheap SAM3 decode (reuses the encoded image), like the floor pass.
        self.doorway_gate = bool(self.declare_parameter("doorway_gate", False).value)  # REMOVED (2026-08): default off
        self._doorway_prompt = list(self.declare_parameter("doorway_prompt", ["open doorway"]).value)
        self.doorway_min_score = float(self.declare_parameter("doorway_min_score", 0.5).value)
        self.doorway_overlap_frac = float(self.declare_parameter("doorway_overlap_frac", 0.5).value)
        self.doorway_lock_skip_frac = float(self.declare_parameter("doorway_lock_skip_frac", 0.5).value)
        self._doorway_feats = None
        if self.doorway_gate:
            try:
                import copy as _copy
                _cached = torch.load(a.cached_text_features, map_location="cpu", weights_only=False)
                _cp = [str(p).lower() for p in _cached.get("prompts", [])]
                if not all(str(p).lower() in _cp for p in self._doorway_prompt):
                    # load_per_prompt_feats SILENTLY falls back to prompt 0 (window) for a missing prompt,
                    # which would mass-declare glass masks as doorways -> disable the gate instead.
                    self.doorway_gate = False
                    self.get_logger().warn(f"doorway prompt {self._doorway_prompt} NOT in cached .pt {_cp} "
                                           f"-- doorway gate DISABLED (regenerate the .pt with 'open doorway')")
                else:
                    _da = _copy.copy(a); _da.prompt = self._doorway_prompt
                    self._doorway_feats = bsp.load_per_prompt_feats(_da, self.device)
                    self.get_logger().info(f"open-doorway gate ON  (prompt = {self._doorway_prompt})")
            except Exception as e:
                self._doorway_feats = None; self.doorway_gate = False
                self.get_logger().warn(f"doorway prompt features failed ({e}) -- doorway gate disabled")
        # DA2 has TWO INDEPENDENT uses: (1) depth-JUMP seed filtering (use_da2), and (2) the pinhole
        # lidar<->cam ALIGNMENT (use_pinhole_align). Build the model if EITHER is on, so alignment can run
        # with jump-filtering OFF (use_da2:=false + use_pinhole_align:=true still loads DA2 for alignment).
        self.use_pinhole_align = bool(self.declare_parameter("use_pinhole_align", False).value)
        if self.use_da2 or self.use_pinhole_align:
            self.da2_model = bsp.build_da2(a, self.device)
            self.da2_model.eval()
        else:
            self.da2_model = None
        if not self.use_da2:
            self.get_logger().warn("DA2 depth-jump seed filtering OFF (use_da2:=false)")

        # DA2 pinhole <-> last-scan alignment (corrects residual lidar<->cam extrinsic/latency per frame).
        self._pinhole_cfg = pda.PinholeCfg(
            out_w=int(self.declare_parameter("pinhole_w", 640).value),
            out_h=int(self.declare_parameter("pinhole_h", 480).value),
            fx=float(self.declare_parameter("pinhole_fx", 320.0).value),
            fy=float(self.declare_parameter("pinhole_fy", 320.0).value),
            cx=float(self.declare_parameter("pinhole_cx", 320.0).value),
            cy=float(self.declare_parameter("pinhole_cy", 240.0).value),
            yaw_deg=float(self.declare_parameter("pinhole_yaw_deg", 0.0).value),
            pitch_deg=float(self.declare_parameter("pinhole_pitch_deg", 0.0).value),
            pano_pixel_offset=float(a.pano_pixel_offset),
            min_score=float(self.declare_parameter("pinhole_align_min_score", 0.90).value))
        if self.use_pinhole_align and self.da2_model is None:   # model build failed -> can't align
            self.use_pinhole_align = False
            self.get_logger().warn("pinhole align disabled -- DA2 model unavailable")
        if self.use_pinhole_align:
            self.get_logger().info(
                f"DA2 pinhole align ON ({self._pinhole_cfg.out_w}x{self._pinhole_cfg.out_h}, "
                f"min_score={self._pinhole_cfg.min_score})")
        # GK-PINHOLE live mode: run the DETECTOR itself on a synthetic pinhole view remapped from the
        # pano (same camera_config.json + same remap formula as the offline gen_pinhole/eval path, so
        # live results match batch --pinhole-cfg replays exactly). Switches ALL of the batch core's
        # pixel<->ray math to the pinhole model via set_pinhole_model().
        self.pinhole_cfg_path = str(self.declare_parameter("pinhole_cfg", "").value)
        self._pin_det_cfg = None
        self._pin_det_maps = None       # cached pano->pinhole cv2.remap maps (built on first frame)
        self._pin_det_wh = None
        if self.pinhole_cfg_path:
            with open(self.pinhole_cfg_path) as _f:
                self._pin_det_cfg = json.load(_f)
            bsp.set_pinhole_model(self._pin_det_cfg)
            if self.use_pinhole_align:
                self.use_pinhole_align = False
                self.get_logger().warn("GK-PINHOLE mode: DA2 pano alignment disabled (pano-specific)")
            self.get_logger().info(
                f"GK-PINHOLE live mode ON ({self._pin_det_cfg['width']}x{self._pin_det_cfg['height']}, "
                f"fx={self._pin_det_cfg['camera_internal']['fx']})")
        # Save a DA2 | BEFORE | AFTER alignment panel EACH frame that aligns (opt-in; off in running mode).
        self.save_align_panel = bool(self.declare_parameter("save_align_panel", False).value)
        self._align_panel_dir = str(self.declare_parameter(
            "align_panel_dir", "./glass_killer_ros_align").value)
        self.get_logger().info(f"Models ready in {time.perf_counter() - t0:.1f}s")

    def _on_cloud(self, msg: PointCloud2):
        with self._lock:
            self._latest_cloud = msg
            self._t_last_cloud = time.time()

    def _on_terrain(self, msg: PointCloud2):
        try:
            xyz, inten = _parse_cloud_xyzi(msg)
            with self._lock:
                self._latest_terrain = (xyz, inten)
        except Exception as e:
            self.get_logger().warn(f"[terrain] parse failed: {e}", throttle_duration_sec=5.0)

    def _on_last_scan(self, msg: PointCloud2):
        with self._lock:
            self._latest_last_scan = msg

    def _gpu_keepwarm_worker(self):
        """Keep the (laptop) GPU clocked up so SAM3 doesn't run cold. Sustained GPU matmul load through
        everything EXCEPT the SAM3 forward (which sets _sam3_running) -- so the clocks are already high
        when SAM3 starts, with no contention during it. All GPU-only (no per-iteration CPU sync) so it
        does NOT fight the executor for the GIL during the CPU-heavy clench phase. Daemon; node lifetime."""
        try:
            a = torch.randn(2048, 2048, device=self.device, dtype=torch.float16)
            b = torch.randn(2048, 2048, device=self.device, dtype=torch.float16)
        except Exception as e:
            self.get_logger().warn(f"GPU keep-warm disabled ({e})")
            return
        while rclpy.ok() and not self._stop_keepwarm:               # exits as soon as the node shuts down
            if self._sam3_running:                                  # pause ONLY for the SAM3 forward
                time.sleep(0.002)
                continue
            try:
                for _ in range(24):                                 # sustained load -> holds clocks high
                    b = (a @ b) * 0.001                             # GPU-only bound (no .item()/CPU sync)
                torch.cuda.synchronize()                            # ONE sync per batch (frees GIL while GPU runs)
            except Exception:
                pass
            time.sleep(0.001)
        del a, b
        if self.device.type == "cuda":
            torch.cuda.empty_cache()                                 # release the keep-warm buffer on exit

    def _watchdog(self):
        """Background thread: logs the node state every few seconds, independent of the
        executor, so a hang inside _process() is visible instead of silent."""
        period = 5.0
        while rclpy.ok():
            time.sleep(period)
            now = time.time()
            with self._lock:
                busy = self._busy
                busy_since = self._busy_since
                t_cloud = self._t_last_cloud
                t_frame = self._t_last_frame
            if busy and busy_since > 0.0 and (now - busy_since) > period:
                self.get_logger().warn(
                    f"[watchdog] STUCK in _process for {now - busy_since:.1f}s "
                    f"(likely a CUDA OOM/wedge or a huge cloud); consider lowering "
                    f"max_cloud_points or the C++ stackTimeWindow")
            elif t_cloud == 0.0:
                self.get_logger().warn("[watchdog] no /glass_killer/cloud received yet (is the C++ node / launch up?)")
            elif (now - t_cloud) > period:
                self.get_logger().warn(
                    f"[watchdog] idle: no /glass_killer/cloud for {now - t_cloud:.1f}s "
                    f"(upstream stopped? check `ros2 topic hz /glass_killer/cloud`)")
            # healthy: stay silent (only the per-frame "inference ..." line is printed)

    def _on_image(self, msg: Image):
        with self._lock:
            self._latest_image = msg
            t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self._img_buf.append((t, msg))

    def _on_odom_at_scan(self, msg):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        entry = (stamp, bsp._quat_to_R(q.x, q.y, q.z, q.w),
                 np.array([p.x, p.y, p.z], dtype=np.float64))
        with self._lock:
            self._pose_buf.append(entry)

    def _on_cloud_pose(self, msg: PoseStamped):
        p = msg.pose.position; q = msg.pose.orientation
        k = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        entry = (bsp._quat_to_R(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z], dtype=np.float64))
        with self._lock:
            if k not in self._cloud_pose:
                if len(self._cloud_pose_keys) >= 64:
                    self._cloud_pose.pop(self._cloud_pose_keys.popleft(), None)
                self._cloud_pose_keys.append(k)
            self._cloud_pose[k] = entry

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position; q = msg.pose.orientation
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        entry = (stamp, bsp._quat_to_R(q.x, q.y, q.z, q.w),
                 np.array([p.x, p.y, p.z], dtype=np.float64))
        with self._lock:
            self._latest_pose = msg
            self._pose_recv_mono = time.monotonic()
            self._pose_buf.append(entry)

    def _pose_at(self, stamp_msg):
        """The buffered pose (R_body, T) nearest a given ROS stamp -- used to lock a frame's world
        transform to its CAPTURE time. Falls back to the latest pose if the stamp is 0/unbuffered."""
        k = int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)
        with self._lock:
            exact = self._cloud_pose.get(k)
            buf = list(self._pose_buf)
        if exact is not None:                           # the pose this cloud was BUILT with
            self._pose_match_gap = 0.0
            self._pose_feed_lag = 0.0
            self._pose_src = "cloud_pose"
            return exact
        self._pose_src = "buffer"                       # fallback: nearest odom (old behaviour)
        if not buf:
            return None
        target = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        if target <= 0.0:
            return (buf[-1][1], buf[-1][2])
        best = min(buf, key=lambda e: abs(e[0] - target))
        self._pose_match_gap = abs(best[0] - target)   # |matched pose stamp - cloud stamp| (s)
        self._pose_feed_lag = target - buf[-1][0]      # cloud stamp beyond NEWEST pose (>0 = feed stuck)
        return (best[1], best[2])

    def _tick(self):
        if self.role == "mapping":
            return                                   # mapping process does no perception; it
            #                                          runs the tracker from /gkpipe/geom instead
        with self._lock:
            if self._busy or self._latest_cloud is None or self._latest_image is None:
                return
            cloud_msg = self._latest_cloud
            _ck = int(cloud_msg.header.stamp.sec) * 1_000_000_000 + int(cloud_msg.header.stamp.nanosec)
            if _ck not in self._cloud_pose and (time.time() - self._t_last_cloud) < 0.10:
                return           # its exact pose is published right behind it; retry next tick
            # Match the IMAGE to the CLOUD's capture stamp (buffered), not just "latest": while the
            # robot turns, a latest-of-each pairing puts the mask and the LiDAR stack at different
            # yaws -> camera<->LiDAR shift (worst on pinhole's narrow FOV). Snap to the nearest stamp.
            _ct = float(cloud_msg.header.stamp.sec) + float(cloud_msg.header.stamp.nanosec) * 1e-9
            if self._img_buf and _ct > 0.0:
                _bi = min(self._img_buf, key=lambda e: abs(e[0] - _ct))
                image_msg = _bi[1]
                self._img_match_gap = abs(_bi[0] - _ct)
            else:
                image_msg = self._latest_image
            last_scan_msg = self._latest_last_scan   # newest single scan (may be None early on)
            self._latest_cloud = None  # consume; only process fresh pairs
            self._busy = True
            self._busy_since = time.time()
        try:
            self._process(cloud_msg, image_msg, last_scan_msg)
        except Exception as e:  # keep the node alive across bad frames
            self.get_logger().error(f"frame failed: {e}")
        finally:
            with self._lock:
                self._busy = False
                self._busy_since = 0.0
                self._t_last_frame = time.time()

    # ---- clench vertical-rectangle pipeline (no file I/O; publishes results) ---- #
    def _build_pin_det_maps(self, W360: int, H360: int):
        """cv2.remap maps for pano -> detector pinhole. EXACTLY the gen_pinhole.py / pano_to_pinhole_node
        spherical mapping, so live GK-pinhole frames are pixel-identical to the offline/eval images."""
        c = self._pin_det_cfg
        ci = c["camera_internal"]
        Wp, Hp = int(c["width"]), int(c["height"])
        yaw = np.radians(float(c.get("pinhole_yaw_deg", 0.0)))
        pit = np.radians(float(c.get("pinhole_pitch_deg", 0.0)))
        u, v = np.meshgrid(np.arange(Wp, dtype=np.float64), np.arange(Hp, dtype=np.float64))
        X = (u - float(ci["cx"])) / float(ci["fx"])
        Y = (v - float(ci["cy"])) / float(ci["fy"])
        th = np.arctan2(X, 1.0) + yaw
        ph = np.arctan2(Y, np.sqrt(X * X + 1.0)) + pit
        m1 = (((th + np.pi) / (2.0 * np.pi) * W360) % W360).astype(np.float32)
        # UNIFORM-ANGULAR pano: W/(2pi) px/rad on BOTH axes (1920x640 = 360x120 deg, NOT +/-90),
        # matching the batch core's equirect model. The old (ph+pi/2)/pi*H mapping stretched the
        # pinhole 1.5x vertically (true fy ~780 vs the config's 520).
        m2 = np.clip(ph * W360 / (2.0 * np.pi) + H360 / 2.0, 0, H360 - 1).astype(np.float32)
        return m1, m2

    def _process(self, cloud_msg: PointCloud2, image_msg: Image, last_scan_msg: PointCloud2 = None):
        a = self.args
        # CAPTURE-TIME pose: lock this frame's world transform to the CLOUD's timestamp, sampled ONCE,
        # so inference latency (jitter from a co-running pipeline) can't drift the placement. Used for
        # every world transform below instead of re-fetching the ever-newer "latest" pose.
        self._frame_pose = self._pose_at(cloud_msg.header.stamp)
        # Arm the batch core's world-coordinate debug-ply writer for THIS frame (spill tracks
        # and any synchronous ply saves; the async keysave writer re-arms per saved item).
        bsp._WORLD_SAVE_POSE = self._frame_pose
        # STALE-POSE / SYNC alarms: a stuck pose feed or an image far from the cloud stamp rotates
        # the whole solve into the map (planes appear where the robot USED to be / at a slanted yaw).
        _gap = float(getattr(self, "_pose_match_gap", 0.0))
        _lag = float(getattr(self, "_pose_feed_lag", 0.0))
        if _gap > 0.15:
            self.get_logger().warn(
                f"[pose] STALE: nearest buffered pose is {_gap:.2f}s from the cloud stamp "
                f"(pose feed {'STUCK, ' + format(_lag, '.2f') + 's behind' if _lag > 0.15 else 'ok'}) "
                f"-- placement rotated by whatever the robot moved in that time")
        _img_t = float(image_msg.header.stamp.sec) + float(image_msg.header.stamp.nanosec) * 1e-9
        _cld_t = float(cloud_msg.header.stamp.sec) + float(cloud_msg.header.stamp.nanosec) * 1e-9
        if _img_t > 0.0 and _cld_t > 0.0 and abs(_img_t - _cld_t) > 0.15:
            self.get_logger().warn(
                f"[sync] image-cloud stamp gap {_img_t - _cld_t:+.2f}s -- while yawing, the mask and "
                f"the LiDAR stack disagree by that much rotation (slanted planes)")
        t_start = time.perf_counter()
        _e3 = np.empty((0, 3), np.float32)
        _e3u = np.empty((0, 3), np.uint8)
        PT: dict = {}                       # per-phase wall time
        _m = time.perf_counter()
        def _pt(key):                       # record time since the last marker, then re-arm
            nonlocal _m
            PT[key] = PT.get(key, 0.0) + (time.perf_counter() - _m)
            _m = time.perf_counter()

        # RGB pano from the Image message (bgr8).
        H, W = image_msg.height, image_msg.width
        bgr = np.frombuffer(bytes(image_msg.data), dtype=np.uint8).reshape(H, W, -1)[:, :, :3]
        bgr = np.ascontiguousarray(bgr)
        pano_bgr = bgr                                  # keep the raw pano for save_input
        H_orig, W_orig = bgr.shape[:2]
        # GK-PINHOLE mode: remap pano -> pinhole (deterministic, same pose/stamp) and hand the
        # detector the pinhole view; the batch core's _PINHOLE global handles all geometry.
        if getattr(self, "_pin_det_cfg", None) is not None:
            if self._pin_det_maps is None or self._pin_det_wh != (W_orig, H_orig):
                self._pin_det_maps = self._build_pin_det_maps(W_orig, H_orig)
                self._pin_det_wh = (W_orig, H_orig)
            bgr = cv2.remap(pano_bgr, self._pin_det_maps[0], self._pin_det_maps[1],
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            H_orig, W_orig = bgr.shape[:2]              # downstream sizes = detector image
            # publish the DETECTOR'S pinhole view so rviz shows what the method actually sees
            if not hasattr(self, "pub_pinhole_img"):
                self.pub_pinhole_img = self.create_publisher(Image, "/glass_killer/pinhole_image", 2)
            self._publish_image(self.pub_pinhole_img, bgr)
        pil_img = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        _pt("rgb")

        # Cloud -> camera frame (reference_viewer_zup convention).
        xyz_raw = _parse_cloud_xyz(cloud_msg)
        if xyz_raw.shape[0] == 0:
            return
        # Bound work/VRAM: the 5-second stacked cloud can grow large; subsample if needed.
        if self._max_cloud_points > 0 and xyz_raw.shape[0] > self._max_cloud_points:
            sel = np.random.default_rng(0).choice(xyz_raw.shape[0], self._max_cloud_points, replace=False)
            xyz_raw = xyz_raw[sel]
        _pt("cloud_parse")
        xyz_cam_all = bsp._convert_input_cloud_to_pano_camera(xyz_raw, mode=a.projection_coord_mode)

        # DA2 pinhole <-> last-scan alignment: solve the residual lidar<->cam extrinsic correction from a
        # front pinhole view (single scan avoids the stack's glass see-through), then apply it to the whole
        # cloud BEFORE projection so every downstream stage (range image, seeds, clench) uses aligned points.
        align_debug = None
        if self.use_pinhole_align and self.da2_model is not None and last_scan_msg is not None:
            try:
                _ls_raw = _parse_cloud_xyz(last_scan_msg)
                if _ls_raw.shape[0] >= 50:
                    _ls_cam = bsp._convert_input_cloud_to_pano_camera(_ls_raw, mode=a.projection_coord_mode)
                    if self._pinhole_remap is None:
                        self._pinhole_remap = pda.build_equirect_to_pinhole_remap(W_orig, H_orig, self._pinhole_cfg)
                    _isz = int(a.da2_input_size)
                    _t_al = time.perf_counter()
                    _res = pda.align_scan_to_da2(
                        bgr, _ls_cam,
                        lambda _b: self.da2_model.infer_image(_b, _isz),
                        self._pinhole_cfg, remap=self._pinhole_remap)
                    _al_ms = (time.perf_counter() - _t_al) * 1000.0
                    _sol = _res["pose"]; _sc = float(_res["score"])
                    _applied = _sc >= self._pinhole_cfg.min_score
                    if _applied:
                        xyz_cam_all = pda.transform_pts(
                            xyz_cam_all, _sol["rx"], _sol["ry"], _sol["rz"],
                            _sol["tx"], _sol["ty"], _sol["tz"])
                        self.get_logger().info(
                            f"pinhole-align APPLIED score={_sc:.3f} ({_al_ms:.0f}ms) "
                            f"r=({_sol['rx']:+.2f},{_sol['ry']:+.2f},{_sol['rz']:+.2f}) "
                            f"t=({_sol['tx']:+.3f},{_sol['ty']:+.3f},{_sol['tz']:+.3f})")
                    else:
                        self.get_logger().warn(f"pinhole-align SKIP score={_sc:.3f} < {self._pinhole_cfg.min_score} ({_al_ms:.0f}ms)")
                    # BEFORE = raw last-scan pinhole depth; AFTER = pose-transformed. Kept for the SPACE panel.
                    _ls_after = pda.transform_pts(_ls_cam, _sol["rx"], _sol["ry"], _sol["rz"],
                                                  _sol["tx"], _sol["ty"], _sol["tz"])
                    align_debug = {
                        "rgb": _res["rgb_pinhole"], "da2": _res["da2_depth"],
                        "lidar_before": _res["lidar_depth"],
                        "lidar_after": pda.project_cloud_to_pinhole_depth(_ls_after, self._pinhole_cfg),
                        "pose": _sol, "score": _sc, "ms": _al_ms, "applied": bool(_applied),
                        "cell": int(self._pinhole_cfg.cell_size),
                    }
                    if self.save_align_panel:              # DA2 | BEFORE | AFTER panel EACH align (opt-in)
                        try:
                            os.makedirs(self._align_panel_dir, exist_ok=True)
                            self._save_alignment_panel(
                                os.path.join(self._align_panel_dir, f"align_{self._save_counter:06d}.png"), align_debug)
                        except Exception as _pe:
                            self.get_logger().warn(f"[align-panel] save failed ({_pe})", throttle_duration_sec=5.0)
            except Exception as _e:
                self.get_logger().warn(f"pinhole-align failed ({_e}) -- using raw cloud")
        _pt("pinhole_align")

        ui, vi, rr, hh, valid_pts = bsp._project_pano_xyz(
            xyz_cam_all, W_orig, H_orig,
            min_range=a.depth_min, max_range=a.depth_max, pixel_offset=a.pano_pixel_offset)
        pc_xyz = xyz_cam_all[valid_pts].astype(np.float32)
        _pt("project")
        if getattr(self, "viz_full", False):
            # RAW INPUT streams -- published HERE, not on the panel thread. The pano and the
            # LiDAR projection are sensor data, not inference products: gating them behind a
            # completed inference would throttle them to the algorithm's rate for no reason.
            # Both are cheap (a colormap + a dilate) and carry the source stamp, so a faster,
            # constant publish rate still aligns with every other stream at encode time.
            try:
                # (rgb pano goes out from the mapping burst; bgr_jpg is already in the payload)
                _lv = np.zeros((H_orig, W_orig, 3), np.uint8)
                _rn = (255.0 * (1.0 - np.clip(np.asarray(rr, np.float32) / 12.0,
                                              0.0, 1.0))).astype(np.uint8)
                _lc = cv2.applyColorMap(_rn.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
                _lv[np.clip(vi, 0, H_orig - 1), np.clip(ui, 0, W_orig - 1)] = _lc
                _lv = cv2.dilate(_lv, np.ones((2, 2), np.uint8))     # 1px dots -> visible
                # rendered here (cheap), PUBLISHED from the mapping burst with everything else
                self._viz_lidar_jpg = cv2.imencode(".jpg", _lv, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tobytes()
            except Exception as e:
                self.get_logger().warn(f"[viz_full] input viz failed: {e}",
                                       throttle_duration_sec=5.0)
            _pt("viz_input")

        # DA2 depth + geometric jump mask. DISABLED (use_da2:=false) -> no depth-jump filtering: the jump
        # mask is empty (keeps every seed) and depth is a flat placeholder.
        if self.use_da2:
            with torch.no_grad():
                depth_raw = self.da2_model.infer_image(bgr, int(a.da2_input_size))
            depth_da2 = np.asarray(depth_raw, dtype=np.float32)
            if depth_da2.shape[:2] != (H_orig, W_orig):
                depth_da2 = cv2.resize(depth_da2, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            jump_bool = bsp.geometric_jump_mask_pano(
                depth_da2, min_range=a.depth_min, max_range=a.depth_max,
                abs_thresh=float(a.da2_jump_abs_thresh_m), spacing_multiplier=float(a.da2_jump_spacing_multiplier),
                pixel_offset=a.pano_pixel_offset)
            jump_dil_full = cv2.dilate(jump_bool.astype(np.uint8),
                                       bsp.make_disk_kernel(int(a.da2_jump_dilate_px)), iterations=1).astype(bool)
        else:
            depth_da2 = np.zeros((H_orig, W_orig), np.float32)
            jump_dil_full = np.zeros((H_orig, W_orig), bool)
        _pt("da2")

        # SAM3 masks + small/big bbox splat (coverage-based). Flag it so the GPU keep-warm PAUSES only
        # for this forward (it warms through the CPU-heavy clench/seed/save gap that follows).
        self._sam3_running = True
        try:
            # Encode the image ONCE (the expensive backbone), then decode the glass prompt off it. The floor
            # prompt (below) reuses this same _sam3_state so the backbone is not paid twice.
            _sam3_state = bsp.sam3_encode_image(self.processor, pil_img, self.device, a)
            all_masks, all_scores, all_boxes, H_enc, W_enc = bsp.sam3_decode_prompts(
                self.sam_model, _sam3_state, self.per_prompt_feats, a.prompt, self.device, a)
        finally:
            self._sam3_running = False
        if not all_masks:
            self._publish_planes(cloud_msg.header, _e3, _e3u)
            self._publish_seeds(cloud_msg.header, _e3, _e3u)
            return
        _pt("sam3")                                   # SAM3 model forward ONLY (the neural inference)
        small_idx, big_idx, _, _ = bsp._bbox_dual_sorted_splat_indices_with_debug(
            all_masks, all_boxes, contain_tol=float(a.bbox_splat_tol),
            cover_frac=float(getattr(a, "splat_cover_frac", 0.75)))

        _link_px = int(getattr(a, "big_mask_link_break_px", 4))   # sever thin links, keep the biggest region
        _irr_frac = float(getattr(a, "big_mask_irregular_min_part_frac", 0.2))  # >=2 big parts -> discard mask
        big_masks_full = {}
        small_masks_full = {}                      # filled by the seed extractor below
        irr_fix = []                                              # (mask_id, before, after|None, is_irregular) for the SPACE cleanup panel
        for bi in big_idx:
            bi = int(bi)
            if bi >= len(all_masks):
                continue
            m = cv2.resize(all_masks[bi].astype(np.uint8), (W_orig, H_orig),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            # NEVER discard for irregularity: just sever thin links / drop stray parts and keep the biggest
            # region (link-break). A mask with multiple parts or a thin-neck connection is fixed, not rejected.
            cleaned = bsp._largest_region_break_links(m, _link_px).astype(bool)
            if int((m != cleaned).sum()) > 0:                    # link-break actually trimmed a blob
                irr_fix.append((bi, m, cleaned, False))
            big_masks_full[bi] = cleaned
        big_idx = [bi for bi in big_idx if int(bi) in big_masks_full]   # drop the discarded masks
        _pt("splat")                                  # bbox splat + big-mask largest-CC (should be tiny)
        if self.publish_overlays and not getattr(self, "viz_full", False):   # viz_full: in the mapping burst
            self._publish_big_mask_overlay(cloud_msg.header, bgr, big_idx, big_masks_full)
        _pt("big_overlay")

        # OPEN-DOORWAY declaration: a big mask is an open doorway (an OPENING, not a plane) if an
        # "open doorway" mask overlaps it (>= doorway_overlap_frac of the mask) with a score >= the mask's
        # glass score (and >= doorway_min_score). Cheap 2nd decode reusing the encoded image.
        door_declared = set()
        if self.doorway_gate and self._doorway_feats is not None and big_idx:
            try:
                _dm, _ds, _db, _, _ = bsp.sam3_decode_prompts(
                    self.sam_model, _sam3_state, self._doorway_feats, self._doorway_prompt, self.device, a)
                _dfull = [cv2.resize(m.astype(np.uint8), (W_orig, H_orig),
                                     interpolation=cv2.INTER_NEAREST).astype(bool) for m in _dm]
                for bi in big_idx:
                    gm = big_masks_full[int(bi)]; ga = int(gm.sum())
                    if ga == 0:
                        continue
                    gscore = float(all_scores[int(bi)]) if int(bi) < len(all_scores) else 0.0
                    for dm, dscore in zip(_dfull, _ds):
                        if float(dscore) < self.doorway_min_score or float(dscore) < gscore:
                            continue
                        if int(np.logical_and(gm, dm).sum()) / ga >= self.doorway_overlap_frac:
                            door_declared.add(int(bi)); break
                if door_declared:
                    self.get_logger().info(f"open-doorway declared: {sorted(door_declared)}")
            except Exception as _e:
                self.get_logger().warn(f"doorway decode failed ({_e})")
        _pt("doorway")

        # 4 corner rays per big mask.
        mask_ray_records = bsp._build_mask_corner_ray_records(
            small_idx=big_idx, small_masks_full=big_masks_full, W=W_orig, H=H_orig, args=a)
        _pt("rays")

        # Small-mask grid seeds, then assign each seed to its big-mask owner (coverage).
        # skip_full_masks: the full-res small-mask resizes are ONLY needed by the overlay
        # publisher and the debug savers -- when neither is active, seed extraction runs
        # entirely at grid resolution (encoder-res masks -> INTER_AREA coverage fractions).
        _need_full_small = bool(self.publish_overlays or self.debug_save
                                or self._save_q is not None or self.keysave_full)
        seed_records, small_masks_full = bsp._extract_original_grid_seed_records(
            small_idx=small_idx, all_masks=all_masks, H=H_orig, W=W_orig,
            ui=ui, vi=vi, rr=rr, hh=hh, pc_xyz=pc_xyz, args=a, jump_dil_full=jump_dil_full,
            big_idx=big_idx, big_masks_full=big_masks_full,
            skip_full_masks=not _need_full_small)
        _pt("seed_extract")
        if self.publish_overlays and not getattr(self, "viz_full", False):   # viz_full: in the mapping burst
            self._publish_small_mask_overlay(cloud_msg.header, bgr, small_idx, small_masks_full)
        _pt("small_overlay")
        seed_records, _owned, _s2b = bsp._assign_big_owners_to_seeds(
            seed_records=seed_records, small_idx=small_idx, big_idx=big_idx,
            all_masks=all_masks, all_boxes=all_boxes, bbox_splat_tol=float(a.bbox_splat_tol),
            overlap_frac=float(getattr(a, "own_overlap_frac", 0.75)), big_masks_full=big_masks_full)
        # Ownership may PROMOTE orphan small masks (no big owner) into big_idx: add them to
        # big_masks_full and rebuild the corner rays so they get a plane like any big mask.
        added = False
        for bi in [int(x) for x in big_idx]:
            if bi not in big_masks_full and bi < len(all_masks):
                m = cv2.resize(all_masks[bi].astype(np.uint8), (W_orig, H_orig),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                cleaned = bsp._largest_region_break_links(m, _link_px).astype(bool)   # link-break, never discard
                if int((m != cleaned).sum()) > 0:
                    irr_fix.append((bi, m, cleaned, False))
                big_masks_full[bi] = cleaned
                added = True
        big_idx = [bi for bi in big_idx if int(bi) in big_masks_full]   # drop discarded / unbuilt masks
        if added:
            mask_ray_records = bsp._build_mask_corner_ray_records(
                small_idx=big_idx, small_masks_full=big_masks_full, W=W_orig, H=H_orig, args=a)
        _pt("assign")

        # DOORWAY lock update + skip -- on PILLAR cells only (a doorway locks just its structural columns,
        # not every seed). A declared mask's valid vertical-pillar cells LOCK; a glass mask's pillar landing
        # on a locked cell advances the cancel streak (2 glass pillars unlock). A mask whose pillar sits on a
        # locked cell is skipped.
        if self.doorway_gate and seed_records:
            def _owns(r):
                return [int(o) for o in (r.get("owner_big_idxs") or [int(r.get("owner_big_idx", -1))])]
            _by = {}
            for r in seed_records:
                _p3 = np.asarray(r["pt"], np.float32)
                for o in _owns(r):
                    _by.setdefault(o, []).append(_p3)
            _pil = {bi: bsp._mask_valid_pillar_reps(np.stack(v), a) for bi, v in _by.items() if v}
            _pose = self._frame_pose
            _dp = [_pil[bi] for bi in door_declared if len(_pil.get(bi, ()))]
            _gp = [_pil[bi] for bi in _pil if bi not in door_declared and len(_pil[bi])]
            _dp_arr = np.concatenate(_dp) if _dp else np.empty((0, 3), np.float32)
            _gp_arr = np.concatenate(_gp) if _gp else np.empty((0, 3), np.float32)
            self._plane_tracker.update_doorways(_dp_arr, _gp_arr, _pose)
            if self.enable_final_map:                 # feed the overview map the same doorway evidence
                self._final_tracker.update_doorways(_dp_arr, _gp_arr, _pose)
            _skip = set(door_declared)
            for bi in [int(x) for x in big_idx]:
                if bi in _skip:
                    continue
                pr = _pil.get(bi)
                if pr is not None and len(pr) and self._plane_tracker.doorway_locked_frac(pr, _pose) > 0.0:
                    _skip.add(bi)
            if _skip:
                big_idx = [bi for bi in big_idx if int(bi) not in _skip]
                self.get_logger().info(f"doorway-skip masks {sorted(_skip)}  locked_cells={len(self._plane_tracker.doorway_world)}")
            door_skipped = _skip - set(door_declared)      # lock-skipped (NOT declared) -> keysave audit
        else:
            door_skipped = set()
        _pt("doorway_lock")

        # ACCUMULATED-seed coverage: add THIS frame's seeds to the world seed-count map, then tag each
        # seed with its accumulated cell count so the clench weights coverage by persistence. Runs before
        # clench so the current frame is included. OFF -> uniform coverage (unchanged).
        accum_note = ""
        if self.use_accum_seed_cover and seed_records:
            _pose = self._frame_pose
            if not self.enable_global_map:            # global-map path accumulates AFTER clench -> avoid double
                self._plane_tracker.accumulate_seeds(seed_records, _pose)
            _pts = np.asarray([r["pt"] for r in seed_records], np.float32).reshape(-1, 3)
            _sw = self._plane_tracker.accum_counts_for(_pts, _pose)
            for _i, _r in enumerate(seed_records):
                _r["accum_w"] = float(_sw[_i]) if _sw[_i] > 0 else 1.0
            if _sw.size:
                accum_note = (f"accum_w mean={float(_sw.mean()):.1f} max={float(_sw.max()):.0f} "
                              f"cells={len(self._plane_tracker.seed_world)}")
        _pt("accum_seed")

        # Seed points (big-mask owner color), camera frame -> viewer_zup for _to_map. Vectorized:
        # ONE _camera_to_viewer_zup on the whole (N,3) array instead of one call per seed.
        if seed_records:
            seeds_cam = np.asarray([r["pt"] for r in seed_records], dtype=np.float32).reshape(-1, 3)
            seeds_rgb = np.asarray([r["color"] for r in seed_records], dtype=np.uint8).reshape(-1, 3)
            seeds_xyz = bsp._camera_to_viewer_zup(seeds_cam).astype(np.float32)
        else:
            seeds_xyz, seeds_rgb = _e3, _e3u
        if not getattr(self, "viz_full", False):  # viz_full: seeds + rays go out in the mapping burst
            self._publish_seeds(cloud_msg.header, seeds_xyz, seeds_rgb)
        # Feed the top-down map node with THIS frame's world-frame seed + depth-jump points (non-
        # stacked). Only when something is subscribed, so it costs nothing otherwise.
        if self.pub_map_seeds.get_subscription_count() > 0 or self.pub_map_jump.get_subscription_count() > 0:
            s_map, s_fid = self._to_map(cloud_msg.header, seeds_xyz)
            self._publish_map_cloud(self.pub_map_seeds, cloud_msg.header, s_map, s_fid)
            jm = jump_dil_full[vi, ui] if (getattr(vi, "size", 0) and getattr(ui, "size", 0)) else np.zeros(0, bool)
            jump_view = bsp._camera_to_viewer_zup(pc_xyz[jm]) if bool(np.any(jm)) else _e3
            j_map, j_fid = self._to_map(cloud_msg.header, jump_view)
            self._publish_map_cloud(self.pub_map_jump, cloud_msg.header, j_map, j_fid)
        _pt("seed_pub")

        # Clench one vertical rectangle per big mask (2 extreme pillars -> rectangle,
        # validated by in-mask + no-occlusion against the scene depth). Run once.
        # OCCLUSION DISABLED in the live path: passing scene_depth=None makes the clench's _validate
        # False, so NO occlusion is ever computed or logged (independent of the occ_check flag). The
        # scene depth is still built + stashed for the on-demand SPACE-key snapshot, which is the
        # "revisit occlusion later" path. To re-enable occ in the live fit, pass scene_depth=scene_depth.
        # Gated OFF for now: occlusion is disabled, so scene_depth is only useful for the debug saver's
        # occlusion visuals. Build it ONLY when a saver is active; otherwise skip (~54ms/frame back).
        scene_depth = (bsp._build_scene_depth_map(ui, vi, rr, W_orig, H_orig)
                       if (self.debug_save or self._save_q is not None) else None)
        _pt("scene_depth")
        # FLOOR-touch gate: 2nd SAM3 "floor" prompt -> this frame's floor points -> accumulate in the world
        # map -> query the accumulated floor near the camera (cam frame). That is what the gate rejects
        # floor-lying planes against. None when the floor gate is off (gate then never rejects).
        floor_gate_xyz = None
        self._sam_floor_cam = None                     # this frame's floor points (cam frame)
        _floor_full = None                             # this frame's 2D floor mask (for the edge-only seed-clear)
        _floor_xyz = None
        _ft = {}; _ft0 = time.perf_counter()
        def _fmark(name):
            _ft[name] = time.perf_counter() - _ft0
        if self.use_terrain_floor and self._frame_pose is not None:
            # FLOOR evidence from the autonomy stack's REGISTERED /terrain_map instead of the SAM3
            # floor decode (saves the 2nd decode, ~0.19s/frame). Terrain intensity = height above
            # ground; points at <= terrain_floor_thresh are ground. Terrain is MAP-frame ->
            # transform to cam so the existing accumulate/query path is unchanged.
            with self._lock:
                _terr = self._latest_terrain
            _floor_xyz = np.empty((0, 3), np.float32)
            if _terr is not None:
                _txyz, _tint = _terr
                _fw = _txyz[_tint <= self.terrain_floor_thresh]
                if len(_fw) > 30000:                    # cap: projection+dilate cost, floor is dense anyway
                    _fw = _fw[::int(np.ceil(len(_fw) / 30000.0))]
                # DISCOUNT floor AT obstacle positions: terrain marks ground right under the glass
                # line AND through-glass outdoor ground hugging it (LiDAR penetrates) -- floor
                # evidence there green-trims the very plane that stands on it. Clear a lateral
                # band around every tracked plane segment. (SAM floor never had this problem:
                # its semantic mask stops at the glass.)
                _clear = float(self.terrain_floor_plane_clear_m)
                if _clear > 0.0 and len(_fw) and self._plane_tracker is not None:
                    _P2 = _fw[:, :2]
                    _keep = np.ones(len(_fw), bool)
                    for _tp in self._plane_tracker.planes:
                        _a = np.asarray(_tp.p0[:2], np.float64); _b = np.asarray(_tp.p1[:2], np.float64)
                        _ab = _b - _a; _L2 = float(_ab @ _ab)
                        if _L2 < 1e-9:
                            continue
                        _t = np.clip(((_P2 - _a) @ _ab) / _L2, 0.0, 1.0)
                        _d2 = ((_P2 - (_a + _t[:, None] * _ab)) ** 2).sum(axis=1)
                        _keep &= _d2 > _clear * _clear
                    _fw = _fw[_keep]
                _fmark("clear_band")
                if len(_fw):
                    _floor_xyz = bsp._world_to_cam(np.asarray(_fw, np.float64),
                                                   self._frame_pose).astype(np.float32)
            _fmark("to_cam")
            # 2D floor mask (edge-only seed-clear): project the cam-frame floor points into the pano
            if len(_floor_xyz):
                _fu, _fv, _fr = bsp._cam_pts_to_pano_uv(_floor_xyz.astype(np.float64), W_orig, H_orig,
                                                        pixel_offset=float(a.pano_pixel_offset))
                _fu = np.round(_fu).astype(int) % W_orig
                _fv = np.clip(np.round(_fv).astype(int), 0, H_orig - 1)
                _floor_full = np.zeros((H_orig, W_orig), dtype=bool)
                _floor_full[_fv, _fu] = True
                _floor_full = cv2.dilate(_floor_full.astype(np.uint8),
                                         np.ones((9, 9), np.uint8)).astype(bool)
                _fmark("mask_dilate")
        elif self._floor_feats is not None:
            self._sam3_running = True
            try:
                # Reuse the already-encoded image state -> only the cheap grounding head runs (no re-encode).
                _fmasks, _, _, _, _ = bsp.sam3_decode_prompts(
                    self.sam_model, _sam3_state, self._floor_feats, self._floor_prompt, self.device, a)
            finally:
                self._sam3_running = False
            _floor_full = np.zeros((H_orig, W_orig), dtype=bool)
            for _fm in (_fmasks or []):
                _floor_full |= cv2.resize(_fm.astype(np.uint8), (W_orig, H_orig),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
            _fe = int(getattr(a, "floor_mask_erode_px", 0))
            if _fe > 0 and _floor_full.any():   # shave mask boundary: glass-base bleed is not floor
                _floor_full = cv2.erode(_floor_full.astype(np.uint8),
                                        np.ones((2 * _fe + 1, 2 * _fe + 1), np.uint8)).astype(bool)
            _floor_xyz = np.empty((0, 3), np.float32)
            if _floor_full.any() and len(pc_xyz):
                _fsel = _floor_full[np.clip(vi, 0, H_orig - 1), np.clip(ui, 0, W_orig - 1)]
                _floor_xyz = pc_xyz[_fsel].astype(np.float32)
        if _floor_xyz is not None:
            _fpose = self._frame_pose
            self._plane_tracker.accumulate_floor(_floor_xyz, _fpose)
            _fmark("accum_cur")
            if self.enable_final_map:                 # overview map keeps its own floor accumulation
                self._final_tracker.accumulate_floor(_floor_xyz, _fpose)
            _fmark("accum_final")
            floor_gate_xyz = self._plane_tracker.floor_points_near_camera_cam(
                _fpose, float(getattr(a, "floor_accum_query_radius_m", 8.0)))
            _fmark("query_near")
        if _ft:
            _fp = "  ".join(f"{k}={int(1000*(v - p0))}ms" for (k, v), p0 in
                            zip(_ft.items(), [0.0] + list(_ft.values())[:-1]))
            self.get_logger().info(f"    floor[{_fp}]", throttle_duration_sec=10.0)
        if _floor_xyz is not None:
            self._sam_floor_cam = _floor_xyz               # THIS frame's floor points (cam frame)
        _pt("floor")
        _t_clench = time.perf_counter()
        clench_by_mask, _, horiz_by_mask, clench_timings = bsp._compute_all_clench(
            big_idx, mask_ray_records, seed_records, a,
            big_masks_full=big_masks_full, scene_depth=None, W=W_orig, H=H_orig, pc_xyz=pc_xyz,
            floor_xyz=floor_gate_xyz, floor_mask=_floor_full)
        self._clench_wall = time.perf_counter() - _t_clench
        self._clench_timings = clench_timings
        _pt("clench")
        if self._hz_q is not None:                     # per-frame top-down horizontal-pillar PNG (async)
            try:
                self._hz_q.put_nowait((int(self._save_counter), [int(x) for x in big_idx],
                                       horiz_by_mask, seed_records, mask_ray_records))
            except queue.Full:
                pass                                   # disk can't keep up -> skip this frame's PNG
        su, sv = int(a.plane_patch_samples_u), int(a.plane_patch_samples_v)

        xyz_parts, rgb_parts = [], []
        placed = 0
        for bi in [int(x) for x in big_idx]:
            res, _info = clench_by_mask.get(bi, (None, {}))
            if res is None:
                continue
            corners = np.asarray(res[0], dtype=np.float32)  # TL,TR,BR,BL (camera frame)
            patch = bsp._make_plane_patch_from_quad_corners(corners, su, sv).astype(np.float32)
            rec = mask_ray_records.get(bi)
            col = np.asarray(rec["color_rgb"], np.uint8) if rec is not None else np.array([0, 0, 255], np.uint8)
            xyz_parts.append(bsp._camera_to_viewer_zup(patch))
            rgb_parts.append(np.tile(col.reshape(1, 3), (patch.shape[0], 1)))
            placed += 1

        out_xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32) if xyz_parts else _e3
        out_rgb = np.concatenate(rgb_parts, axis=0).astype(np.uint8) if rgb_parts else _e3u
        self._publish_planes(cloud_msg.header, out_xyz, out_rgb)
        _pt("publish_planes")

        # Package this inference's data (cheap: references, one small subsample) so it can be
        # dumped later on a SPACE keypress, and (optionally) streamed to the always-on saver.
        fid = self._save_counter
        self._save_counter += 1
        scene_sub = pc_xyz
        if scene_sub.shape[0] > 60000:
            # stride subsample (was rng.choice without replacement: ~25ms on a 500k cloud, every
            # frame, for a debug buffer that is usually never dumped)
            _st = int(np.ceil(scene_sub.shape[0] / 60000.0))
            scene_sub = scene_sub[::_st]
        item = {
            "fid": fid, "bgr": bgr, "depth_da2": depth_da2, "jump_dil_full": jump_dil_full,
            "ui": ui, "vi": vi, "rr": rr, "pc_xyz": pc_xyz, "scene_sub": scene_sub.astype(np.float32),
            "small_idx": [int(x) for x in small_idx], "big_idx": [int(x) for x in big_idx],
            "big_masks_full": big_masks_full, "small_masks_full": small_masks_full,
            "mask_ray_records": mask_ray_records, "seed_records": seed_records,
            "clench_by_mask": clench_by_mask, "scene_depth": scene_depth,
            "W": int(W_orig), "H": int(H_orig), "irregular_fix": irr_fix,
            "horiz_by_mask": horiz_by_mask, "floor_gate_xyz": floor_gate_xyz,
            "door_declared": set(int(x) for x in door_declared),
            "door_skipped": set(int(x) for x in door_skipped),
            "align_debug": align_debug,
            "pose": self._frame_pose,   # (R_body, T) -- world pose for the debug PLY writers
        }
        # GT-ALIGN: accumulate a small live obstacle cloud (world frame) so the GT overlay can be
        # ICP-registered onto THIS session's map (recorded and live SLAM frames rarely coincide).
        if getattr(self, "_gt_cloud_msg", None) is not None and not getattr(self, "_gt_aligned", True):
            try:
                _pw = bsp._cam_to_world(xyz_cam_all.astype(np.float64), self._frame_pose)
                _gz = float(np.percentile(_pw[:, 2], 3))
                _pw = _pw[_pw[:, 2] > _gz + 0.3]              # obstacles only (drop floor)
                if len(_pw) > 1500:
                    _pw = _pw[np.random.default_rng(0).choice(len(_pw), 1500, replace=False)]
                self._live_world_buf.append(_pw.astype(np.float32))
            except Exception:
                pass
        self._recent.append(item)                     # rolling last-5 buffer
        _pt("save")                                   # close out packaging so global_map is isolated

        # --- cross-frame ACCUMULATED map (OFF unless enable_global_map): feed this frame's placed planes
        # into the tracker (merge/compete in the map frame), snapshot the geometry, and publish the
        # accumulated wall cloud. Skipped entirely when disabled so the live inference isn't slowed.
        tracks = []; seed_centroids = []; pose_tuple = None; floor_world = None
        _want_panel = self.publish_global_map_image or self.publish_topdown   # the ONLY consumers of the image data
        if self.role == "perception":
            # PIPELINE cut: ship this frame's placed planes + evidence to the mapping process
            # (which owns the tracker) and skip the local tracker update entirely.
            self._publish_geom(cloud_msg.header, clench_by_mask, seed_records, mask_ray_records,
                               floor_gate_xyz, pc_xyz, H_orig, W_orig, big_masks_full,
                               _viz_bgr=(bgr if getattr(self, "viz_full", False) else None),
                               _viz_extra=(self._viz_pack_extra(big_idx, big_masks_full, small_idx,
                                                                small_masks_full, horiz_by_mask)
                                           if getattr(self, "viz_full", False) else None))
            _pt("global_map")
        elif self.enable_global_map:
            pose_tuple = self._frame_pose
            # FLOOR evidence = THIS frame's SAM3 'floor' mask points -> world (capture-time pose).
            floor_world = None
            if pose_tuple is not None and self._sam_floor_cam is not None and len(self._sam_floor_cam):
                floor_world = bsp._cam_to_world(
                    np.asarray(self._sam_floor_cam, np.float64), pose_tuple)
            # OBSTACLE evidence (orange) = REGISTERED /terrain_map obstacle points (already MAP frame,
            # world-consistent -- no pose transform, does NOT move with the camera).
            obst_world = None
            if self.use_terrain_obstacle and pose_tuple is not None:
                with self._lock:
                    terr = self._latest_terrain
                if terr is not None:
                    _txyz, _tint = terr
                    obst_world = _txyz[_tint >= self.terrain_obstacle_thresh]
            # REPROJECT-SPILL gate needs THIS frame's glass mask (union of big masks). Occlusion uses the
            # scene cloud (pc_xyz) directly inside the tracker (dense coarse angular depth), so no full-res
            # depth map is built here. Built only when the gate (or its panel) is on, to avoid the cost.
            _glass_mask = None
            if self.reproject_evict or self.depth_sweep_evict or self.save_reproject_panel or self.mask_clamp:
                _glass_mask = np.zeros((H_orig, W_orig), bool)
                for _mk in (big_masks_full or {}).values():
                    _glass_mask |= np.asarray(_mk, bool)
            # snapshot geometry+color so an evicted pid can be described (bearing/range/color)
            _pre_geo = {int(tp.pid): (np.asarray(tp.p0, np.float64).copy(),
                                      np.asarray(tp.p1, np.float64).copy(),
                                      tuple(int(c) for c in tp.color),
                                      float(tp.v0) if tp.v0 is not None else 0.0,
                                      float(tp.v1) if tp.v1 is not None else 2.0)
                        for tp in self._plane_tracker.planes}
            # PARALLEL overview map: the FINAL tracker is an independent object reading the SAME
            # read-only inputs (verified: no in-place writes to seeds/cloud/masks/args; separate
            # instance state), so its update runs CONCURRENTLY with the current-map tracker here
            # and is joined+published below at THIS frame -> identical outputs/order, no planner
            # -feed lag. GIL caps the win to the cKDTree/large-numpy portions that release it;
            # the t_cur/t_fin split on the inference line shows the realized overlap.
            _final_box = {}
            def _run_final_tracker():
                if not self.enable_final_map:
                    return
                _tf0 = time.perf_counter()
                try:
                    self._final_tracker.update_from_frame(
                        clench_by_mask, seed_records, pose_tuple, mask_ray_records=mask_ray_records,
                        floor_cam=floor_gate_xyz, scene_cam=pc_xyz, floor_world=floor_world,
                        obst_world=obst_world, glass_mask=_glass_mask,
                        img_wh=(W_orig, H_orig), pano_offset=float(a.pano_pixel_offset))
                    _final_box["tracks"] = [(tp.apex.copy(), tp.p0.copy(), tp.p1.copy(),
                                (int(tp.color[0]), int(tp.color[1]), int(tp.color[2])),
                                bool(tp.fixed), int(tp.mask)) for tp in self._final_tracker.planes]
                    _final_box["vr"] = [(tp.v0, tp.v1) for tp in self._final_tracker.planes]
                except Exception as e:
                    _final_box["err"] = e
                _final_box["dt"] = time.perf_counter() - _tf0
            _final_thread = threading.Thread(target=_run_final_tracker, name="final_tracker", daemon=True)
            _final_thread.start()
            _tc0 = time.perf_counter()
            try:
                try:
                    self._plane_tracker._dbg_bgr = bgr
                except NameError:
                    pass
                tracker_logs = self._plane_tracker.update_from_frame(
                    clench_by_mask, seed_records, pose_tuple, mask_ray_records=mask_ray_records,
                    floor_cam=floor_gate_xyz, scene_cam=pc_xyz, floor_world=floor_world,
                    obst_world=obst_world, glass_mask=_glass_mask,
                    img_wh=(W_orig, H_orig), pano_offset=float(a.pano_pixel_offset))
                item["tracker_logs"] = list(tracker_logs or [])   # per-frame compete/prune lines -> txt
                # make EVICTIONS obvious: every PRUNE / SPILL event goes loudly to the console,
                # tagged with WHERE (bearing vs robot facing + range) and WHICH (track color)
                import re as _re
                def _whereis(_ln):
                    _m = _re.search(r"PRUNE #(\d+)|#(\d+)", _ln)
                    _pid = int(_m.group(1) or _m.group(2)) if _m else -1
                    if _pid not in _pre_geo or pose_tuple is None:
                        return ""
                    _q0, _q1, _col = _pre_geo[_pid][:3]
                    _c = 0.5 * (_q0 + _q1)
                    _Rb, _T = pose_tuple
                    _v = _c[:2] - np.asarray(_T[:2], np.float64)
                    _rng = float(np.linalg.norm(_v))
                    _fwd = (_Rb @ np.array([1.0, 0.0, 0.0]))[:2]
                    _ang = float(np.degrees(np.arctan2(_fwd[0]*_v[1]-_fwd[1]*_v[0], _fwd @ _v)))
                    _side = "left" if _ang > 0 else "right"
                    _name = bsp._rgb_name(_col[2], _col[1], _col[0])   # tracker color is BGR
                    return f" [{_name}, {abs(_ang):.0f}deg {_side}, {_rng:.1f}m]"
                for _ln in (tracker_logs or []):
                    if "PRUNE" in _ln:
                        self.get_logger().warn(f"[EVICT]{_whereis(_ln)} {_ln.strip()}")
                        # (BLACK-GHOST RViz visual removed by request -- evicted planes just vanish;
                        #  the [EVICT] console line above remains the eviction signal)
                    # ([SPILL] console echo removed by request -- spill activity still appears in
                    #  plane_tracker_log.txt and the reproject panels; evictions still log [EVICT])
            except Exception as e:
                self.get_logger().warn(f"[global-map] update failed: {e}", throttle_duration_sec=5.0)
            # canonical-run recording + trajectory-anchored GT (no-op unless configured)
            if self._canon_dir or self._canon_traj is not None:
                try:
                    self._canonical_tick(pose_tuple, pc_xyz, item.get("tracker_logs"), _pre_geo)
                except Exception as e:
                    self.get_logger().warn(f"[canon] tick failed: {e}", throttle_duration_sec=5.0)
            tracks = [(tp.apex.copy(), tp.p0.copy(), tp.p1.copy(),
                       (int(tp.color[0]), int(tp.color[1]), int(tp.color[2])),
                       bool(tp.fixed), int(tp.mask)) for tp in self._plane_tracker.planes]
            track_vr = [(tp.v0, tp.v1) for tp in self._plane_tracker.planes]
            # eviction mirror: current-tracker pids that vanished this frame -> suppress their
            # segment in the FINAL-sourced obstacle feeds (see evict_suppress_s)
            _now_m = time.monotonic()
            _cur_pids = {int(tp.pid): (np.asarray(tp.p0[:2], np.float64),
                                       np.asarray(tp.p1[:2], np.float64))
                         for tp in self._plane_tracker.planes if hasattr(tp, "pid")}
            for _pid, (_sa, _sb) in self._prev_cur_pids.items():
                if _pid not in _cur_pids:
                    self._evict_suppress.append((_sa, _sb, _now_m))
            self._prev_cur_pids = _cur_pids
            self._publish_global_planes(cloud_msg.header, tracks, pose_tuple, track_vr)
            if _want_panel:                           # world seed centroids are ONLY for the top-down image panels
                seed_centroids = self._plane_tracker.world_seed_centroids()
            _cur_trk_dt = time.perf_counter() - _tc0
            # FINAL / OVERVIEW map: join the concurrent update and publish at THIS frame (same
            # detections, its own accumulation, looser 5 m spill radius). Publishing here (main
            # thread, after publish_global) preserves the original topic order & the /added_obstacles
            # planner feed timing -- only the UPDATE compute overlapped the current-map tracker.
            if self.enable_final_map:
                _final_thread.join()
                if "err" in _final_box:
                    self.get_logger().warn(f"[final-map] update failed: {_final_box['err']}",
                                           throttle_duration_sec=5.0)
                elif "tracks" in _final_box:
                    self._publish_final_planes(cloud_msg.header, _final_box["tracks"],
                                               pose_tuple, _final_box["vr"])
            self._trk_split = (_cur_trk_dt, float(_final_box.get("dt", 0.0)))
            _pt("global_map")                         # accumulation update + plane cloud publish
            if self.save_local_tug and pose_tuple is not None:
                try:
                    self._save_local_tug_view(fid, pose_tuple)
                except Exception as e:
                    self.get_logger().warn(f"[local-tug] save failed: {e}", throttle_duration_sec=5.0)

        # Snapshot AFTER the tracker update so both the SPACE debug dump and the auto global-map
        # reflect THIS frame's accumulated tracks / tug map.
        item["tracks"] = tracks
        item["pose"] = pose_tuple
        if self.enable_global_map:                    # snapshot for the GLOBAL map PNG (auto-map thread)
            _tk = self._plane_tracker
            item["map_snap"] = {
                "cell_m": _tk.map_cell_m, "obst": set(_tk.obst_cells),
                "tug_floor": dict(_tk.tug_floor), "tug_seed": dict(_tk.tug_seed),
                "seed_weight": _tk.seed_floor_weight,
                "gi": _tk.gi, "gj": _tk.gj, "path": list(_tk.path),
            }
        if self._sam_floor_cam is not None and len(self._sam_floor_cam):
            # this frame's SAM3 floor points (already CAMERA frame) -> the recorded floor_planes PNG
            # shows the SAME floor evidence the tracker tug judged with
            item["terrain_floor_cam"] = np.asarray(self._sam_floor_cam, np.float32)
        # REPROJECT panel: LEFT top-down map (evicted red) | RIGHT rgb + reprojected silhouettes.
        if self.save_reproject_panel and self.enable_global_map and item.get("map_snap") is not None:
            try:
                self._save_reproject_panel(bgr, dict(self._plane_tracker._spill_debug),
                                           item["map_snap"], tracks, fid, glass_mask=_glass_mask)
            except Exception as e:
                self.get_logger().warn(f"[reproject] panel failed: {e}", throttle_duration_sec=5.0)
        # AUTO global-map PNG -- THROTTLED. The render iterates thousands of cells and holds the GIL,
        # which was making the SPACE key sluggish; cap it to one save per auto_map_min_period seconds.
        if self.save_auto_map and self.enable_global_map:
            _now = time.time()
            if _now - self._last_auto_map_t >= self.auto_map_min_period:
                self._last_auto_map_t = _now
                try:
                    self._auto_q.put_nowait(item)
                except queue.Full:
                    pass                              # writer behind -> skip this frame (never block)

        if _want_panel:                               # hand data to the async render thread only if an image is on
            with self._td_lock:
                self._td_data = {
                    "pc_xyz": pc_xyz, "seed_records": seed_records, "clench_by_mask": clench_by_mask,
                    "mask_ray_records": mask_ray_records, "pose": pose_tuple, "tracks": tracks,
                    "seed_centroids": seed_centroids,
                    # viz_full extras: the pano (silhouette overlay) + frame id for the tug header
                    "bgr": (bgr if getattr(self, "viz_full", False) else None),
                    "stamp": cloud_msg.header.stamp,   # SOURCE stamp -> every demo stream shares it
                    "small_masks_full": (small_masks_full if getattr(self, "viz_full", False) else None),
                    "fid": int(self._save_counter),
                    "big_idx": ([int(x) for x in big_idx] if getattr(self, "viz_full", False) else None),
                    "horiz_by_mask": (horiz_by_mask if getattr(self, "viz_full", False) else None),
                }
                self._td_seq += 1
        if self._save_q is not None:                  # optional always-on saver (debug_save:=true)
            try:
                self._save_q.put_nowait(item)
            except queue.Full:
                self.get_logger().warn("[debug-save] saver busy; dropped this frame's debug visuals",
                                       throttle_duration_sec=5.0)
        _canon_save = bool(self._record_run and getattr(self, "_canon_placed_now", False)
                           and self._canon_dir and self._input_q_canon is not None)
        if self._input_q is not None or _canon_save:  # save the RAW input for offline batch replay
            # CAPTURE-TIME pose (the pose inference itself used), NOT the latest pose at save
            # time -- the latest-pose version lagged by processing latency, shifting every
            # saved frame by robot-speed x latency (the "registered cloud drifts each pose"
            # bug seen while annotating).
            pose7 = None
            if self._frame_pose is not None:
                _Rb, _Tb = self._frame_pose
                _q = _R_to_quat(_Rb)
                pose7 = (float(_Tb[0]), float(_Tb[1]), float(_Tb[2]),
                         float(_q[0]), float(_q[1]), float(_q[2]), float(_q[3]))
            # Per-pixel RANGE map (metres, camera frame) for the pano: the cloud projected to (u,v), nearest
            # range per pixel. This IS the depth image that corresponds 1:1 to rgb_{fid}.png.
            _depth_m = (scene_depth if scene_depth is not None
                        else bsp._build_scene_depth_map(ui, vi, rr, W_orig, H_orig))
            try:
                if self._input_q is not None:
                    self._input_q.put_nowait((fid, xyz_raw, pano_bgr, pose7, _depth_m,
                                              self.input_save_dir))
                if _canon_save:                       # frame where a plane was PLACED -> canonical
                    self._input_q_canon.put_nowait((fid, xyz_raw, pano_bgr, pose7, _depth_m,
                                                    os.path.join(self._canon_dir, "placed_input")))
            except queue.Full:
                self.get_logger().warn("[input-save] saver busy; dropped this frame's input",
                                       throttle_duration_sec=5.0)

        _pt("save")
        # Optionally release the caching-allocator pool so nvidia-smi drops to the true live size
        # (~1 GB) instead of holding the forward's ~3.4 GB reserve. Costs a re-cudaMalloc of the next
        # forward's activations (~100-300 ms), so it's PERIODIC: empty_cache_every=0 keeps the pool
        # warm (fastest); N releases every N frames. Combined with expandable_segments (set above).
        if self.empty_cache_every > 0 and (self._save_counter % self.empty_cache_every) == 0 \
                and self.device.type == "cuda":
            torch.cuda.empty_cache()
        _pt("empty_cache")

        ct = getattr(self, "_clench_timings", {}) or {}
        # print ALL clench sub-timings (sorted, biggest first) so nothing is hidden -- vertical /
        # borrow_search / pillars_in_clench were previously not shown.
        cbrk = "  ".join(f"{k}={v * 1e3:.0f}ms" for k, v in
                         sorted(ct.items(), key=lambda kv: -kv[1]) if v * 1e3 >= 0.5)
        pbrk = "  ".join(f"{k}={PT.get(k, 0.0) * 1e3:.0f}ms" for k in
                         ("rgb", "cloud_parse", "pinhole_align", "project", "da2", "sam3", "splat",
                          "big_overlay", "rays", "seed_extract", "small_overlay", "assign", "accum_seed",
                          "seed_pub", "scene_depth", "floor", "clench", "publish_planes", "save",
                          "global_map", "empty_cache"))
        _sc, _sf = getattr(self, "_trk_split", (0.0, 0.0))
        _split = f"  [global_map split: t_cur={_sc * 1e3:.0f}ms || t_fin={_sf * 1e3:.0f}ms]" if _sf > 0 else ""
        _es = getattr(self._plane_tracker, "_evict_split", None)
        if _es:
            _split += (f"  [cur-tracker: merge={_es[0]*1e3:.0f}ms  ground_evict={_es[1]*1e3:.0f}ms  "
                       f"spill_evict={_es[2]*1e3:.0f}ms]")
        # SYNC line, every frame: how far the paired image/pose stamps sit from the cloud stamp,
        # and how much stamp time passed since the previous PROCESSED frame. The existing [sync]
        # and [pose] warnings only fire above 0.15 s, so a steady sub-threshold offset -- which
        # still shows up as a camera<->LiDAR shift -- was invisible. dt_proc is the real driver:
        # inference serializes _process, so the robot moves that far between captured frames.
        _sync_img = float(getattr(self, "_img_match_gap", 0.0))
        _sync_pose = float(getattr(self, "_pose_match_gap", 0.0))
        _cld_now = float(cloud_msg.header.stamp.sec) + float(cloud_msg.header.stamp.nanosec) * 1e-9
        _dt_proc = _cld_now - float(getattr(self, "_prev_proc_stamp", 0.0) or _cld_now)
        self._prev_proc_stamp = _cld_now
        self.get_logger().info(
            f"inference {time.perf_counter() - t_start:.2f}s  planes={placed}"
            f"{('  ' + accum_note) if accum_note else ''}\n"
            f"    SYNC: img_gap={_sync_img * 1e3:.0f}ms  pose_gap={_sync_pose * 1e3:.0f}ms  "
            f"pose_src={getattr(self, '_pose_src', '?')}  "
            f"dt_proc={_dt_proc:.2f}s\n"
            f"    PHASES: {pbrk}{_split}\n"
            f"    clench[{cbrk}]")

    def _pose_tuple(self):
        """Latest /habitat/state_estimation as the batch pose (R_body, T) -- the exact form
        bsp._cam_to_world / _world_to_cam expect. None until a pose has arrived (accumulate in the
        camera frame until then, mirroring the batch NOPOSE path)."""
        with self._lock:
            pose = self._latest_pose
        if pose is None:
            return None
        p = pose.pose.position; q = pose.pose.orientation
        return (bsp._quat_to_R(q.x, q.y, q.z, q.w),
                np.array([p.x, p.y, p.z], dtype=np.float64))

    def _to_map(self, src_header: Header, xyz: np.ndarray):
        """Transform camera-frame points into the map frame via this frame's CAPTURE-TIME pose
        (locked at intake, not the latest). Returns (xyz_out, frame_id)."""
        pose = getattr(self, "_frame_pose", None)
        # Emit directly in map: P_map = R_body (R_static P_F + t_static) + T_lidar.
        if self.publish_frame == "map" and pose is not None and xyz.shape[0] > 0:
            R_body, T_lidar = pose[0], pose[1]
            P_sensor = xyz.astype(np.float64) @ _R_STATIC.T + _T_STATIC
            P_map = P_sensor @ R_body.T + T_lidar
            return P_map.astype(np.float32), "map"
        if self.publish_frame == "map" and pose is None:
            self.get_logger().warn(
                "publish_frame=map but no /habitat/state_estimation yet; "
                "emitting in camera_link until a pose arrives", once=True)
        return xyz, src_header.frame_id

    def _publish_stacked(self, publisher, hist, src_header, xyz, rgb):
        """Transform to map, roll the last N non-empty rounds together, publish.
        An empty frame does NOT evict prior points, so they persist (no flashing)
        until new points replace them."""
        xyz, frame_id = self._to_map(src_header, xyz)
        if xyz.shape[0] > 0:
            hist.append((xyz, rgb))
        if hist:
            xyz = np.concatenate([h[0] for h in hist], axis=0)
            rgb = np.concatenate([h[1] for h in hist], axis=0)
        out = Header()
        out.stamp = src_header.stamp
        out.frame_id = frame_id
        self._pub(publisher, _make_xyzrgb_cloud(out, xyz, rgb))

    def _publish_planes(self, src_header: Header, xyz: np.ndarray, rgb: np.ndarray):
        # When the global map is on, /glass_killer/planes is driven by the TRACKER in
        # _publish_global_planes (so it respects eviction). Skip the per-frame stacked placements here
        # -- otherwise an evicted plane would linger in the stack visual.
        if self.enable_global_map:
            return
        # FREEZE planes in the WORLD (map) frame at placement time: once transformed to map they no
        # longer move with the camera. Retain the last `stack_rounds` non-empty placements, so a
        # 'no detection' frame keeps the previously placed planes instead of clearing/flashing.
        # In map mode we only retain WORLD-fixed points (frame_id=='map'); if no pose has arrived we
        # skip rather than glue planes to the moving camera.
        xyz, frame_id = self._to_map(src_header, xyz)
        world_ok = (self.publish_frame != "map") or (frame_id == "map")
        if xyz.shape[0] > 0 and world_ok:
            self._plane_hist.append((xyz, rgb, frame_id))
        if not self._plane_hist:
            return
        out_xyz = np.concatenate([h[0] for h in self._plane_hist], axis=0)
        out_rgb = np.concatenate([h[1] for h in self._plane_hist], axis=0)
        out = Header()
        out.stamp = src_header.stamp
        out.frame_id = self._plane_hist[-1][2]         # frame of the retained (world) points
        self.pub_planes.publish(_make_xyzrgb_cloud(out, out_xyz, _viz_recolor(out_rgb)))

    def _publish_seeds(self, src_header: Header, xyz: np.ndarray, rgb: np.ndarray):
        self._publish_stacked(self.pub_seeds, self._seed_hist, src_header, xyz, rgb)

    def _topdown_worker(self):
        """Own-thread render loop for the live top-down cloud-grid map. Reads only the LATEST
        inference's data (never blocks the plane algorithm) and publishes it as a ROS Image."""
        period = 1.0 / max(0.1, self.topdown_hz)
        last_seq = -1
        while rclpy.ok():
            time.sleep(period)
            with self._td_lock:
                data = self._td_data
                seq = self._td_seq
            if data is None or seq == last_seq:
                continue
            last_seq = seq
            if self.publish_topdown:                    # OFF by default (image render steals the GIL)
                try:
                    vis = bsp._render_topdown_cloud_grid(
                        data["pc_xyz"], data["seed_records"], self.args,
                        clench_by_mask=data["clench_by_mask"], mask_ray_records=data["mask_ray_records"])
                    self._publish_image(self.pub_topdown, vis)
                except Exception as e:
                    self.get_logger().warn(f"[topdown] render failed: {e}", throttle_duration_sec=5.0)
            if self.enable_global_map and self.publish_global_map_image:   # two-panel IMAGE (opt-in; pricey render)
                try:
                    panel = self._render_global_panel(data)
                    self._publish_image(self.pub_global_map, panel)
                except Exception as e:
                    self.get_logger().warn(f"[global-map] render failed: {e}", throttle_duration_sec=5.0)

    def _publish_pane_lines(self, data, stamp):
        """viz_full: VERTICAL lines for every SMALL mask owned by a big mask, placed on that big
        mask's ACCEPTED plane. The pane's left (TL,BL) and right (TR,BR) corner rays are intersected
        with the vertical plane through the accepted ground line -- the paper's metric-extent
        construction, applied per pane -- so each small mask yields its two mullion segments ON the
        plane, in the OWNING big mask's colour. Only big masks with an accepted plane contribute.
        Runs on the panel thread with THIS frame's pose, so inference is untouched."""
        a = self.args
        pose = data.get("pose"); bgr = data.get("bgr")
        cbm = data.get("clench_by_mask") or {}
        mrr = data.get("mask_ray_records") or {}
        smf = data.get("small_masks_full") or {}
        seed_records = data.get("seed_records") or []
        def _emit(cam_pts, cols, why):
            """publish (possibly EMPTY) so the topic is visibly alive, and say why it is empty."""
            if cam_pts is None or cam_pts.shape[0] == 0:
                hdr = Header(); hdr.frame_id = "map"
                if stamp is not None: hdr.stamp = stamp
                self._pub(self.pub_viz_pane_lines, _make_xyzrgb_cloud(hdr, np.empty((0, 3), np.float32),
                                                                   np.empty((0, 3), np.uint8)))
                self.get_logger().info(f"[viz_full] pane_lines: EMPTY -- {why}", throttle_duration_sec=5.0)
        if pose is None or bgr is None or not cbm:
            return _emit(None, None, f"pose={pose is not None} bgr={bgr is not None} clench_masks={len(cbm)}")
        H, W = bgr.shape[:2]
        po = float(getattr(a, "pano_pixel_offset", 1.0))
        min_seeds = int(getattr(a, "clench_h_min_pts", 5))
        # owned smalls per big, from seed ownership (a small that IS a big mask is skipped)
        owned = {}; cnt = {}
        for r in seed_records:
            si = int(r.get("mask_idx", -1))
            if si < 0 or si in cbm:
                continue
            for o in (r.get("owner_big_idxs") or [int(r.get("owner_big_idx", -1))]):
                o = int(o)
                if o in cbm:
                    owned.setdefault(o, set()).add(si); cnt[si] = cnt.get(si, 0) + 1
        want = sorted({si for ss in owned.values() for si in ss
                       if cnt.get(si, 0) >= min_seeds and si in smf})
        n_seed = len(seed_records); n_small_seen = len({int(r.get("mask_idx", -1)) for r in seed_records})
        if not want:
            return _emit(None, None, f"no owned small mask with >={min_seeds} seeds: seeds={n_seed} "
                                     f"small_ids_in_seeds={n_small_seen} owned_bigs={len(owned)} "
                                     f"smf_keys={len(smf)} clench_keys={sorted(cbm.keys())[:8]}")
        masks = {}
        for si in want[:12]:                              # bound the per-frame ray work: measured 39 ms @12,
                                                          # 78 ms @24 -- pure-Python RANSAC that holds the GIL
            m = np.asarray(smf[si], bool)
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                masks[si] = m
        if not masks:
            return _emit(None, None, f"owned smalls had empty masks: want={want[:8]}")
        srec = bsp._build_mask_corner_ray_records(small_idx=list(masks.keys()), small_masks_full=masks,
                                                  W=W, H=H, args=a)
        up = np.array([0.0, 1.0, 0.0])                    # camera frame: +y DOWN, so this is vertical
        xs, cs = [], []

        def _seg(p, q, rgb):
            n = max(2, int(np.linalg.norm(q - p) / 0.03))
            t = np.linspace(0.0, 1.0, n)[:, None]
            xs.append((p[None, :] * (1.0 - t) + q[None, :] * t).astype(np.float32))
            cs.append(np.tile(np.array([rgb], np.uint8), (n, 1)))

        for bi, smalls in owned.items():
            res = cbm[bi][0] if isinstance(cbm[bi], (tuple, list)) else None
            if res is None:
                continue                                  # no accepted plane for this big mask
            lc, rc = bsp._plane_ground_line_cam(res[0])
            lc = np.asarray(lc, np.float64); rc = np.asarray(rc, np.float64)
            d = rc - lc; d[1] = 0.0
            L = float(np.linalg.norm(d))
            if L < 0.05:
                continue
            n = np.cross(d, up); n /= (np.linalg.norm(n) + 1e-12)   # normal of the vertical plane
            rec_b = mrr.get(int(bi)); c = np.asarray(rec_b["color_rgb"], np.uint8).reshape(3) if rec_b else np.array([255, 255, 255], np.uint8)
            rgb = [int(c[0]), int(c[1]), int(c[2])]
            for si in sorted(smalls):
                rec = srec.get(int(si))
                if rec is None:
                    continue
                cuv = np.asarray(rec.get("corners_uv", []), np.float64).reshape(-1, 2)
                if cuv.shape[0] != 4:
                    continue
                rays = bsp._pano_uv_to_unit_rays(cuv, W, H, pixel_offset=po)   # TL, TR, BR, BL
                pts = []
                for r in rays:
                    den = float(r @ n)
                    if abs(den) < 1e-6:
                        pts.append(None); continue
                    t = float(lc @ n) / den
                    P = t * r
                    sp = float((P - lc) @ d) / (L * L)         # position along the ground line
                    pts.append(P if (0.1 < t < 60.0 and -0.15 <= sp <= 1.15) else None)
                for (i, j) in ((0, 3), (1, 2)):               # left edge TL->BL, right edge TR->BR
                    if pts[i] is not None and pts[j] is not None:
                        _seg(pts[i], pts[j], rgb)
        n_acc = sum(1 for bi in owned if isinstance(cbm.get(bi), (tuple, list)) and cbm[bi][0] is not None)
        if not xs:
            return _emit(None, None, f"no segments: want={len(want)} records={len(srec)} "
                                     f"accepted_planes={n_acc}/{len(owned)} (rays missed the plane or no accepted plane)")
        cam = np.concatenate(xs, 0); col = np.concatenate(cs, 0)
        self.get_logger().info(f"[viz_full] pane_lines: {len(xs)} segments from {len(srec)} panes on "
                               f"{n_acc} accepted plane(s)", throttle_duration_sec=5.0)
        vz = bsp._camera_to_viewer_zup(cam).astype(np.float64)
        R_body, T_lidar = pose[0], pose[1]                # same transform as _to_map, THIS frame's pose
        pm = ((vz @ _R_STATIC.T + _T_STATIC) @ R_body.T + T_lidar).astype(np.float32)
        hdr = Header(); hdr.frame_id = "map"
        if stamp is not None:
            hdr.stamp = stamp
        self._pub(self.pub_viz_pane_lines, _make_xyzrgb_cloud(hdr, pm, col))

    def _publish_viz_full(self, data):
        """DEMO (viz_full:=true): render the four --full-visual demo images and PUBLISH them as
        live topics instead of writing PNGs. Runs on the async panel thread, so a slow render
        never blocks inference. Each render is guarded separately: one failing must not take the
        others down."""
        a = self.args
        pose = data.get("pose")
        fid = int(data.get("fid") or 0)
        _vstamp = data.get("stamp")        # source frame stamp, shared by every stream
        # (tug map + spill verdict are published by the MAPPING role -- see _publish_viz_tracker)
        # 0c) per-PANE vertical lines on each accepted plane (small masks owned by the big mask)
        try:
            self._publish_pane_lines(data, _vstamp)
        except Exception as e:
            self.get_logger().warn(f"[viz_full] pane lines failed: {e}", throttle_duration_sec=5.0)
        # (rgb pano + lidar pano are published INLINE in _process -- they are raw INPUTS, not
        #  inference products, so they go out at the sensor's own rate, not once per inference)
        # (plane compete is published by the MAPPING role -- it needs the live tracker)
        # 3) pillars top-down: horizontal bars + vertical pillar cells in each mask's colour
        try:
            _cbm = data.get("clench_by_mask") or {}
            _mrr = data.get("mask_ray_records") or {}
            _vcells = []
            for _bi_v, (_res_v, _inf_v) in _cbm.items():
                _rc = _mrr.get(int(_bi_v))
                if _rc is not None:
                    _c = np.asarray(_rc["color_rgb"], np.uint8).reshape(3)
                    _col = (int(_c[2]), int(_c[1]), int(_c[0]))
                else:
                    _col = (0, 165, 255)
                for _q in ((_res_v[3] if _res_v is not None else _inf_v.get("valid")) or []):
                    _cx, _cz = _q["center"]
                    _vcells.append((float(_cx), float(_cz), _col))
            import tempfile as _tf2, os as _os2
            _tmp2 = _os2.path.join(_tf2.gettempdir(), f"gk_viz_pillars_{fid:06d}.png")
            bsp._save_topdown_horizontal_pillars_image(
                _tmp2, big_idx=(data.get("big_idx") or []),
                horiz_by_mask=(data.get("horiz_by_mask") or {}),
                seed_records=data.get("seed_records") or [],
                mask_ray_records=_mrr, args=a, vert_cells_cam=_vcells)
            _im2 = cv2.imread(_tmp2)
            try: _os2.remove(_tmp2)
            except OSError: pass
            if _im2 is not None:
                self._publish_image(self.pub_viz_pillars, _im2, stamp=_vstamp)
        except Exception as e:
            self.get_logger().warn(f"[viz_full] pillars render failed: {e}", throttle_duration_sec=5.0)
        # 4) all-masks silhouette + support edge + 2 support rays
        try:
            _bgr = data.get("bgr")
            if _bgr is not None:
                import tempfile as _tf3, os as _os3
                _tmp3 = _os3.path.join(_tf3.gettempdir(), f"gk_viz_sil_{fid:06d}.png")
                _H, _W = _bgr.shape[:2]
                bsp._save_silhouette_allmasks(_tmp3, _bgr, (data.get("big_idx") or []),
                                              (data.get("mask_ray_records") or {}), _W, _H)
                _im3 = cv2.imread(_tmp3)
                try: _os3.remove(_tmp3)
                except OSError: pass
                if _im3 is not None:
                    self._publish_image(self.pub_viz_silhouette, _im3, stamp=_vstamp)
        except Exception as e:
            self.get_logger().warn(f"[viz_full] silhouette render failed: {e}", throttle_duration_sec=5.0)

    def _render_viz_burst(self, payload, pose_tuple, hdr):
        """viz_full, MAPPING role: render EVERY per-frame visual (formerly split between the
        perception process and its panel thread) into the hold list. Caller flushes after planes."""
        from builtin_interfaces.msg import Time as _T
        vz = payload.get("viz") or {}
        jpg = payload.get("bgr_jpg")
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg else None
        if bgr is None or pose_tuple is None:
            return
        H, W = bgr.shape[:2]
        st = _T(); st.sec, st.nanosec = int(payload["hstamp"][0]), int(payload["hstamp"][1])
        self._frame_pose = pose_tuple                   # _to_map / _publish_stacked use this
        mrr = payload.get("mask_ray_records") or {}
        big_idx = vz.get("big_idx") or list(mrr.keys())
        small_idx = vz.get("small_idx") or []
        def _unpack(keys, packed):
            try:
                ms = tx.unpack_masks(packed) if packed else []
                out = {}
                for k, m in zip(keys, ms):
                    if m.shape[0] != H or m.shape[1] != W:
                        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    out[int(k)] = m
                return out
            except Exception:
                return {}
        big_masks_full = _unpack(vz.get("big_keys") or [], vz.get("big_packed"))
        small_masks_full = _unpack(vz.get("small_keys") or [], vz.get("small_packed"))
        # 1) camera inputs
        self._publish_image(self.pub_viz_rgb, bgr, stamp=st)
        lj = vz.get("lidar_jpg")
        if lj:
            lv = cv2.imdecode(np.frombuffer(lj, np.uint8), cv2.IMREAD_COLOR)
            if lv is not None:
                self._publish_image(self.pub_viz_lidar, lv, stamp=st)
        # 2) SAM3 mask overlays (same colouring rule as perception: by POSITION in big/small_idx)
        if big_masks_full:
            self._publish_image(self.pub_big_masks, self._render_mask_overlay(bgr, big_idx, big_masks_full), stamp=st)
        if small_masks_full:
            self._publish_image(self.pub_small_masks, self._render_mask_overlay(bgr, small_idx, small_masks_full), stamp=st)
        # 3) seeds (stacked, map frame) + the 4 reference rays per big mask, in the mask colour
        seed_records = payload.get("seed_records") or []
        if seed_records:
            sc = np.asarray([r["pt"] for r in seed_records], np.float32).reshape(-1, 3)
            srgb = np.asarray([r["color"] for r in seed_records], np.uint8).reshape(-1, 3)
            self._publish_seeds(hdr, bsp._camera_to_viewer_zup(sc).astype(np.float32), srgb)
        a = self.args
        rl = float(getattr(a, "ray_length", 20.0)); rs = int(getattr(a, "ray_samples", 128))
        rx, rg = [], []
        for _bi, rec in mrr.items():
            r4 = rec.get("rays_robust")
            if r4 is None or len(r4) < 4:
                continue
            c = np.asarray(rec["color_rgb"], np.uint8).reshape(3)
            for k in range(4):
                pts = bsp._make_ray_points(np.asarray(r4[k], np.float32), ray_length=rl, ray_samples=rs)
                rx.append(pts.astype(np.float32)); rg.append(np.repeat(c.reshape(1, 3), len(pts), axis=0).astype(np.uint8))
        rxyz = bsp._camera_to_viewer_zup(np.concatenate(rx, 0)).astype(np.float32) if rx else np.empty((0, 3), np.float32)
        rrgb = np.concatenate(rg, 0) if rg else np.empty((0, 3), np.uint8)
        rm, rfid = self._to_map(hdr, rxyz)
        oh = Header(); oh.stamp = st; oh.frame_id = rfid
        self._pub(self.pub_viz_rays, _make_xyzrgb_cloud(oh, rm, rrgb))
        # 4) silhouette + pillars + pane lines (the old panel-thread renderers, unchanged)
        self._publish_viz_full({
            "bgr": bgr, "big_idx": big_idx, "mask_ray_records": mrr,
            "clench_by_mask": payload.get("clench_by_mask") or {},
            "horiz_by_mask": vz.get("horiz_by_mask") or {},
            "seed_records": seed_records, "pose": pose_tuple,
            "fid": int(payload.get("fid") or 0), "stamp": st,
            "small_masks_full": small_masks_full,
        })
        # 5) tracker-side: tug map, compete BEFORE/AFTER, spill verdict
        self._publish_viz_tracker(payload, pose_tuple)

    def _publish_viz_tracker(self, payload, pose_tuple):
        """viz_full, MAPPING role: publish the two visuals that need the live tracker --
        the seed-vs-floor tug map and the multi-view spill verdict."""
        a = self.args
        fid = int(payload.get("fid") or 0)
        tk = self._plane_tracker
        _vstamp = None                      # SOURCE stamp, shared with the perception streams
        try:
            _hs = payload.get("hstamp")
            if _hs is not None:
                from builtin_interfaces.msg import Time as _T
                _vstamp = _T(); _vstamp.sec, _vstamp.nanosec = int(_hs[0]), int(_hs[1])
        except Exception:
            _vstamp = None
        # 1) seed-vs-floor TUG map (robot-facing)
        if pose_tuple is not None:
            _ntf = len(getattr(tk, "tug_floor", {}) or {})
            _nts = len(getattr(tk, "tug_seed", {}) or {})
            if _ntf == 0 and _nts == 0:
                self.get_logger().warn(
                    f"[viz_full] tug map still empty (tug_floor={_ntf} tug_seed={_nts}, "
                    f"floor_check={getattr(tk, 'floor_check', '?')})", throttle_duration_sec=10.0)
            else:
                import tempfile as _tf, os as _os
                _tmp = _os.path.join(_tf.gettempdir(), f"gk_viz_tug_{fid:06d}.png")
                bsp._save_local_tug_png(_tmp, tk, pose_tuple,
                                        half_cells=int(getattr(a, "local_tug_half_cells", 50)),
                                        fid=fid)
                _im = cv2.imread(_tmp)
                try: _os.remove(_tmp)
                except OSError: pass
                if _im is not None:
                    self._publish_image(self.pub_viz_tug, _im, stamp=_vstamp)
        # 2) PLANE COMPETE: winners GREEN solid, in-compete losers RED dotted, one circled pair
        try:
            _cv = self._render_compete_view(payload, pose_tuple)
            if _cv is not None:
                self._publish_image(self.pub_viz_compete, _cv, stamp=_vstamp)
        except Exception as e:
            self.get_logger().warn(f"[viz_full] compete render failed: {e}", throttle_duration_sec=5.0)
        # 3) MULTI-VIEW SPILL verdict on the pano shipped from perception
        _jpg = payload.get("bgr_jpg")
        _spill = dict(getattr(tk, "_spill_debug", {}) or {})
        if _jpg and _spill:
            _bgr = cv2.imdecode(np.frombuffer(_jpg, np.uint8), cv2.IMREAD_COLOR)
            if _bgr is not None:
                self._publish_image(self.pub_viz_spill, self._render_spill_verdict(_bgr, _spill, fid), stamp=_vstamp)
        elif _jpg and not _spill:
            self.get_logger().warn(
                f"[viz_full] spill empty (reproject_evict={getattr(self, 'reproject_evict', '?')})",
                throttle_duration_sec=10.0)

    # colour per eviction mechanism (RGB), and how long a ghost stays on screen
    _EVICT_RGB = {"merge": (60, 120, 255), "floor": (60, 220, 60), "multiview": (255, 60, 60),
                  "path": (200, 200, 60), "depth": (255, 60, 60)}
    _EVICT_TTL_S = 4.0

    def _publish_evicted_planes(self, out_hdr, up, hh):
        """viz_full: every plane REMOVED by a global filter, drawn BLACK with a coloured BORDER
        naming the mechanism that removed it -- BLUE = merge/absorb, GREEN = floor evidence,
        RED = multi-view spill. Ghosts linger _EVICT_TTL_S seconds so a removal is visible even
        though it happens in a single frame, then expire."""
        a = self.args
        su, sv = int(a.plane_patch_samples_u), int(a.plane_patch_samples_v)
        su = max(4, su); sv = max(4, sv)          # need an interior for the border to read
        _now = time.time()
        # border mask over the (sv, su) grid, flattened v-major to match the patch builder
        _bw = 1                                    # border thickness, in grid cells
        _gj, _gi = np.meshgrid(np.arange(su), np.arange(sv))
        _edge = ((_gi < _bw) | (_gi >= sv - _bw) | (_gj < _bw) | (_gj >= su - _bw)).reshape(-1)
        for e in list(getattr(self._plane_tracker, "_evict_vis", []) or []):
            why = str(e.get("why", ""))
            rgb = self._EVICT_RGB.get(why)
            if rgb is None:
                continue
            p0 = np.asarray(e["p0"], np.float64); p1 = np.asarray(e["p1"], np.float64)
            vr = e.get("vr")
            if vr is not None and vr[0] is not None and vr[1] is not None:
                corners = np.array([[p0[0], p0[1], float(vr[1])], [p1[0], p1[1], float(vr[1])],
                                    [p1[0], p1[1], float(vr[0])], [p0[0], p0[1], float(vr[0])]],
                                   np.float32)
            else:
                corners = np.stack([p0 + up * hh, p1 + up * hh,
                                    p1 - up * hh, p0 - up * hh]).astype(np.float32)
            patch = bsp._make_plane_patch_from_quad_corners(corners, su, sv).astype(np.float32)
            cols = np.zeros((patch.shape[0], 3), np.uint8)          # BLACK interior
            if _edge.shape[0] == patch.shape[0]:
                cols[_edge] = np.array(rgb, np.uint8)               # coloured BORDER = mechanism
            self._evict_vis_ghosts.append((_now + self._EVICT_TTL_S, patch, cols))
        self._evict_vis_ghosts = [g for g in self._evict_vis_ghosts if g[0] > _now]
        if self._evict_vis_ghosts:
            gx = np.concatenate([g[1] for g in self._evict_vis_ghosts], axis=0)
            gc = np.concatenate([g[2] for g in self._evict_vis_ghosts], axis=0)
        else:
            gx = np.empty((0, 3), np.float32); gc = np.empty((0, 3), np.uint8)
        self.pub_viz_evicted.publish(_make_xyzrgb_cloud(out_hdr, gx, gc))

    def _render_compete_view(self, payload, pose):
        """Top-down COMPETE view, same styling as the batch demo visual:
        GREEN solid = the tracks that WON (the global merged planes), RED dotted = planes that
        LOST this frame's competition, plus ONE yellow-circled competition with its ray casts.
        Built from the live tracker, so this only runs in the mapping role."""
        if pose is None:
            return None
        a = self.args
        tk = self._plane_tracker
        pc_xyz = payload.get("pc_xyz")
        fid = int(payload.get("fid") or 0)
        # seed-count overlay: world centroids -> this camera's XZ, as (x, z, count) tuples.
        # (accum_counts_for returns a bare (N,) array, which is NOT the shape this renderer wants.)
        seed_nums = []
        try:
            wc = tk.world_seed_centroids()
            if wc:
                cc = bsp._world_to_cam(
                    np.array([[x, y, z] for (x, y, z, _n) in wc], np.float64), pose)
                seed_nums = [(float(cc[i][0]), float(cc[i][2]), wc[i][3]) for i in range(len(wc))]
        except Exception:
            seed_nums = []
        lines = []; rays = []; mid_by_pid = {}; ray_by_pid = {}; nfix = 0
        for tp in tk.planes:                       # WINNERS: all green, no label
            nfix += int(getattr(tp, "fixed", 0))
            pc = bsp._world_to_cam(np.stack([tp.apex, tp.p0, tp.p1]), pose)
            ap = (float(pc[0][0]), float(pc[0][2]))
            la = (float(pc[1][0]), float(pc[1][2])); ra = (float(pc[2][0]), float(pc[2][2]))
            mid_by_pid[tp.pid] = (0.5 * (la[0] + ra[0]), 0.5 * (la[1] + ra[1]))
            lines.append((la, ra, (0, 220, 0), bool(getattr(tp, "fixed", 0)), None))
            ray_by_pid[tp.pid] = (ap, la, ra, tp.color)
        cvis = list(getattr(tk, "_compete_vis", []) or [])
        lost_mids = []
        for e in cvis:                             # LOSERS: dotted red
            if not e.get("lost"):
                continue
            pc = bsp._world_to_cam(np.stack([e["w0"], e["w1"]]), pose)
            la = (float(pc[0][0]), float(pc[0][2])); ra = (float(pc[1][0]), float(pc[1][2]))
            lines.append((la, ra, (0, 0, 255), False, None, True))
            e["_cam_ray"] = ((0.0, 0.0), la, ra, (0, 0, 255))
            lost_mids.append(((0.5 * (la[0] + ra[0]), 0.5 * (la[1] + ra[1])), e))
        circles = []                               # ONE circled competition, with both sides' rays
        cands = lost_mids + [(mid_by_pid.get(e2.get("pid")), e2) for e2 in cvis
                             if not e2.get("lost") and e2.get("strk") is not None
                             and mid_by_pid.get(e2.get("pid")) is not None]
        cands = [x for x in cands if x[1].get("strk") is not None]
        np.random.default_rng(fid).shuffle(cands)
        for (mn, e) in cands[:1]:
            mt = mid_by_pid.get(e.get("pid"))
            ctr = (0.5 * (mn[0] + mt[0]), 0.5 * (mn[1] + mt[1])) if mt is not None else mn
            rad = (max(1.0, 0.6 * float(np.hypot(mn[0] - mt[0], mn[1] - mt[1])) + 0.8)
                   if mt is not None else 1.2)
            circles.append((ctr[0], ctr[1], rad))
            if e.get("_cam_ray") is not None:
                rays.append(e["_cam_ray"])
            if e.get("pid") in ray_by_pid:
                rays.append(ray_by_pid[e["pid"]])
        after = bsp._render_topdown_cam_view(
            pc_xyz, seed_nums, lines, a,
            f"AFTER: global map  {len(tk.planes)} tracks ({nfix} fixed)",
            ray_sets=rays, compete_circle=circles)
        # BEFORE: the planes ENTERING the scene this frame -- every placement the perception side
        # solved, in its own big-mask colour, with its camera rays. This is what is about to be
        # competed/merged against the global map; the AFTER panel is the outcome.
        in_lines = []; in_rays = []
        for e in cvis:
            pc = bsp._world_to_cam(np.stack([e["w0"], e["w1"]]), pose)
            la = (float(pc[0][0]), float(pc[0][2])); ra = (float(pc[1][0]), float(pc[1][2]))
            mc = getattr(self, "_viz_mask_rgb", {}).get(int(e.get("mask", -1)))
            col = ((int(mc[2]), int(mc[1]), int(mc[0])) if mc is not None else (0, 200, 255))
            in_lines.append((la, ra, col, False, f"m{int(e.get('mask', -1))}"))
            in_rays.append(((0.0, 0.0), la, ra, col))
        before = bsp._render_topdown_cam_view(
            pc_xyz, seed_nums, in_lines, a,
            f"BEFORE: entering this frame  {len(in_lines)} plane(s)",
            ray_sets=in_rays)
        return bsp._topdown_panel(before, after,
                                  "BEFORE -- planes entering the scene",
                                  "AFTER -- compete / merge outcome")

    def _render_spill_verdict(self, bgr, spill, fid):
        """Reprojected plane samples on the RGB, coloured by the multi-view spill verdict:
        GREEN = on-mask, RED = spill, GRAY = not judged (no 2D detection there). Each plane is
        labelled with its iou/spill percentages, red once it is over the eviction threshold."""
        thr = float(getattr(self.args, "track_spill_thresh", 0.30))
        vis = bgr.copy()
        for d in spill.values():
            u = np.asarray(d.get("u", [])); v = np.asarray(d.get("v", []))
            if u.size == 0:
                continue
            det = bool(d.get("det", True))
            inm = np.asarray(d.get("inmask", []), bool)
            step = max(1, u.size // 400)                    # cap the dots drawn
            for k in range(0, u.size, step):
                if not det:
                    col = (160, 160, 160)
                else:
                    col = (0, 200, 0) if (k < inm.size and inm[k]) else (0, 0, 255)
                cv2.circle(vis, (int(u[k]), int(v[k])), 1, col, -1)
            cu, cvv = int(np.median(u)), int(np.median(v))
            iou = float(d.get("iou", 0.0))
            if not bool(d.get("judged", det)):              # not judged this frame -> say WHY
                lab = (f"m{d.get('mask')} iou={iou*100:.0f}% spill={d.get('spill', 0)*100:.0f}% "
                       f"(skip:{d.get('skip_why', '?')})"); lcol = (200, 200, 200)
            else:
                lab = (f"m{d.get('mask')} vis={float(d.get('vis_frac', 1.0))*100:.0f}% "
                       f"spill={float(d.get('spill', 0))*100:.0f}%"
                       f"{' EVICT' if d.get('evicted') else ''}")
                lcol = (0, 0, 255) if float(d.get("spill", 0)) >= thr else (0, 255, 255)
            cv2.putText(vis, lab, (cu, max(cvv, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol, 2, cv2.LINE_AA)
        _txt = f"frame {fid}  multi-view spill: GREEN=on-mask RED=spill GRAY=not judged"
        (_tw, _th), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (0, 0), (_tw + 20, _th + 18), (18, 18, 18), -1)
        cv2.putText(vis, _txt, (10, _th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    def _pub(self, publisher, msg):
        """Publish, or HOLD when a burst is being assembled (see _run_mapping)."""
        if self._viz_hold is not None:
            self._viz_hold.append((publisher, msg))
        else:
            publisher.publish(msg)

    def _publish_image(self, publisher, vis, stamp=None):
        """Publish a BGR image. `stamp` should be the SOURCE FRAME's stamp so every demo stream
        carries the same timestamp for the same pipeline frame -- that is what makes the separately
        recorded streams alignable afterwards (the renders finish at different wall times, and the
        panel thread skips frames when it falls behind, so wall time alone would not line up)."""
        msg = Image()
        msg.header.frame_id = "map"
        if stamp is not None:
            msg.header.stamp = stamp
        msg.height, msg.width = vis.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = np.ascontiguousarray(vis).tobytes()
        self._pub(publisher, msg)

    def _track_wall_patch(self, p0, p1, vr, up, grid_m, end_inset=0.0):
        """Dense metric-grid samples of one tracked wall: the ground line p0->p1 extruded along `up`
        over the track's observed world height range vr=(v0,v1) (fallback: +/- half global_plane_height_m
        around the line height). Sample pitch is `grid_m` in BOTH directions, so a big wall gets a
        proportionally dense filled patch instead of a fixed-count dot line.
        end_inset (m): pull the sampling in from EACH end by this much, so the emitted obstacle stops
        short of the plane's edges. Glass walls end at corners/doorframes, and the local planner
        inflates every obstacle point by the robot clearance -- without the inset that inflation bleeds
        into an adjacent doorway/opening and closes it. Insetting keeps open space open."""
        d = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
        length = float(np.linalg.norm(d))
        v0, v1 = (vr if (vr is not None and vr[0] is not None and vr[1] is not None)
                  else (None, None))
        if v0 is None:
            hh = 0.5 * float(self.global_plane_height_m)
            base = float(np.dot(np.asarray(p0, np.float64), up))
            v0, v1 = base - hh, base + hh
        v0 = float(v0) - max(0.0, float(self.obstacle_extend_bottom_m))
        t_in = min(0.45, (end_inset / length) if length > 1e-6 else 0.0)   # cap so a short wall survives
        nu = max(2, int(np.ceil(length * (1.0 - 2.0 * t_in) / grid_m)) + 1)
        nv = max(2, int(np.ceil((v1 - v0) / grid_m)) + 1)
        tu = np.linspace(t_in, 1.0 - t_in, nu)
        heights = np.linspace(v0, v1, nv)
        line = np.asarray(p0, np.float64)[None, :] + tu[:, None] * d[None, :]   # (nu,3) along ground line
        line = line - np.dot(line, up)[:, None] * up[None, :]                   # strip height component
        pts = (line[:, None, :] + heights[None, :, None] * up[None, None, :]).reshape(-1, 3)
        return pts.astype(np.float32)

    def _publish_global_planes(self, src_header: Header, tracks, pose, track_vr=None):
        """Publish the accumulated tracks as world(map)-frame clouds, on THREE topics:
          /glass_killer/global_planes   -- the ORIGINAL colored patch visual (fixed su x sv samples,
                                           +/- half global_plane_height_m around the ground line).
          /glass_killer/obstacle_walls  -- RViz copy of the OBSTACLE walls: dense metric-pitch patches
                                           over each track's observed height range, colored like the
                                           track but dimmed, so the two visuals can be compared.
          /added_obstacles              -- the same dense walls as PointXYZI (intensity=200) for the
                                           local planner (its handler clears + replaces per message, so
                                           the FULL set goes out each frame; an empty cloud clears
                                           stale planes when no tracks exist)."""
        out = Header(); out.stamp = src_header.stamp; out.frame_id = "map"
        a = self.args
        su, sv = int(a.plane_patch_samples_u), int(a.plane_patch_samples_v)
        hh = 0.5 * float(self.global_plane_height_m)
        up = np.array([0.0, 0.0, 1.0]) if pose is not None else np.array([0.0, 1.0, 0.0])
        xyz_parts, rgb_parts = [], []                  # original visual
        oxyz_parts, orgb_parts = [], []                # dense obstacle walls
        for ti, (apex, p0, p1, color, fixed, mask) in enumerate(tracks):
            b, g, r = int(color[0]), int(color[1]), int(color[2])      # track colour is BGR
            if getattr(self, "viz_full", False):
                # viz_full: recolour to the OWNING BIG MASK's colour so a plane matches the rays,
                # seeds and silhouette it came from. The tracker assigns its own palette slot on
                # spawn (unrelated to the mask), and mask colours are indexed by POSITION in
                # big_idx -- so the colour must be looked up, never recomputed from the mask id.
                _map = getattr(self, "_viz_mask_rgb", {}) or {}
                _mc = _map.get(int(mask))
                if _mc is not None:
                    r, g, b = int(_mc[0]), int(_mc[1]), int(_mc[2])
                elif ti == 0:                 # say WHY the colour did not match, once per frame
                    self.get_logger().warn(
                        f"[viz_full] plane colour miss: track mask={int(mask)} not in "
                        f"mask_rgb keys={sorted(_map.keys())[:12]} (n={len(_map)})",
                        throttle_duration_sec=5.0)
            vr = track_vr[ti] if track_vr is not None else None
            if vr is not None and vr[0] is not None and vr[1] is not None and pose is not None:
                # TRUE vertical extent (v0..v1, world z) -- the display previously drew a fixed
                # +/-hh slab about the ground line, hiding the upper half of tall glass
                corners = np.array([[p0[0], p0[1], float(vr[1])], [p1[0], p1[1], float(vr[1])],
                                    [p1[0], p1[1], float(vr[0])], [p0[0], p0[1], float(vr[0])]],
                                   np.float32)
            else:
                corners = np.stack([p0 + up * hh, p1 + up * hh,             # TL, TR
                                    p1 - up * hh, p0 - up * hh]).astype(np.float32)   # BR, BL
            patch = bsp._make_plane_patch_from_quad_corners(corners, su, sv).astype(np.float32)
            xyz_parts.append(patch)
            rgb_parts.append(np.tile(np.array([[r, g, b]], np.uint8), (patch.shape[0], 1)))
            wall = self._track_wall_patch(p0, p1, vr, up, float(self.obstacle_grid_m),
                                          end_inset=float(self.obstacle_end_inset_m))
            oxyz_parts.append(wall)
            orgb_parts.append(np.tile(np.array([[r // 2, g // 2, b // 2]], np.uint8),
                                      (wall.shape[0], 1)))
        xyz = (np.concatenate(xyz_parts, 0).astype(np.float32)
               if xyz_parts else np.empty((0, 3), np.float32))
        rgb = (np.concatenate(rgb_parts, 0).astype(np.uint8)
               if rgb_parts else np.empty((0, 3), np.uint8))
        # append the BLACK ghosts of recently evicted planes (3s lifetime) so evictions are
        # visible in the SAME display as the live map; no live track ever renders pure black
        _now = time.time()
        self._evict_ghosts = [g for g in self._evict_ghosts if g[0] > _now]
        if self._evict_ghosts:
            _gx = np.concatenate([g[1] for g in self._evict_ghosts], axis=0)
            xyz = np.concatenate([xyz, _gx], axis=0)
            rgb = np.concatenate([rgb, np.zeros((_gx.shape[0], 3), np.uint8)], axis=0)
        self.pub_global_planes.publish(_make_xyzrgb_cloud(out, xyz, _viz_recolor(rgb)))
        if getattr(self, "viz_full", False):      # EVICTED planes, coloured by removal mechanism
            try:
                self._publish_evicted_planes(out, up, hh)
            except Exception as e:
                self.get_logger().warn(f"[viz_full] evicted planes failed: {e}",
                                       throttle_duration_sec=5.0)
        if getattr(self, "viz_full", False):      # plane WIREFRAME in the big-mask colour
            try:
                _lx, _lc = [], []

                def _seg(a3, b3, rgb3):          # sample a 3D segment at ~3 cm pitch
                    a3 = np.asarray(a3, np.float64); b3 = np.asarray(b3, np.float64)
                    n = max(2, int(np.linalg.norm(b3 - a3) / 0.03))
                    t = np.linspace(0.0, 1.0, n)[:, None]
                    _lx.append((a3[None, :] * (1.0 - t) + b3[None, :] * t).astype(np.float32))
                    _lc.append(np.tile(np.array([rgb3], np.uint8), (n, 1)))

                for _ti, (apex, p0, p1, color, fixed, mask) in enumerate(tracks):
                    _mc = getattr(self, "_viz_mask_rgb", {}).get(int(mask))
                    _rgb = ([int(_mc[0]), int(_mc[1]), int(_mc[2])] if _mc is not None
                            else [int(color[2]), int(color[1]), int(color[0])])
                    _a = np.asarray(p0, np.float64); _bp = np.asarray(p1, np.float64)
                    # GROUND line (the plane's footprint)
                    _seg(_a, _bp, _rgb)
                    # VERTICAL PILLAR lines: the pane's true height range at each end, plus the top
                    # edge, so the wireframe shows the standing extent rather than a bare floor line.
                    _vr = track_vr[_ti] if (track_vr is not None and _ti < len(track_vr)) else None
                    if _vr is not None and _vr[0] is not None and _vr[1] is not None:
                        _v0, _v1 = float(_vr[0]), float(_vr[1])
                    else:                         # no observed range -> the display slab about the line
                        _v0, _v1 = float(_a[2]) - hh, float(_a[2]) + hh
                    _aL = np.array([_a[0], _a[1], _v0]); _aH = np.array([_a[0], _a[1], _v1])
                    _bL = np.array([_bp[0], _bp[1], _v0]); _bH = np.array([_bp[0], _bp[1], _v1])
                    _seg(_aL, _aH, _rgb)          # left vertical pillar
                    _seg(_bL, _bH, _rgb)          # right vertical pillar
                    _seg(_aH, _bH, _rgb)          # top edge
                _lxyz = (np.concatenate(_lx, 0) if _lx else np.empty((0, 3), np.float32))
                _lrgb = (np.concatenate(_lc, 0) if _lc else np.empty((0, 3), np.uint8))
                self.pub_viz_lines.publish(_make_xyzrgb_cloud(out, _lxyz, _lrgb))
            except Exception as e:
                self.get_logger().warn(f"[viz_full] plane lines failed: {e}",
                                       throttle_duration_sec=5.0)
        # Drive /glass_killer/planes from the TRACKER too, so an EVICTED plane disappears from BOTH
        # topics (the per-frame _publish_planes is skipped while the global map is on).
        self.pub_planes.publish(_make_xyzrgb_cloud(out, xyz, _viz_recolor(rgb)))
        oxyz = (np.concatenate(oxyz_parts, 0).astype(np.float32)
                if oxyz_parts else np.empty((0, 3), np.float32))
        orgb = (np.concatenate(orgb_parts, 0).astype(np.uint8)
                if orgb_parts else np.empty((0, 3), np.uint8))
        self.pub_obstacle_walls.publish(_make_xyzrgb_cloud(out, oxyz, _viz_recolor(orgb)))
        # --- obstacle injection (only once poses arrive: tracker world == /state_estimation frame) ---
        if pose is not None:
            # CURRENT-map walls, STABILIZED: only points whose 0.3m voxel also existed in the
            # LAST frame's current walls join the scan union -- a plane must survive 2
            # consecutive frames before it blocks, so single-frame misplacements (the no-path
            # burst source) never reach the planner, at the cost of ~1 frame extra latency.
            _pv = getattr(self, "_cur_wall_vox_prev", None)
            if oxyz.shape[0] and _pv is not None:
                _kk = np.floor(oxyz / 0.3).astype(np.int64)
                _keep = np.fromiter((tuple(k) in _pv for k in _kk), bool, len(_kk))
                self._cur_wall_xyz = oxyz[_keep]
            else:
                self._cur_wall_xyz = np.empty((0, 3), np.float32)
            self._cur_wall_vox_prev = ({tuple(k) for k in np.floor(oxyz / 0.3).astype(np.int64)}
                                       if oxyz.shape[0] else set())
            if not (self.obstacle_from_final and self.enable_final_map):
                self._latest_wall_xyz = oxyz               # cached for the scan-rate republish timer
                if self.obstacle_mode in ("scan", "both"): # let terrain-analysis decide (height-aware)
                    self.pub_scan_obstacles.publish(_make_xyzi_rgb_cloud(out, oxyz, 199.0))
            if (self.publish_added_obstacles and self.obstacle_mode in ("added", "both")
                    and not (self.obstacle_from_final and self.enable_final_map)):
                self.pub_added_obstacles.publish(_make_xyzi_rgb_cloud(out, oxyz, 200.0))

    def _publish_final_planes(self, src_header: Header, tracks, pose, track_vr=None):
        """Publish the FINAL/OVERVIEW map's accumulated tracks as the colored plane visual only, on
        /glass_killer/final_global_planes. Same colored-patch style as the current map's global_planes,
        and, when obstacle_from_final (default), ALSO the /added_obstacles planner feed --
        the persistence-biased map is what navigation consumes."""
        out = Header(); out.stamp = src_header.stamp; out.frame_id = "map"
        a = self.args
        su, sv = int(a.plane_patch_samples_u), int(a.plane_patch_samples_v)
        hh = 0.5 * float(self.global_plane_height_m)
        up = np.array([0.0, 0.0, 1.0]) if pose is not None else np.array([0.0, 1.0, 0.0])
        xyz_parts, rgb_parts = [], []
        for ti, (apex, p0, p1, color, fixed, mask) in enumerate(tracks):
            b, g, r = int(color[0]), int(color[1]), int(color[2])
            vrf = track_vr[ti] if track_vr is not None and ti < len(track_vr) else None
            if vrf is not None and vrf[0] is not None and vrf[1] is not None and pose is not None:
                corners = np.array([[p0[0], p0[1], float(vrf[1])], [p1[0], p1[1], float(vrf[1])],
                                    [p1[0], p1[1], float(vrf[0])], [p0[0], p0[1], float(vrf[0])]],
                                   np.float32)
                patch = bsp._make_plane_patch_from_quad_corners(corners, su, sv).astype(np.float32)
                xyz_parts.append(patch)
                rgb_parts.append(np.tile(np.array([[r, g, b]], np.uint8), (patch.shape[0], 1)))
                continue
            corners = np.stack([p0 + up * hh, p1 + up * hh,
                                p1 - up * hh, p0 - up * hh]).astype(np.float32)
            patch = bsp._make_plane_patch_from_quad_corners(corners, su, sv).astype(np.float32)
            xyz_parts.append(patch)
            rgb_parts.append(np.tile(np.array([[r, g, b]], np.uint8), (patch.shape[0], 1)))
        xyz = (np.concatenate(xyz_parts, 0).astype(np.float32)
               if xyz_parts else np.empty((0, 3), np.float32))
        rgb = (np.concatenate(rgb_parts, 0).astype(np.uint8)
               if rgb_parts else np.empty((0, 3), np.uint8))
        self.pub_final_global_planes.publish(_make_xyzrgb_cloud(out, xyz, _viz_recolor(rgb)))
        # FINAL-map planner feed: dense metric walls from the persistent tracks
        if self.obstacle_from_final and pose is not None:
            _wparts = []
            for ti, (apex, p0, p1, color, fixed, mask) in enumerate(tracks):
                vrf = track_vr[ti] if track_vr is not None and ti < len(track_vr) else None
                _wparts.append(self._track_wall_patch(
                    p0, p1, vrf, up, float(self.obstacle_grid_m),
                    end_inset=float(self.obstacle_end_inset_m)))
            _wxyz = (np.concatenate(_wparts, 0).astype(np.float32)
                     if _wparts else np.empty((0, 3), np.float32))
            # apply the eviction-mirror suppression to the FINAL-sourced walls
            _now_m = time.monotonic()
            self._evict_suppress = [e for e in self._evict_suppress
                                    if _now_m - e[2] < self.evict_suppress_s]
            if _wxyz.shape[0] and self._evict_suppress:
                _lat2 = float(self.evict_suppress_lat_m) ** 2
                _P2 = _wxyz[:, :2].astype(np.float64)
                _keep = np.ones(len(_wxyz), bool)
                for _sa, _sb, _t0 in self._evict_suppress:
                    _ab = _sb - _sa
                    _L2 = float(_ab @ _ab)
                    if _L2 < 1e-9:
                        continue
                    _tt = np.clip(((_P2 - _sa) @ _ab) / _L2, 0.0, 1.0)
                    _d2 = ((_P2 - (_sa + _tt[:, None] * _ab)) ** 2).sum(axis=1)
                    _keep &= _d2 > _lat2
                _wxyz = np.ascontiguousarray(_wxyz[_keep])
            if self.publish_added_obstacles and self.obstacle_mode in ("added", "both"):
                self.pub_added_obstacles.publish(_make_xyzi_rgb_cloud(out, _wxyz, 200.0))
            if self.obstacle_mode in ("scan", "both"):
                # scan injection = FINAL walls (stable) UNION CURRENT walls (immediate): a
                # just-placed pane blocks now instead of after final-tracker confirmation
                _cur = getattr(self, "_cur_wall_xyz", None)
                if _cur is not None and _cur.shape[0]:
                    _wxyz = np.concatenate([_wxyz, _cur], 0) if _wxyz.shape[0] else _cur
                if _wxyz.shape[0]:
                    _sv = max(0.05, self.obstacle_scan_voxel_m)
                    _kk, _idx = np.unique(np.floor(_wxyz / _sv).astype(np.int64),
                                          axis=0, return_index=True)
                    _wxyz = np.ascontiguousarray(_wxyz[_idx])
                self._latest_wall_xyz = _wxyz
                self.pub_scan_obstacles.publish(_make_xyzi_rgb_cloud(out, _wxyz, 199.0))

    def _save_local_tug_view(self, fid, pose):
        """Per-frame PNG of the seed-vs-floor tug in a local window (+/- local_tug_half cells) around the
        robot. Each fine (map_cell_m) cell is colored by which side leads and how strong (count 0..3):
        GREEN = floor-dominant, MAGENTA = seed-dominant, GRAY = tied, ORANGE = obstacle. The dominant
        count digit is drawn in each non-empty cell. Robot at the center (cyan cross)."""
        tk = self._plane_tracker
        tf, ts = tk.tug_floor, tk.tug_seed
        if not tf and not ts:
            return
        gi, gj = tk.gi, tk.gj; mk = tk.map_cell_m
        apex_w = bsp._cam_to_world(np.zeros((1, 3), np.float32), pose)[0]
        ri = int(np.floor(apex_w[gi] / mk)); rj = int(np.floor(apex_w[gj] / mk))
        R = int(self.local_tug_half); cp = 14                     # half-window cells, cell pixels
        N = 2 * R + 1
        img = np.full((N * cp, N * cp, 3), 20, np.uint8)          # near-black background
        tmax = max(1, int(getattr(tk, "tug_max", 3)))
        obst = tk.obst_cells
        for dj in range(-R, R + 1):
            for di in range(-R, R + 1):
                c = (ri + di, rj + dj)
                f = int(tf.get(c, 0)); s = int(ts.get(c, 0))
                if f == 0 and s == 0 and c not in obst:
                    continue
                # image: +i -> right (col), +j -> UP (so row = R - dj)
                col = (di + R) * cp; row = (R - dj) * cp
                if c in obst:
                    color = (0, 140, 255); dig = None            # orange (BGR)
                elif s >= f and s > 0:
                    # SEED shown on co-occurrence/ties: glass presence takes display priority
                    v = int(90 + 55 * min(s, tmax)); color = (v, 0, v); dig = str(s)    # magenta
                else:
                    v = int(90 + 55 * min(f, tmax)); color = (0, v, 0); dig = str(f)    # green
                cv2.rectangle(img, (col, row), (col + cp - 1, row + cp - 1), color, -1)
                if dig is not None and cp >= 12:
                    cv2.putText(img, dig, (col + 2, row + cp - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1, cv2.LINE_AA)
        # TRACKED PLANES overlaid as ground lines in their own track colors
        for tp in tk.planes:
            try:
                x0 = (float(tp.p0[gi]) / mk - (ri - R)) * cp
                y0 = ((rj + R) - float(tp.p0[gj]) / mk) * cp
                x1 = (float(tp.p1[gi]) / mk - (ri - R)) * cp
                y1 = ((rj + R) - float(tp.p1[gj]) / mk) * cp
                b_, g_, r_ = int(tp.color[0]), int(tp.color[1]), int(tp.color[2])
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (b_, g_, r_), 3, cv2.LINE_AA)
            except Exception:
                pass
        # robot marker at window center + gridlines every 10 cells (=1 m at 0.1 m cells)
        cc = R * cp + cp // 2
        cv2.drawMarker(img, (cc, cc), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        for k in range(0, N + 1, 10):
            cv2.line(img, (k * cp, 0), (k * cp, N * cp), (45, 45, 45), 1)
            cv2.line(img, (0, k * cp), (N * cp, k * cp), (45, 45, 45), 1)
        hdr = (f"frame {fid}  MAGENTA=seed(wins/ties) GREEN=floor ORANGE=obst LINES=planes  "
               f"win +/-{R}cell ({R*mk:.1f}m)  cell={mk:.2f}m")
        cv2.rectangle(img, (0, 0), (N * cp, 20), (0, 0, 0), -1)
        cv2.putText(img, hdr, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.local_tug_dir, f"localtug_{fid:06d}.png"), img)

    def _scan_obstacle_pump(self):
        """Keep the glass points fresh in terrain-analysis's rolling voxel grid (scan mode).
        Re-emits the last wall cloud onto the scan topic at obstacle_scan_hz, from its OWN thread
        so inference can't starve it. Stamped in the SENSOR/BAG clock, extrapolated: latest pose
        stamp + wall time elapsed since that pose arrived -- pose callbacks are also blocked
        during inference, and a stamp staler than terrain's 2s decay drops the points on arrival
        (a wall-clock stamp is worse still: it desyncs terrain's whole decay reference)."""
        period = 1.0 / max(0.5, float(self.obstacle_scan_hz))
        while rclpy.ok():
            time.sleep(period)
            if self.obstacle_mode not in ("scan", "both"):
                continue
            wall = self._latest_wall_xyz
            if wall.shape[0] == 0:
                continue
            with self._lock:
                lp, mono = self._latest_pose, self._pose_recv_mono
            if lp is None:
                continue
            t = (float(lp.header.stamp.sec) + float(lp.header.stamp.nanosec) * 1e-9
                 + max(0.0, time.monotonic() - mono))
            hdr = Header(); hdr.frame_id = "map"
            hdr.stamp.sec = int(t); hdr.stamp.nanosec = int((t - int(t)) * 1e9)
            try:
                self.pub_scan_obstacles.publish(_make_xyzi_rgb_cloud(hdr, wall, 199.0))
            except Exception:
                pass

    def _render_global_panel(self, data):
        """Two-panel top-down (ported from the batch process_frame): LEFT = THIS frame's placed planes
        in the native camera frame; RIGHT = the GLOBAL accumulated tracks, world -> current camera (only
        those in view show). Both overlay the world-accumulated seed-count numbers + the world grid."""
        a = self.args
        pose = data["pose"]
        pc_xyz = data["pc_xyz"]
        clench_by_mask = data["clench_by_mask"]
        mask_ray_records = data["mask_ray_records"]
        tracks = data["tracks"]
        wc = data["seed_centroids"]

        # world seed centroids -> this camera's XZ for the seed-count overlay
        if wc:
            cc = bsp._world_to_cam(np.array([[x, y, z] for (x, y, z, _n) in wc], np.float64), pose)
            seed_nums = [(float(cc[i][0]), float(cc[i][2]), wc[i][3]) for i in range(len(wc))]
        else:
            seed_nums = []

        # world pillar grid (same bins the tracker uses) projected into this camera's XZ
        grid_lines_cam = []
        gi, gj, g3 = (0, 1, 2) if pose is not None else (0, 2, 1)
        bm = float(getattr(a, "clench_bin_m", 1.0))
        rng = float(getattr(a, "topdown_max_range_m", 12.0)) + bm
        camw = bsp._cam_to_world(np.zeros((1, 3), np.float32), pose)[0]
        c0, c1, cz = float(camw[gi]), float(camw[gj]), float(camw[g3])
        k0 = int(np.floor((c0 - rng) / bm)); k1 = int(np.ceil((c0 + rng) / bm))
        l0 = int(np.floor((c1 - rng) / bm)); l1 = int(np.ceil((c1 + rng) / bm))

        def wg(u, v):
            p = np.zeros(3, np.float64); p[gi] = u; p[gj] = v; p[g3] = cz; return p
        for k in range(k0, k1 + 1):
            cc2 = bsp._world_to_cam(np.stack([wg(k * bm, l0 * bm), wg(k * bm, l1 * bm)]), pose)
            grid_lines_cam.append(((float(cc2[0][0]), float(cc2[0][2])), (float(cc2[1][0]), float(cc2[1][2]))))
        for l in range(l0, l1 + 1):
            cc2 = bsp._world_to_cam(np.stack([wg(k0 * bm, l * bm), wg(k1 * bm, l * bm)]), pose)
            grid_lines_cam.append(((float(cc2[0][0]), float(cc2[0][2])), (float(cc2[1][0]), float(cc2[1][2]))))

        # LEFT: this frame's placed planes (native camera frame)
        left_lines = []; left_rays = []
        for bi, (res, _inf) in clench_by_mask.items():
            if res is None:
                continue
            lc, rc = bsp._plane_ground_line_cam(res[0])
            rec = mask_ray_records.get(int(bi))
            crgb = np.asarray(rec["color_rgb"], np.uint8).reshape(3) if rec is not None else np.array([0, 0, 255], np.uint8)
            col = (int(crgb[2]), int(crgb[1]), int(crgb[0]))
            la = (float(lc[0]), float(lc[2])); ra = (float(rc[0]), float(rc[2]))
            left_lines.append((la, ra, col, False, f"m{int(bi)}"))
            left_rays.append(((0.0, 0.0), la, ra, col))
        left = bsp._render_topdown_cam_view(pc_xyz, seed_nums, left_lines, a,
                                            "current frame planes + rays", ray_sets=left_rays,
                                            grid_lines_cam=grid_lines_cam)

        # DOORWAY-locked world cells -> this camera's XZ centres (marked red on the RIGHT/global panel).
        lock_cells_cam = []
        for (ci, cj) in self._plane_tracker.doorway_locked_cells():
            wp = np.zeros(3, np.float64); wp[gi] = (ci + 0.5) * bm; wp[gj] = (cj + 0.5) * bm; wp[g3] = cz
            cc3 = bsp._world_to_cam(wp[None, :], pose)[0]
            lock_cells_cam.append((float(cc3[0]), float(cc3[2])))

        # RIGHT: global accumulated tracks (world -> this camera)
        right_lines = []; right_rays = []; nfix = 0
        for (apex, p0, p1, color, fixed, mask) in tracks:
            nfix += int(fixed)
            pc = bsp._world_to_cam(np.stack([apex, p0, p1]), pose)
            ap = (float(pc[0][0]), float(pc[0][2]))
            la = (float(pc[1][0]), float(pc[1][2])); ra = (float(pc[2][0]), float(pc[2][2]))
            right_lines.append((la, ra, color, fixed, f"m{mask}"))
            right_rays.append((ap, la, ra, color))
        right = bsp._render_topdown_cam_view(pc_xyz, seed_nums, right_lines, a,
                                             f"global merged in view: {len(tracks)} ({nfix} fixed)  "
                                             f"doorway-locked cells: {len(lock_cells_cam)}",
                                             ray_sets=right_rays, grid_lines_cam=grid_lines_cam,
                                             mark_cells_cam=lock_cells_cam)
        return bsp._topdown_panel(left, right, "CURRENT FRAME", "GLOBAL ACCUMULATED (in view)")

    def _publish_map_cloud(self, publisher, src_header: Header, xyz: np.ndarray, frame_id: str):
        """Publish this-frame-only world points (xyz already _to_map'd) for the top-down map node.
        Colour is irrelevant (white); the accumulator only reads xyz."""
        out = Header()
        out.stamp = src_header.stamp
        out.frame_id = frame_id
        rgb = np.full((int(xyz.shape[0]), 3), 255, np.uint8)
        publisher.publish(_make_xyzrgb_cloud(out, xyz.astype(np.float32), rgb))

    def _save_reproject_panel(self, bgr, spill, map_snap, tracks, fid, glass_mask=None):
        """LEFT = SAM big-mask (GREEN) vs the reprojected planes (BLUE), with their OVERLAP in YELLOW
        (this is what the IoU gate measures). RIGHT = the RGB with each plane's reprojected surface
        samples (GREEN=on-mask, RED=spill, GRAY=no-detection) + a per-plane spill%/iou label."""
        thr = float(self.args.track_spill_thresh)
        H, W = bgr.shape[:2]
        # LEFT: green = SAM glass mask, blue = reprojected plane footprint, yellow = overlap
        gm = np.zeros((H, W), bool) if glass_mask is None else np.asarray(glass_mask, bool)
        plane_fp = np.zeros((H, W), bool)                      # union of all reprojected plane footprints
        for d in spill.values():
            u = np.asarray(d.get("u", [])); v = np.asarray(d.get("v", []))
            if u.size:
                plane_fp[np.clip(v, 0, H - 1).astype(np.int64), np.clip(u, 0, W - 1).astype(np.int64)] = True
        if plane_fp.any():                                     # dilate sparse samples into a region
            plane_fp = cv2.dilate(plane_fp.astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(bool)
        left = (bgr.astype(np.float32) * 0.35).astype(np.uint8)   # dim RGB for context
        left[gm & ~plane_fp] = (0, 200, 0)                     # GREEN = SAM mask only
        left[plane_fp & ~gm] = (255, 60, 0)                    # BLUE  = reprojected plane only
        left[gm & plane_fp] = (0, 255, 255)                   # YELLOW = overlap (intersection)
        # RIGHT: reprojected samples on the RGB
        right = bgr.copy()
        for d in spill.values():
            u = np.asarray(d.get("u", [])); v = np.asarray(d.get("v", []))
            if u.size == 0:
                continue
            det = bool(d.get("det", True))                     # was there a 2D detection where it reprojects?
            inm = np.asarray(d.get("inmask", []), bool)
            step = max(1, u.size // 400)                       # cap points drawn
            for k in range(0, u.size, step):
                if not det:
                    col = (160, 160, 160)                      # GRAY = NOT spill-judged (no 2D detection here)
                else:
                    col = (0, 200, 0) if (k < inm.size and inm[k]) else (0, 0, 255)
                cv2.circle(right, (int(u[k]), int(v[k])), 1, col, -1)
            cu, cv2v = int(np.median(u)), int(np.median(v))
            iou = float(d.get("iou", 0.0))
            _judged = bool(d.get("judged", det))
            if not _judged:                                    # NOT judged this frame -> say WHY
                lab = (f"m{d['mask']} iou={iou*100:.0f}% spill={d.get('spill',0)*100:.0f}% "
                       f"(skip:{d.get('skip_why','?')})"); lcol = (200, 200, 200)
            else:
                hot = d["spill"] >= thr
                lab = (f"m{d['mask']} iou={iou*100:.0f}% spill={d['spill']*100:.0f}%"
                       f"{' EVICT' if d.get('evicted') else ''}")
                lcol = (0, 0, 255) if hot else (0, 255, 255)
            cv2.putText(right, lab, (cu, max(cv2v, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol, 2, cv2.LINE_AA)
        # stack to a common height
        h = 520
        lw = int(left.shape[1] * h / left.shape[0]); rw = int(right.shape[1] * h / right.shape[0])
        left = cv2.resize(left, (lw, h)); right = cv2.resize(right, (rw, h))
        panel = np.hstack([left, right])
        cv2.putText(panel, f"frame {fid}  LEFT: GREEN=SAM mask BLUE=reproj plane YELLOW=overlap(IoU)  "
                           f"RIGHT: GREEN=on-mask RED=spill GRAY=not judged (skip:novis|cov|far|move)",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.reproject_panel_dir, f"reproject_{fid:06d}.png"), panel)

    def _render_mask_overlay(self, bgr, idx_list, masks_full):
        # Tint + outline each mask in its palette color; returns a bgr8 image.
        img = bgr.copy().astype(np.float32)
        for color_i, raw_si in enumerate([int(x) for x in idx_list]):
            mask = masks_full.get(int(raw_si))
            if mask is None or not mask.any():
                continue
            c_bgr = np.array(bsp.MASK_COLORS_BGR[color_i % len(bsp.MASK_COLORS_BGR)], dtype=np.float32)
            img[mask] = img[mask] * 0.55 + c_bgr * 0.45
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, tuple(float(x) for x in c_bgr.tolist()), 1)
        return np.ascontiguousarray(img.clip(0, 255).astype(np.uint8))

    def _publish_mask_overlay(self, publisher, src_header, bgr, idx_list, masks_full):
        img = self._render_mask_overlay(bgr, idx_list, masks_full)
        msg = Image()
        msg.header.stamp = src_header.stamp
        msg.header.frame_id = src_header.frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        publisher.publish(msg)

    def _publish_big_mask_overlay(self, src_header, bgr, big_idx, big_masks_full):
        self._publish_mask_overlay(self.pub_big_masks, src_header, bgr, big_idx, big_masks_full)

    def _publish_small_mask_overlay(self, src_header, bgr, small_idx, small_masks_full):
        self._publish_mask_overlay(self.pub_small_masks, src_header, bgr, small_idx, small_masks_full)

    # ---- async RAW-input saver: cloud_XXXXXX.ply + rgb_XXXXXX.png for batch replay ---- #
    def _write_input_conventions(self, out_dir):
        """Drop a CONVENTIONS.md in the input dir so another agent can decode the ply/rgb/depth/pose."""
        txt = (
            "# glass_killer input conventions\n\n"
            "Per-frame (XXXXXX = frame id), same instant:\n"
            "  cloud_XXXXXX.ply  raw 3D points (xyzrgb ascii; rgb is gray placeholder, ignore)\n"
            "  rgb_XXXXXX.png    equirectangular panorama, OpenCV BGR\n"
            "  depth_XXXXXX.png  per-pixel RANGE map for the pano, uint16 MILLIMETRES (0 = no point)\n"
            "  pose_XXXXXX.txt   sensor pose in map frame: px py pz qx qy qz qw\n\n"
            "FRAMES\n"
            "  Cloud PLY = VIEWER Z-UP (default --projection-coord-mode reference_viewer_zup):\n"
            "    +X right, +Y forward, +Z up, metres, origin = sensor.\n"
            "  Camera/optical frame (pano rays + fitting): +X right, +Y down, +Z forward.\n"
            "    cam = [x,-z,y] from viewer;  viewer = [x,z,-y] from cam.\n"
            "  Map frame (z-up), for poses:\n"
            "    world = ( viewer_zup(cam) @ R_STATIC.T + T_STATIC ) @ R_body.T + T\n"
            "    R_STATIC = Rz(-90 deg),  T_STATIC = [-0.12,-0.075,0.255]  (fixed mount)\n"
            "    R_body = quat_to_R(qx,qy,qz,qw),  T = [px,py,pz]  (from pose file)\n\n"
            "PANORAMA (rgb + depth), W x H (e.g. 1920x640):\n"
            "  alpha = (u - W/2 - 1)*2*pi/W   (azimuth, +u -> right/+X)\n"
            "  beta  = (v - H/2 - 1)*2*pi/W   (elevation, +v -> down/+Y)\n"
            "  ray_cam = [cos(beta)*sin(alpha), sin(beta), cos(beta)*cos(alpha)]  (unit)\n"
            "  centre pixel -> +Z (forward).\n\n"
            "DEPTH VALUE = RANGE (Euclidean distance from sensor along the ray), NOT Z-depth.\n"
            "  range_m = depth_png[v,u]/1000.0            # 0 = no point\n"
            "  point_cam = range_m * ray_cam(u,v)         # +X right, +Y down, +Z fwd\n"
            "  point_viewer = [point_cam[0], point_cam[2], -point_cam[1]]  # back to z-up cloud frame\n"
        )
        try:
            with open(os.path.join(out_dir, "CONVENTIONS.md"), "w") as f:
                f.write(txt)
        except Exception as e:
            self.get_logger().warn(f"[input-save] could not write CONVENTIONS.md: {e}")

    def _input_saver_worker(self, q=None):
        q = q if q is not None else self._input_q
        while rclpy.ok():
            got = q.get()
            if got is None:
                break
            fid, xyz_raw, bgr, pose7, depth_m, outdir = got
            try:
                xyz = np.asarray(xyz_raw, np.float32)
                rgb = np.full((xyz.shape[0], 3), 128, np.uint8)   # placeholder color; batch recomputes
                bsp.write_ply_xyzrgb_ascii(
                    os.path.join(outdir, f"cloud_{fid:06d}.ply"), xyz, rgb)
                cv2.imwrite(os.path.join(outdir, f"rgb_{fid:06d}.png"), bgr)
                if depth_m is not None:                           # per-pixel RANGE (m) -> uint16 PNG in MILLIMETRES
                    dmm = np.clip(np.nan_to_num(np.asarray(depth_m, np.float32), nan=0.0) * 1000.0,
                                  0, 65535).astype(np.uint16)     # 0 = no cloud point at that pixel
                    cv2.imwrite(os.path.join(outdir, f"depth_{fid:06d}.png"), dmm)
                if pose7 is not None:                             # px py pz qx qy qz qw (sensor in map)
                    with open(os.path.join(outdir, f"pose_{fid:06d}.txt"), "w") as pf:
                        pf.write(" ".join(f"{v:.6f}" for v in pose7) + "\n")
            except Exception as e:
                self.get_logger().error(f"[input-save] frame {fid} failed: {e}")
            finally:
                q.task_done()

    # ---- async debug-visual saver (runs on its own thread, off the algorithm) ---- #
    def _debug_saver_worker(self):
        while rclpy.ok():
            item = self._save_q.get()
            if item is None:
                break
            try:
                self._save_debug_frame(item)
            except Exception as e:
                self.get_logger().error(f"[debug-save] frame {item.get('fid')} failed: {e}")
            finally:
                self._save_q.task_done()

    def _topdown_horiz_saver_worker(self):
        """Render + write the per-frame top-down HORIZONTAL-pillar PNG on its own thread (off the algorithm)."""
        while rclpy.ok():
            got = self._hz_q.get()
            if got is None:
                break
            fid, big_idx, horiz_by_mask, seed_records, mask_ray_records = got
            try:
                bsp._save_topdown_horizontal_pillars_image(
                    os.path.join(self._hz_dir, f"topdown_horiz_pillars_frame{fid:06d}.png"),
                    big_idx, horiz_by_mask, seed_records, mask_ray_records, self.args)
            except Exception as e:
                self.get_logger().warn(f"[topdown-horiz] frame {fid} failed: {e}", throttle_duration_sec=5.0)
            finally:
                self._hz_q.task_done()

    def _save_alignment_panel(self, path, ad):
        """3-panel pinhole alignment: LEFT = DA2 inference (dense pinhole depth, turbo); CENTER = BEFORE
        (front pinhole RGB dimmed + raw last-1s lidar depth as grid dots); RIGHT = AFTER the solved pose.
        Title carries score, the 6-DOF pose, and the alignment inference time (ms)."""
        rgb = np.ascontiguousarray(ad["rgb"])
        cell = int(ad.get("cell", 8))
        da2 = ad.get("da2")
        db, da = ad["lidar_before"], ad["lidar_after"]
        # shared colour scale from the valid BEFORE depths (fallback to AFTER), so both panels compare
        valid = db[db > 0.01]
        if valid.size < 8:
            valid = da[da > 0.01]
        lo, hi = (float(np.percentile(valid, 2)), float(np.percentile(valid, 98))) if valid.size else (0.5, 8.0)
        hi = max(hi, lo + 1e-3)

        def _dots(depth):
            vis = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
            gd, gv = pda._build_grid_fast(depth, cell)
            rr, cc = np.where(gv)
            for r, c in zip(rr.tolist(), cc.tolist()):
                t = int(np.clip((gd[r, c] - lo) / (hi - lo), 0.0, 1.0) * 255)
                col = cv2.applyColorMap(np.array([[t]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
                cv2.circle(vis, (c * cell + cell // 2, r * cell + cell // 2),
                           max(2, cell // 2 - 1), (int(col[0]), int(col[1]), int(col[2])), -1)
            return vis

        def _dense(depth):                                # DA2: full dense depth map as a turbo colormap
            h, w = rgb.shape[:2]
            vis = np.zeros((h, w, 3), np.uint8)
            if depth is None:
                return vis
            dv = np.asarray(depth, np.float32)
            m = dv > 0.01
            if m.any():
                v = dv[m]; lo2, hi2 = float(np.percentile(v, 2)), float(np.percentile(v, 98))
                hi2 = max(hi2, lo2 + 1e-3)
                t = np.clip((dv - lo2) / (hi2 - lo2), 0.0, 1.0)
                cm = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                vis[m] = cm[m]
            return vis

        def _label(img, txt):
            cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        p = ad["pose"]
        left = _label(_dense(da2), "DA2 inference (pinhole depth)")
        center = _label(_dots(db), "BEFORE (last-1s scan)")
        right = _label(_dots(da),
                       f"AFTER {'APPLIED' if ad['applied'] else 'SKIP'} score={ad['score']:.3f} "
                       f"{ad['ms']:.0f}ms  r=({p['rx']:+.2f},{p['ry']:+.2f},{p['rz']:+.2f}) "
                       f"t=({p['tx']:+.3f},{p['ty']:+.3f},{p['tz']:+.3f})")
        div = np.full((left.shape[0], 4, 3), (90, 90, 90), np.uint8)
        cv2.imwrite(path, np.hstack([left, div, center, div, right]))

    def _save_floor_planes_png(self, it, out_dir):
        """GLOBAL (world-frame) accumulated map PNG: WHOLE floor accumulator auto-scaled to fit --
        GREEN floor / PURPLE determined-nonfloor / ORANGE obstacle + global tracks + robot path."""
        fid = int(it["fid"])
        try:
            snap = it.get("map_snap")
            if snap is None:
                return
            _gp = bsp._render_global_floor_map(snap, it.get("tracks") or [])
            cv2.imwrite(os.path.join(out_dir, f"floor_planes_frame{fid:06d}.png"), _gp)
        except Exception as e:
            self.get_logger().warn(f"[keysave] global map render failed for frame {fid}: {e}")

    def _save_debug_frame(self, it, base_dir=None):
        """Write per-frame debug files. keysave_full:=true (DEFAULT) saves EVERY output
        (masks/curves/DA2/tries/occlusion/...). keysave_full:=false saves only the minimal
        plane-compete set: top-down clench PNG (+snapshot-root copy), seed-points PLY and
        accumulated-floor PLY."""
        a = self.args
        fid = int(it["fid"])
        frame_dir = os.path.join(base_dir or self.debug_save_dir, f"frame_{fid:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        bgr = it["bgr"]
        full = bool(getattr(self, "keysave_full", True))
        # WORLD-coordinate debug plys: the batch's _save_points_ply SKIPS entirely when
        # _WORLD_SAVE_POSE is unset (debug50 world-save contract). Set it from THIS frame's
        # captured pose so every custom ply (seeds/scene/floor/rays/planes/tries/...) is
        # written, in map coordinates, exactly like the batch replays.
        bsp._WORLD_SAVE_POSE = it.get("pose")
        if bsp._WORLD_SAVE_POSE is None:
            self.get_logger().warn(f"[keysave] frame {fid}: no pose captured -- debug PLYs "
                                   f"will be SKIPPED (pngs only)")
        scene_depth = None
        if full:
            jq = int(getattr(a, "jpeg_quality", 92))
            # Occlusion is OFF in the live path, so scene_depth may be None. The occlusion-based debug
            # outputs need it, so rebuild it here (on the saver thread) from the stashed ui/vi/rr.
            scene_depth = it.get("scene_depth")
            if scene_depth is None and it.get("rr") is not None:
                scene_depth = bsp._build_scene_depth_map(it["ui"], it["vi"], it["rr"], it["W"], it["H"])

            # CORNER RAY POINTS overlay: each big mask's 4 quad corners drawn in the MASK's
            # color (filled dots + connecting quad), off-image corners clamped to the border
            # with an annotation so degenerate quads are visible instead of silently absent.
            try:
                self._save_corner_ray_points(
                    os.path.join(frame_dir, f"corner_rays_frame{fid:06d}.png"),
                    bgr, it["big_masks_full"], it["mask_ray_records"])
            except Exception as e:
                self.get_logger().warn(f"[keysave] corner-ray overlay failed for frame {fid}: {e}")
            # big-mask overlay PNG (masks_frame*, same as the batch)
            bsp._save_big_mask_debug_image(
                os.path.join(frame_dir, f"masks_frame{fid:06d}.png"),
                bgr, it["big_masks_full"], it["mask_ray_records"], door_declared=it.get("door_declared"))
            # CURVE-REPAIR overlay (replaces the old corner_move/ irregularity debug): per big mask the
            # two through-corner great circles drawn through the FINAL corners + the repair verdict.
            try:
                self._save_curve_repair_overlay(
                    os.path.join(frame_dir, f"curve_repair_frame{fid:06d}.png"),
                    bgr, it["big_masks_full"], it["mask_ray_records"])
            except Exception as e:
                self.get_logger().warn(f"[keysave] curve-repair overlay failed for frame {fid}: {e}")
            # small-mask overlay PNG (matches the /glass_killer/small_masks topic)
            cv2.imwrite(os.path.join(frame_dir, f"small_masks_frame{fid:06d}.png"),
                        self._render_mask_overlay(bgr, it["small_idx"], it["small_masks_full"]))
            # DA2 pinhole<->lidar alignment BEFORE|AFTER panel (original vs transformed lidar depth dots).
            if it.get("align_debug") is not None:
                try:
                    self._save_alignment_panel(
                        os.path.join(frame_dir, f"pinhole_align_frame{fid:06d}.png"), it["align_debug"])
                except Exception as e:
                    self.get_logger().warn(f"[align-panel] frame {fid} failed: {e}")
            # DA2 depth + jump overlay images, and the DA2 jump points PLY
            bsp._save_da2_debug_outputs(frame_dir, fid, bgr, it["depth_da2"], it["jump_dil_full"], jq)
            bsp._save_da2_jump_points_ply(frame_dir, fid, it["pc_xyz"], it["ui"], it["vi"], it["jump_dil_full"])
            # downscaled scene cloud (the same subsample the top-down uses), gray, camera->viewer
            _scene = it.get("scene_sub")
            if _scene is not None and len(_scene) > 0:
                bsp._save_points_ply(os.path.join(frame_dir, f"scene_downscaled_frame{fid:06d}.ply"),
                                     np.asarray(_scene, np.float32), np.full((len(_scene), 3), 160, np.uint8))
            # seed points PLY (owner-colored)
            bsp._save_seed_points_ply(
                os.path.join(frame_dir, f"seed_points_frame{fid:06d}.ply"), it["seed_records"])
        # accumulated near-camera floor: GREEN = kept (gate verdicts against), RED = REMOVED (seed-clear or
        # occupancy dropped it). Same removal the gate applies, so the PLY shows what got cleared.
        _floor = it.get("floor_gate_xyz")
        if full and _floor is not None and len(_floor) > 0:
            _floor = np.asarray(_floor, np.float32)
            _seed_pts = (np.stack([np.asarray(r["pt"], np.float32) for r in it.get("seed_records", [])])
                         if it.get("seed_records") else np.empty((0, 3), np.float32))
            _fkeep = bsp._floor_gate_keep_mask(
                _floor, _seed_pts, float(getattr(a, "clench_floor_seed_clear_m", 0.25)), scene_xyz=it.get("pc_xyz"),
                occ_cell_m=float(getattr(a, "clench_floor_occ_cell_m", 0.2)),
                occ_above_min_m=float(getattr(a, "clench_floor_occ_above_min_m", 0.2)),
                occ_above_max_m=float(getattr(a, "clench_floor_occ_above_max_m", 2.0)),
                occ_min_pts=int(getattr(a, "clench_floor_occ_min_pts", 3)),
                occ_ground_pct=float(getattr(a, "clench_floor_occ_ground_pct", 90.0)))
            _frgb = np.where(_fkeep[:, None], np.array([0, 220, 0], np.uint8),
                             np.array([230, 0, 0], np.uint8)).astype(np.uint8)
            bsp._save_points_ply(os.path.join(frame_dir, f"floor_points_frame{fid:06d}.ply"), _floor, _frgb)
        # top-down clench PNG (in the frame folder, AND copied to the SNAPSHOT ROOT so one keysave
        # folder shows all 5 frames' top-downs side by side without opening each frame folder)
        _td_path = os.path.join(frame_dir, f"topdown_clench_frame{fid:06d}.png")
        bsp._save_topdown_seed_ray_clench_image(
            _td_path,
            big_idx=it["big_idx"], mask_ray_records=it["mask_ray_records"],
            seed_records=it["seed_records"], scene_xyz=it["scene_sub"], args=a,
            clench_by_mask=it["clench_by_mask"],
            floor_xyz=_floor)                              # per-cell WHITE=seed / CYAN=floor counts
        try:
            import shutil as _sh
            _sh.copyfile(_td_path, os.path.join(base_dir or self.debug_save_dir,
                                                f"topdown_clench_frame{fid:06d}.png"))
        except Exception as e:
            self.get_logger().warn(f"[keysave] topdown root copy failed for frame {fid}: {e}")
        # FLOOR-GRID + GLOBAL-PLANES PNG (same as the batch floor_planes_frame*.png). Saved in BOTH
        # full and minimal modes.
        self._save_floor_planes_png(it, frame_dir)
        if not full:                                       # MINIMAL keysave ends here
            return
        # ray PLY + plane PLY (reuses the precomputed clench, no recompute)
        bsp._save_clench_rays_and_planes(
            frame_dir=frame_dir, frame_id=fid, big_idx=it["big_idx"],
            mask_ray_records=it["mask_ray_records"], seed_records=it["seed_records"],
            args=a, clench_by_mask=it["clench_by_mask"])
        # occlusion visual: planes drawn solid, eating the RGB they occlude
        bsp._save_clench_occlusion_image(
            os.path.join(frame_dir, f"plane_occlusion_frame{fid:06d}.png"),
            bgr, it["clench_by_mask"], it["mask_ray_records"], scene_depth,
            it["W"], it["H"], a)
        # per-try clench dump: each pillar-pair try's plane PNG + PLY (+ occluded points)
        bsp._save_clench_debug_tries(
            frame_dir, fid, it["clench_by_mask"], it["mask_ray_records"],
            bgr, scene_depth, it["W"], it["H"], a)
        # txt log of every plane try: occlusion, params, coverage
        self._save_tries_log(os.path.join(frame_dir, f"plane_tries_frame{fid:06d}.txt"), it)
        # BEFORE|AFTER panels for the irregular-mask cleanup, at the snapshot ROOT (not per-frame) so all
        # verdicts sit together: link_break_masks/ (trimmed) + irregular_masks/ (discarded). Only masks the
        # cleanup actually touched are written -> no touched mask means the folders never get created.
        root = base_dir or self.debug_save_dir
        for (mid, before, after, is_irr) in it.get("irregular_fix", []):
            bsp._save_mask_cleanup_debug(bgr, it["W"], it["H"], before, after, is_irr, root, fid, mid)
        # mask corner-move FALLBACK: per-frame subfolder with a BEFORE|AFTER panel (raw vs moved 4 ray
        # corners) for each mask that used it -> frame_XXXXXX/mask_fallback/f{frame}_m{mask}.png. Folder is
        # only created if at least one mask in this frame used the fallback.
        bsp._save_corner_move_debug(bgr, it["W"], it["H"], it["clench_by_mask"], it["mask_ray_records"],
                                    frame_dir, fid, subdir="mask_fallback")
        # top-down HORIZONTAL pillars (bars), only when at least one big mask actually has a bar
        _hbm = it.get("horiz_by_mask") or {}
        if any(_hbm.get(int(b)) for b in it["big_idx"]):
            bsp._save_topdown_horizontal_pillars_image(
                os.path.join(frame_dir, f"topdown_horiz_pillars_frame{fid:06d}.png"),
                it["big_idx"], _hbm, it["seed_records"], it["mask_ray_records"], a)

    def _save_corner_ray_points(self, path, bgr, big_masks_full, mask_ray_records):
        """Per big mask: the 4 support-quad corners (TL,TR,BR,BL) as filled dots in the mask's
        color, connected as a quad. Off-image corners are clamped to the border and annotated
        with their true (u,v) so a degenerate/escaped quad is immediately visible."""
        H, W = bgr.shape[:2]
        vis = (bgr.astype(np.float32) * 0.55).astype(np.uint8)
        LBL = ("TL", "TR", "BR", "BL")
        for bi, rec in (mask_ray_records or {}).items():
            if rec is None:
                continue
            C = np.asarray(rec.get("corners_uv", []), np.float64).reshape(-1, 2)
            if C.shape[0] != 4:
                continue
            col = tuple(int(x) for x in np.asarray(rec["color_rgb"])[::-1])
            mk = (big_masks_full or {}).get(int(bi))
            if mk is not None and np.asarray(mk).any():
                cont, _ = cv2.findContours(np.asarray(mk, np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cont, -1, col, 1)
            P = []
            for k in range(4):
                u, v = float(C[k][0]), float(C[k][1])
                cu = int(np.clip(u, 4, W - 5)); cv_ = int(np.clip(v, 4, H - 5))
                off = (u < 0 or u >= W or v < 0 or v >= H)
                P.append((cu, cv_))
                cv2.circle(vis, (cu, cv_), 7 if off else 5, col, -1)
                if off:                                     # escaped corner: ring + true coords
                    cv2.circle(vis, (cu, cv_), 11, (0, 0, 255), 2)
                    cv2.putText(vis, f"{LBL[k]}({u:.0f},{v:.0f})", (cu + 8, cv_ - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(vis, f"m{int(bi)}{LBL[k]}", (cu + 6, cv_ - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                cv2.line(vis, P[a], P[b], col, 1, cv2.LINE_AA)
        cv2.imwrite(path, vis)

    def _save_curve_repair_overlay(self, path, bgr, big_masks_full, mask_ray_records):
        """Per big mask: mask contour + the top/bottom through-corner great circles (through the
        FINAL corners) + corners as dots + the curve-repair verdict. Verdict colors: normal
        (yellow/magenta) = clean, GREEN = this line was repaired, RED = nofix-both."""
        H, W = bgr.shape[:2]
        po = float(getattr(self.args, "pano_pixel_offset", 1.0))
        vis = (bgr.astype(np.float32) * 0.6).astype(np.uint8)

        def _curve(vis_, n, u0, u1, col, thick):
            us = np.arange(int(u0), int(u1) + 1, 2, dtype=np.float64)
            ny = float(n[1])
            if abs(ny) < 1e-8 or us.size < 2:
                return
            if bsp._PINHOLE is not None:
                # PINHOLE: the interpretation-plane trace n.ray=0 is a STRAIGHT image line
                # v = cy - fy*(n2 + n0*X)/n1  (matches the batch's pinhole edge-line math).
                _fx, _fy = bsp._PINHOLE["fx"], bsp._PINHOLE["fy"]
                _cx, _cy = bsp._PINHOLE["cx"], bsp._PINHOLE["cy"]
                X = (us - _cx) / _fx
                vs = _cy - _fy * (float(n[2]) + float(n[0]) * X) / ny
                ok = np.isfinite(vs) & (vs >= 0) & (vs < H)
                pts = np.stack([us, vs], 1)
                for s in np.split(np.arange(len(us)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1):
                    if len(s) > 1 and ok[s[0]]:
                        cv2.polylines(vis_, [np.round(pts[s]).astype(np.int32)], False, col, thick, cv2.LINE_AA)
                return
            th = (us - W / 2.0 - po) * (2.0 * np.pi / W)
            ph = np.arctan(-(float(n[0]) * np.sin(th) + float(n[2]) * np.cos(th)) / ny)
            vs = H / 2.0 + po + ph * W / (2.0 * np.pi)
            ok = np.isfinite(vs) & (vs >= 0) & (vs < H)
            pts = np.stack([us, vs], 1)
            for s in np.split(np.arange(len(us)), np.where(np.diff(ok.astype(int)) != 0)[0] + 1):
                if len(s) > 1 and ok[s[0]]:
                    cv2.polylines(vis_, [np.round(pts[s]).astype(np.int32)], False, col, thick, cv2.LINE_AA)

        for bi, mk in (big_masks_full or {}).items():
            rec = mask_ray_records.get(int(bi))
            if rec is None or mk is None or not np.asarray(mk).any():
                continue
            cr = rec.get("curve_repair") or {}
            status = str(cr.get("status", "off"))
            C = np.asarray(rec.get("corners_uv", []), np.float64).reshape(-1, 2)
            cont, _ = cv2.findContours(np.asarray(mk, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cb = tuple(int(x) for x in np.asarray(rec["color_rgb"])[::-1])
            cv2.drawContours(vis, cont, -1, cb, 1)
            if C.shape[0] != 4:
                continue
            ys, xs = np.where(np.asarray(mk, bool))
            u0, u1 = max(0, int(xs.min()) - 40), min(W - 1, int(xs.max()) + 40)
            r = [np.asarray(bsp._pano_pixel_to_unit_ray_cam(float(u), float(v), W, H, pixel_offset=po),
                            np.float64) for (u, v) in C]
            for (a, b, base_col, lbl) in ((0, 1, (0, 255, 255), "top"), (3, 2, (255, 0, 255), "bot")):
                n = np.cross(r[a], r[b]); nn = float(np.linalg.norm(n))
                if nn < 1e-9:
                    continue
                col, thick = base_col, 1
                if status == f"fixed-{lbl}":
                    col, thick = (80, 255, 80), 2                  # this line was solved/repaired
                elif status == "nofix-both":
                    col, thick = (0, 0, 255), 2
                _curve(vis, n / nn, u0, u1, col, thick)
            # SUPPORT-QUAD debug: the solid RANSAC edge segment that oriented the quad, in
            # ORANGE (thick) with ringed endpoints -- shows exactly WHICH pixels the support
            # construction trusted.
            seg = rec.get("pin_support_seg")
            if seg is not None:
                a = (int(round(seg[0][0])), int(round(seg[0][1])))
                b = (int(round(seg[1][0])), int(round(seg[1][1])))
                # the USED RANSAC support line: dark halo + thick orange so it stands out from
                # the thin curve traces; label says which mask edge d3 was derived from.
                cv2.line(vis, a, b, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.line(vis, a, b, (0, 165, 255), 3, cv2.LINE_AA)
                for q in (a, b):
                    cv2.circle(vis, q, 6, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(vis, q, 5, (0, 165, 255), 2, cv2.LINE_AA)
                _hs = rec.get("pin_hside") or "?"
                _lb = f"RANSAC sup:{_hs}"
                cv2.putText(vis, _lb, (a[0], max(14, a[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, _lb, (a[0], max(14, a[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 165, 255), 1, cv2.LINE_AA)
            Cp = rec.get("corners_uv_pre")                     # BEFORE-repair corners: gray + pull-line
            if Cp is not None:
                Cp = np.asarray(Cp, np.float64).reshape(-1, 2)
                for k in range(min(4, Cp.shape[0])):
                    q = (int(round(Cp[k][0])), int(round(Cp[k][1])))
                    p = (int(round(C[k][0])), int(round(C[k][1])))
                    if abs(q[0] - p[0]) + abs(q[1] - p[1]) > 2:
                        cv2.circle(vis, q, 3, (160, 160, 160), 1, cv2.LINE_AA)
                        cv2.line(vis, q, p, (160, 160, 160), 1, cv2.LINE_AA)
            for k in range(4):
                p = (int(round(C[k][0])), int(round(C[k][1])))
                cv2.circle(vis, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, p, 4, (0, 0, 0), 1, cv2.LINE_AA)
            _f = lambda x: "n/a" if x is None else f"{x*100:.0f}%"
            txt = f"b{int(bi)} {status} t={_f(cr.get('top'))} b={_f(cr.get('bot'))}"
            if cr.get("expanded_px"):
                txt += f" exp={cr.get('expanded_px', 0):.0f}px"
            if status.startswith("fixed"):
                txt += f" moved={cr.get('moved_px', 0):.0f}px"
            cv2.putText(vis, txt, (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "curve repair: yellow/magenta=clean  GREEN=repaired  RED=nofix-both  o=final corners  "
                         "ORANGE=used RANSAC support line",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(path, vis)

    def _save_tries_log(self, path, it):
        """Write a human-readable txt: the solver params, then per big mask every pillar-pair try
        with its occlusion %, parallelogram pass, occlusion pass, and seed-coverage score."""
        a = self.args
        pnames = ["clench_max_occ_frac", "clench_occ_expand_frac", "clench_random_tries",
                  "clench_side_diff_frac", "clench_vert_stretch_frac", "clench_seed_cover_tol_m",
                  "clench_bin_m", "clench_layer_m", "clench_min_layers",
                  "clench_floor_touch_reject_frac", "clench_floor_touch_radius_m",
                  "clench_diag_robust_quad", "clench_robust_min_az_deg",
                  "clench_pillar_min_fill", "clench_pillar_max_xz_ext_m",
                  "clench_h_clamp_height_floor_m"]
        lines = [f"frame {int(it['fid']):06d}", "params:"]
        lines += [f"  {p} = {getattr(a, p, None)}" for p in pnames]
        # Doorway audit: DECLARED masks are openings (never solved); SKIPPED masks had pillars on
        # doorway-locked world cells (removed from big_idx BEFORE clench) -- both are why a mask
        # can appear in masks_frame*.png but be absent from the topdown / tries below.
        _dd, _ds = sorted(it.get("door_declared") or []), sorted(it.get("door_skipped") or [])
        if _dd or _ds:
            lines.append(f"doorway: declared={_dd}  lock-skipped={_ds}")
        for bi, (res, info) in it["clench_by_mask"].items():
            rec = it["mask_ray_records"].get(int(bi))
            crgb = np.asarray(rec["color_rgb"], np.uint8).reshape(3) if rec is not None else np.zeros(3, np.uint8)
            reason = (info or {}).get("reason", "no result")
            cr = rec.get("curve_repair") if rec is not None else None
            crs = ""
            if cr is not None:
                _f = lambda x: "n/a" if x is None else f"{x*100:.0f}%"
                crs = (f"  curve[{cr.get('status')} top={_f(cr.get('top'))} bot={_f(cr.get('bot'))}"
                       + (f" moved={cr.get('moved_px', 0):.0f}px" if str(cr.get('status', '')).startswith('fixed') else "")
                       + "]")
            lines.append(f"\nmask {int(bi)} RGB({int(crgb[0])},{int(crgb[1])},{int(crgb[2])})  "
                         f"placed={res is not None}  {reason}{crs}")
            # Show every plane try with its OWN dv/dh + coverage. Sort so the ACCEPTED plane (the
            # highest-coverage passer -- the one actually placed) is listed FIRST, then the rest.
            tries = list((info or {}).get("debug_tries", []) or [])
            tries.sort(key=lambda t: (0 if t.get("passed") else 1, -float(t.get("cov", 0.0))))
            marked = False
            for t in tries:
                C = np.asarray(t["corners"], np.float32)
                w = float(np.linalg.norm(C[0] - C[1])); h = float(np.linalg.norm(C[0] - C[3]))
                occ = t.get("occ", float("nan")); o1 = t.get("occ1", float("nan")); o2 = t.get("occ2", float("nan"))
                jj = int(t.get("j", 0))
                lbl = (f"HORIZ({t['i']})" if jj == -3 else
                       (f"BORROW m{t.get('borrow_mask', -1)}" if jj < 0 else f"pair({t['i']},{t['j']})"))
                ps = t.get("par_sides")                       # sides of the shape the gate actually tested
                pr = t.get("par_robust")                      # robust-diagonal gate: per-corner residuals
                if pr is not None and not pr.get("degen") and pr.get("res"):
                    _seg = ",".join(
                        (f"{r_['corner']}:no-touch" if not r_.get("touched", False) else
                         f"{r_['corner']}:dh={r_['dh']*100:.0f}%/dw={r_['dw']*100:.0f}%{'' if r_['ok'] else '!'}")
                        for r_ in pr["res"])
                    _out = pr.get("outlier")
                    _st = f" az={pr['az_span_deg']:.0f}deg STRICT" if pr.get("strict") else ""
                    par_str = (f"robust[{_seg} {int(pr.get('n_ok', 0))}/2"
                               f"{' outlier=' + _out if _out else ''}{_st}]")
                elif pr is not None:
                    par_str = "robust[DEGENERATE]"
                elif ps is not None:
                    vL_, vR_, hT_, hB_ = [float(x) for x in ps]
                    dv = abs(vL_ - vR_) / max(vL_, vR_, 1e-9) * 100.0
                    dh = abs(hT_ - hB_) / max(hT_, hB_, 1e-9) * 100.0
                    par_str = f"dv={dv:4.0f}% dh={dh:4.0f}%"
                else:
                    par_str = "dv= n/a dh= n/a"
                mark = ""
                if res is not None and t.get("passed") and not marked:
                    mark = "  <== ACCEPTED"; marked = True
                _ff = t.get("floor_frac", None)                # floor-touch coverage of this try's footprint
                floor_str = f"  floor={_ff * 100:3.0f}%" if _ff is not None else "  floor= n/a"
                _cl = t.get("cov_local")                       # long-bar azimuth-windowed coverage
                if _cl is not None:
                    floor_str += f"  covL={_cl * 100:3.0f}%"
                _px = t.get("par_px")                          # pixel-space par defect (companion metric)
                if _px is not None:
                    par_str += f" px={_px:.0f}"
                _why = f"  why[{t['why']}]" if t.get("why") else ""
                lines.append(
                    f"  {lbl:<16s}  cov={float(t.get('cov', 0.0)) * 100:5.1f}%  {par_str}  "
                    f"par_ok={str(bool(t.get('par_ok'))):5s}  passed={str(bool(t.get('passed'))):5s}{_why}  "
                    f"plane={w:.2f}x{h:.2f}m  occ={occ * 100:5.1f}%(f={o1 * 100:.0f}% b={o2 * 100:.0f}%)"
                    f"{floor_str}{mark}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ---- keyboard control (own thread, needs a TTY) ---- #
    #   s          -> dump the last 5 inferences' FULL debug
    #   SPACE      -> pause/resume the rosbag (rosbag2 player TogglePaused service)
    #   UP / DOWN  -> rosbag playback faster / slower (SetRate service)
    def _key_listener(self):
        import sys, select
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
        except Exception:
            self.get_logger().warn("[keys] no TTY on stdin; keyboard control disabled "
                                   "(run the node in a terminal to enable)")
            return
        self._bag_setup_clients()
        try:
            tty.setcbreak(fd)
            self.get_logger().info("[keys] s=save 5 frames | SPACE=pause/resume bag | UP/DOWN=faster/slower")
            while rclpy.ok():
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("s", "S"):
                    self._trigger_snapshot_save()
                elif ch == " ":
                    self._bag_toggle_pause()
                elif ch == "\x1b":                        # ESC -> maybe an arrow escape sequence
                    seq = ""
                    if select.select([sys.stdin], [], [], 0.02)[0]:
                        seq = sys.stdin.read(2)
                    if seq == "[A":
                        self._bag_change_rate(1.5)        # UP -> faster
                    elif seq == "[B":
                        self._bag_change_rate(1.0 / 1.5)  # DOWN -> slower
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    def _bag_setup_clients(self):
        """Service clients that drive the rosbag2 player (bag runs with --disable-keyboard-controls,
        so the node is the single keyboard owner and forwards pause/rate to it)."""
        self._bag_rate = 1.0
        self._bag_toggle_cli = self._bag_rate_cli = None
        try:
            from rosbag2_interfaces.srv import TogglePaused, SetRate
            player = str(self.declare_parameter("bag_player_node", "/rosbag2_player").value)
            self._bag_toggle_cli = self.create_client(TogglePaused, f"{player}/toggle_paused")
            self._bag_rate_cli = self.create_client(SetRate, f"{player}/set_rate")
            self._SetRate = SetRate
        except Exception as e:
            self.get_logger().warn(f"[keys] rosbag2 control unavailable ({e}); SPACE/arrows disabled")

    def _bag_toggle_pause(self):
        from rosbag2_interfaces.srv import TogglePaused
        if self._bag_toggle_cli is None or not self._bag_toggle_cli.service_is_ready():
            self.get_logger().warn("[keys] bag player not up yet; SPACE ignored", throttle_duration_sec=3.0)
            return
        self._bag_toggle_cli.call_async(TogglePaused.Request())
        self.get_logger().info("[keys] SPACE -> toggled bag pause")

    def _bag_change_rate(self, factor):
        if self._bag_rate_cli is None or not self._bag_rate_cli.service_is_ready():
            self.get_logger().warn("[keys] bag player not up yet; rate change ignored",
                                   throttle_duration_sec=3.0)
            return
        self._bag_rate = float(np.clip(self._bag_rate * factor, 0.1, 10.0))
        req = self._SetRate.Request(); req.rate = self._bag_rate
        self._bag_rate_cli.call_async(req)
        self.get_logger().info(f"[keys] bag rate -> {self._bag_rate:.2f}x")

    def _trigger_snapshot_save(self):
        snap = list(self._recent)                     # shallow copy of the rolling last-5 buffer
        if not snap:
            self.get_logger().warn("[keys] no inferences buffered yet")
            return
        self._snap_counter += 1
        try:
            self._snap_q.put_nowait((self._snap_counter, snap))
            self.get_logger().info(f"[keys] s -> snapshot #{self._snap_counter} "
                                   f"({len(snap)} frames, full debug)")
        except queue.Full:
            self.get_logger().warn("[keys] previous snapshot still saving; ignored this 's'")

    def _snapshot_saver_worker(self):
        """SPACE snapshots: save all 5 buffered frames' FULL debug set into one keysave dir."""
        while rclpy.ok():
            got = self._snap_q.get()
            if got is None:
                break
            snap_id, items = got
            try:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_dir = os.path.join(self._keysave_dir, f"keysave_{stamp}_{snap_id:03d}")
                os.makedirs(out_dir, exist_ok=True)
                for it in items:
                    self._save_debug_frame(it, base_dir=out_dir)
            except Exception as e:
                self.get_logger().error(f"[keysave] snapshot #{snap_id} failed: {e}")
            finally:
                self._snap_q.task_done()

    def _auto_map_worker(self):
        """AUTOMATIC per-inference global-map PNG: save the seed-vs-floor tug map + the accumulating
        plane_tracker_log.txt to auto_map_dir. Runs from the first inference on, no key press."""
        while rclpy.ok():
            item = self._auto_q.get()
            if item is None:
                break
            try:
                self._save_floor_planes_png(item, self.auto_map_dir)
                self._append_tracker_log(item, self.auto_map_dir)
            except Exception as e:
                self.get_logger().error(f"[auto-map] save failed: {e}")
            finally:
                self._auto_q.task_done()

    def _append_tracker_log(self, it, out_dir):
        """Append this inference's plane-tracker decisions to <out_dir>/plane_tracker_log.txt: the
        per-plane compete rows (mask, colour, compete-rule, VS, seeds, RESULT) and every PRUNE /
        GREEN-HIT line -- so the floor % that removed a plane is recorded next to each removal."""
        logs = it.get("tracker_logs")
        if not logs:
            return
        fid = int(it["fid"])
        try:
            with open(os.path.join(out_dir, "plane_tracker_log.txt"), "a") as f:
                f.write("=" * 78 + "\n")
                f.write(f"  FRAME {fid:06d}   tracks={len(it.get('tracks') or [])}\n")
                f.write("=" * 78 + "\n")
                for ln in logs:
                    f.write(ln + "\n")
                f.write("\n")
        except Exception as e:
            self.get_logger().warn(f"[keysave] tracker-log append failed for frame {fid}: {e}")


def main():
    rclpy.init()
    node = GlassKillerNode()
    # SINGLE-THREADED executor, deliberately. A MultiThreadedExecutor was tried to stop the pose
    # feed starving during inference (it did: pose_gap 55ms -> 15ms). But with the sensor callbacks
    # on separate threads the cloud and image streams advance INDEPENDENTLY, so a _tick could pair
    # a cloud from one instant against an image buffer that had already moved past it -- during
    # rotation the whole seed projection came out lagged and yawed. Serial draining keeps the
    # streams in step, which matters more than the pose gap.
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_keepwarm = True          # stop the GPU keep-warm promptly (GPU idles back down)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()




# # terminal 1 — the C++ provider (5-sec stacked cloud + rgb + pose, no disk writes)
# ros2 launch extrinsic_latency_calib glass_killer.launch

# # terminal 2 — the algorithm node (int8 SAM 2816, clench, publish-only)
# source /opt/ros/jazzy/setup.bash
# source <ROS_WS>/install/setup.bash
# 


# source /opt/ros/jazzy/setup.bash
# source <ROS_WS>/install/setup.bash
# conda run -n sam3 --no-capture-output python ./glass_killer_ros_node.py --ros-args -p use_da2:=false -p save_input:=true

# run glass killer:
# source /opt/ros/jazzy/setup.bash
# source <ROS_WS>/install/setup.bash
# conda run -n sam3 --no-capture-output python ./glass_killer_ros_node.py --ros-args -p use_da2:=false
