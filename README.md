# GlassGuard — code

Anonymous code release accompanying the paper *GlassGuard: Verified Glass Plane Mapping for Robot
Navigation* (under review). Project page: https://glassguardproject.github.io/

## Layout

| Path | Contents |
|---|---|
| `glassguard_node/glassguard_node.py` | ROS 2 node. Runs as two processes: `role:=perception` (Slim SAM3 detection + pillar construction + ray-cast orientation verification) and `role:=mapping` (global plane manager: merging, seed–floor evidence, multi-view verification, planner feed). `pipeline_transport.py` carries geometry between them. |
| `glassguard_core.py` | The algorithm library used by the node (candidate construction, angle gate, tracker) and an offline replay driver for recorded inputs. |
| `glass_frame_ring.py`, `glassguard_deterministic.py`, `debug_combined_frame.py`, `train_normal_head.py`, `plane_predict.py` | Helper modules imported by the core (silhouette/frame-ring geometry, deterministic placement, per-frame debug rendering, optional learned normal head). |
| `pinhole_da2_align.py` | Optional pinhole depth-prior alignment (off in the evaluated configuration). |
| `run_glassguard.sh` | Launcher: bag playback + autonomy stack + provider + GlassGuard. `METHOD=360|pinhole`, `RANGE_M`, `VIZ_FULL`, `VIZ_REC`, ablation switches (`PAR_CHECK`, `REPROJECT_EVICT`, `FLOOR_EVICT`, `TRACK_MERGE`). |
| `ros/extrinsic_latency_calib/` | The LiDAR/camera provider node (`glassGuardProvider`, launched by `glassguard.launch`) (registered-scan stack, de-rotation with the exact cloud pose, `/glassguard/cloud`). |
| `tools/` | Recording (`capture_input_node.py`), evaluation (`eval_occupancy.py`, `eval_abl_scene.sh`), batch experiment drivers (`run_main_rerecord.sh`, `run_pin_ablation.sh`), demo-video capture (`record_viz.sh`, `encode_viz.sh`, `pick_regions.py`), protection maps. |
| `slim_sam3/` | Slim SAM3: confidence-guided Taylor pruning, distillation fine-tune, loader, VRAM benchmark, and the pruned-channel metadata (`mlp_pruned_meta.json`). Weights are not included (2.7 GB); see below. |
| `rviz/` | RViz layout for the full-visual demo. |

## Paths

Machine-specific paths were replaced by environment variables (defaults in parentheses):

* `GG_DATA_ROOT` — recordings, ground truth and evaluation outputs (`~/glassguard_data`)
* `GG_ROS_WS` — the ROS 2 workspace containing `ros/extrinsic_latency_calib` (`~/ros_ws`).
  Build it with `colcon build --packages-select extrinsic_latency_calib` from a shell where
  conda is **not** active: a conda `libcurl` on the library path breaks the PCL/GDAL link step.
* `GG_AUTONOMY_STACK` — the LiDAR autonomy stack (`~/autonomy_stack`). See *Input interface* below;
  any stack that publishes the listed topics can be used.
* `GG_BASELINES`, `GG_DA2` — baseline checkouts, used only for the baseline comparisons

Scene identifiers in the scripts (`bldgA_f5`, `bldgB_atrium`, …) match the project page.

## Input interface

GlassGuard does not read raw sensors. The provider node (`glassGuardProvider`) subscribes to
three topics that a LiDAR SLAM / autonomy stack is expected to publish:

| Topic | Type | Contents |
|---|---|---|
| `/registered_scan` | `sensor_msgs/PointCloud2` | the current LiDAR scan registered into the world (`map`) frame |
| `/state_estimation` | `nav_msgs/Odometry` | the robot pose in the same world frame, time-stamped with the scan |
| `/camera/image` | `sensor_msgs/Image` | the raw RGB image (the launcher republishes it from `/camera/image/compressed`) |

Optionally, with `TERRAIN_FLOOR=true`, the node also subscribes to `/terrain_map`
(`sensor_msgs/PointCloud2`, intensity = height above ground; points with intensity ≤ 0.1 m are
treated as floor). Set `TERRAIN_FLOOR=false` to run without it.

Message details the provider relies on:

* `/registered_scan` is read as `PointXYZI`; only `x, y, z` are used, but an `intensity` field
  must be present (points with intensity 199 are GlassGuard's own injected glass markers and are
  dropped on the way back in). Scans are stacked over `stackTimeWindow` (5 s) and voxelized.
* `/state_estimation` poses are matched to scans and images by **nearest timestamp** with no
  tolerance, so the odometry must be time-stamped on the same clock as the sensors.
* `/camera/image` is converted with `cv_bridge` to `bgr8`; any encoding `cv_bridge` can convert
  is accepted.

Any stack meeting this contract works. Our experiments used the TARE autonomy stack
(Cao et al., RSS 2021), as cited in the paper, which also supplies the local planner and terrain
analysis that consume the published planes.

### Camera and mounting

The provider is written for a 360° equirectangular camera and the evaluated values are set in
`ros/extrinsic_latency_calib/launch/glassguard.launch`: `imageWidth`/`imageHeight` (1920×640),
the camera-to-LiDAR extrinsic (`camX/camY/camZ`, `camRoll/camPitch/camYaw`), and
`imageLatencyOffset` (camera stamp lag, tune with `latencyCalib`). For another robot these
must be changed to match the mount. A pinhole camera is supported with `is360Cam:=false`
and its intrinsics `fx/fy/cx/cy` and distortion `k1/k2/p1/p2`; the pinhole path is the
`METHOD=pinhole` configuration of the paper.

## Dependencies

`slim_sam3/load_slim_sam3.py` imports the upstream `sam3` package (`sam3.model.*`), which is not
vendored here. Install the public SAM 3 release so that `sam3` is importable, then add this
repository to `PYTHONPATH`. `depth_anything_v2` (pointed at by `GG_DA2`) is likewise an external
checkout, used only by the pinhole depth-alignment path. The remaining requirements are the usual
ROS 2 / PyTorch stack: `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `scipy`,
`matplotlib`, and `open3d` for the offline evaluation and annotation scripts.

## Weights

The Slim-2816 student checkpoint and the cached text embedding are distributed separately
(too large for this repository); place them under `slim_sam3/checkpoints/` and
`slim_sam3/prompt_features/`. The pruning recipe in `slim_sam3/` reproduces the student from the
public SAM 3 release.

## Evaluated configuration

All thresholds are set in `run_glassguard.sh` and were frozen across the experiments
(`PIPELINE=true`, `PAR_CHECK=true`, `SPILL_VIS_MIN=0.5`, `SPILL_BASE_N=2`, `SPILL_PERSIST=2`,
`SPILL_HARD=0.55`, `RANGE_M=10`, or `20` for the two large outdoor scenes).
