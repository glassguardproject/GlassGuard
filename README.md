# GlassGuard — code

Anonymous code release accompanying the paper *GlassGuard: Verified Glass Plane Mapping for Robot
Navigation* (under review). Project page: https://glassguardproject.github.io/

## Layout

| Path | Contents |
|---|---|
| `glass_killer_pipeline/gk_node.py` | ROS 2 node. Runs as two processes: `role:=perception` (Slim SAM3 detection + pillar construction + ray-cast orientation verification) and `role:=mapping` (global plane manager: merging, seed–floor evidence, multi-view verification, planner feed). `pipeline_transport.py` carries geometry between them. |
| `batch_bigmask_4ray_randomopt.py` | The algorithm library used by the node (candidate construction, angle gate, tracker) and an offline replay driver for recorded inputs. |
| `pinhole_da2_align.py` | Optional pinhole depth-prior alignment (off in the evaluated configuration). |
| `run_glass_killer_full.sh` | Launcher: bag playback + autonomy stack + provider + GlassGuard. `METHOD=360|pinhole`, `RANGE_M`, `VIZ_FULL`, `VIZ_REC`, ablation switches (`PAR_CHECK`, `REPROJECT_EVICT`, `FLOOR_EVICT`, `TRACK_MERGE`). |
| `ros/extrinsic_latency_calib/` | The LiDAR/camera provider node (registered-scan stack, de-rotation with the exact cloud pose, `/glass_killer/cloud`). |
| `tools/` | Recording (`capture_input_node.py`), evaluation (`eval_occupancy.py`, `eval_abl_scene.sh`), batch experiment drivers (`run_main_rerecord.sh`, `run_pin_ablation.sh`), demo-video capture (`record_viz.sh`, `encode_viz.sh`, `pick_regions.py`), protection maps. |
| `slim_sam3/` | Slim SAM3: confidence-guided Taylor pruning, distillation fine-tune, loader, VRAM benchmark, and the pruned-channel metadata (`mlp_pruned_meta.json`). Weights are not included (2.7 GB); see below. |
| `rviz/` | RViz layout for the full-visual demo. |

## Paths

Machine-specific paths were replaced by environment variables (defaults in parentheses):

* `GG_DATA_ROOT` — recordings, ground truth and evaluation outputs (`~/glassguard_data`)
* `GG_ROS_WS` — the ROS 2 workspace containing `ros/extrinsic_latency_calib` (`~/ros_ws`)
* `GG_AUTONOMY_STACK` — the LiDAR autonomy stack (`~/autonomy_stack`)
* `GG_BASELINES`, `GG_DA2` — baseline checkouts, used only for the baseline comparisons

Scene identifiers in the scripts (`bldgA_f5`, `bldgB_atrium`, …) match the project page.

## Weights

The Slim-2816 student checkpoint and the cached text embedding are distributed separately
(too large for this repository); place them under `slim_sam3/checkpoints/` and
`slim_sam3/prompt_features/`. The pruning recipe in `slim_sam3/` reproduces the student from the
public SAM 3 release.

## Evaluated configuration

All thresholds are set in `run_glass_killer_full.sh` and were frozen across the experiments
(`PIPELINE=true`, `PAR_CHECK=true`, `SPILL_VIS_MIN=0.5`, `SPILL_BASE_N=2`, `SPILL_PERSIST=2`,
`SPILL_HARD=0.55`, `RANGE_M=10`, or `20` for the two large outdoor scenes).
