#!/bin/bash
# no-SHM transport: stale /dev/shm segments from killed nodes break later runs (see ~/.ros/fastdds_no_shm.xml)
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastdds_no_shm.xml
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastdds_no_shm.xml
# One-shot launcher for the FULL Glass Killer bag pipeline. The Glass Killer NODE starts FIRST (loads
# its models in the foreground); everything else (stack, provider, republish, bag) is brought up in
# parallel right after, so they load while the node loads.
#   node (foreground) + [stack -> provider -> republish -> bag play] in parallel
# Ctrl-C tears the whole group down (SIGINT trap + kill 0).
# ============================ KNOBS (edit these) ============================
METHOD=${METHOD:-glassrecon}  # WHICH method: 360 | pinhole | monoglass | glassrecon | monoglass_live | glassrecon_live
                      #   (env-overridable: METHOD=monoglass_live ./run_glass_killer_full.sh)
                      #   360/pinhole   -> the Glass Killer node runs LIVE inference on the bag
                      #   monoglass/glassrecon -> replays their PRECOMPUTED per-frame predictions
                      #     (preds_mg3d / preds_gr in SCENE_DIR) pose-synced to the bag, on the
                      #     same /glass_killer/planes rviz topic (they have no live pipelines)
SCENE_DIR=""          # scene dir for pinhole/monoglass/glassrecon; "" = auto:
                      # ${GG_DATA_ROOT:-$HOME/glassguard_data}/$(basename $BAG)_test
PINHOLE_CFG=""        # set automatically by METHOD=pinhole (or point at any camera_config.json)
USE_ALIGN=false       # DA2 pinhole<->lidar alignment          (node: use_pinhole_align)
USE_DA2=false    # DA2 depth-jump seed filtering           (node: use_da2)
PAR_CHECK=${PAR_CHECK:-true}   # the ANGLE gate (tau/length-adaptive facing check; falls back to dv/dh when
                   # the silhouette can't vouch). false = NO par gate at all (slanted planes admitted)
PAR_MIN_SPAN_PX=${PAR_MIN_SPAN_PX:-140.0}  # disarm span threshold (px at 520px/rad focal, angular-scaled on 360). MUST
                   # env-overridable: 999999.0 = angle gate OFF everywhere (dv/dh fallback only)
                   # have a decimal point (ROS DOUBLE param). 0.0 = NO disarm: any trusted support
                   # line always judges via the ANGLE test (no dv/dh fallback for trusted masks);
                   # 140.0 = the old conditioning disarm
DIR_MIN_COND=0.02  # min CONDITIONING (|ra x rb| * |n x g|) a support edge needs to be trusted:
                   # 0 at the CAMERA-HEIGHT degeneracy (eye-level line = no yaw info; frame80 b15
                   # occluder bottom line was 0.004, true top edge 0.23). Both camera models;
                   # pinhole tries the next edge, 360 picks best of top/bottom (no forced top)
H_EDGE_CURVE=false  # horizontal-bar: false = use ALL edge columns (recovers silhouette bars; true=restrict span)
H_RAY_HIT_TOL=0.0   # 0 = direction gate OFF (curve-repair + facing check own direction validation)
SAVE_INPUT=false       # dump raw cloud+rgb per frame (replayable)(node: save_input)
SAVE_OUTPUT=${SAVE_OUTPUT:-true}   # auto-save the global floor/plane map PNG (node: save_auto_map); env-overridable
MINIMAL_SAVE=false     # SPACE dump: true=minimal, false=full     (node: keysave_full = !MINIMAL_SAVE)
PRECISION=bf16        # bf16 | int8 | fp32                       (node: precision)
OBSTACLE_MODE=both     # scan | added | both | off (off = NO glass obstacle injection anywhere)
CONF_TH=0.3           # SAM3 detection confidence               (node: conf_th)
COV_TH=0.1           # min seed-coverage frac to place a plane  (node: min_cov; 0.0=pick top-cov, 0.20=old)
EMPTY_CACHE_EVERY=0  # torch empty_cache every N frames to cap VRAM (0=keep-warm/fast; 1=~1GB, slower)
MASK_CLAMP=false      # trim every placed plane to the 2D glass mask (all RViz plane topics show it)
DOORWAY_GATE=false    # OFF: no "open doorway" decode/lock/skip (the seed-vs-floor green-trim handles doorways)
FLOOR_CHECK=true      # global-map floor check: OFF = no floor accumulation / green cells / green-trim
TERRAIN_FLOOR=true    # floor evidence from the autonomy stack's /terrain_map (intensity<=0.1m = ground)
                      # instead of the SAM3 floor decode -> saves ~0.19s/frame; false = SAM3 floor
GROUNDING_CELL=6      # seed grid cell size (px): finer = more/tighter contact seeds (10=old, 6=finer)
SEED_GRID_RING=true   # seeds = ring of grid cells JUST OUTSIDE the mask (no pixel dilate/cut)
SEED_BAND_SYM=false   # OFF: outward-only band (symmetric showed flat-pane spill; revisit with depth filter)
                      # seeds; watch spill on flat-pane scenes (through-glass exposure)
SMALL_FALLBACK=true   # failed BIG mask retries with its largest owned SMALL masks (~5ms/frame)
QUAD_SUPPORT=true     # pinhole quads 360-style: solid edge orients (VP), MASK sizes (support lines)
EXTEND_FULL_MASK=false # widen solved quads over mask parts DROPPED by largest-component cleaning
                       # (thin mullion/occluder splits; gap+row-band guarded so separate panes
                       # across a real pillar are NOT merged). Geometry-only: fit/orientation untouched
SEED_RING_CELLS=1     # ring THICKNESS in grid cells (1 = tight clinging ring on the mask edge)
SEED_DILATE=2         # seed spread rings on the 0.10m tug grid: 2 rings = 0.2m pollute radius
SEED_FLOOR_VTOL=0.5   # a seed within this height (m) above a floor point keeps that cell SEED (low glass)
SAVE_LOCAL_TUG=false   # per-frame PNG of seed-vs-floor tug in a local window around the robot
REPROJECT_EVICT=${REPROJECT_EVICT:-true}   # SPILL-RISE eviction (coverage gate + high-water increase counter)
DEPTH_SWEEP_EVICT=false # multi-hypothesis depth sweep (OFF: spill-rise is the active depth evictor)
PATH_EVICT=${PATH_EVICT:-true}        # robot-path-through-plane eviction
FLOOR_EVICT=${FLOOR_EVICT:-true}       # green-floor eviction/trim
SPILL_COV_MIN=0.1     # judge only when >= this frac of footprint on mask (coverage); below it the tug count STAYS
SPILL_VIS_MIN=${SPILL_VIS_MIN:-0.5}  # judge only when >= this frac of the pane is IN VIEW; a correct pane leaving
                        # the view otherwise evicts itself (spill is measured over the visible sliver only)
SPILL_TUG_START=0.10   # EXCESS over the plane's own floor must exceed this before a RISE counts +1
SPILL_BASE_N=${SPILL_BASE_N:-2}   # well-observed checks to calibrate each plane's own under-segmentation spill floor (0 = no calibration)
SPILL_HARD=0.55       # judged spill >= this frac -> DIRECT evict (no baseline calibrate / no tug)
SPILL_PERSIST=2        # spill-TUG net count (rise=+1, fall=-1, min 0) to evict
SPILL_CHECK_MOVE=0.1  # robot movement (m) that paces ONE spill check
SPILL_THRESH=0.30      # (legacy hard-threshold: UNUSED by the rise rule; kept for arg compat)
SPILL_IOU_MIN=0.5      # IoU (plane footprint vs glass mask) a plane needs before its spill is judged
SPILL_IOU_GATE=${SPILL_IOU_GATE:-false}  # true = judge spill only when IoU >= SPILL_IOU_MIN (default false = coverage gate only)
SPILL_MAX=0.50         # (legacy immediate-evict: UNUSED by the rise rule)
SPILL_BASELINE=0.4     # (unused by the move-paced check; kept for reference)
PLANE_HEIGHT_SHRINK=0.05  # shrink each fitted plane's top+bottom inward by this frac (0.05 = 5% each end)
SPILL_PLANE_OCC=true   # spill test: a plane can be OCCLUDED by another plane in front of it (not just LiDAR)
CORNER_SEG_CURVE=true  # seed the corner curves from the occlusion-robust good edge segment (not raw silhouette corners)
RANGE_M=${RANGE_M:-10}     # sensing/placement range (m): RANGE_M=20 -> provider crop,
                           # placement + spill radii all follow (outdoor scenes)
SPILL_DIST_MAX=${RANGE_M}.0     # no spill comparison for planes farther than this (m) from the robot
PLACE_DIST_MAX=${RANGE_M}.0    # never place a plane farther than this (m) from the robot
OPT_DEPTH_MAX=$((RANGE_M+5)).0  # plane-fit depth ceiling; must cover PLACE_DIST_MAX (batch default 15)
DEPTH_MAX=20; [ "$RANGE_M" -gt 20 ] && DEPTH_MAX=$RANGE_M   # projection range crop; keep >=20 (node
                           # default) so short-range runs are unchanged, extend to RANGE_M for long ones
SPILL_DEBUG_DIR=${SPILL_DEBUG_DIR:-}   # non-empty -> dump per-evicted-plane spill-check history JSONs there
RECORD_RUN=${RECORD_RUN:-false}   # true -> CANONICAL run: dump trajectory.txt + scene_cloud.ply +
                       # planes_run.json (ever+evicted) into SCENE_DIR/canonical_run. Annotate GT
                       # against that scene_cloud; later runs auto-anchor GT to its trajectory.
RECORD_PLACED_INPUT=false  # ALSO dump the raw algorithm input (cloud PLY + rgb + depth + pose)
                       # on every plane-PLACEMENT frame -> canonical_run/placed_input. With many
                       # placements this is near per-frame recording -- keep OFF unless you need
                       # frame-exact replays; annotation/eval only needs the light artifacts above
FINAL_MAP=true         # publish a 2nd looser OVERVIEW map (/glass_killer/final_global_planes)
FINAL_SPILL_DIST=2.0   # overview map only spill-evicts within this radius (m); farther planes persist
FINAL_SPILL_THRESH=0.4 # overview map spill frac to count as a hit (higher = more tolerant than current)
SAVE_REPROJECT=false    # save per-frame panel: topdown(evicted red) | rgb+reprojection(spill)

# Which stages to launch (turn off any you start by hand)
LAUNCH_STACK=true
WITH_TARE=${WITH_TARE:-false}  # true -> stack launch includes TARE exploration planner
                               # (waypoints consume /added_obstacles; no pacing, no logger)
DRONE_BAND=${DRONE_BAND:-false} # true -> DRONE-LIKE traversability: terrain analysis judges the
                                # full vertical corridor, not a ~1m ground band. Restarts both
                                # terrain nodes with the band ceiling raised to DRONE_BAND_TOP.
DRONE_BAND_TOP=${DRONE_BAND_TOP:-2.5}  # band ceiling (m above sensor) for obstacle analysis
OCC_PLANNER=${OCC_PLANNER:-false}  # true -> 3D OCCUPANCY local planner (A* through free SPACE,
                                   # not over terrain): rolling voxel grid from /registered_scan
                                   # (glass injection included) -> nav_msgs/Path on /drone_path
FRONTIER_PLANNER=${FRONTIER_PLANNER:-false}  # true -> REPLACE TARE with the simple frontier
                                   # planner (kills tare_planner_node; frontier goals on
                                   # /way_point, cells on /frontier_cells). Frontiers form
                                   # THROUGH glass when occupancy is glass-blind
LAUNCH_PROVIDER=true
LAUNCH_REPUBLISH=true
PLAY_BAG=true
# ${GG_DATA_ROOT:-$HOME/glassguard_data}/bldgC_office



# BAG=./bldgB_int
# BAG=./bldgB_f2
# BAG=./bldgB_atrium
# BAG=${BAG:-./bldgA_ext_night}
# BAG=./bldgD_ext_night
#${GG_DATA_ROOT:-$HOME/glassguard_data}/bldgA_f5.mcap

# NOTE: BAG must be a RECORDING (ros2 bag dir/.mcap), NOT a *_test frames folder.
BAG=${BAG:-./bldgA_ext_night}
BAG_LOOP=false         # true -> --loop the bag
IMAGE_LATENCY=${IMAGE_LATENCY:-0.0}  # camera capture latency (s) used to de-rotate the stacked
                        # cloud. The camera header stamp lags true exposure, so at 0.0 the cloud is
                        # projected with a stale pose and the seeds/rays yaw while the robot turns.
                        # Sweep e.g. IMAGE_LATENCY=0.03; the provider logs the value at startup.
# Default rate depends on the mode: the VIZ_FULL demo plays at 0.5x (gives the per-frame renders
# and RViz time to keep up); EVERYTHING ELSE -- eval replays, live_abl_scene.sh, the TARE wrapper --
# plays at rosbag2's native 1.0x, exactly as the benchmark protocol did. Override with BAG_RATE=.
if [ "${VIZ_FULL:-false}" = "true" ]; then _BAG_RATE_DEF=0.5; else _BAG_RATE_DEF=""; fi
BAG_RATE=${BAG_RATE:-$_BAG_RATE_DEF}  # playback speed, e.g. 0.7 = 70% (empty -> rosbag2 default 1.0). Slowing the
                        # bag gives the per-frame renders time to keep up, so the RViz image panels
                        # stay lined up with the 3D view instead of lagging behind it.
# ===========================================================================

ROS_SETUP=/opt/ros/jazzy/setup.bash
STACK_DIR=${GG_AUTONOMY_STACK:-$HOME/autonomy_stack}
CAM_INSTALL=${GG_ROS_WS:-$HOME/ros_ws}/install/setup.bash
KEYSAVE_FULL=true; [ "$MINIMAL_SAVE" = "true" ] && KEYSAVE_FULL=false

# METHOD dispatch: resolve the scene dir and, for pinhole, the camera config.
# (strip any .mcap/_test from the bag name so both spellings resolve to the same scene)
SCENE_BASE=$(basename "$BAG"); SCENE_BASE=${SCENE_BASE%.mcap}; SCENE_BASE=${SCENE_BASE%_test}
[ -z "$SCENE_DIR" ] && SCENE_DIR=${GG_DATA_ROOT:-$HOME/glassguard_data}/${SCENE_BASE}_test
case "$METHOD" in
  360) ;;
  pinhole)
    [ -z "$PINHOLE_CFG" ] && PINHOLE_CFG=$SCENE_DIR/camera_config.json
    [ -f "$PINHOLE_CFG" ] || { echo "[launcher] METHOD=pinhole but no $PINHOLE_CFG"; exit 1; } ;;
  monoglass|glassrecon)
    [ -d "$SCENE_DIR" ] || { echo "[launcher] METHOD=$METHOD but no scene dir $SCENE_DIR"; exit 1; } ;;
  monoglass_live|glassrecon_live) ;;   # LIVE baseline inference: needs only the provider streams
  *) echo "[launcher] unknown METHOD=$METHOD (360|pinhole|monoglass|glassrecon|monoglass_live|glassrecon_live)"; exit 1 ;;
esac

# The screen recorder is a snap app: signals sent from a VS Code terminal can be refused (AppArmor).
# Stop it through its systemd scope when a plain signal does not work.
stop_snap_proc() { local p=$1 sc
  [ -n "$p" ] && [ -d "/proc/$p" ] || return 0          # (kill -0 can be refused too, so test /proc)
  kill -INT "$p" 2>/dev/null; for _ in 1 2 3 4 5 6; do [ -d "/proc/$p" ] || return 0; sleep 0.5; done
  sc=$(grep -oE "snap\.[^/]*\.scope" /proc/$p/cgroup 2>/dev/null | head -1)
  [ -n "$sc" ] && systemctl --user stop "$sc" 2>/dev/null; }
cleanup() { echo; echo "[launcher] Ctrl-C -> shutting everything down..."
            trap '' SIGINT SIGTERM; trap - EXIT     # survive our own 'kill 0' below; run once
            if [ -n "${VIZ_REC_DIR:-}" ]; then            # VIZ_REC: close the recording cleanly first
              date +%s.%N > "$VIZ_REC_DIR/screen_end.txt"
              stop_snap_proc "${VIZ_REC_FF:-}"
              [ -n "${VIZ_REC_SAVER:-}" ] && kill -INT "$VIZ_REC_SAVER" 2>/dev/null
              sleep 2
            fi
            kill 0 2>/dev/null
            pkill -f "system_bagfile.launch" 2>/dev/null
            pkill -f "system_bagfile_with_exploration_planner.launc[h]" 2>/dev/null
            pkill -f "tare_planner_nod[e]"  2>/dev/null
            pkill -f "glass_killer.launch"  2>/dev/null
            pkill -f "ros2 bag play"        2>/dev/null
            if [ -n "${VIZ_REC_DIR:-}" ] && [ "${VIZ_REC_ENCODE:-true}" = "true" ]; then
              echo "[viz-rec] encoding the panel videos (synced to screen.mp4) ..."
              ( trap - SIGINT SIGTERM; setsid ./tools/encode_viz.sh "$(basename "$VIZ_REC_DIR")" \
                  > "$VIZ_REC_DIR/encode.log" 2>&1 < /dev/null & )
              echo "[viz-rec] running in the background; log: $VIZ_REC_DIR/encode.log"
              echo "[viz-rec] videos will be in $VIZ_REC_DIR/"
            fi; }
trap cleanup SIGINT SIGTERM EXIT

# VIZ_REC=<name>  (with VIZ_FULL=true): ONE-TERMINAL demo recording. Before anything else starts,
#   record the whole screen to demo_rec/<name>/screen.mp4 and save every full-visual image stream with
#   its wall-clock arrival time. On Ctrl-C the recording is closed and the panel videos are encoded
#   so that each one starts and ends WITH screen.mp4 (tools/encode_viz.sh). VIZ_REC_SCREEN=false
#   skips the screen capture. Regions: run tools/pick_regions.py once and every later VIZ_REC run also
#   writes region1.mp4, region2.mp4, ... (VIZ_REC_FULL=false drops the full-screen file;
#   VIZ_REC_USE_REGIONS=false ignores the saved regions; VIZ_REC_REGION=WxH+X+Y gives one region inline);
#   VIZ_REC_ENCODE=false leaves the encoding for later.
VIZ_REC_DIR=""; VIZ_REC_FF=""; VIZ_REC_SAVER=""
if [ -n "${VIZ_REC:-}" ]; then
  [ "${VIZ_FULL:-false}" = "true" ] || { echo "[viz-rec] VIZ_REC needs VIZ_FULL=true"; exit 1; }
  VIZ_REC_DIR=./demo_rec/$VIZ_REC
  if [ -d "$VIZ_REC_DIR" ] && [ -n "$(ls -A "$VIZ_REC_DIR" 2>/dev/null)" ]; then
    echo "[viz-rec] $VIZ_REC_DIR already has data -- pick another VIZ_REC name"; VIZ_REC_DIR=""; exit 1
  fi
  mkdir -p "$VIZ_REC_DIR"
  if [ "${VIZ_REC_SCREEN:-true}" = "true" ]; then
    # ONE screen grab feeds every output (full screen + each picked region), so all of them share the
    # exact same first frame and timestamps. Regions come from VIZ_REC_REGION ("WxH+X+Y", one) or from
    # demo_rec/regions.txt written by tools/pick_regions.py (any number). VIZ_REC_FULL=false drops
    # the full-screen file when regions are used.
    _vr_disp=${DISPLAY:-:1}; _vr_regs=()
    if [ -n "${VIZ_REC_REGION:-}" ]; then _vr_regs=("$VIZ_REC_REGION")
    elif [ -s ./demo_rec/regions.txt ] && [ "${VIZ_REC_USE_REGIONS:-true}" = "true" ]; then
      while read -r _n _g; do [ -n "${_g:-}" ] && _vr_regs+=("$_g"); done < ./demo_rec/regions.txt
    fi
    # -g 30: a keyframe (= an mp4 fragment) every second, so stopping the recorder loses < 1 s of any output
    _vr_enc=(-c:v libx264 -preset veryfast -crf 18 -g 30 -color_range tv -colorspace bt709 -color_primaries bt709
             -color_trc bt709 -movflags +frag_keyframe+empty_moov)
    _vr_full=true; [ ${#_vr_regs[@]} -gt 0 ] && [ "${VIZ_REC_FULL:-true}" != "true" ] && _vr_full=false
    _vr_n=${#_vr_regs[@]}; [ "$_vr_full" = "true" ] && _vr_n=$((_vr_n + 1))
    _vr_fc="[0:v]scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p,split=${_vr_n}"
    _vr_maps=(); _vr_k=0
    if [ "$_vr_full" = "true" ]; then _vr_fc+="[vf]"; fi
    for _g in "${_vr_regs[@]}"; do _vr_k=$((_vr_k + 1)); _vr_fc+="[s${_vr_k}]"; done
    if [ "$_vr_full" = "true" ]; then _vr_maps+=(-map "[vf]" "${_vr_enc[@]}" "$VIZ_REC_DIR/screen.mp4"); fi
    _vr_k=0
    for _g in "${_vr_regs[@]}"; do
      _vr_k=$((_vr_k + 1)); _wh=${_g%%+*}; _xy=${_g#*+}
      _vr_fc+=";[s${_vr_k}]crop=${_wh%x*}:${_wh#*x}:${_xy%%+*}:${_xy#*+}[r${_vr_k}]"
      _vr_maps+=(-map "[r${_vr_k}]" "${_vr_enc[@]}" "$VIZ_REC_DIR/region${_vr_k}.mp4")
      echo "[viz-rec] region${_vr_k}: $_g -> $VIZ_REC_DIR/region${_vr_k}.mp4"
    done
    /snap/bin/ffmpeg -hide_banner -y -f x11grab -framerate 30 -video_size 1920x1080 -i ${_vr_disp}.0+0,0 \
        -filter_complex "$_vr_fc" "${_vr_maps[@]}" > "$VIZ_REC_DIR/screen_ffmpeg.log" 2>&1 &
    VIZ_REC_FF=$!
    for _i in $(seq 1 50); do grep -q "start: " "$VIZ_REC_DIR/screen_ffmpeg.log" 2>/dev/null && break; sleep 0.2; done
    grep -m1 -oE "start: [0-9.]+" "$VIZ_REC_DIR/screen_ffmpeg.log" | awk '{print $2}' > "$VIZ_REC_DIR/screen_start.txt"
    if [ -s "$VIZ_REC_DIR/screen_start.txt" ]; then echo "[viz-rec] screen recording running ($_vr_n output file(s)) -> $VIZ_REC_DIR/"
    else echo "[viz-rec] !! screen recording did not start (see $VIZ_REC_DIR/screen_ffmpeg.log)"; rm -f "$VIZ_REC_DIR/screen_start.txt"; fi
  fi
  ( set +u; source /opt/ros/jazzy/setup.bash
    exec /usr/bin/python3 ./tools/viz_stream_saver.py "$VIZ_REC_DIR" ) > "$VIZ_REC_DIR/saver.log" 2>&1 &
  VIZ_REC_SAVER=$!
  echo "[viz-rec] saving the 7 image streams -> $VIZ_REC_DIR   (Ctrl-C ends the run AND the recording)"
fi

RECORD_IO=${RECORD_IO:-false}   # true -> SIDECAR recording during the live run (zero impact
                                # on inference): capture node saves every input frame (WORLD-
                                # frame scan stack + pano + pose + stamps) and ros2 bag records
                                # the plane/obstacle outputs with their stamps.
IO_DIR=${IO_DIR:-./live_io_$(date +%m%d_%H%M%S)}
# Bring up everything else in parallel, right after the node starts (no warmup wait).
deferred_start() {
  if [ "$RECORD_IO" = "true" ]; then
    mkdir -p "$IO_DIR"
    echo "[launcher] RECORD_IO -> $IO_DIR (inputs + output bag; separate processes)"
    ( source "$ROS_SETUP"
      exec /usr/bin/python3 ./tools/capture_input_node.py         --ros-args -p out_dir:="'$IO_DIR/inputs'" -p crop_m:=${RANGE_M}.0 ) > /tmp/capture_io.log 2>&1 &
    ( source "$ROS_SETUP"
      exec ros2 bag record -o "$IO_DIR/outputs_$(date +%H%M%S)"         /glass_killer/planes /glass_killer/global_planes         /glass_killer/final_global_planes         /monoglass3d/glass /glassrecon/glass         /habitat/state_estimation ) > /tmp/record_io.log 2>&1 &
  fi
  # CLEAN display scan (glass-marked msgs dropped) for the RegScan rviz display; pinhole-FOV
  # methods additionally crop the DISPLAY to the camera frustum. Planner topics untouched.
  FOV_CROP=true; [ "$METHOD" = "360" ] && FOV_CROP=false
  ( source "$ROS_SETUP"
    exec /usr/bin/python3 ./tools/scan_glass_relay.py \
      --ros-args -p fov_crop:=$FOV_CROP ) > /tmp/scan_glass_relay.log 2>&1 &
  if [ "$LAUNCH_STACK" = "true" ]; then
    if [ "$WITH_TARE" = "true" ]; then
      echo "[launcher] starting autonomy stack WITH TARE exploration planner..."
      # FULL-TEST visual + TARE topics: the vehicle_simulator rviz with an extra unchecked
      # "TARE" group (LocalPath / LocalPlanningHorizon / ExploredAreas) -- tick to show.
      ( cd "$STACK_DIR" && source ./install/setup.bash
        ros2 launch vehicle_simulator system_bagfile_with_exploration_planner.launch & sleep 1
        if [ "${VIZ_FULL:-false}" = "true" ]; then     # demo layout: the 4 pipeline-stage images
          exec ros2 run rviz2 rviz2 -d ./rviz/glass_killer_video.rviz
        else
          exec ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator_tare.rviz
        fi ) &
      DRONE_LOG_DIR=./planner_logs/drone
      mkdir -p $DRONE_LOG_DIR
      DRONE_TAG=${METHOD}_${RUN_TAG:-$(date +%m%d_%H%M%S)}
      if [ "$FRONTIER_PLANNER" = "true" ] || [ "$OCC_PLANNER" = "true" ]; then
        # per-run recording: frontier goals + poses + /added_obstacles snapshots
        ( source /opt/ros/jazzy/setup.bash
          exec /usr/bin/python3 ./tools/waypoint_logger.py --ros-args \
            -p out_csv:="'$DRONE_LOG_DIR/waypoints_$DRONE_TAG.csv'" \
            -p pose_topic:="'/state_estimation'" ) > /tmp/drone_wplog.log 2>&1 &
        echo "[launcher] drone-run recording -> $DRONE_LOG_DIR/*_$DRONE_TAG.*"
      fi
      if [ "$FRONTIER_PLANNER" = "true" ]; then
        echo "[launcher] REPLACING TARE with the simple frontier planner..."
        ( sleep 10
          for P in $(pgrep -f 'tare_planner_nod[e]'); do kill -9 $P 2>/dev/null; done
          source /opt/ros/jazzy/setup.bash
          exec /usr/bin/python3 ./tools/frontier_planner.py \
            --ros-args -p z_above:=$(python3 -c "print(float(${DRONE_BAND_TOP:-2.5}))") ) \
          > /tmp/frontier_planner.log 2>&1 &
      fi
      if [ "$OCC_PLANNER" = "true" ]; then
        echo "[launcher] starting 3D occupancy local planner (/drone_path)..."
        ( source /opt/ros/jazzy/setup.bash
          exec /usr/bin/python3 ./tools/occupancy_local_planner.py \
            --ros-args -p z_above:=$(python3 -c "print(float(${DRONE_BAND_TOP:-2.5}))") \
            -p log_jsonl:="'$DRONE_LOG_DIR/paths_$DRONE_TAG.jsonl'" ) \
          > /tmp/occ_planner.log 2>&1 &
      fi
      if [ "$DRONE_BAND" = "true" ]; then
        # DRONE-BAND: replace both terrain nodes with wide-vertical-band variants. The launch-file
        # params are replicated verbatim except the band: obstacles anywhere in the corridor up to
        # DRONE_BAND_TOP above the sensor count (glass at head height blocks a drone; a ground
        # rover only cares about ~1m). NOTE: TARE stays a GROUND planner -- this widens what
        # counts as blocked, it does not make paths fly.
        ( sleep 8
          for P in $(pgrep -f 'terrainAnalysi[s]'); do kill -9 $P 2>/dev/null; done
          sleep 1
          source "$STACK_DIR/install/setup.bash"
          ros2 run terrain_analysis terrainAnalysis --ros-args \
            -p scanVoxelSize:=0.05 -p decayTime:=1.0 -p noDecayDis:=1.75 -p clearingDis:=8.0 \
            -p useSorting:=true -p quantileZ:=0.25 -p considerDrop:=false -p limitGroundLift:=false \
            -p maxGroundLift:=0.15 -p clearDyObs:=true -p minDyObsDis:=0.14 -p absDyObsRelZThre:=0.2 \
            -p minDyObsVFOV:=-30.0 -p maxDyObsVFOV:=35.0 -p minDyObsPointNum:=1 \
            -p minOutOfFovPointNum:=10 -p obstacleHeightThre:=0.1 -p noDataObstacle:=false \
            -p noDataBlockSkipNum:=0 -p minBlockPointNum:=10 -p vehicleHeight:=$DRONE_BAND_TOP \
            -p voxelPointUpdateThre:=100 -p voxelTimeUpdateThre:=2.0 \
            -p minRelZ:=-2.5 -p maxRelZ:=$DRONE_BAND_TOP -p disRatioZ:=0.2 > /tmp/terrain_drone.log 2>&1 &
          ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
            -p scanVoxelSize:=0.1 -p decayTime:=4.0 -p noDecayDis:=0.0 -p clearingDis:=30.0 \
            -p useSorting:=true -p quantileZ:=0.1 -p vehicleHeight:=$DRONE_BAND_TOP \
            -p voxelPointUpdateThre:=100 -p voxelTimeUpdateThre:=2.0 \
            -p lowerBoundZ:=-2.5 -p upperBoundZ:=$DRONE_BAND_TOP -p disRatioZ:=0.1 \
            -p checkTerrainConn:=false -p terrainConnThre:=0.5 -p terrainUnderVehicle:=-0.75 \
            -p ceilingFilteringThre:=$(python3 -c "print(float($DRONE_BAND_TOP)+1.5)") \
            -p localTerrainMapRadius:=4.0 > /tmp/terrain_ext_drone.log 2>&1 &
          # localPlanner has its OWN vertical crop (maxRelZ 0.3): it discards high terrain points
          # before collision-checking paths, so the raised terrain band alone changes nothing.
          # Restart it with the crop opened to the same band (pathFollower untouched).
          for P in $(pgrep -f 'localPlanne[r]'); do kill -9 $P 2>/dev/null; done
          sleep 1
          LPSHARE=$(ros2 pkg prefix local_planner)/share/local_planner
          ros2 run local_planner localPlanner --ros-args --params-file $LPSHARE/config/omniDir.yaml \
            -p pathFolder:="'$LPSHARE/paths'" -p vehicleLength:=0.5 -p vehicleWidth:=0.5 \
            -p sensorOffsetX:=0.0 -p sensorOffsetY:=0.0 -p twoWayDrive:=true \
            -p laserVoxelSize:=0.05 -p terrainVoxelSize:=0.2 -p useTerrainAnalysis:=true \
            -p checkObstacle:=true -p checkRotObstacle:=false -p adjacentRange:=3.5 \
            -p obstacleHeightThre:=0.05 -p groundHeightThre:=0.05 -p costHeightThre1:=0.1 \
            -p costHeightThre2:=0.05 -p useCost:=false -p slowPathNumThre:=5 \
            -p slowGroupNumThre:=1 -p pointPerPathThre:=2 \
            -p minRelZ:=-0.4 -p maxRelZ:=$DRONE_BAND_TOP \
            -p maxSpeed:=0.875 -p dirWeight:=0.02 -p dirThre:=90.0 -p dirToVehicle:=false \
            -p pathScale:=0.875 -p minPathScale:=0.675 -p pathScaleStep:=0.1 \
            -p pathScaleBySpeed:=true -p minPathRange:=0.8 -p pathRangeStep:=0.6 \
            -p pathRangeBySpeed:=true -p pathCropByGoal:=true -p autonomyMode:=false \
            -p autonomySpeed:=0.875 -p joyToSpeedDelay:=2.0 -p joyToCheckObstacleDelay:=5.0 \
            -p goalClearRange:=0.35 -p goalBehindRange:=0.35 -p freezeAng:=90.0 -p freezeTime:=0.0 \
            -p goalX:=0.0 -p goalY:=0.0 > /tmp/localplanner_drone.log 2>&1 &
        ) &
      fi
    else
      echo "[launcher] starting autonomy stack (system_bagfile.sh)..."
      # system_bagfile.sh hardcodes vehicle_simulator.rviz, so for VIZ_FULL we inline the same
      # two steps here and point RViz at the demo layout instead (the stack script is untouched).
      if [ "${VIZ_FULL:-false}" = "true" ]; then
        ( cd "$STACK_DIR" && source ./install/setup.bash
          ros2 launch vehicle_simulator system_bagfile.launch & sleep 1
          exec ros2 run rviz2 rviz2 -d ./rviz/glass_killer_video.rviz ) &
      elif [ "${RVIZ_FULLSCREEN:-false}" = "true" ]; then
        # standard stack RViz layout, but fullscreen so a screen recorder captures only RViz
        ( cd "$STACK_DIR" && source ./install/setup.bash
          ros2 launch vehicle_simulator system_bagfile.launch & sleep 1
          exec ros2 run rviz2 rviz2 --fullscreen -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz ) &
      elif [ "${HEADLESS:-false}" = "true" ]; then
        # HEADLESS=true: same stack launch as system_bagfile.sh, without opening RViz
        ( cd "$STACK_DIR" && source ./install/setup.bash
          exec ros2 launch vehicle_simulator system_bagfile.launch ) &
      else
        ( cd "$STACK_DIR" && ./system_bagfile.sh ) &
      fi
    fi
    sleep 6
  fi
  if [ "$LAUNCH_PROVIDER" = "true" ]; then
    echo "[launcher] starting glass_killer provider..."
    ( source "$ROS_SETUP"; [ -f "$CAM_INSTALL" ] && source "$CAM_INSTALL"
      exec ros2 launch extrinsic_latency_calib glass_killer.launch maxRange:=${RANGE_M}.0 \
        imageLatencyOffset:=${IMAGE_LATENCY:-0.0} ) &
    sleep 3
  fi
  if [ "$LAUNCH_REPUBLISH" = "true" ]; then
    echo "[launcher] starting image republish (compressed -> raw)..."
    ( source "$ROS_SETUP"
      exec ros2 run image_transport republish --ros-args \
        -p in_transport:=compressed -p out_transport:=raw \
        --remap in/compressed:=/camera/image/compressed --remap out:=/camera/image ) &
    sleep 1
  fi
  if [ "$PLAY_BAG" = "true" ]; then
    echo "[launcher] playing bag: $BAG"
    LOOP=""; [ "$BAG_LOOP" = "true" ] && LOOP="--loop"
    # --disable-keyboard-controls + stdin from /dev/null so bag play never grabs the terminal. The
    # NODE owns the keyboard and drives the bag via the rosbag2 player SERVICES (toggle_paused,
    # set_rate): in the node terminal -> s=save 5 frames, SPACE=pause/resume bag, UP/DOWN=faster/slower.
    ( source "$ROS_SETUP"; [ -f "$STACK_DIR/install/setup.bash" ] && source "$STACK_DIR/install/setup.bash"
      exec ros2 bag play --disable-keyboard-controls $LOOP \
        ${BAG_RATE:+--rate $BAG_RATE} \
        ${BAG_START_OFFSET:+--start-offset $BAG_START_OFFSET} "$BAG" < /dev/null ) &
  fi
}
deferred_start &

# BASELINE replay mode: no GK node at all -- publish the precomputed predictions pose-synced.
if [ "$METHOD" = "monoglass" ] || [ "$METHOD" = "glassrecon" ]; then
  echo "[launcher] METHOD=$METHOD -> replaying precomputed predictions from $SCENE_DIR"
  source "$ROS_SETUP"; [ -f "$CAM_INSTALL" ] && source "$CAM_INSTALL"
  # OFFSET bags re-anchor the live SLAM origin (bag = raw sensors only) -> absolute pose
  # matching stalls. Pass offset + bag duration so the node switches to arc-length sync.
  BAG_DUR_S=0.0
  if [ -n "${BAG_START_OFFSET:-}" ]; then
    BAG_DUR_S=$(ros2 bag info "$BAG" 2>/dev/null | awk '/Duration:/{gsub(/s/,"",$2); print $2}')
    BAG_DUR_S=${BAG_DUR_S:-0.0}
  fi
  conda run -n sam3 --no-capture-output python ./baselines/baseline_replay_node.py \
    --ros-args -p scene:="'$SCENE_DIR'" -p method:="'$METHOD'" \
    -p start_offset_s:=${BAG_START_OFFSET:-0}.0 -p bag_dur_s:=$BAG_DUR_S
  exit 0
fi

# LIVE baseline mode: real inference on the provider streams, frame-snapped pose+cloud+image
# (capture-time pose, no latency shift) -- exactly the GK pattern, no precomputed preds needed.
if [ "$METHOD" = "monoglass_live" ] || [ "$METHOD" = "glassrecon_live" ]; then
  echo "[launcher] METHOD=$METHOD -> LIVE baseline inference"
  source "$ROS_SETUP"; [ -f "$CAM_INSTALL" ] && source "$CAM_INSTALL"
  if [ "$METHOD" = "monoglass_live" ]; then
    # display runs (no recording) publish the ACCUMULATED map; recordings stay per-frame (small)
    PUB_ACCUM=true; [ "$RECORD_IO" = "true" ] && PUB_ACCUM=false
    exec conda run -n sam3 --no-capture-output python ./baselines/monoglass3d_ros_node.py \
      --ros-args -p planner_feed:=true -p scan_feed:=true -p pub_accum:=$PUB_ACCUM
  else
    # MonoGlass runs as the MASK PROVIDER only (planner feed muted); GlassRecon owns the terminal
    ( source "$ROS_SETUP"
      exec conda run -n sam3 --no-capture-output python ./baselines/monoglass3d_ros_node.py \
        --ros-args -p planner_feed:=false -p scan_feed:=false ) > /tmp/gr_mask_provider.log 2>&1 &
    sleep 5
    exec conda run -n sam3 --no-capture-output python ./baselines/glassrecon_ros_node.py \
      --ros-args -p planner_feed:=true -p scan_feed:=true -p align:=${GR_ALIGN:-ransac}
  fi
fi

# NOTE: full-input capture during a live run PROVED TOO SLOW (ascii PLY writes stall the loop,
# frame rate tanks) -- RECORD_RUN no longer forces SAVE_INPUT. The canonical recording keeps the
# LIGHT artifacts only (ledger/trajectory/scene cloud); placed_input is opt-in via
# RECORD_PLACED_INPUT (it approached per-frame recording once the gate placed many planes).
INPUT_SAVE_DIR=./glass_killer_ros_input

# GT ANCHOR: when a canonical pinhole run exists, align by SCENE CLOUD (structure ICP onto
# the live map) using the canonical-frame GT; otherwise fall back to raw frames-world GT +
# trajectory anchoring.
GT_AUTO_ALIGN=false   # true = scene-cloud ICP anchoring of the GT overlay; false = NO ICP
                      # anywhere -- raw GT shown, manual alignment via align_gt_tool.py after
GT_ALIGN_CLOUD=false
GT_REF_CLOUD=""
GT_OVERLAY=${GT_OVERLAY:-false}   # true -> publish the white GT-plane overlay (off for videos)
GT_JSON=$SCENE_DIR/gts/gt_planes.json
# CANONICAL-frame GT is preferred for DISPLAY whenever it exists: with the deterministic
# anchor gate, a new run's map frame reproduces the canonical run's (~cm), so the raw
# overlay should already line up -- NO ICP unless GT_AUTO_ALIGN=true.
if [ -f "$SCENE_DIR/gts/gt_planes_canonical.json" ]; then
  GT_JSON=$SCENE_DIR/gts/gt_planes_canonical.json
fi
[ "$GT_OVERLAY" = "true" ] || GT_JSON=""   # GT overlay OFF unless explicitly requested
if [ "$GT_AUTO_ALIGN" = "true" ] && [ -f "$SCENE_DIR/canonical_run_pinhole/scene_cloud.ply" ] && [ -f "$SCENE_DIR/gts/gt_planes_canonical.json" ]; then
  GT_ALIGN_CLOUD=true
  GT_REF_CLOUD=$SCENE_DIR/canonical_run_pinhole/scene_cloud.ply
fi

# The Glass Killer node -- FIRST and FOREGROUND (loads models now; keeps stdin for SPACE-key save).
echo "[launcher] starting Glass Killer node FIRST (loading models; the rest comes up in parallel)"
echo "[launcher] method=$METHOD align=$USE_ALIGN da2=$USE_DA2 par_check=$PAR_CHECK save_input=$SAVE_INPUT" \
     "save_output=$SAVE_OUTPUT keysave_full=$KEYSAVE_FULL precision=$PRECISION obstacle_mode=$OBSTACLE_MODE"
source "$ROS_SETUP"; [ -f "$CAM_INSTALL" ] && source "$CAM_INSTALL"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # trim reserved pool / avoid fragmentation OOM
launch_gk_node() {  # $1 = node .py, $2 = extra --ros-args (role); PIPELINE reuses this verbatim
conda run -n sam3 --no-capture-output python "$1" --ros-args $2 \
  -p pinhole_cfg:="'$PINHOLE_CFG'" \
  -p gt_planes_json:="'$GT_JSON'" \
  -p gt_align_icp:=$GT_ALIGN_CLOUD \
  -p gt_ref_cloud:="'$GT_REF_CLOUD'" \
  -p canonical_dir:="'$SCENE_DIR/canonical_run_$METHOD'" \
  -p record_run:=$RECORD_RUN \
  -p record_placed_input:=$RECORD_PLACED_INPUT \
  -p use_pinhole_align:=$USE_ALIGN \
  -p use_da2:=$USE_DA2 \
  -p par_check:=$PAR_CHECK \
  -p tug_max:=${TUG_MAX:-2} \
  -p track_merge:=${TRACK_MERGE:-true} \
  -p viz_full:=${VIZ_FULL:-false} \
  -p par_360_angle:=$PAR_CHECK \
  -p par_min_span_px:=$PAR_MIN_SPAN_PX \
  -p dir_trust_min_cond:=$DIR_MIN_COND \
  -p h_edge_curve:=$H_EDGE_CURVE \
  -p h_ray_hit_tol_px:=$H_RAY_HIT_TOL \
  -p save_input:=$SAVE_INPUT \
  -p input_save_dir:="'$INPUT_SAVE_DIR'" \
  -p save_auto_map:=$SAVE_OUTPUT \
  -p keysave_full:=$KEYSAVE_FULL \
  -p precision:=$PRECISION \
  -p obstacle_mode:="'$OBSTACLE_MODE'" \
  -p conf_th:=$CONF_TH \
  -p min_cov:=$COV_TH \
  -p floor_reject_frac:=0.50 \
  -p empty_cache_every:=$EMPTY_CACHE_EVERY \
  -p mask_clamp:=$MASK_CLAMP \
  -p doorway_gate:=$DOORWAY_GATE \
  -p floor_check:=$FLOOR_CHECK \
  -p use_terrain_floor:=$TERRAIN_FLOOR \
  -p grounding_cell_size:=$GROUNDING_CELL \
  -p seed_grid_ring:=$SEED_GRID_RING \
  -p seed_band_symmetric:=$SEED_BAND_SYM \
  -p small_fallback:=$SMALL_FALLBACK \
  -p quad_support:=$QUAD_SUPPORT \
  -p extend_full_mask:=$EXTEND_FULL_MASK \
  -p seed_ring_cells:=$SEED_RING_CELLS \
  -p seed_dilate:=$SEED_DILATE \
  -p seed_floor_vtol:=$SEED_FLOOR_VTOL \
  -p save_local_tug:=$SAVE_LOCAL_TUG \
  -p reproject_evict:=$REPROJECT_EVICT \
  -p depth_sweep_evict:=$DEPTH_SWEEP_EVICT \
  -p path_evict:=$PATH_EVICT \
  -p floor_evict:=$FLOOR_EVICT \
  -p spill_thresh:=$SPILL_THRESH \
  -p spill_iou_min:=$SPILL_IOU_MIN \
  -p spill_iou_gate:=$SPILL_IOU_GATE \
  -p spill_cov_min:=$SPILL_COV_MIN \
  -p spill_vis_min:=$SPILL_VIS_MIN \
  -p spill_tug_start:=$SPILL_TUG_START \
  -p spill_base_n:=$SPILL_BASE_N \
  -p spill_hard_frac:=$SPILL_HARD \
  -p spill_persist:=$SPILL_PERSIST \
  -p spill_baseline_min_m:=$SPILL_BASELINE \
  -p spill_check_move_m:=$SPILL_CHECK_MOVE \
  -p spill_max:=$SPILL_MAX \
  -p plane_height_shrink:=$PLANE_HEIGHT_SHRINK \
  -p spill_plane_occ:=$SPILL_PLANE_OCC \
  -p corner_seg_curve:=$CORNER_SEG_CURVE \
  -p spill_dist_max_m:=$SPILL_DIST_MAX \
  -p place_dist_max_m:=$PLACE_DIST_MAX \
  -p opt_depth_max_m:=$OPT_DEPTH_MAX \
  -p depth_max:=${DEPTH_MAX}.0 \
  -p spill_debug_dir:="'$SPILL_DEBUG_DIR'" \
  -p enable_final_map:=$FINAL_MAP \
  -p final_spill_dist_max_m:=$FINAL_SPILL_DIST \
  -p final_spill_thresh:=$FINAL_SPILL_THRESH \
  -p save_reproject_panel:=$SAVE_REPROJECT
}
# PIPELINE=true (opt-in): 2-process split -- background MAPPING node (owns the tracker) +
# foreground PERCEPTION node (detect+geometry -> /gkpipe/geom). Default = original mono node.
if [ "${PIPELINE:-true}" = "true" ]; then
  echo "[pipeline] 2-process: background MAPPING + foreground PERCEPTION (gk_node.py)"
  mkdir -p /dev/shm/gkpipe && rm -f /dev/shm/gkpipe/*.pkl 2>/dev/null
  launch_gk_node ./glass_killer_pipeline/gk_node.py "-p role:=mapping" > /tmp/pipe_mapping.log 2>&1 &
  sleep 8    # let the mapping node subscribe to /gkpipe/geom before perception starts publishing
  launch_gk_node ./glass_killer_pipeline/gk_node.py "-p role:=perception"
else
  launch_gk_node ./glass_killer_ros_node.py ""
fi
