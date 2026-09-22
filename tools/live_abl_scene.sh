#!/bin/bash
# LIVE ablation replays for one scene: five ablated configs through the full ROS stack,
# each recorded with RECORD_IO (Full GG = the existing live benchmark run, not re-run).
#
# Completion is detected by the BAG PLAYER EXITING (config-independent), then a drain
# delay so the node finishes the frames already queued. An earlier version watched the
# recorded frame count for a plateau; configs that publish sparsely (e.g. no-merge) hit
# a mid-run lull and were torn down early, truncating the recording.
#
# Usage: live_abl_scene.sh <scene_key> <bag_path> <range_m> <min_expected_frames>
set -u
SCENE=${1:?scene key}; BAG=${2:?bag path}; RANGE=${3:-10}; MINFR=${4:-100}
GK=.
# ABL_METHOD=pinhole records GG-pin ablations into <scene>_ablpin_<cfg> (360 dirs untouched).
ABL_METHOD=${ABL_METHOD:-360}
if [ "$ABL_METHOD" = "pinhole" ]; then ABL_TAG=ablpin; else ABL_TAG=abl; fi
OUTROOT=${GG_DATA_ROOT:-$HOME/glassguard_data}

teardown() {
  pkill -f "[g]lass_killer_plane_node"; pkill -f "[g]lass_killer_ros_node"
  pkill -f "[r]os2 bag play"; pkill -f "[c]apture_input_node"; pkill -f "[r]os2 bag record"
  pkill -f "[s]ystem_real_robot"; pkill -f "[l]oam"
  # the launcher and the stack launch themselves (needed when HEADLESS: no RViz exit to end them)
  pkill -f "[r]un_glass_killer_full.sh"; pkill -f "[s]ystem_bagfile.launch"
  pkill -f "[g]lass_killer.launch"; pkill -f "[g]k_node.py"; sleep 10
  left=$(pgrep -fc "[g]lass_killer_plane_node|[g]k_node.py|[s]ystem_bagfile.launch|[r]os2 bag play|[r]os2 bag record")
  if [ "$left" -gt 0 ]; then echo "!! teardown: $left process(es) still alive, forcing"; 
    pkill -9 -f "[g]lass_killer_plane_node|[g]k_node.py|[s]ystem_bagfile.launch|[r]os2 bag play|[r]os2 bag record"; sleep 5; fi
}

run_cfg() {
  cfg=$1; shift
  IO=$OUTROOT/${SCENE}_${ABL_TAG}_${cfg}
  n0=$(ls $IO/inputs/pose_*.txt 2>/dev/null | wc -l)
  if [ "$n0" -ge "$MINFR" ]; then echo "== $cfg already recorded ($n0 frames) =="; return; fi
  rm -rf "$IO"
  echo "=== LIVE $SCENE/$cfg $(date +%H:%M:%S) ==="
  ( cd $GK && env "$@" METHOD=$ABL_METHOD BAG="$BAG" RANGE_M=$RANGE RECORD_RUN=false RECORD_IO=true \
      IO_DIR="$IO" setsid ./run_glass_killer_full.sh ) > /tmp/live_${ABL_TAG}_${SCENE}_${cfg}.log 2>&1 &
  # 1) wait for the bag player to appear (stack startup can take ~60 s)
  for i in $(seq 1 40); do sleep 5; pgrep -f "[r]os2 bag play" >/dev/null && break; done
  # 2) wait for it to EXIT = bag fully played
  for i in $(seq 1 600); do
    pgrep -f "[r]os2 bag play" >/dev/null || break
    sleep 5
  done
  sleep 45                                  # drain: let the node finish queued frames
  n=$(ls $IO/inputs/pose_*.txt 2>/dev/null | wc -l)
  teardown
  if [ "$n" -lt "$MINFR" ]; then echo "!! $SCENE/$cfg SHORT: $n frames (expected >= $MINFR)"; fi
  echo "== $SCENE/$cfg done: $n frames =="
}

# pinhole ablation also records its own Full row, so every row shares the same code version
[ "$ABL_METHOD" = "pinhole" ] && run_cfg full
run_cfg nopar      PAR_CHECK=false
run_cfg nospill    REPROJECT_EVICT=false
run_cfg nofloor    FLOOR_EVICT=false
run_cfg nomerge    TRACK_MERGE=false
run_cfg nomanager  TRACK_MERGE=false FLOOR_EVICT=false PATH_EVICT=false REPROJECT_EVICT=false
echo "LIVE_ABL_${SCENE}_DONE"
