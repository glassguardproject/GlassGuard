#!/bin/bash
# Re-record the MAIN-TABLE runs on the current code: 9 scenes x (GG-pin, GG-360) = 18 live runs,
# each with standard RViz fullscreen and a screen recording, then evaluated and trimmed.
#
#   tools/run_main_rerecord.sh                       # all nine scenes
#   tools/run_main_rerecord.sh bldgB_f2            # or name scenes
#
# Recordings : <real_test>/<scene>_v5_pin , <scene>_v5_360
# Results    : <scene>_live/inputs/gts/occupancy_eval_v5_pin.json , ..._v5_360.json
#              (Ever / by-distance bands / current-map coverage / retention / spill, all voxel sizes)
# Videos     : ~/Videos/glassguard_runs/<scene>/<scene>_GG-pin.mp4 , <scene>_GG-360.mp4
# Log        : /tmp/main_rerecord.log
set -u
GK=.
RT=${GG_DATA_ROOT:-$HOME/glassguard_data}
VID=./demo_rec/screen
LOG=/tmp/main_rerecord.log
FF=/snap/bin/ffmpeg
export DISPLAY=${DISPLAY:-:1}
MIN_FREE_GB=20
# SCREEN_REC=true: fullscreen RViz + one screen video per run. Default false: headless, no video
# (same conditions as the headless pinhole ablation).
SCREEN_REC=${SCREEN_REC:-false}

# scene -> "bag | range (m) | min frames"     (20 m: bldgB_atrium, bldgD_ext)
declare -A CFG=(
  [bldgB_f2]="./bldgB_f2|10|200"
  [bldgD_int]="$RT/bldgD_int|10|300"
  [bldgA_ext_night]="./bldgA_ext_night|10|300"
  [bldgB_int]="./bldgB_int|10|400"
  [bldgC_office]="$RT/bldgC_office|10|450"
  [bldgD_ext]="$RT/bldgD_ext|20|600"
  [bldgA_f5]="$RT/bldgA_f5.mcap|10|800"
  [bldgB_atrium]="./bldgB_atrium|20|900"
  [bldgA_f4]="$RT/bldgA_f4.mcap|10|1000"
)
SCENES=(bldgB_f2 bldgD_int bldgA_ext_night bldgB_int bldgC_office bldgD_ext bldgA_f5 bldgB_atrium bldgA_f4)
[ $# -gt 0 ] && SCENES=("$@")

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

teardown() {
  pkill -f "[g]lass_killer_plane_node"; pkill -f "[g]lass_killer_ros_node"
  pkill -f "[r]os2 bag play"; pkill -INT -f "[r]os2 bag record"; pkill -f "[c]apture_input_node"
  pkill -f "[r]un_glass_killer_full.sh"; pkill -f "[s]ystem_bagfile.launch"
  pkill -f "[g]lass_killer.launch"; pkill -f "[g]k_node.py"; pkill -f "[r]viz2 --fullscreen"
  pkill -f "[l]oam"; sleep 10
  local pat="[g]lass_killer_plane_node|[g]k_node.py|[s]ystem_bagfile.launch|[r]os2 bag play|[r]os2 bag record|[r]viz2 --fullscreen"
  if [ "$(pgrep -fc "$pat")" -gt 0 ]; then say "   teardown: forcing leftovers"; pkill -9 -f "$pat"; sleep 5; fi
}

run_one() {   # scene method(pinhole|360) tag(pin|360)
  local SCENE=$1 METHOD=$2 TAG=$3 BAG RANGE MINFR
  IFS='|' read -r BAG RANGE MINFR <<< "${CFG[$SCENE]}"
  local IO=$RT/${SCENE}_v5_$TAG  OUTJ=$RT/${SCENE}_live/inputs/gts/occupancy_eval_v5_$TAG.json
  local MP4=$VID/$SCENE/${SCENE}_GG-$TAG.mp4
  if [ -s "$OUTJ" ]; then say "== $SCENE/GG-$TAG already evaluated, skipping =="; return; fi
  local FREE; FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge "$MIN_FREE_GB" ] || { say "!! only ${FREE} GB free (< $MIN_FREE_GB); stopping before $SCENE/GG-$TAG"; exit 1; }
  rm -rf "$IO"
  say "=== LIVE $SCENE / GG-$TAG  range=${RANGE}m  free=${FREE}GB ==="
  local VIEW="HEADLESS=true"; [ "$SCREEN_REC" = "true" ] && VIEW="RVIZ_FULLSCREEN=true"
  ( cd $GK && env METHOD=$METHOD $VIEW BAG="$BAG" RANGE_M=$RANGE RECORD_RUN=false \
      RECORD_IO=true IO_DIR="$IO" setsid ./run_glass_killer_full.sh ) > /tmp/main_${SCENE}_$TAG.log 2>&1 &
  # screen recording: fragmented mp4 so the file stays playable even if the recorder is killed
  local FFPID=""
  if [ "$SCREEN_REC" = "true" ]; then
    mkdir -p "$VID/$SCENE"
    $FF -hide_banner -loglevel error -y -f x11grab -framerate 15 -video_size 1920x1080 -i ${DISPLAY}.0 \
        -c:v libx264 -preset ultrafast -crf 24 -pix_fmt yuv420p -movflags +frag_keyframe+empty_moov "$MP4" \
        > /tmp/main_${SCENE}_${TAG}_ffmpeg.log 2>&1 &
    FFPID=$!
  fi
  for i in $(seq 1 40); do sleep 5; pgrep -f "[r]os2 bag play" >/dev/null && break; done
  for i in $(seq 1 900); do pgrep -f "[r]os2 bag play" >/dev/null || break; sleep 5; done
  sleep 45                                   # drain queued frames
  if [ -n "$FFPID" ]; then kill -INT $FFPID 2>/dev/null; sleep 3; kill -9 $FFPID 2>/dev/null; fi
  local n; n=$(ls $IO/inputs/pose_*.txt 2>/dev/null | wc -l)
  teardown
  say "== $SCENE/GG-$TAG recorded: $n frames =="
  if [ "$n" -lt "$MINFR" ]; then say "!! $SCENE/GG-$TAG SHORT ($n < $MINFR): NOT evaluated, NOT trimmed"; return; fi
  ABL_METHOD=$METHOD ABL_DIR="$IO" ABL_OUT=v5_$TAG "$GK/tools/eval_abl_scene.sh" "$SCENE" v5 2>&1 | tee -a "$LOG"
  if [ -s "$OUTJ" ]; then
    find "$IO/inputs" -maxdepth 1 -type f \( -name "cloud_*" -o -name "rgb_*" \) -delete; rm -rf "$IO"/outputs_*
    say "   $SCENE/GG-$TAG evaluated and trimmed -> $(du -sh "$IO" | cut -f1)"
  else
    say "!! $SCENE/GG-$TAG evaluation produced no JSON: recording kept untrimmed"
  fi
}

for SCENE in "${SCENES[@]}"; do
  [ -n "${CFG[$SCENE]:-}" ] || { say "!! unknown scene: $SCENE"; exit 1; }
  IFS='|' read -r BAG _ _ <<< "${CFG[$SCENE]}"; [ -e "$BAG" ] || { say "!! bag not found: $BAG"; exit 1; }
  say "===== $SCENE ====="
  run_one "$SCENE" pinhole pin
  run_one "$SCENE" 360 360
done
say "MAIN_RERECORD_ALL_DONE"
