#!/bin/bash
# Headless GG-pin component ablation: for each scene, record 6 configs (full + 5 ablations),
# evaluate them inside the pinhole FOV, then trim the raw input copies to give the disk back.
#
#   tools/run_pin_ablation.sh                 # scenes listed in SCENES below
#   tools/run_pin_ablation.sh bldgA_ext_night     # or name scenes on the command line
#
# Log: /tmp/pin_ablation.log   (per-config node logs: /tmp/live_ablpin_<scene>_<cfg>.log)
set -u
GK=.
RT=${GG_DATA_ROOT:-$HOME/glassguard_data}
LOG=/tmp/pin_ablation.log
MIN_FREE_GB=40          # refuse to start a scene with less free space than this

# scene key -> "bag path | range (m) | min frames for a complete recording"
declare -A CFG=(
  [bldgB_f2]="./bldgB_f2|10|200"
  [bldgA_ext_night]="./bldgA_ext_night|10|300"
  [bldgA_f5]="${GG_DATA_ROOT:-$HOME/glassguard_data}/bldgA_f5.mcap|10|800"
  [bldgB_atrium]="./bldgB_atrium|20|900"     # outdoor scene: 20 m range
)

# First pass: shortest scene only. Add the rest once this one checks out:
#   SCENES=(bldgB_f2 bldgA_ext_night bldgA_f5 bldgB_atrium)
SCENES=(bldgB_f2)
[ $# -gt 0 ] && SCENES=("$@")

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for SCENE in "${SCENES[@]}"; do
  [ -n "${CFG[$SCENE]:-}" ] || { say "!! unknown scene: $SCENE"; exit 1; }
  IFS='|' read -r BAG RANGE MINFR <<< "${CFG[$SCENE]}"
  [ -e "$BAG" ] || { say "!! bag not found: $BAG"; exit 1; }
  FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge "$MIN_FREE_GB" ] || { say "!! only ${FREE} GB free (< $MIN_FREE_GB); stopping before $SCENE"; exit 1; }

  say "===== $SCENE  range=${RANGE}m  bag=$BAG  free=${FREE}GB ====="
  ABL_METHOD=pinhole HEADLESS=true "$GK/tools/live_abl_scene.sh" "$SCENE" "$BAG" "$RANGE" "$MINFR" 2>&1 | tee -a "$LOG"

  for c in full nopar nospill nofloor nomerge nomanager; do
    d=$RT/${SCENE}_ablpin_$c
    n=$(ls "$d"/inputs/pose_*.txt 2>/dev/null | wc -l)
    if [ "$n" -lt "$MINFR" ]; then say "!! $SCENE/$c has $n frames (< $MINFR): NOT evaluated, NOT trimmed"; continue; fi
    ABL_METHOD=pinhole "$GK/tools/eval_abl_scene.sh" "$SCENE" "$c" 2>&1 | tee -a "$LOG"
    if [ -s "$RT/${SCENE}_live/inputs/gts/occupancy_eval_ablpin_$c.json" ]; then
      find "$d/inputs" -maxdepth 1 -type f \( -name "cloud_*" -o -name "rgb_*" \) -delete
      rm -rf "$d"/outputs_*
      say "   $SCENE/$c evaluated ($n frames) and trimmed -> $(du -sh "$d" | cut -f1)"
    else
      say "!! $SCENE/$c evaluation produced no JSON: recording kept untrimmed"
    fi
  done
  say "===== $SCENE finished, free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)GB ====="
done
say "PIN_ABLATION_ALL_DONE"
