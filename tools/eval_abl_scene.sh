#!/bin/bash
# Prep -> align -> time-remap -> evaluate ONE live ablation recording against its
# scene's live benchmark session (win-0 metric, one-voxel tolerance).
#
# Usage: eval_abl_scene.sh <scene_key> <config>
#   e.g. eval_abl_scene.sh bldgB_atrium nopar
# Writes <scene>_live/inputs/gts/occupancy_eval_abl_<config>.json
set -u
SCENE=${1:?scene key}; CFG=${2:?config}
RT=${GG_DATA_ROOT:-$HOME/glassguard_data}
GK=.
# ABL_METHOD=pinhole scores <scene>_ablpin_<cfg> inside the pinhole frustum (like gkpin_fov).
ABL_METHOD=${ABL_METHOD:-360}
if [ "$ABL_METHOD" = "pinhole" ]; then ABL_TAG=ablpin; FOV="--cam-fov-cfg $RT/bldgA_f5_test/camera_config.json"; else ABL_TAG=abl; FOV=""; fi
PY="conda run -n sam3 --no-capture-output python"
REF=$RT/${SCENE}_live/scene_cloud.ply
ROOT=$RT/${SCENE}_live/inputs
d=$RT/${SCENE}_${ABL_TAG}_${CFG}
# ABL_DIR / ABL_OUT override the recording folder and the result-file tag (main-table re-records)
[ -n "${ABL_DIR:-}" ] && d=$ABL_DIR
OUT_TAG=${ABL_OUT:-${ABL_TAG}_${CFG}}

[ -d "$d" ] || { echo "!! missing recording: $d"; exit 1; }
echo "=== $SCENE/$CFG prep"
$PY $GK/tools/prep_live.py --dir "$d" > /tmp/prep_${SCENE}_${CFG}.log 2>&1 || { echo "!! prep failed"; exit 1; }
echo "   preds: $(ls $d/inputs/preds_live/*.ply 2>/dev/null | wc -l)"

AL=$($PY $GK/tools/framecheck_align.py "$REF" "$d" 2>/dev/null | grep "^ALIGN"); echo "   $AL"
pdir=$d/inputs/preds_live
echo "$AL" | grep -q preds_live_aligned && pdir=$d/inputs/preds_live_aligned

RM=$($PY $GK/tools/remap_preds_by_time.py "$ROOT" "$d" 2>/dev/null | grep "^REMAP"); echo "   $RM"
[ -n "$RM" ] && pdir=$d/inputs/preds_by_roottime

W=$RT/${SCENE}_live/w_${ABL_TAG}_${CFG}; rm -rf "$W"; mkdir -p "$W"
for f in $ROOT/*; do ln -s "$f" "$W"/ 2>/dev/null; done
rm -rf "$W"/gts; mkdir -p "$W"/gts; ln -s $ROOT/gts/gt_planes.json "$W"/gts/
ln -s $ROOT/gts/bad_frames.json "$W"/gts/ 2>/dev/null

( cd $RT && $PY eval_occupancy.py --root ${SCENE}_live/w_${ABL_TAG}_${CFG} \
    --pred-dir "$pdir" --pred-suffix glasskiller_glass --pred-world-frame \
    --scene-voxel 0.1 --expel-win 0 --cov-dilate 1 $FOV ) > /tmp/ev_${SCENE}_${CFG}.log 2>&1 \
  && cp "$W"/gts/occupancy_eval_glasskiller_glass.json \
        $ROOT/gts/occupancy_eval_${OUT_TAG}.json && echo "   EVAL OK" || echo "   !! EVAL FAIL"
rm -rf "$W"
echo "DONE_${SCENE}_${CFG}"
