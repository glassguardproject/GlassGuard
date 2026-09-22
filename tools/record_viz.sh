#!/bin/bash
# Record every full-visual image stream (PNG frames + source stamp + WALL-CLOCK arrival time).
#   tools/record_viz.sh <name>            -> demo_rec/<name>/<stream>/frame%06d.png + stamps.csv
#   tools/record_viz.sh <name> --screen   -> also records the whole screen to demo_rec/<name>/screen.mp4
#                                            and notes the exact wall time of its first frame, so
#                                            encode_viz.sh makes clips that start and end WITH it.
# Start this FIRST, then start the VIZ_FULL run in another terminal. Ctrl-C here when the run ends.
set -u
NAME=${1:?usage: record_viz.sh <name> [--screen]}; SCREEN=${2:-}
OUT=./demo_rec/$NAME
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "!! $OUT already has data. Pick another name or remove it first."; exit 1
fi
mkdir -p "$OUT"
set +u; source /opt/ros/jazzy/setup.bash; set -u
FF=""
if [ "$SCREEN" = "--screen" ]; then
  export DISPLAY=${DISPLAY:-:1}
  /snap/bin/ffmpeg -hide_banner -y -f x11grab -framerate 30 -video_size 1920x1080 -i ${DISPLAY}.0 \
      -vf "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p" -c:v libx264 -preset veryfast -crf 18 -g 30 -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -movflags +frag_keyframe+empty_moov \
      "$OUT/screen.mp4" > "$OUT/screen_ffmpeg.log" 2>&1 &
  FF=$!
  for i in $(seq 1 50); do grep -q "start: " "$OUT/screen_ffmpeg.log" 2>/dev/null && break; sleep 0.2; done
  grep -m1 -oE "start: [0-9.]+" "$OUT/screen_ffmpeg.log" | awk '{print $2}' > "$OUT/screen_start.txt"
  [ -s "$OUT/screen_start.txt" ] && echo "[record_viz] screen recording started at wall time $(cat "$OUT/screen_start.txt")" \
                                  || echo "!! could not read the screen recording's start time"
fi
# The screen recorder is a snap app: signals sent from a VS Code terminal can be refused (AppArmor).
# Stop it through its systemd scope when a plain signal does not work.
stop_snap_proc() { local p=$1 sc
  [ -n "$p" ] && [ -d "/proc/$p" ] || return 0          # (kill -0 can be refused too, so test /proc)
  kill -INT "$p" 2>/dev/null; for _ in 1 2 3 4 5 6; do [ -d "/proc/$p" ] || return 0; sleep 0.5; done
  sc=$(grep -oE "snap\.[^/]*\.scope" /proc/$p/cgroup 2>/dev/null | head -1)
  [ -n "$sc" ] && systemctl --user stop "$sc" 2>/dev/null; }
finish() { date +%s.%N > "$OUT/screen_end.txt"; stop_snap_proc "$FF"; }
trap finish EXIT
echo "[record_viz] saving to $OUT   (free disk: $(df -h / | tail -1 | awk '{print $4}'))"
echo "[record_viz] now start the VIZ_FULL run in another terminal; Ctrl-C here when it is done."
/usr/bin/python3 ./tools/viz_stream_saver.py "$OUT"
