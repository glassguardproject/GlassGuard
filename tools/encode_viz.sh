#!/bin/bash
# Turn a record_viz.sh capture into time-aligned videos (one per stream, same length, in sync).
#   tools/encode_viz.sh <name>                 clips on the WALL-CLOCK timeline = exactly what RViz showed, when.
#                                              With a --screen capture they start/end with screen.mp4.
#   tools/encode_viz.sh <name> --start 14:03:10   align to a screen recording made with another tool (its start time)
#   tools/encode_viz.sh <name> --clock stamp      old behaviour: bag time, independent of the 0.5x playback
set -u
NAME=${1:?usage: encode_viz.sh <name> [--start HH:MM:SS] [--clock stamp]}; shift
REC=./demo_rec/$NAME
[ -d "$REC" ] || { echo "!! no recording at $REC"; exit 1; }
for d in "$REC"/*/; do echo "  $(basename "$d"): $(ls "$d"/frame*.png 2>/dev/null | wc -l) frames"; done
exec /usr/bin/python3 ./tools/viz_encode.py "$REC" --fps 30 "$@"
