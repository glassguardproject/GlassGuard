#!/usr/bin/env python3
"""Encode each captured viz stream to a video on a COMMON timeline, holding every frame until the
next one arrives.

The streams publish at different, irregular rates, so gluing PNGs at a fixed framerate would let
them drift apart. Instead each frame is given a real duration (stamp[i+1] - stamp[i]) via an ffmpeg
concat list, and every video starts at the same t0. A frame therefore PERSISTS on screen until the
algorithm produces the next one -- so all clips stay in sync and can be laid side by side.

Usage: viz_encode.py <rec_dir> [--fps 30] [--speed 1.0] [--clock wall|stamp] [--start EPOCH|HH:MM:SS]

--clock wall  (default when the capture has wall times): the timeline is the WALL CLOCK at which each
              image arrived, i.e. exactly what RViz showed and when -- use this to match a screen
              recording. If <rec_dir>/screen_start.txt exists (record_viz.sh --screen), every clip
              starts at that instant and lasts as long as screen.mp4, so they line up frame for frame.
--clock stamp the old behaviour: bag/source time from the message headers (independent of bag rate).
--start       begin every clip at this wall time (for a screen recording made with another tool).
"""
import os, sys, csv, subprocess, shutil

def encoder():
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        enc = ""
    for c in ("libx264", "libopenh264"):
        if f" {c} " in enc:
            return c
    return "mpeg4"

CLOCK = "stamp"

def load(d):
    p = os.path.join(d, "stamps.csv")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                if CLOCK == "wall":
                    t = float(r["wall"])
                else:
                    t = float(r["sec"]) + float(r["nanosec"]) * 1e-9
            except (ValueError, KeyError, TypeError):
                continue
            fn = os.path.join(d, f"frame{int(r['index']):06d}.png")
            if t > 0 and os.path.exists(fn):
                rows.append((t, fn))
    rows.sort()
    return rows

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    rec = sys.argv[1]
    global CLOCK
    fps = 30.0; speed = 1.0; clock = None; start = None
    for i, a in enumerate(sys.argv):
        if a == "--fps" and i + 1 < len(sys.argv): fps = float(sys.argv[i + 1])
        if a == "--speed" and i + 1 < len(sys.argv): speed = float(sys.argv[i + 1])
        if a == "--clock" and i + 1 < len(sys.argv): clock = sys.argv[i + 1]
        if a == "--start" and i + 1 < len(sys.argv): start = sys.argv[i + 1]
    has_wall = False
    for name in os.listdir(rec):
        q = os.path.join(rec, name, "stamps.csv")
        if os.path.exists(q):
            with open(q) as f:
                has_wall = "wall" in (f.readline() or "")
            break
    CLOCK = clock or ("wall" if has_wall else "stamp")
    if CLOCK == "wall" and not has_wall:
        print("this capture has no wall-clock times (made with the old recorder): re-record it, or "
              "use --clock stamp"); return 1

    streams = {}
    for name in sorted(os.listdir(rec)):
        d = os.path.join(rec, name)
        if os.path.isdir(d):
            rows = load(d)
            if rows:
                streams[name] = rows
    if not streams:
        print(f"no stamped streams under {rec}"); return 1

    t0 = min(r[0][0] for r in streams.values())          # COMMON zero across all streams
    tend = max(r[-1][0] for r in streams.values())
    anchor = "first frame"
    if CLOCK == "wall":
        ss = os.path.join(rec, "screen_start.txt"); sv = os.path.join(rec, "screen.mp4")
        if start is not None:                              # --start EPOCH or HH:MM:SS (same day as t0)
            if ":" in start:
                import time as _t
                lt = _t.localtime(t0); hh, mm, sec = (start.split(":") + ["0"])[:3]
                t0 = _t.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, int(hh), int(mm), 0, 0, 0, lt.tm_isdst)) + float(sec)
            else:
                t0 = float(start)
            anchor = f"--start {start}"
        elif os.path.exists(ss):
            t0 = float(open(ss).read().split()[0]); anchor = "screen.mp4 start"
            se = os.path.join(rec, "screen_end.txt")       # make every clip exactly as long as the screen video
            if os.path.exists(se):
                tend = float(open(se).read().split()[0])
    print(f"clock={CLOCK}  t0 anchored to {anchor}")
    total = (tend - t0) / speed
    nframes = max(1, int(round(total * fps)))
    vcodec = encoder()
    print(f"encoder={vcodec}  span={total:.1f}s  {nframes} frames @ {fps}fps  speed={speed}x")

    # Resample each stream onto the SAME fixed grid ourselves and pipe raw frames to ffmpeg.
    # (ffmpeg's concat demuxer drops the final entry and its PTS fight -t, so clip lengths came
    # out inconsistent; sampling here is deterministic -- every clip gets exactly nframes.)
    import numpy as _np, cv2 as _cv2
    for name, rows in streams.items():
        times = [r[0] for r in rows]
        first = _cv2.imread(rows[0][1])
        if first is None:
            print(f"  {name:22s} SKIP (unreadable frames)"); continue
        h, w = first.shape[:2]
        h += h % 2; w += w % 2                            # yuv420p needs even dimensions
        out = os.path.join(rec, f"{name}.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", vcodec, "-pix_fmt", "yuv420p", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        j = 0
        cur = None
        blank = _np.zeros((h, w, 3), _np.uint8)
        for k in range(nframes):
            t = t0 + (k / fps) * speed                    # wall time of this output frame
            while j < len(times) and times[j] <= t:       # advance to the newest frame due by now
                img = _cv2.imread(rows[j][1]); j += 1
                if img is not None:
                    cur = _cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1],
                                              _cv2.BORDER_CONSTANT, value=(0, 0, 0)) \
                          if (img.shape[0] != h or img.shape[1] != w) else img
            proc.stdin.write((cur if cur is not None else blank).tobytes())   # HOLD until next
        proc.stdin.close(); proc.wait()
        print(f"  {name:22s} {len(rows):5d} frames -> {os.path.basename(out)}")
    print(f"\nall clips: {nframes} frames, {total / 1.0:.1f}s, shared t0 -- lay them side by side.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
