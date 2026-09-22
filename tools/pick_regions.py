#!/usr/bin/env python3
"""Pick the screen regions to record during a VIZ_REC run.

Shows a picture of the WHOLE screen (scaled to fit) and lets you drag rectangles on it. The picture is
either a live screenshot (have RViz open in the layout you will record) or a frame from an earlier
screen recording, so RViz does not need to be running.

  tools/pick_regions.py                      live screenshot of the screen right now
  tools/pick_regions.py --from <mp4|png>     use a frame of a previous screen.mp4 (taken at --at seconds)
  tools/pick_regions.py --from last          newest demo_rec/*/screen*.mp4

Mouse : drag = draw the current region.
Keys  : 1..4 choose which region to (re)draw   n next region   d delete current
        arrows nudge by 2 px (Shift: resize)   s / Enter save   q / Esc quit without saving

Saves ./demo_rec/regions.txt  (one line per region:  name WxH+X+Y, full-screen
pixels, even sizes). run_glass_killer_full.sh reads it when VIZ_REC is set.
"""
import os, sys, glob, subprocess, tempfile
import tkinter as tk
from PIL import Image, ImageTk

REC = "./demo_rec"
OUT = os.path.join(REC, "regions.txt")
COLORS = ["#00e05a", "#ff8a00", "#2aa8ff", "#ff3df0"]
FF = "/snap/bin/ffmpeg"


def grab(src, at):
    tmp = os.path.join(REC, "_pick_src.png")          # under $HOME: the snap ffmpeg cannot write to /tmp
    os.makedirs(REC, exist_ok=True)
    if src is None:
        disp = os.environ.get("DISPLAY", ":1")
        cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "x11grab", "-video_size", "1920x1080",
               "-i", f"{disp}.0+0,0", "-frames:v", "1", tmp]
    else:
        if src == "last":
            c = sorted(glob.glob(os.path.join(REC, "*", "screen*.mp4")), key=os.path.getmtime)
            if not c:
                sys.exit("no demo_rec/*/screen*.mp4 found; run without --from for a live screenshot")
            src = c[-1]
        if src.lower().endswith(".png"):
            return Image.open(src).convert("RGB"), src
        cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(at), "-i", src, "-frames:v", "1", tmp]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(tmp):
        sys.exit("could not get a picture of the screen")
    im = Image.open(tmp).convert("RGB"); im.load(); os.remove(tmp)
    return im, (src or "live screenshot")


def load_existing():
    regs = []
    if os.path.exists(OUT):
        for l in open(OUT):
            p = l.split()
            if len(p) == 2 and "x" in p[1]:
                wh, x, y = p[1].split("+"); w, h = wh.split("x")
                regs.append([int(x), int(y), int(x) + int(w), int(y) + int(h)])
    return regs


class Picker:
    def __init__(self, im, title):
        self.im = im; self.W, self.H = im.size
        self.root = tk.Tk(); self.root.title("Pick recording regions  --  " + title)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.k = min((sw - 80) / self.W, (sh - 190) / self.H, 1.0)      # display scale
        self.dw, self.dh = int(self.W * self.k), int(self.H * self.k)
        self.photo = ImageTk.PhotoImage(im.resize((self.dw, self.dh), Image.LANCZOS))
        self.cv = tk.Canvas(self.root, width=self.dw, height=self.dh, highlightthickness=0, cursor="crosshair")
        self.cv.pack(); self.cv.create_image(0, 0, image=self.photo, anchor="nw")
        self.info = tk.Label(self.root, font=("DejaVu Sans Mono", 11), justify="left", anchor="w"); self.info.pack(fill="x")
        tk.Label(self.root, fg="#555", text="drag = draw   1-4 = choose region   n = next   d = delete   "
                 "arrows = nudge (Shift: resize)   s / Enter = SAVE   q / Esc = quit").pack(fill="x")
        self.regs = load_existing() or []
        self.cur = 0; self.saved = False; self.drag = None
        self.cv.bind("<ButtonPress-1>", self.down); self.cv.bind("<B1-Motion>", self.move)
        self.cv.bind("<ButtonRelease-1>", self.up); self.root.bind("<Key>", self.key)
        self.draw(); self.root.mainloop()

    def full(self, e):  # canvas -> full-screen pixels
        return (max(0, min(self.W, round(e.x / self.k))), max(0, min(self.H, round(e.y / self.k))))

    def down(self, e): self.drag = self.full(e)

    def move(self, e):
        if self.drag is None: return
        x0, y0 = self.drag; x1, y1 = self.full(e)
        while len(self.regs) <= self.cur: self.regs.append(None)
        self.regs[self.cur] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]; self.draw()

    def up(self, e):
        self.move(e); self.drag = None
        r = self.regs[self.cur] if self.cur < len(self.regs) else None
        if r and (r[2] - r[0] < 16 or r[3] - r[1] < 16): self.regs[self.cur] = None
        self.draw()

    @staticmethod
    def even(r):
        x0, y0, x1, y1 = r; w = (x1 - x0) // 2 * 2; h = (y1 - y0) // 2 * 2
        return w, h, x0, y0

    def key(self, e):
        k = e.keysym
        if k in "1234": self.cur = int(k) - 1
        elif k == "n": self.cur = min(3, self.cur + 1)
        elif k == "d" and self.cur < len(self.regs): self.regs[self.cur] = None
        elif k in ("Left", "Right", "Up", "Down") and self.cur < len(self.regs) and self.regs[self.cur]:
            r = self.regs[self.cur]; dx = {"Left": -2, "Right": 2}.get(k, 0); dy = {"Up": -2, "Down": 2}.get(k, 0)
            if e.state & 0x1: r[2] += dx; r[3] += dy                     # Shift: resize
            else: r[0] += dx; r[2] += dx; r[1] += dy; r[3] += dy
            r[0] = max(0, r[0]); r[1] = max(0, r[1]); r[2] = min(self.W, max(r[0] + 16, r[2])); r[3] = min(self.H, max(r[1] + 16, r[3]))
        elif k in ("s", "Return"): self.save(); return
        elif k in ("q", "Escape"): self.root.destroy(); return
        self.draw()

    def draw(self):
        self.cv.delete("r"); lines = []
        for i, r in enumerate(self.regs):
            if not r: continue
            w, h, x, y = self.even(r); c = COLORS[i % 4]; k = self.k
            if w < 2 or h < 2: continue                      # drag just started: nothing to show yet
            self.cv.create_rectangle(x * k, y * k, (x + w) * k, (y + h) * k, outline=c, width=3 if i == self.cur else 2, tags="r")
            self.cv.create_text(x * k + 6, y * k + 6, anchor="nw", fill=c, font=("DejaVu Sans", 13, "bold"),
                                text=f"{i + 1}: {w}x{h}", tags="r")
            lines.append(f"region {i + 1}{' <-- drawing' if i == self.cur else ''}:  {w}x{h}+{x}+{y}   (aspect {(w / h if h else 0):.3f})")
        if not lines: lines = [f"drag on the picture to draw region {self.cur + 1}"]
        elif self.cur >= len(self.regs) or not self.regs[self.cur]: lines.append(f"drag to draw region {self.cur + 1}")
        self.info.config(text="\n".join(lines))

    def save(self):
        regs = [self.even(r) for r in self.regs if r]
        if not regs:
            self.info.config(text="nothing to save -- draw a region first"); return
        with open(OUT, "w") as f:
            for i, (w, h, x, y) in enumerate(regs):
                f.write(f"region{i + 1} {w}x{h}+{x}+{y}\n")
        self.saved = True; self.regs_out = regs; self.root.destroy()


def main():
    src = None; at = 30.0
    a = sys.argv[1:]
    for i, v in enumerate(a):
        if v == "--from" and i + 1 < len(a): src = a[i + 1]
        if v == "--at" and i + 1 < len(a): at = float(a[i + 1])
    im, title = grab(src, at)
    p = Picker(im, os.path.basename(str(title)))
    if p.saved:
        print(f"saved {len(p.regs_out)} region(s) -> {OUT}")
        for i, (w, h, x, y) in enumerate(p.regs_out): print(f"  region{i + 1}  {w}x{h}+{x}+{y}")
        print("they are used automatically by the next  VIZ_REC=<name>  run.")
    else:
        print("quit without saving; regions unchanged.")


if __name__ == "__main__":
    main()
