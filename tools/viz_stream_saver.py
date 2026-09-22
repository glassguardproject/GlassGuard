#!/usr/bin/env python3
"""Save viz_full demo image streams as PNGs PLUS their publish timestamps.

image_saver drops the header stamp, which makes the streams impossible to align afterwards: each
stream publishes at its own irregular rate (a render that takes 30 ms and one that takes 2 ms do
not land together, and the panel thread skips frames when it falls behind). Writing the stamp next
to every frame lets the encoder rebuild a COMMON timeline and HOLD each frame until the next one
arrives -- which is what makes the separate videos play in sync.

Out: <out>/<topic>/frame%06d.png  and  <out>/<topic>/stamps.csv  (index,sec,nanosec,t_rel)
"""
import os, sys, csv, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import numpy as np, cv2

TOPICS = ["rgb_pano", "lidar_pano", "silhouette_allmasks", "floor_seed_tug",
          "plane_compete", "multiview_spill", "pillars_topdown"]


class Saver(Node):
    def __init__(self, out):
        super().__init__("viz_stream_saver")
        self.out = out
        self.n = {t: 0 for t in TOPICS}
        self.csv = {}
        self.t0 = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        for t in TOPICS:
            d = os.path.join(out, t)
            os.makedirs(d, exist_ok=True)
            f = open(os.path.join(d, "stamps.csv"), "w", newline="")
            w = csv.writer(f); w.writerow(["index", "sec", "nanosec", "t_rel", "wall"])
            self.csv[t] = (f, w)
            self.create_subscription(Image, f"/glass_killer/viz/{t}",
                                     self._make_cb(t), qos)
        self.get_logger().info(f"saving {len(TOPICS)} streams -> {out}")

    def _make_cb(self, topic):
        def cb(msg):
            try:
                img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            except Exception:
                return
            wall = time.time()                     # arrival = the moment RViz shows this image too
            i = self.n[topic]
            cv2.imwrite(os.path.join(self.out, topic, f"frame{i:06d}.png"), img)
            t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            if self.t0 is None and t > 0:
                self.t0 = t
            self.csv[topic][1].writerow(
                [i, msg.header.stamp.sec, msg.header.stamp.nanosec,
                 f"{(t - self.t0):.6f}" if (self.t0 is not None and t > 0) else "", f"{wall:.6f}"])
            self.csv[topic][0].flush()
            self.n[topic] = i + 1
        return cb

    def report(self):
        for t in TOPICS:
            print(f"  {t:22s} {self.n[t]:6d} frames")
        for f, _ in self.csv.values():
            try: f.close()
            except Exception: pass


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "./viz_rec"
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    node = Saver(out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[saver] captured:")
        node.report()
        node.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == "__main__":
    main()
