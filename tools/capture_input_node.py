#!/usr/bin/env python3
"""CAPTURE-ONLY input recorder: subscribes to the provider's cloud + the camera + state
estimation and dumps algorithm-input frames at the HIGHEST rate the provider pairs them --
NO inference, NO pipeline, so recording is never throttled by the algorithm.

Output (realTestSaver format, batch-replayable):
  cloud_%06d.ply   binary little-endian xyz+rgb (fast; batch loader reads binary)
  rgb_%06d.jpg     decoded pano image nearest the cloud stamp
  pose_%06d.txt    px py pz qx qy qz qw -- pose INTERPOLATED to the cloud's header stamp
  times.txt        fid + cloud header stamp (sec) -- for TRUE timestamp subsampling later

Params: out_dir, max_img_dt (default 0.35s pairing window).
"""
import os
import struct
import threading
import queue
from collections import deque

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, CompressedImage
from geometry_msgs.msg import PoseStamped


def write_ply_binary(path, xyz):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n").encode()
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"] = rec["g"] = rec["b"] = 128
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(rec.tobytes())


class CaptureNode(Node):
    def __init__(self):
        super().__init__("capture_input_node")
        self.out = str(self.declare_parameter("out_dir",
                       "./capture_only_input").value)
        self.max_img_dt = float(self.declare_parameter("max_img_dt", 0.35).value)
        # WORLD-SCAN mode (default): save the cloud as a 5s stack of /registered_scan in
        # the MAP frame -- annotation and eval consume world coordinates directly, no
        # viewer->world alignment ever. The provider cloud still TRIGGERS the pairing.
        self.world_scan = bool(self.declare_parameter("world_scan", True).value)
        self.stack_s = float(self.declare_parameter("stack_s", 5.0).value)
        self.crop_m = float(self.declare_parameter("crop_m", 10.0).value)
        self.save_voxel = float(self.declare_parameter("save_voxel", 0.05).value)
        self.scans = deque()              # (t, xyz world) from /registered_scan
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "frame_convention.txt"), "w") as cf:
            cf.write("cloud frame: %s\n" % ("world/map (registered scans, no alignment needed)"
                                            if self.world_scan else "viewer (provider cloud)"))
        self.fid = 0
        self.imgs = deque(maxlen=60)      # (t, CompressedImage)
        self.poses = deque(maxlen=400)    # (t, x,y,z,qx,qy,qz,qw)
        self.q = queue.Queue(maxsize=64)
        self.dropped = 0
        self.times = open(os.path.join(self.out, "times.txt"), "w")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PointCloud2, "/glass_killer/cloud", self._cloud, qos)
        if self.world_scan:
            self.create_subscription(PointCloud2, "/registered_scan", self._scan, qos)
        self.create_subscription(CompressedImage, "/camera/image/compressed", self._img, qos)
        self.create_subscription(PoseStamped, "/habitat/state_estimation", self._pose, qos)
        threading.Thread(target=self._writer, daemon=True).start()
        self.n_cloud = 0; self.n_img = 0; self.n_pose = 0
        self.create_timer(5.0, self._stats)
        self.get_logger().info(f"capture-only -> {self.out}")

    def _stats(self):
        self.get_logger().info(f"[rx 5s] cloud={self.n_cloud} img={self.n_img} "
                               f"pose={self.n_pose} saved={self.fid} "
                               f"pair_dt={getattr(self,'last_pair_dt',-1):+.2f}s "
                               f"(cloud_t-img_t; + = cloud AHEAD of newest img)")
        self.n_cloud = self.n_img = self.n_pose = 0

    @staticmethod
    def _t(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _scan(self, msg):
        if any(f.name == "rgb" for f in msg.fields):
            return                                     # glass injection, not a real scan
        t = self._t(msg.header.stamp)
        n = msg.width * msg.height
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
        offs = {f.name: f.offset for f in msg.fields}
        xyz = np.stack([data[:, offs[k]:offs[k] + 4].copy().view("<f4").ravel()
                        for k in ("x", "y", "z")], axis=1)
        xyz = xyz[np.isfinite(xyz).all(axis=1)].astype(np.float32)
        self.scans.append((t, xyz))
        cut = t - self.stack_s - 2.0
        while self.scans and self.scans[0][0] < cut:
            self.scans.popleft()

    def _img(self, msg):
        self.n_img += 1
        self.imgs.append((self._t(msg.header.stamp), msg))

    def _pose(self, msg):
        self.n_pose += 1
        p, q = msg.pose.position, msg.pose.orientation
        self.poses.append((self._t(msg.header.stamp),
                           p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def _pose_at(self, t):
        P = list(self.poses)
        if not P:
            return None
        lo = None; hi = None
        for rec in P:
            if rec[0] <= t:
                lo = rec
            elif hi is None:
                hi = rec
                break
        if lo is None:
            return P[0][1:]
        if hi is None or hi[0] - lo[0] < 1e-6:
            return lo[1:]
        a = (t - lo[0]) / (hi[0] - lo[0])
        pos = [(1 - a) * lo[i] + a * hi[i] for i in (1, 2, 3)]
        # nlerp quaternion (small inter-sample rotations)
        q0 = np.array(lo[4:8]); q1 = np.array(hi[4:8])
        if float(q0 @ q1) < 0:
            q1 = -q1
        qn = (1 - a) * q0 + a * q1
        qn /= (np.linalg.norm(qn) + 1e-12)
        return (*pos, *qn.tolist())

    def _cloud(self, msg):
        self.n_cloud += 1
        t = self._t(msg.header.stamp)
        if self.imgs:
            self.last_pair_dt = t - self.imgs[-1][0]
        img = None; best = self.max_img_dt
        for it, m in self.imgs:
            if abs(it - t) <= best:
                best = abs(it - t); img = m
        pose = self._pose_at(t)
        if img is None or pose is None:
            return
        payload = msg
        if self.world_scan:
            win = [x for (ts, x) in self.scans if t - self.stack_s <= ts <= t + 0.05]
            if not win:
                return
            payload = np.concatenate(win)
        try:
            self.q.put_nowait((self.fid, t, payload, img, pose))
            self.fid += 1
        except queue.Full:
            self.dropped += 1
            if self.dropped % 10 == 1:
                self.get_logger().warn(f"writer busy: {self.dropped} frames dropped")

    def _writer(self):
        while rclpy.ok():
            fid, t, cloud, img, pose = self.q.get()
            try:
                if isinstance(cloud, np.ndarray):
                    # WORLD-frame scan stack: crop around the robot, voxel-thin, save AS-IS
                    xyz = cloud
                    ctr = np.asarray(pose[:3], np.float32)
                    xyz = xyz[np.linalg.norm(xyz - ctr[None, :], axis=1) <= self.crop_m]
                    if self.save_voxel > 0 and len(xyz):
                        kk = np.floor(xyz / self.save_voxel).astype(np.int64)
                        _, ui = np.unique(kk, axis=0, return_index=True)
                        xyz = xyz[ui]
                else:
                    n = cloud.width * cloud.height
                    data = np.frombuffer(cloud.data, dtype=np.uint8).reshape(n, cloud.point_step)
                    offs = {f.name: f.offset for f in cloud.fields}
                    xyz = np.stack([data[:, offs[k]:offs[k] + 4].copy().view("<f4").ravel()
                                    for k in ("x", "y", "z")], axis=1)
                    xyz = xyz[np.isfinite(xyz).all(axis=1)]
                write_ply_binary(os.path.join(self.out, f"cloud_{fid:06d}.ply"), xyz)
                bgr = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(self.out, f"rgb_{fid:06d}.jpg"), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(os.path.join(self.out, f"pose_{fid:06d}.txt"), "w") as pf:
                    pf.write(" ".join(f"{v:.6f}" for v in pose) + "\n")
                self.times.write(f"{fid:06d} {t:.6f}\n")
                self.times.flush()
                if fid % 20 == 0:
                    self.get_logger().info(f"captured {fid + 1} frames")
            except Exception as e:
                self.get_logger().error(f"frame {fid} failed: {e}")
            finally:
                self.q.task_done()


def main():
    rclpy.init()
    n = CaptureNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
