#!/usr/bin/env python3
"""Clean-scan relay for rviz: forward /registered_scan -> /registered_scan_clean, DROPPING
the glass-injection messages so the display shows only the real SLAM scan.

/registered_scan keeps carrying real scan + glass (terrain, local planner, and TARE all
consume it unchanged -- planners MUST see glass). Glass injections are identified
deterministically: every glass publisher marks its messages with an extra 'rgb' field;
real SLAM scans have none.

FOV mode (display-only, for pinhole/baseline videos): with fov_crop:=true the clean scan
is additionally cropped to the pinhole camera frustum around the live robot heading, so
rviz shows exactly what the method's camera observes. The planner still gets the full
360 scan -- this never touches /registered_scan itself."""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped


class CleanScanRelay(Node):
    def __init__(self):
        super().__init__("scan_clean_relay")
        self.fov_crop = bool(self.declare_parameter("fov_crop", False).value)
        # pinhole-C frustum: 1260x1008 @ f=520 -> half-angles atan(630/520), atan(504/520)
        self.half_h = math.radians(float(self.declare_parameter("fov_h_deg", 101.0).value) / 2.0)
        self.half_v = math.radians(float(self.declare_parameter("fov_v_deg", 88.0).value) / 2.0)
        self.pose = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(PointCloud2, "/registered_scan_clean", 5)
        # ADAPTIVE TARE FEED: TARE subscribes /registered_scan_tare. 360 method -> full scan;
        # pinhole/baselines (fov_crop) -> scan cropped to the camera frustum, so the planner
        # explores with each method's actual perception budget. Glass messages pass through
        # untouched (already FOV-limited by the method itself).
        self.pub_tare = self.create_publisher(PointCloud2, "/registered_scan_tare", 5)
        self.create_subscription(PointCloud2, "/registered_scan", self._on_scan, qos)
        if self.fov_crop:
            self.create_subscription(PoseStamped, "/habitat/state_estimation",
                                     self._on_pose, qos)
        self.get_logger().info(
            f"clean /registered_scan -> /registered_scan_clean "
            f"(glass dropped{'; FOV-cropped to pinhole frustum' if self.fov_crop else ''})")

    def _on_pose(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, p.z, yaw)

    def _sane(self, msg: PointCloud2):
        """Remove non-finite / absurdly far points: ONE 300m rogue return makes TARE's 0.2m
        voxel filter overflow its int32 indices (the '[pcl::VoxelGrid] Leaf size is too small'
        spam) and the scan stack stops downsampling."""
        off = {f.name: f.offset for f in msg.fields}
        step = msg.point_step
        raw = np.frombuffer(msg.data, np.uint8).reshape(-1, step)
        xyz = np.stack([raw[:, off[a]:off[a] + 4].copy().view(np.float32)[:, 0]
                        for a in ("x", "y", "z")], 1)
        keep = np.isfinite(xyz).all(1) & (np.abs(xyz) < 500.0).all(1)
        if keep.all():
            return msg
        return PointCloud2(header=msg.header, height=1, width=int(keep.sum()),
                           fields=msg.fields, is_bigendian=msg.is_bigendian,
                           point_step=step, row_step=step * int(keep.sum()),
                           data=raw[keep].tobytes(), is_dense=True)

    def _on_scan(self, msg: PointCloud2):
        if any(f.name == "rgb" for f in msg.fields):
            self.pub_tare.publish(msg)                   # glass: TARE yes, clean view no
            return
        try:
            msg = self._sane(msg)                        # drop rogue far points (PCL overflow)
        except Exception:
            pass
        if not self.fov_crop or self.pose is None:
            self.pub.publish(msg)
            self.pub_tare.publish(msg)
            return
        try:
            off = {f.name: f.offset for f in msg.fields}
            step = msg.point_step
            raw = np.frombuffer(msg.data, np.uint8).reshape(-1, step)
            xyz = np.stack([raw[:, off[a]:off[a] + 4].copy().view(np.float32)[:, 0]
                            for a in ("x", "y", "z")], 1)
            px, py, pz, yaw = self.pose
            dx, dy, dz = xyz[:, 0] - px, xyz[:, 1] - py, xyz[:, 2] - pz
            bearing = np.arctan2(dy, dx) - yaw
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
            horiz = np.hypot(dx, dy)
            elev = np.arctan2(dz, np.maximum(horiz, 1e-6))
            keep = (np.abs(bearing) <= self.half_h) & (np.abs(elev) <= self.half_v)
            if not keep.any():
                return
            out = PointCloud2(header=msg.header, height=1, width=int(keep.sum()),
                              fields=msg.fields, is_bigendian=msg.is_bigendian,
                              point_step=step, row_step=step * int(keep.sum()),
                              data=raw[keep].tobytes(), is_dense=True)
            self.pub.publish(out)
            self.pub_tare.publish(out)
        except Exception:
            self.pub.publish(msg)                        # never break the display on a parse hiccup
            self.pub_tare.publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(CleanScanRelay())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
