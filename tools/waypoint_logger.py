#!/usr/bin/env python3
"""Log TARE's proposals for the planner experiment: every /way_point together with the
robot pose at that instant -> CSV (t, rx, ry, rz, gx, gy, gz).

Pose source: /state_estimation (nav_msgs/Odometry, from the stack's SLAM) so the logger
works in every method mode including the no-provider control.

Params: out_csv (required).
"""
import numpy as np  # noqa: F401  (kept for parity; not strictly needed)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


class WaypointLogger(Node):
    def __init__(self):
        super().__init__("waypoint_logger")
        out = str(self.declare_parameter("out_csv", "/tmp/waypoints.csv").value)
        self.f = open(out, "w")
        self.f.write("t,rx,ry,rz,gx,gy,gz\n")
        self.pose = None
        self.n = 0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        pt = str(self.declare_parameter("pose_topic", "/state_estimation").value)
        self.create_subscription(Odometry, pt, self._pose, qos)
        self.create_subscription(PointStamped, "/way_point", self._goal, qos)
        # METRIC-B capture: keep the LATEST /added_obstacles map on disk (obstacles only
        # accumulate, so the last snapshot = the method's final map for path-blockage analysis)
        self.obst_ply = out.replace(".csv", "_obstacles.xyz")
        self._obst_n = 0
        self.create_subscription(PointCloud2, "/added_obstacles", self._obst, qos)
        self.get_logger().info(f"waypoint logger -> {out}")

    def _obst(self, msg):
        self._obst_n += 1
        if self._obst_n % 10 != 1:        # snapshot every 10th message
            return
        try:
            import numpy as np
            n = msg.width * msg.height
            if n == 0:
                return
            data = np.frombuffer(msg.data, np.uint8).reshape(n, msg.point_step)
            offs = {f.name: f.offset for f in msg.fields}
            xyz = np.stack([data[:, offs[k]:offs[k] + 4].copy().view("<f4").ravel()
                            for k in ("x", "y", "z")], axis=1)
            np.savetxt(self.obst_ply, xyz, fmt="%.3f")
        except Exception as e:
            self.get_logger().warn(f"obstacle snapshot failed: {e}")

    def _pose(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, p.z)

    def _goal(self, msg):
        if self.pose is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        g = msg.point
        self.f.write(f"{t:.3f},{self.pose[0]:.3f},{self.pose[1]:.3f},{self.pose[2]:.3f},"
                     f"{g.x:.3f},{g.y:.3f},{g.z:.3f}\n")
        self.f.flush()
        self.n += 1
        if self.n % 50 == 1:
            self.get_logger().info(f"{self.n} proposals logged")


def main():
    rclpy.init()
    rclpy.spin(WaypointLogger())


if __name__ == "__main__":
    main()
