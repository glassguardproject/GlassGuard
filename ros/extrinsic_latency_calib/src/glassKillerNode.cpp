// extrinsicCalib_real.cpp
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include <filesystem>

#include "rclcpp/rclcpp.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "tf2/transform_datatypes.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace {
  // latest scan stamp
  std::atomic<bool> g_have_scan_stamp{false};
  builtin_interfaces::msg::Time g_last_scan_stamp;

  // pose buffer keyed by nanoseconds
  std::mutex g_state_mu;
  std::unordered_map<int64_t, geometry_msgs::msg::PoseStamped> g_state_map;
  std::deque<int64_t> g_state_order;           // for pruning
  constexpr size_t kMaxStateBuf = 500;
}

// Always pick the closest pose to t_target (no tolerance). Returns false if buffer empty.
// Also returns |Δt| in nanoseconds via out_abs_dt_ns.
static bool lookup_pose_nearest(const builtin_interfaces::msg::Time& t_target,
                                geometry_msgs::msg::PoseStamped& out_pose,
                                int64_t& out_abs_dt_ns)
{
  const int64_t k = rclcpp::Time(t_target).nanoseconds();
  std::scoped_lock lk(g_state_mu);

  if (g_state_order.empty()) return false;

  // Collect candidates: (|Δt|, key)
  std::vector<std::pair<int64_t,int64_t>> cand;
  cand.reserve(g_state_order.size());
  for (auto it = g_state_order.begin(); it != g_state_order.end(); ++it) {
    const int64_t kt = *it;
    cand.emplace_back(std::llabs(kt - k), kt);
  }

  const size_t K = std::min<size_t>(3, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });

  if (cand.empty()) return false;

  out_pose = g_state_map.at(cand[0].second);
  out_abs_dt_ns = cand[0].first;

  return true;
}

using namespace std;
using namespace cv;

const double PI = 3.1415926;

double minRange = 0.5;
double maxRange = 10.0;
double angAdjustment = 0.01;
double voxelSize = 0.02;
double imageSkipYaw = 0.05;
int imageSkipNum = 4;
int imageSkipCount = 0;
bool is360Cam = true;

int imageWidth = 1920;
int imageHeight = 640;

// 360 output saving controls
std::string outputFolder = "./360_output";
bool saveOutputs = true;
int saveInterval = 10;
double depthPngScale = 1000.0;
int frameCounter = 0;
int saveCounter = 0;
FILE* poseFile = nullptr;

// pano intrinsics (only used if is360Cam == false for undistort)
double kImage[9] = {480.0, 0, 960.5, 0, 480.0, 320.5, 0, 0, 1};
double dImage[4] = {0, 0, 0, 0};

double fx = kImage[0];
double fy = kImage[4];
double cx = kImage[2];
double cy = kImage[5];
double k1 = dImage[0];
double k2 = dImage[1];
double p1 = dImage[2];
double p2 = dImage[3];

Mat mapx, mapy;
Mat kMat, dMat;

pcl::PointCloud<pcl::PointXYZ>::Ptr scanCloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr scanCloudStack(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr scanCloudCrop(new pcl::PointCloud<pcl::PointXYZ>());
// Newest SINGLE registered scan (not stacked). Published separately so the DA2 pinhole-alignment
// step sees a single viewpoint -> no glass "see-through" that the 5s stack produces.
pcl::PointCloud<pcl::PointXYZ>::Ptr lastScanRaw(new pcl::PointCloud<pcl::PointXYZ>());

// Sliding time window: keep only the last stackTimeWindow seconds of registered scans
// so the accumulated cloud does not grow unbounded.
double stackTimeWindow = 5.0;  // seconds
double alignTimeWindow = 1.0;  // seconds of scans (~10 MID360 frames) accumulated for the DA2 align cloud
// Camera timestamp latency (s): the pose used to project the lidar into the image is looked up at
// imageTime + imageLatencyOffset. On the real robot the camera stamp lags true capture, so during motion
// the lidar lands on the wrong pixels; a small offset (usually NEGATIVE, e.g. -0.03..-0.08) re-times the
// pose to capture and removes the image<->lidar shift WITHOUT any DA2 correction. Calibrate once (rotate
// the robot, adjust until lidar edges snap to image edges; or use latencyCalib).
double imageLatencyOffset = 0.0;
// SYNC DEBUG: per-frame, log the image<->lidar time skew + robot velocity; when |skew| exceeds syncThresh
// (or the pose could not straddle the image), flag OUT-OF-SYNC, save the RGB + lidar-overlay images, and
// append a CSV row. Lets us pinpoint which stream desyncs on the real robot. Set syncDebug=false to silence.
double syncThresh = 0.05;          // seconds; above this the frame is OUT-OF-SYNC
bool   syncDebug  = false;         // OFF by default (each run can dump GBs of PNGs); enable via param
std::string syncDebugDir = "./debug/sync";
FILE*  syncLog = nullptr;
std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZ>::Ptr>> scanWindow;

const int odomStackNum = 400;
float lidarXStack[odomStackNum];
float lidarYStack[odomStackNum];
float lidarZStack[odomStackNum];
float lidarRollStack[odomStackNum];
float lidarPitchStack[odomStackNum];
float lidarYawStack[odomStackNum];
double odomTimeStack[odomStackNum];
int odomLastIDPointer = -1;
int odomFrontIDPointer = 0;

double odomTime = 0;
float odomX = 0, odomY = 0, odomZ = 0;

double camRoll = -1.5707963, camPitch = 0, camYaw = -1.5707963;
double camX = 0, camY = 0, camZ = 0;

const int imageStackNum = 10;
Mat imageStack[imageStackNum];
double imageTimeStack[imageStackNum];
builtin_interfaces::msg::Time imageStampStack[imageStackNum];
int imageLastIDPointer = -1;
int imageFrontIDPointer = 0;

float *depthArray = nullptr;
pcl::VoxelGrid<pcl::PointXYZ> downSizeFilter;

rclcpp::Node::SharedPtr nh;

void odomHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odomIn)
{
  odomTime = rclcpp::Time(odomIn->header.stamp).seconds();
  odomX = odomIn->pose.pose.position.x;
  odomY = odomIn->pose.pose.position.y;
  odomZ = odomIn->pose.pose.position.z;

  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odomIn->pose.pose.orientation;
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  odomLastIDPointer = (odomLastIDPointer + 1) % odomStackNum;
  odomTimeStack[odomLastIDPointer] = odomTime;
  lidarXStack[odomLastIDPointer] = odomX;
  lidarYStack[odomLastIDPointer] = odomY;
  lidarZStack[odomLastIDPointer] = odomZ;
  lidarRollStack[odomLastIDPointer] = roll;
  lidarPitchStack[odomLastIDPointer] = pitch;
  lidarYawStack[odomLastIDPointer] = yaw;
}

void scanHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr scanIn)
{
  g_last_scan_stamp = scanIn->header.stamp;
  g_have_scan_stamp.store(true, std::memory_order_release);

  scanCloud->clear();
  {
    // Drop GK's OWN injected glass points (marker intensity 199): with obstacle_mode
    // scan/both they ride on /registered_scan for the planner stack, and feeding them
    // back into the algorithm's input cloud self-confirms planes and poisons the
    // floor/seed depth evidence near the glass line.
    pcl::PointCloud<pcl::PointXYZI> scanXYZI;
    pcl::fromROSMsg(*scanIn, scanXYZI);
    scanCloud->reserve(scanXYZI.size());
    for (const auto &p : scanXYZI.points) {
      if (p.intensity > 198.5f && p.intensity < 199.5f) continue;
      pcl::PointXYZ q; q.x = p.x; q.y = p.y; q.z = p.z;
      scanCloud->push_back(q);
    }
  }

  // Sliding time-window accumulation: push this scan, drop scans older than
  // stackTimeWindow seconds, then rebuild the stack from what remains.
  double scanT = rclcpp::Time(scanIn->header.stamp).seconds();
  scanWindow.emplace_back(scanT,
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(*scanCloud)));
  while (!scanWindow.empty() && (scanT - scanWindow.front().first) > stackTimeWindow) {
    scanWindow.pop_front();
  }

  // DA2 alignment cloud = the last alignTimeWindow (~1s, ~10 MID360 scans) -- a SINGLE viewpoint window
  // (no 5s glass see-through) but far denser than one 100ms scan, so the pinhole lidar depth is usable.
  lastScanRaw->clear();
  for (const auto &e : scanWindow) {
    if (scanT - e.first <= alignTimeWindow) *lastScanRaw += *e.second;
  }

  scanCloudStack->clear();
  for (const auto &e : scanWindow) {
    *scanCloudStack += *e.second;
  }

  scanCloudCrop->clear();
  int scanCloudStackSize = (int)scanCloudStack->points.size();
  for (int i = 0; i < scanCloudStackSize; i++) {
    float x1 = scanCloudStack->points[i].x - odomX;
    float y1 = scanCloudStack->points[i].y - odomY;
    float z1 = scanCloudStack->points[i].z - odomZ;
    float dis = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    if (dis < maxRange) scanCloudCrop->push_back(scanCloudStack->points[i]);
  }

  scanCloudStack->clear();
  downSizeFilter.setInputCloud(scanCloudCrop);
  downSizeFilter.filter(*scanCloudStack);
}

void imageHandler(const sensor_msgs::msg::Image::ConstSharedPtr imageIn)
{
  imageSkipCount--;
  if (imageSkipCount >= 0) return;
  imageSkipCount = imageSkipNum;

  imageLastIDPointer = (imageLastIDPointer + 1) % imageStackNum;
  imageTimeStack[imageLastIDPointer] = rclcpp::Time(imageIn->header.stamp).seconds();
  imageStampStack[imageLastIDPointer] = imageIn->header.stamp;

  cv_bridge::CvImageConstPtr imageInCv = cv_bridge::toCvShare(imageIn, "bgr8");
  if (is360Cam) {
    imageInCv->image.copyTo(imageStack[imageLastIDPointer]);
  } else {
    remap(imageInCv->image, imageStack[imageLastIDPointer], mapx, mapy, cv::INTER_LINEAR);
  }
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("Glass_killer_node");

  // Publish ONLY Habitat pinhole outputs
  auto pub_rgb   = nh->create_publisher<sensor_msgs::msg::Image>("/habitat/rgb", 10);
  auto pub_depth = nh->create_publisher<sensor_msgs::msg::Image>("/habitat/depth", 10);
  auto pub_pose  = nh->create_publisher<geometry_msgs::msg::PoseStamped>("/habitat/state_estimation", 10);
  // The EXACT pose each published cloud was de-rotated with (see the publish block). depth 16 so
  // the consumer still finds the matching entry after it has been blocked in inference for a while.
  auto pub_cloud_pose = nh->create_publisher<geometry_msgs::msg::PoseStamped>("/glass_killer/cloud_pose", 16);
  // Same as extrinsicCalib, plus: publish the exact cloud that gets saved to PLY.
  auto pub_cloud = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/glass_killer/cloud", 2);
  // Newest SINGLE scan (same camera-centered viewer frame as /glass_killer/cloud) for DA2 alignment.
  auto pub_last_scan = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/glass_killer/last_scan", 2);

  // Params
  nh->declare_parameter<double>("minRange", minRange);
  nh->declare_parameter<double>("maxRange", maxRange);
  nh->declare_parameter<double>("angAdjustment", angAdjustment);
  nh->declare_parameter<double>("voxelSize", voxelSize);
  nh->declare_parameter<double>("stackTimeWindow", stackTimeWindow);
  // Camera capture latency (s): the pose used to de-rotate the stacked cloud is looked up at
  // imageTime + imageLatencyOffset. The camera's header stamp lags true exposure, so at 0.0 the
  // cloud is projected with a stale pose and the whole projection (seeds included) yaws while
  // the robot turns. Positive values move the lookup LATER. Also live-tunable with keys 7/8.
  nh->declare_parameter<double>("imageLatencyOffset", imageLatencyOffset);
  nh->declare_parameter<double>("imageSkipYaw", imageSkipYaw);
  nh->declare_parameter<int>("imageSkipNum", imageSkipNum);
  nh->declare_parameter<double>("camRoll", camRoll);
  nh->declare_parameter<double>("camPitch", camPitch);
  nh->declare_parameter<double>("camYaw", camYaw);
  nh->declare_parameter<double>("camX", camX);
  nh->declare_parameter<double>("camY", camY);
  nh->declare_parameter<double>("camZ", camZ);
  nh->declare_parameter<bool>("is360Cam", is360Cam);
  nh->declare_parameter<int>("imageWidth", imageWidth);
  nh->declare_parameter<int>("imageHeight", imageHeight);
  nh->declare_parameter<double>("fx", fx);
  nh->declare_parameter<double>("fy", fy);
  nh->declare_parameter<double>("cx", cx);
  nh->declare_parameter<double>("cy", cy);
  nh->declare_parameter<double>("k1", k1);
  nh->declare_parameter<double>("k2", k2);
  nh->declare_parameter<double>("p1", p1);
  nh->declare_parameter<double>("p2", p2);
  nh->declare_parameter<std::string>("outputFolder", outputFolder);
  nh->declare_parameter<bool>("syncDebug", syncDebug);
  nh->get_parameter("syncDebug", syncDebug);
  nh->declare_parameter<bool>("saveOutputs", saveOutputs);
  nh->declare_parameter<int>("saveInterval", saveInterval);
  nh->declare_parameter<double>("depthPngScale", depthPngScale);

  nh->get_parameter("minRange", minRange);
  nh->get_parameter("maxRange", maxRange);
  nh->get_parameter("angAdjustment", angAdjustment);
  nh->get_parameter("voxelSize", voxelSize);
  nh->get_parameter("stackTimeWindow", stackTimeWindow);
  nh->get_parameter("imageLatencyOffset", imageLatencyOffset);
  RCLCPP_INFO(nh->get_logger(), "imageLatencyOffset = %.3f s (pose lookup for cloud de-rotation; "
                                "keys 7/8 adjust by -/+5ms at runtime)", imageLatencyOffset);
  nh->get_parameter("imageSkipYaw", imageSkipYaw);
  nh->get_parameter("imageSkipNum", imageSkipNum);
  nh->get_parameter("camRoll", camRoll);
  nh->get_parameter("camPitch", camPitch);
  nh->get_parameter("camYaw", camYaw);
  nh->get_parameter("camX", camX);
  nh->get_parameter("camY", camY);
  nh->get_parameter("camZ", camZ);
  nh->get_parameter("is360Cam", is360Cam);
  nh->get_parameter("imageWidth", imageWidth);
  nh->get_parameter("imageHeight", imageHeight);
  nh->get_parameter("fx", fx);
  nh->get_parameter("fy", fy);
  nh->get_parameter("cx", cx);
  nh->get_parameter("cy", cy);
  nh->get_parameter("k1", k1);
  nh->get_parameter("k2", k2);
  nh->get_parameter("p1", p1);
  nh->get_parameter("p2", p2);
  nh->get_parameter("outputFolder", outputFolder);
  nh->get_parameter("saveOutputs", saveOutputs);
  nh->get_parameter("saveInterval", saveInterval);
  nh->get_parameter("depthPngScale", depthPngScale);

  if (saveInterval < 1) saveInterval = 1;
  if (depthPngScale <= 0.0) depthPngScale = 1000.0;

  if (saveOutputs) {
    if (!std::filesystem::exists(outputFolder)) {
      std::filesystem::create_directories(outputFolder);
      RCLCPP_INFO(nh->get_logger(), "Created output folder: %s", outputFolder.c_str());
    } else {
      RCLCPP_INFO(nh->get_logger(), "Using existing output folder: %s", outputFolder.c_str());
    }

    std::string poseFilePath = outputFolder + "/poses.csv";
    poseFile = fopen(poseFilePath.c_str(), "w");
    if (poseFile) {
      fprintf(poseFile, "frame,timestamp,x,y,z,qx,qy,qz,qw\n");
      fflush(poseFile);
      RCLCPP_INFO(nh->get_logger(), "Created pose file: %s", poseFilePath.c_str());
    } else {
      RCLCPP_WARN(nh->get_logger(), "Failed to create pose file: %s", poseFilePath.c_str());
    }
  } else {
    RCLCPP_INFO(nh->get_logger(), "360 output saving is disabled (saveOutputs=false)");
  }

  if (syncDebug) {
    std::string mk = "mkdir -p " + syncDebugDir; int _rc = system(mk.c_str()); (void)_rc;
    syncLog = fopen((syncDebugDir + "/sync_log.csv").c_str(), "w");
    if (syncLog) {
      fprintf(syncLog, "frame,image_time,scan_time,pose_time,skew_img_scan_s,pose_straddled,lin_vel_mps,ang_vel_dps,out_of_sync\n");
      fflush(syncLog);
      RCLCPP_INFO(nh->get_logger(), "[sync] logging to %s/sync_log.csv (thresh=%.0fms)", syncDebugDir.c_str(), syncThresh*1000.0);
    }
  }

  // Subs
  auto subOdom  = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odomHandler);
  auto subScan  = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan", 2, scanHandler);
  auto subImage = nh->create_subscription<sensor_msgs::msg::Image>("/camera/image", 2, imageHandler);

  // Pose buffer (for publishing nearest pose to scan stamp)
  auto subState = nh->create_subscription<nav_msgs::msg::Odometry>(
    "/state_estimation", rclcpp::SensorDataQoS(),
    [](const nav_msgs::msg::Odometry::SharedPtr msg){
      geometry_msgs::msg::PoseStamped ps;
      ps.header = msg->header;
      ps.pose   = msg->pose.pose;

      const int64_t k = rclcpp::Time(ps.header.stamp).nanoseconds();
      std::scoped_lock lk(g_state_mu);
      g_state_map[k] = std::move(ps);
      g_state_order.push_back(k);
      if (g_state_order.size() > kMaxStateBuf) {
        const int64_t oldk = g_state_order.front();
        g_state_order.pop_front();
        g_state_map.erase(oldk);
      }
    });

  RCLCPP_INFO(nh->get_logger(), "\nPress buttons to adjust camera orientation\n");

  // Undistort map (only used when is360Cam == false)
  kImage[0] = fx;  kImage[4] = fy;  kImage[2] = cx;  kImage[5] = cy;
  dImage[0] = k1;  dImage[1] = k2;  dImage[2] = p1;  dImage[3] = p2;

  cv::Size imageSize(imageWidth, imageHeight);
  kMat = cv::Mat(3, 3, CV_64FC1, kImage);
  dMat = cv::Mat(4, 1, CV_64FC1, dImage);
  mapx.create(imageSize, CV_32FC1);
  mapy.create(imageSize, CV_32FC1);
  initUndistortRectifyMap(kMat, dMat, cv::Mat(), kMat, imageSize, CV_32FC1, mapx, mapy);

  // Depth buffer for pano z-buffering
  const int imagePixelNum = imageWidth * imageHeight;
  depthArray = new float[imagePixelNum];

  downSizeFilter.setLeafSize(voxelSize, voxelSize, voxelSize);

  // OpenCV preview windows. OFF by default -- rendering three 1920x640 images per frame to the GUI adds
  // real per-frame latency and is not needed in running mode. Set showWindows=true to see them.
  bool showWindows = false;
  if (showWindows) {
    cv::namedWindow("360 RGB", cv::WINDOW_NORMAL);
    cv::namedWindow("360 Depth (Range)", cv::WINDOW_NORMAL);
    cv::namedWindow("360 Alignment", cv::WINDOW_NORMAL);
  }

  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);

    double imageTime = imageTimeStack[imageFrontIDPointer];
    if (odomLastIDPointer >= 0 && imageLastIDPointer >= 0 && odomTime > imageTime &&
        imageFrontIDPointer != (imageLastIDPointer + 1) % imageStackNum)
    {
      if (imageTime == 0) {
        imageFrontIDPointer = (imageFrontIDPointer + 1) % imageStackNum;
        (void)cv::waitKey(1);
        status = rclcpp::ok();
        continue;
      }

      cv::Mat panoRGB = imageStack[imageFrontIDPointer]; // 1920x640 (or your pano size)
      cv::Mat panoAlign = panoRGB.clone();
      builtin_interfaces::msg::Time imageStamp = imageStampStack[imageFrontIDPointer];
      imageFrontIDPointer = (imageFrontIDPointer + 1) % imageStackNum;

      // reset z-buffer helper
      std::fill(depthArray, depthArray + imagePixelNum, 0.0f);

      // pano depth as RANGE (we will later convert to pinhole Z)
      cv::Mat panoDepth32(imageHeight, imageWidth, CV_32FC1, cv::Scalar(0));

      // Re-time the POSE lookup to the camera's TRUE capture instant (imageTime + imageLatencyOffset), not
      // the possibly-delayed image header stamp. This is the non-DA2 fix for the real-robot image<->lidar
      // shift under motion; 0 = no change (e.g. for a slow/stationary rosbag).
      double poseQueryTime = imageTime + imageLatencyOffset;

      // advance odomFront pointer to straddle the pose-query time
      while (odomFrontIDPointer != odomLastIDPointer) {
        if (odomTimeStack[odomFrontIDPointer] > poseQueryTime) break;
        odomFrontIDPointer = (odomFrontIDPointer + 1) % odomStackNum;
      }

      bool depthProj = true;
      float lidarRoll = 0, lidarPitch = 0, lidarYaw = 0;
      float lidarX = 0, lidarY = 0, lidarZ = 0;

      if (odomTimeStack[odomFrontIDPointer] < poseQueryTime) {
        lidarX = lidarXStack[odomFrontIDPointer];
        lidarY = lidarYStack[odomFrontIDPointer];
        lidarZ = lidarZStack[odomFrontIDPointer];
        lidarRoll = lidarRollStack[odomFrontIDPointer];
        lidarPitch = lidarPitchStack[odomFrontIDPointer];
        lidarYaw = lidarYawStack[odomFrontIDPointer];
        depthProj = false;
      } else {
        // FIXED: avoid negative modulo
        int odomBackIDPointer = (odomFrontIDPointer - 1 + odomStackNum) % odomStackNum;

        float denom = (float)(odomTimeStack[odomFrontIDPointer] - odomTimeStack[odomBackIDPointer]);
        if (denom <= 1e-6f) {
          depthProj = false;
        } else {
          float ratioFront = (float)((poseQueryTime - odomTimeStack[odomBackIDPointer]) / denom);
          float ratioBack  = (float)((odomTimeStack[odomFrontIDPointer] - poseQueryTime) / denom);

          if (lidarYawStack[odomFrontIDPointer] - lidarYawStack[odomBackIDPointer] > PI) {
            lidarYawStack[odomBackIDPointer] += 2 * PI;
          } else if (lidarYawStack[odomFrontIDPointer] - lidarYawStack[odomBackIDPointer] < -PI) {
            lidarYawStack[odomBackIDPointer] -= 2 * PI;
          }

          lidarX = lidarXStack[odomFrontIDPointer] * ratioFront + lidarXStack[odomBackIDPointer] * ratioBack;
          lidarY = lidarYStack[odomFrontIDPointer] * ratioFront + lidarYStack[odomBackIDPointer] * ratioBack;
          lidarZ = lidarZStack[odomFrontIDPointer] * ratioFront + lidarZStack[odomBackIDPointer] * ratioBack;
          lidarRoll  = lidarRollStack[odomFrontIDPointer]  * ratioFront + lidarRollStack[odomBackIDPointer]  * ratioBack;
          lidarPitch = lidarPitchStack[odomFrontIDPointer] * ratioFront + lidarPitchStack[odomBackIDPointer] * ratioBack;
          lidarYaw   = lidarYawStack[odomFrontIDPointer]   * ratioFront + lidarYawStack[odomBackIDPointer]   * ratioBack;

          float deltaYaw = std::fabs(lidarYawStack[odomFrontIDPointer] - lidarYawStack[odomBackIDPointer]);
          if (deltaYaw > (float)imageSkipYaw) depthProj = false;
        }
      }

      if (depthProj) {
        float sinCamRoll = std::sin((float)camRoll),  cosCamRoll = std::cos((float)camRoll);
        float sinCamPitch = std::sin((float)camPitch), cosCamPitch = std::cos((float)camPitch);
        float sinCamYaw = std::sin((float)camYaw),   cosCamYaw = std::cos((float)camYaw);

        float sinLidarRoll = std::sin(lidarRoll),  cosLidarRoll = std::cos(lidarRoll);
        float sinLidarPitch = std::sin(lidarPitch), cosLidarPitch = std::cos(lidarPitch);
        float sinLidarYaw = std::sin(lidarYaw),   cosLidarYaw = std::cos(lidarYaw);

        // Same lidar->cam extrinsic transform as the per-point loop below, into the viewer z-up frame
        // (x8, z8, -y8). Used to transform the LAST single scan identically to the stacked panoCloud.
        auto toViewerPt = [&](const pcl::PointXYZ &p, pcl::PointXYZ &out) -> bool {
          float x1 = p.x - lidarX, y1 = p.y - lidarY, z1 = p.z - lidarZ;
          float dis = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
          if (dis < (float)minRange || dis > (float)maxRange) return false;
          float x2 = x1 * cosLidarYaw + y1 * sinLidarYaw;
          float y2 = -x1 * sinLidarYaw + y1 * cosLidarYaw;
          float z2 = z1;
          float x3 = x2 * cosLidarPitch - z2 * sinLidarPitch;
          float y3 = y2;
          float z3 = x2 * sinLidarPitch + z2 * cosLidarPitch;
          float x4 = x3;
          float y4 = y3 * cosLidarRoll + z3 * sinLidarRoll;
          float z4 = -y3 * sinLidarRoll + z3 * cosLidarRoll;
          float x5 = x4 - (float)camX, y5 = y4 - (float)camY, z5 = z4 - (float)camZ;
          float x6 = x5 * cosCamYaw + y5 * sinCamYaw;
          float y6 = -x5 * sinCamYaw + y5 * cosCamYaw;
          float z6 = z5;
          float x7 = x6 * cosCamPitch - z6 * sinCamPitch;
          float y7 = y6;
          float z7 = x6 * sinCamPitch + z6 * cosCamPitch;
          float x8 = x7;
          float y8 = y7 * cosCamRoll + z7 * sinCamRoll;
          float z8 = -y7 * sinCamRoll + z7 * cosCamRoll;
          out = pcl::PointXYZ(x8, z8, -y8);
          return true;
        };

        pcl::PointCloud<pcl::PointXYZ> panoCloud;
        panoCloud.points.reserve(scanCloudStack->points.size());

        int scanCloudStackSize = (int)scanCloudStack->points.size();
        for (int i = 0; i < scanCloudStackSize; i++) {
          float x1 = scanCloudStack->points[i].x - lidarX;
          float y1 = scanCloudStack->points[i].y - lidarY;
          float z1 = scanCloudStack->points[i].z - lidarZ;

          float dis = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
          if (dis < (float)minRange || dis > (float)maxRange) continue;

          float x2 = x1 * cosLidarYaw + y1 * sinLidarYaw;
          float y2 = -x1 * sinLidarYaw + y1 * cosLidarYaw;
          float z2 = z1;

          float x3 = x2 * cosLidarPitch - z2 * sinLidarPitch;
          float y3 = y2;
          float z3 = x2 * sinLidarPitch + z2 * cosLidarPitch;

          float x4 = x3;
          float y4 = y3 * cosLidarRoll + z3 * sinLidarRoll;
          float z4 = -y3 * sinLidarRoll + z3 * cosLidarRoll;

          float x5 = x4 - (float)camX;
          float y5 = y4 - (float)camY;
          float z5 = z4 - (float)camZ;

          float x6 = x5 * cosCamYaw + y5 * sinCamYaw;
          float y6 = -x5 * sinCamYaw + y5 * cosCamYaw;
          float z6 = z5;

          float x7 = x6 * cosCamPitch - z6 * sinCamPitch;
          float y7 = y6;
          float z7 = x6 * sinCamPitch + z6 * cosCamPitch;

          float x8 = x7;
          float y8 = y7 * cosCamRoll + z7 * sinCamRoll;
          float z8 = -y7 * sinCamRoll + z7 * cosCamRoll;

          // pano pixel
          float horiDis = std::sqrt(x8 * x8 + z8 * z8);
          int u = (int)(imageWidth  / (2 * PI) * std::atan2(x8, z8) + imageWidth  / 2 + 1);
          int v = (int)(imageWidth  / (2 * PI) * std::atan(y8 / (horiDis + 1e-6f)) + imageHeight / 2 + 1);

          if (u < 1 || u >= imageWidth - 1 || v < 1 || v >= imageHeight - 1) continue;

          // Save in the same Z-up viewer convention as the old pinhole node:
          // camera frame x=right, y=down, z=forward -> viewer x=right, y=forward, z=up.
          panoCloud.points.emplace_back(x8, z8, -y8);

          // Store RANGE depth in pano (Euclidean range)
          float range = std::sqrt(x8*x8 + y8*y8 + z8*z8);

          // 3x3 splat disabled: keep only the center pixel update.
          int pixelID = imageWidth * v + u;
          if (depthArray[pixelID] == 0.0f || depthArray[pixelID] > horiDis) {
            depthArray[pixelID] = horiDis;
            float r_clamped = std::clamp(range, (float)minRange, (float)maxRange);
            float& dref = panoDepth32.at<float>(v, u);
            if (dref == 0.0f || r_clamped < dref) dref = r_clamped;

            int pixelVal = (int)(255.0f * (horiDis - (float)minRange) / ((float)maxRange - (float)minRange));
            pixelVal = std::max(0, std::min(255, pixelVal));
            panoAlign.data[3 * pixelID] = (uchar)pixelVal;
            panoAlign.data[3 * pixelID + 1] = (uchar)(255 - pixelVal);
          }
        }

        // LAST single scan (no stacking -> no glass see-through), transformed identically, for DA2 align.
        pcl::PointCloud<pcl::PointXYZ> lastPanoCloud;
        lastPanoCloud.points.reserve(lastScanRaw->points.size());
        for (const auto &p : lastScanRaw->points) {
          pcl::PointXYZ vp;
          if (toViewerPt(p, vp)) lastPanoCloud.points.emplace_back(vp);
        }

        // Visualize 360 depth as 8-bit (range)
        cv::Mat panoDepth8(imageHeight, imageWidth, CV_8UC1, cv::Scalar(0));
        cv::Mat panoDepth16(imageHeight, imageWidth, CV_16UC1, cv::Scalar(0));
        for (int vv = 0; vv < imageHeight; ++vv) {
          for (int uu = 0; uu < imageWidth; ++uu) {
            float d = panoDepth32.at<float>(vv, uu);
            if (d <= 0.0f) continue;

            uint16_t d_png = (uint16_t)std::min(65535.0, std::round((double)d * depthPngScale));
            panoDepth16.at<uint16_t>(vv, uu) = d_png;

            float t = (d - (float)minRange) / ((float)maxRange - (float)minRange);
            t = std::max(0.0f, std::min(1.0f, t));
            panoDepth8.at<uchar>(vv, uu) = (uchar)std::lround(t * 255.0f);
          }
        }

        // ---- SYNC DEBUG: per-frame image<->lidar time skew + robot velocity; save RGB + lidar overlay
        // (and range colormap) whenever |skew| exceeds syncThresh, plus a CSV row for offline analysis. ----
        if (syncDebug) {
          static int    syncFrame = 0;
          static double prevT = -1, prevX = 0, prevY = 0, prevZ = 0, prevYaw = 0;
          double scanTnow = rclcpp::Time(g_last_scan_stamp).seconds();
          double skew = imageTime - scanTnow;                       // image stamp vs LATEST lidar scan (s)
          double linVel = 0.0, angVel = 0.0;
          if (prevT > 0) {
            double dt = imageTime - prevT;
            if (dt > 1e-4) {
              linVel = std::sqrt((lidarX-prevX)*(lidarX-prevX) + (lidarY-prevY)*(lidarY-prevY) +
                                 (lidarZ-prevZ)*(lidarZ-prevZ)) / dt;
              double dyaw = lidarYaw - prevYaw;
              while (dyaw >  PI) dyaw -= 2*PI;
              while (dyaw < -PI) dyaw += 2*PI;
              angVel = (dyaw / dt) * 180.0 / PI;                    // deg/s (yaw rate = the misalignment driver)
            }
          }
          prevT = imageTime; prevX = lidarX; prevY = lidarY; prevZ = lidarZ; prevYaw = lidarYaw;
          bool outOfSync = std::fabs(skew) > syncThresh;
          RCLCPP_INFO(nh->get_logger(),
            "[sync] f=%d img=%.3f scan=%.3f skew=%+.0fms lin=%.2fm/s ang=%+.0fdeg/s %s",
            syncFrame, imageTime, scanTnow, skew*1000.0, linVel, angVel, outOfSync ? "OUT-OF-SYNC" : "ok");
          if (syncLog) {
            fprintf(syncLog, "%d,%.6f,%.6f,%.6f,%.5f,%d,%.4f,%.4f,%d\n",
                    syncFrame, imageTime, scanTnow, poseQueryTime, skew, 1, linVel, angVel, outOfSync ? 1 : 0);
            fflush(syncLog);
          }
          if (outOfSync) {                                          // save for visual inspection
            std::string base = cv::format("%s/f%06d_skew%+05dms_ang%+04d", syncDebugDir.c_str(),
                                          syncFrame, (int)std::lround(skew*1000.0), (int)std::lround(angVel));
            cv::imwrite(base + "_rgb.png", panoRGB);
            cv::imwrite(base + "_overlay.png", panoAlign);          // RGB with lidar projection colored on it
            cv::imwrite(base + "_range.png", panoDepth8);           // lidar range colormap
          }
          syncFrame++;
        }

        if (showWindows) {
          cv::imshow("360 RGB", panoRGB);
          cv::imshow("360 Depth (Range)", panoDepth8);
          cv::imshow("360 Alignment", panoAlign);
        }

        if (saveOutputs && saveCounter % saveInterval == 0) {
          std::string frameNum = cv::format("%06d", frameCounter);
          std::string rgbFile = outputFolder + "/rgb_" + frameNum + ".png";
          std::string depthFile = outputFolder + "/depth_" + frameNum + ".png";
          std::string depthPreviewFile = outputFolder + "/depth_preview_" + frameNum + ".png";
          std::string alignFile = outputFolder + "/align_" + frameNum + ".png";
          std::string cloudFile = outputFolder + "/cloud_" + frameNum + ".ply";

          cv::imwrite(rgbFile, panoRGB);
          cv::imwrite(depthFile, panoDepth16);
          cv::imwrite(depthPreviewFile, panoDepth8);
          cv::imwrite(alignFile, panoAlign);

          panoCloud.width = (uint32_t)panoCloud.points.size();
          panoCloud.height = 1;
          panoCloud.is_dense = false;
          int plyStatus = pcl::io::savePLYFileBinary(cloudFile, panoCloud);

          if (poseFile) {
            geometry_msgs::msg::PoseStamped pose_out;
            int64_t dt_pose_ns = 0;
            if (lookup_pose_nearest(imageStamp, pose_out, dt_pose_ns)) {
              fprintf(poseFile, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                      frameCounter, rclcpp::Time(imageStamp).seconds(),
                      pose_out.pose.position.x, pose_out.pose.position.y, pose_out.pose.position.z,
                      pose_out.pose.orientation.x, pose_out.pose.orientation.y,
                      pose_out.pose.orientation.z, pose_out.pose.orientation.w);
              fflush(poseFile);
            }
          }

          RCLCPP_INFO(nh->get_logger(),
                      "Saved frame %d: %s, %s, %s, %s, %s (ply=%s, points=%zu)",
                      frameCounter, rgbFile.c_str(), depthFile.c_str(),
                      depthPreviewFile.c_str(), alignFile.c_str(), cloudFile.c_str(),
                      (plyStatus == 0 ? "ok" : "failed"), panoCloud.points.size());
        }

        frameCounter++;
        saveCounter++;

        // ---------------- Publish with scan stamp ----------------
        if (!g_have_scan_stamp.load(std::memory_order_acquire)) {
          RCLCPP_WARN_THROTTLE(nh->get_logger(), *nh->get_clock(), 2000,
                               "No /registered_scan yet; skip publish");
        } else {
          std_msgs::msg::Header header;
          header.stamp    = g_last_scan_stamp;
          header.frame_id = "camera_link";

          auto rgb_msg   = cv_bridge::CvImage(header, "bgr8",  panoRGB).toImageMsg();
          auto depth_msg = cv_bridge::CvImage(header, "32FC1", panoDepth32).toImageMsg();
          pub_rgb->publish(*rgb_msg);
          pub_depth->publish(*depth_msg);

          // The one difference vs extrinsicCalib: also publish the exact cloud
          // that gets written to PLY (camera-centered, same stamp/frame as RGB).
          sensor_msgs::msg::PointCloud2 cloud_msg;
          pcl::toROSMsg(panoCloud, cloud_msg);
          cloud_msg.header = header;
          pub_cloud->publish(cloud_msg);

          // The EXACT pose this cloud was de-rotated with: the interpolated pose at
          // poseQueryTime = imageTime + imageLatencyOffset, i.e. the camera's capture instant.
          // Stamped IDENTICALLY to the cloud. The consumer must transform the cloud back to the
          // world with THIS pose. Matching the cloud's (scan) stamp against a raw odom buffer
          // instead -- which is what happens with only the republish above -- re-rotates every
          // output by however far the robot moved between the image and the newest scan: a
          // random 0-100 ms per frame, which shows up as seeds/rays yawing and jittering in turns.
          {
            geometry_msgs::msg::PoseStamped cp;
            cp.header.stamp = header.stamp;
            cp.header.frame_id = "map";
            cp.pose.position.x = lidarX; cp.pose.position.y = lidarY; cp.pose.position.z = lidarZ;
            tf2::Quaternion q; q.setRPY(lidarRoll, lidarPitch, lidarYaw);   // inverse of odomHandler's getRPY
            cp.pose.orientation.x = q.x(); cp.pose.orientation.y = q.y();
            cp.pose.orientation.z = q.z(); cp.pose.orientation.w = q.w();
            pub_cloud_pose->publish(cp);
          }

          // Newest single scan (same frame/stamp) for the node's DA2 pinhole alignment.
          sensor_msgs::msg::PointCloud2 last_msg;
          pcl::toROSMsg(lastPanoCloud, last_msg);
          last_msg.header = header;
          pub_last_scan->publish(last_msg);

          // Publish nearest pose to scan stamp (without re-stamping)
          geometry_msgs::msg::PoseStamped pose_out;
          int64_t dt_pose_ns = 0;
          if (lookup_pose_nearest(g_last_scan_stamp, pose_out, dt_pose_ns)) {
            pub_pose->publish(pose_out);
          }
        }

      } else {
        RCLCPP_INFO(nh->get_logger(), "Skipping frame\n");
      }
    }

    // Keep your interactive tuning keys alive AND allow imshow to refresh
    char c = (char)cv::waitKey(1);
    if      (c == '1') camRoll  -= angAdjustment;
    else if (c == '2') camRoll  += angAdjustment;
    else if (c == '3') camPitch -= angAdjustment;
    else if (c == '4') camPitch += angAdjustment;
    else if (c == '5') camYaw   -= angAdjustment;
    else if (c == '6') camYaw   += angAdjustment;
    // 7/8: camera time-LATENCY offset (s). Rotate the robot and adjust until the lidar snaps onto the image.
    else if (c == '7') { imageLatencyOffset -= 0.005; RCLCPP_INFO(nh->get_logger(), "imageLatencyOffset = %.3f s", imageLatencyOffset); }
    else if (c == '8') { imageLatencyOffset += 0.005; RCLCPP_INFO(nh->get_logger(), "imageLatencyOffset = %.3f s", imageLatencyOffset); }

    status = rclcpp::ok();
  }

  delete[] depthArray;
  depthArray = nullptr;

  if (poseFile) {
    fclose(poseFile);
    RCLCPP_INFO(nh->get_logger(), "Closed pose file");
  }

  return 0;
}
