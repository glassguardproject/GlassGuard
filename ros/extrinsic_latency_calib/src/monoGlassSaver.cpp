// monoGlassSaver.cpp
//
// Data-saver node for MonoGlass3D: play a rosbag as usual, run this node, and it
// saves everything MonoGlass3D needs, one subfolder per saved frame:
//
//   <outputFolder>/<%06d>/<%06d>_image.png            pinhole RGB (customizable size/intrinsics/view dir)
//   <outputFolder>/<%06d>/<%06d>_camera_config.json   {"camera_internal": {fx, fy, cx, cy}} (MonoGlass3D format)
//   <outputFolder>/<%06d>/<%06d>_pc_depth_map.txt     sparse "u v Z" lidar depth rows (toggle: saveDepth)
//   <outputFolder>/<%06d>/<%06d>_depth.png            dense 16-bit Z depth, mm (toggle: saveDepth)
//   <outputFolder>/<%06d>/<%06d>_cloud.ply            last-5s sliding-window scan stack, viewer frame,
//                                                     same wiring as glassKillerNode (toggle: savePly)
//   <outputFolder>/poses.csv                          frame,timestamp,x,y,z,qx,qy,qz,qw
//
// Input wiring is identical to glassKillerNode: /state_estimation, /registered_scan, /camera/image,
// with the same sliding stackTimeWindow accumulation (NOT an unbounded stack).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <utility>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "tf2/transform_datatypes.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace {
  std::atomic<bool> g_have_scan_stamp{false};
  builtin_interfaces::msg::Time g_last_scan_stamp;

  std::mutex g_state_mu;
  std::unordered_map<int64_t, geometry_msgs::msg::PoseStamped> g_state_map;
  std::deque<int64_t> g_state_order;
  constexpr size_t kMaxStateBuf = 500;
}

static bool lookup_pose_nearest(const builtin_interfaces::msg::Time& t_target,
                                geometry_msgs::msg::PoseStamped& out_pose,
                                int64_t& out_abs_dt_ns)
{
  const int64_t k = rclcpp::Time(t_target).nanoseconds();
  std::scoped_lock lk(g_state_mu);
  if (g_state_order.empty()) return false;

  std::vector<std::pair<int64_t,int64_t>> cand;
  cand.reserve(g_state_order.size());
  for (const int64_t kt : g_state_order) cand.emplace_back(std::llabs(kt - k), kt);

  const size_t K = std::min<size_t>(3, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });

  out_pose = g_state_map.at(cand[0].second);
  out_abs_dt_ns = cand[0].first;
  return true;
}

using namespace std;
using namespace cv;

const double PI = 3.1415926;

double minRange = 0.5;
double maxRange = 10.0;
double voxelSize = 0.02;
double imageSkipYaw = 0.05;
int imageSkipNum = 4;
int imageSkipCount = 0;
bool is360Cam = true;

int imageWidth = 1920;
int imageHeight = 640;

// ── Customizable pinhole view (the MonoGlass3D input) ──
int pinholeWidth = 640;
int pinholeHeight = 480;
double pinholeFx = 388.1910413097385;
double pinholeFy = 422.0475153598262;
double pinholeCx = 320.0;
double pinholeCy = 240.0;
double pinholeYawDeg = 0.0;    // view direction around vertical, 0 = pano forward
double pinholePitchDeg = 0.0;  // positive looks down (camera y is down)

// ── Saving controls ──
std::string outputFolder = "./monoglass_output";
bool saveOutputs = true;
bool saveDepth = true;   // toggle off to skip pc_depth_map.txt + 16-bit depth png
bool savePly = true;     // toggle off to skip the 5s-window cloud PLY
int saveInterval = 5;
double depthPngScale = 1000.0;
bool showWindows = false;
int frameCounter = 0;
int saveCounter = 0;
FILE* poseFile = nullptr;

double imageLatencyOffset = 0.0;

// pano undistort intrinsics (only used if is360Cam == false)
double kImage[9] = {480.0, 0, 960.5, 0, 480.0, 320.5, 0, 0, 1};
double dImage[4] = {0, 0, 0, 0};
double fx = kImage[0], fy = kImage[4], cx = kImage[2], cy = kImage[5];
double k1 = 0, k2 = 0, p1 = 0, p2 = 0;

Mat mapx, mapy;
Mat kMat, dMat;

pcl::PointCloud<pcl::PointXYZ>::Ptr scanCloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr scanCloudStack(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr scanCloudCrop(new pcl::PointCloud<pcl::PointXYZ>());

// Sliding time window: keep only the last stackTimeWindow seconds of registered scans
// (same wiring as glassKillerNode -- no unbounded accumulation).
double stackTimeWindow = 5.0;  // seconds
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
  pcl::fromROSMsg(*scanIn, *scanCloud);

  // Sliding time-window accumulation (glassKillerNode wiring).
  double scanT = rclcpp::Time(scanIn->header.stamp).seconds();
  scanWindow.emplace_back(scanT,
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(*scanCloud)));
  while (!scanWindow.empty() && (scanT - scanWindow.front().first) > stackTimeWindow) {
    scanWindow.pop_front();
  }

  scanCloudStack->clear();
  for (const auto &e : scanWindow) *scanCloudStack += *e.second;

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
  nh = rclcpp::Node::make_shared("monoglass_saver");

  // Preview publishers (pinhole outputs, stamped with the scan stamp)
  auto pub_rgb   = nh->create_publisher<sensor_msgs::msg::Image>("/monoglass/rgb", 10);
  auto pub_depth = nh->create_publisher<sensor_msgs::msg::Image>("/monoglass/depth", 10);
  auto pub_pose  = nh->create_publisher<geometry_msgs::msg::PoseStamped>("/monoglass/pose", 10);

  // Params
  nh->declare_parameter<double>("minRange", minRange);
  nh->declare_parameter<double>("maxRange", maxRange);
  nh->declare_parameter<double>("voxelSize", voxelSize);
  nh->declare_parameter<double>("stackTimeWindow", stackTimeWindow);
  nh->declare_parameter<double>("imageSkipYaw", imageSkipYaw);
  nh->declare_parameter<int>("imageSkipNum", imageSkipNum);
  nh->declare_parameter<double>("imageLatencyOffset", imageLatencyOffset);
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
  nh->declare_parameter<int>("pinholeWidth", pinholeWidth);
  nh->declare_parameter<int>("pinholeHeight", pinholeHeight);
  nh->declare_parameter<double>("pinholeFx", pinholeFx);
  nh->declare_parameter<double>("pinholeFy", pinholeFy);
  nh->declare_parameter<double>("pinholeCx", pinholeCx);
  nh->declare_parameter<double>("pinholeCy", pinholeCy);
  nh->declare_parameter<double>("pinholeYawDeg", pinholeYawDeg);
  nh->declare_parameter<double>("pinholePitchDeg", pinholePitchDeg);
  nh->declare_parameter<std::string>("outputFolder", outputFolder);
  nh->declare_parameter<bool>("saveOutputs", saveOutputs);
  nh->declare_parameter<bool>("saveDepth", saveDepth);
  nh->declare_parameter<bool>("savePly", savePly);
  nh->declare_parameter<int>("saveInterval", saveInterval);
  nh->declare_parameter<double>("depthPngScale", depthPngScale);
  nh->declare_parameter<bool>("showWindows", showWindows);

  nh->get_parameter("minRange", minRange);
  nh->get_parameter("maxRange", maxRange);
  nh->get_parameter("voxelSize", voxelSize);
  nh->get_parameter("stackTimeWindow", stackTimeWindow);
  nh->get_parameter("imageSkipYaw", imageSkipYaw);
  nh->get_parameter("imageSkipNum", imageSkipNum);
  nh->get_parameter("imageLatencyOffset", imageLatencyOffset);
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
  nh->get_parameter("pinholeWidth", pinholeWidth);
  nh->get_parameter("pinholeHeight", pinholeHeight);
  nh->get_parameter("pinholeFx", pinholeFx);
  nh->get_parameter("pinholeFy", pinholeFy);
  nh->get_parameter("pinholeCx", pinholeCx);
  nh->get_parameter("pinholeCy", pinholeCy);
  nh->get_parameter("pinholeYawDeg", pinholeYawDeg);
  nh->get_parameter("pinholePitchDeg", pinholePitchDeg);
  nh->get_parameter("outputFolder", outputFolder);
  nh->get_parameter("saveOutputs", saveOutputs);
  nh->get_parameter("saveDepth", saveDepth);
  nh->get_parameter("savePly", savePly);
  nh->get_parameter("saveInterval", saveInterval);
  nh->get_parameter("depthPngScale", depthPngScale);
  nh->get_parameter("showWindows", showWindows);

  if (saveInterval < 1) saveInterval = 1;
  if (depthPngScale <= 0.0) depthPngScale = 1000.0;

  if (saveOutputs) {
    std::filesystem::create_directories(outputFolder);
    std::string poseFilePath = outputFolder + "/poses.csv";
    poseFile = fopen(poseFilePath.c_str(), "w");
    if (poseFile) {
      fprintf(poseFile, "frame,timestamp,x,y,z,qx,qy,qz,qw\n");
      fflush(poseFile);
    }
    RCLCPP_INFO(nh->get_logger(),
                "Saving to %s (interval=%d, depth=%s, ply=%s)",
                outputFolder.c_str(), saveInterval,
                saveDepth ? "on" : "off", savePly ? "on" : "off");
  } else {
    RCLCPP_INFO(nh->get_logger(), "Saving disabled (saveOutputs=false)");
  }

  // Subs (same wiring as glassKillerNode)
  auto subOdom  = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odomHandler);
  auto subScan  = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan", 2, scanHandler);
  auto subImage = nh->create_subscription<sensor_msgs::msg::Image>("/camera/image", 2, imageHandler);

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

  // Undistort map (only used when is360Cam == false)
  kImage[0] = fx;  kImage[4] = fy;  kImage[2] = cx;  kImage[5] = cy;
  dImage[0] = k1;  dImage[1] = k2;  dImage[2] = p1;  dImage[3] = p2;

  cv::Size imageSize(imageWidth, imageHeight);
  kMat = cv::Mat(3, 3, CV_64FC1, kImage);
  dMat = cv::Mat(4, 1, CV_64FC1, dImage);
  mapx.create(imageSize, CV_32FC1);
  mapy.create(imageSize, CV_32FC1);
  initUndistortRectifyMap(kMat, dMat, cv::Mat(), kMat, imageSize, CV_32FC1, mapx, mapy);

  downSizeFilter.setLeafSize(voxelSize, voxelSize, voxelSize);

  // ── Pinhole view rotation: R maps pinhole-frame dirs into camera frame ──
  // camera frame: x right, y down, z forward. Yaw about y (vertical), pitch about x.
  const double yawR = pinholeYawDeg * PI / 180.0;
  const double pitR = pinholePitchDeg * PI / 180.0;
  const double cyw = std::cos(yawR), syw = std::sin(yawR);
  const double cpt = std::cos(pitR), spt = std::sin(pitR);
  // R = Ry(yaw) * Rx(pitch)
  const float R00 = (float)( cyw), R01 = (float)( syw * spt), R02 = (float)( syw * cpt);
  const float R10 = 0.0f,          R11 = (float)( cpt),       R12 = (float)(-spt);
  const float R20 = (float)(-syw), R21 = (float)( cyw * spt), R22 = (float)( cyw * cpt);

  // ── Pinhole RGB remap from the pano (consistent with the lidar pano projection:
  //    u = W/(2pi)*atan2(x,z) + W/2 + 1,  v = W/(2pi)*atan(y/hori) + H/2 + 1) ──
  cv::Mat pinMapX(pinholeHeight, pinholeWidth, CV_32FC1);
  cv::Mat pinMapY(pinholeHeight, pinholeWidth, CV_32FC1);
  for (int v = 0; v < pinholeHeight; ++v) {
    for (int u = 0; u < pinholeWidth; ++u) {
      float X = (float)((u - pinholeCx) / pinholeFx);
      float Y = (float)((v - pinholeCy) / pinholeFy);
      // pinhole ray (X, Y, 1) rotated into camera frame
      float dx = R00 * X + R01 * Y + R02;
      float dy = R10 * X + R11 * Y + R12;
      float dz = R20 * X + R21 * Y + R22;
      float hori = std::sqrt(dx * dx + dz * dz);
      pinMapX.at<float>(v, u) = (float)(imageWidth / (2.0 * PI) * std::atan2(dx, dz) + imageWidth / 2.0 + 1.0);
      pinMapY.at<float>(v, u) = (float)(imageWidth / (2.0 * PI) * std::atan(dy / (hori + 1e-6f)) + imageHeight / 2.0 + 1.0);
    }
  }

  if (showWindows) {
    cv::namedWindow("Pinhole RGB", cv::WINDOW_NORMAL);
    cv::namedWindow("Pinhole Depth (Z)", cv::WINDOW_NORMAL);
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

      cv::Mat panoRGB = imageStack[imageFrontIDPointer];
      builtin_interfaces::msg::Time imageStamp = imageStampStack[imageFrontIDPointer];
      imageFrontIDPointer = (imageFrontIDPointer + 1) % imageStackNum;

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

        // 5s-window stacked cloud in the viewer z-up frame (x8, z8, -y8) -- the PLY content,
        // exactly like glassKillerNode.
        pcl::PointCloud<pcl::PointXYZ> panoCloud;
        panoCloud.points.reserve(scanCloudStack->points.size());

        // Pinhole Z-depth by DIRECT projection of the stacked cloud (z-buffered).
        cv::Mat pinDepthZ32(pinholeHeight, pinholeWidth, CV_32FC1, cv::Scalar(0));

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

          // Viewer z-up frame for the PLY (same as glassKillerNode)
          panoCloud.points.emplace_back(x8, z8, -y8);

          // Rotate camera-frame point into the pinhole view frame: p_pin = R^T * p_cam
          float px = R00 * x8 + R10 * y8 + R20 * z8;
          float py = R01 * x8 + R11 * y8 + R21 * z8;
          float pz = R02 * x8 + R12 * y8 + R22 * z8;
          if (pz <= 0.05f) continue;  // behind the pinhole

          int uu = (int)std::lround(pinholeFx * px / pz + pinholeCx);
          int vv = (int)std::lround(pinholeFy * py / pz + pinholeCy);
          if (uu < 0 || uu >= pinholeWidth || vv < 0 || vv >= pinholeHeight) continue;

          float& dref = pinDepthZ32.at<float>(vv, uu);
          if (dref == 0.0f || pz < dref) dref = pz;
        }

        // Pinhole RGB from the pano
        cv::Mat pinRGB(pinholeHeight, pinholeWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::remap(panoRGB, pinRGB, pinMapX, pinMapY,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        if (showWindows) {
          cv::Mat pinDepth8(pinholeHeight, pinholeWidth, CV_8UC1, cv::Scalar(0));
          for (int vv = 0; vv < pinholeHeight; ++vv) {
            for (int uu = 0; uu < pinholeWidth; ++uu) {
              float z = pinDepthZ32.at<float>(vv, uu);
              if (z <= 0.0f) continue;
              float t = (z - (float)minRange) / ((float)maxRange - (float)minRange);
              t = std::max(0.0f, std::min(1.0f, t));
              pinDepth8.at<uchar>(vv, uu) = (uchar)std::lround(t * 255.0f);
            }
          }
          cv::imshow("Pinhole RGB", pinRGB);
          cv::imshow("Pinhole Depth (Z)", pinDepth8);
        }

        // ── Save MonoGlass3D-style per-frame folder ──
        if (saveOutputs && saveCounter % saveInterval == 0) {
          std::string idx = cv::format("%06d", frameCounter);
          std::string frameDir = outputFolder + "/" + idx;
          std::filesystem::create_directories(frameDir);

          cv::imwrite(frameDir + "/" + idx + "_image.png", pinRGB);

          // camera_config.json in the MonoGlass3D cam_cfg format
          {
            std::ofstream f(frameDir + "/" + idx + "_camera_config.json");
            f << "{\n  \"camera_internal\": {\n"
              << "    \"fx\": " << pinholeFx << ",\n"
              << "    \"fy\": " << pinholeFy << ",\n"
              << "    \"cx\": " << pinholeCx << ",\n"
              << "    \"cy\": " << pinholeCy << "\n  },\n"
              << "  \"width\": " << pinholeWidth << ",\n"
              << "  \"height\": " << pinholeHeight << ",\n"
              << "  \"pinhole_yaw_deg\": " << pinholeYawDeg << ",\n"
              << "  \"pinhole_pitch_deg\": " << pinholePitchDeg << ",\n"
              << "  \"depth_png_scale\": " << depthPngScale << ",\n"
              << "  \"timestamp\": " << std::fixed << rclcpp::Time(imageStamp).seconds() << "\n}\n";
          }

          size_t nDepth = 0;
          if (saveDepth) {
            // sparse "u v Z" rows (MonoGlass3D pc_depth_map format) + dense 16-bit png
            std::ofstream f(frameDir + "/" + idx + "_pc_depth_map.txt");
            cv::Mat pinDepth16(pinholeHeight, pinholeWidth, CV_16UC1, cv::Scalar(0));
            for (int vv = 0; vv < pinholeHeight; ++vv) {
              for (int uu = 0; uu < pinholeWidth; ++uu) {
                float z = pinDepthZ32.at<float>(vv, uu);
                if (z <= 0.0f) continue;
                f << uu << " " << vv << " " << z << "\n";
                pinDepth16.at<uint16_t>(vv, uu) =
                    (uint16_t)std::min(65535.0, std::round((double)z * depthPngScale));
                nDepth++;
              }
            }
            cv::imwrite(frameDir + "/" + idx + "_depth.png", pinDepth16);
          }

          int plyStatus = -1;
          if (savePly) {
            panoCloud.width = (uint32_t)panoCloud.points.size();
            panoCloud.height = 1;
            panoCloud.is_dense = false;
            plyStatus = pcl::io::savePLYFileBinary(frameDir + "/" + idx + "_cloud.ply", panoCloud);
          }

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
                      "Saved frame %d -> %s (depth px=%zu%s, ply=%s pts=%zu)",
                      frameCounter, frameDir.c_str(), nDepth,
                      saveDepth ? "" : " [off]",
                      savePly ? (plyStatus == 0 ? "ok" : "failed") : "off",
                      panoCloud.points.size());
        }

        frameCounter++;
        saveCounter++;

        // Publish previews with the scan stamp
        if (g_have_scan_stamp.load(std::memory_order_acquire)) {
          std_msgs::msg::Header header;
          header.stamp    = g_last_scan_stamp;
          header.frame_id = "camera_link";

          auto rgb_msg   = cv_bridge::CvImage(header, "bgr8",  pinRGB).toImageMsg();
          auto depth_msg = cv_bridge::CvImage(header, "32FC1", pinDepthZ32).toImageMsg();
          pub_rgb->publish(*rgb_msg);
          pub_depth->publish(*depth_msg);

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

    (void)cv::waitKey(1);
    status = rclcpp::ok();
  }

  if (poseFile) fclose(poseFile);
  return 0;
}
