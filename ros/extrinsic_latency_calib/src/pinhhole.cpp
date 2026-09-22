// extrinsicCalib_real.cpp
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <vector>
#include <algorithm>
#include <limits>

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
  pcl::fromROSMsg(*scanIn, *scanCloud);

  // Use the latest scan only (no accumulation)
  *scanCloudStack = *scanCloud;

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
  nh = rclcpp::Node::make_shared("extrinsicCalib");

  // Publish ONLY Habitat pinhole outputs
  auto pub_rgb   = nh->create_publisher<sensor_msgs::msg::Image>("/habitat/rgb", 10);
  auto pub_depth = nh->create_publisher<sensor_msgs::msg::Image>("/habitat/depth", 10);
  auto pub_pose  = nh->create_publisher<geometry_msgs::msg::PoseStamped>("/habitat/state_estimation", 10);

  // Params
  nh->declare_parameter<double>("minRange", minRange);
  nh->declare_parameter<double>("maxRange", maxRange);
  nh->declare_parameter<double>("angAdjustment", angAdjustment);
  nh->declare_parameter<double>("voxelSize", voxelSize);
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

  nh->get_parameter("minRange", minRange);
  nh->get_parameter("maxRange", maxRange);
  nh->get_parameter("angAdjustment", angAdjustment);
  nh->get_parameter("voxelSize", voxelSize);
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

  // ---------------- Habitat pinhole remap (precompute once) ----------------
  static constexpr int HAB_W = 640;
  static constexpr int HAB_H = 480;

  // Habitat intrinsics (your values)
  const float hab_fx = 388.1910413097385f;
  const float hab_fy = 422.0475153598262f;
  const float hab_cx = 320.0f;
  const float hab_cy = 240.0f;

  cv::Mat hab_map_x(HAB_H, HAB_W, CV_32FC1);
  cv::Mat hab_map_y(HAB_H, HAB_W, CV_32FC1);

  for (int v = 0; v < HAB_H; ++v) {
    for (int u = 0; u < HAB_W; ++u) {
      // pinhole ray in camera coords, z=1
      float X = (u - hab_cx) / hab_fx;
      float Y = (v - hab_cy) / hab_fy;

      float theta = std::atan2(X, 1.0f);                     // [-pi, pi]
      float phi   = std::atan2(Y, std::sqrt(X*X + 1.0f));    // [-pi/2, pi/2]

      float u360 = (theta + (float)M_PI) / (2.0f*(float)M_PI) * (float)imageWidth;
      // UNIFORM-ANGULAR pano: W/(2pi) px/rad on BOTH axes (1920x640 = 360x120 deg, NOT +/-90).
      float v360 = phi * (float)imageWidth / (2.0f*(float)M_PI) + 0.5f*(float)imageHeight;

      hab_map_x.at<float>(v,u) = u360;
      hab_map_y.at<float>(v,u) = v360;
    }
  }

  // OpenCV windows
  cv::namedWindow("Habitat RGB", cv::WINDOW_NORMAL);
  cv::namedWindow("Habitat Depth (Z)", cv::WINDOW_NORMAL);

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
      imageFrontIDPointer = (imageFrontIDPointer + 1) % imageStackNum;

      // reset z-buffer helper
      std::fill(depthArray, depthArray + imagePixelNum, 0.0f);

      // pano depth as RANGE (we will later convert to pinhole Z)
      cv::Mat panoDepth32(imageHeight, imageWidth, CV_32FC1, cv::Scalar(0));

      // advance odomFront pointer to straddle imageTime
      while (odomFrontIDPointer != odomLastIDPointer) {
        if (odomTimeStack[odomFrontIDPointer] > imageTime) break;
        odomFrontIDPointer = (odomFrontIDPointer + 1) % odomStackNum;
      }

      bool depthProj = true;
      float lidarRoll = 0, lidarPitch = 0, lidarYaw = 0;
      float lidarX = 0, lidarY = 0, lidarZ = 0;

      if (odomTimeStack[odomFrontIDPointer] < imageTime) {
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
          float ratioFront = (float)((imageTime - odomTimeStack[odomBackIDPointer]) / denom);
          float ratioBack  = (float)((odomTimeStack[odomFrontIDPointer] - imageTime) / denom);

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

          // Store RANGE depth in pano (Euclidean range)
          float range = std::sqrt(x8*x8 + y8*y8 + z8*z8);

          // Small 3x3 splat with z-buffering using helper depthArray (based on horiDis, original behavior)
          // We’ll keep the “closest” using RANGE in panoDepth32.
          for (int ii = -1; ii <= 1; ii++) {
            for (int jj = -1; jj <= 1; jj++) {
              int uu = u + jj;
              int vv = v + ii;
              int pixelID = imageWidth * vv + uu;

              if (depthArray[pixelID] == 0.0f || depthArray[pixelID] > horiDis) {
                depthArray[pixelID] = horiDis;
                float r_clamped = std::clamp(range, (float)minRange, (float)maxRange);
                float& dref = panoDepth32.at<float>(vv, uu);
                if (dref == 0.0f || r_clamped < dref) dref = r_clamped;
              }
            }
          }
        }

        // ---------------- Build Habitat pinhole outputs ----------------
        cv::Mat habitatRGB(HAB_H, HAB_W, CV_8UC3, cv::Scalar(0,0,0));
        cv::remap(panoRGB, habitatRGB, hab_map_x, hab_map_y,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        // Range on pinhole rays (from remap)
        cv::Mat habDepthRange32(HAB_H, HAB_W, CV_32FC1, cv::Scalar(0));
        cv::remap(panoDepth32, habDepthRange32, hab_map_x, hab_map_y,
                  cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0.0f);

        // Convert RANGE -> Z depth for pinhole convention
        cv::Mat habDepthZ32(HAB_H, HAB_W, CV_32FC1, cv::Scalar(0));
        for (int vv = 0; vv < HAB_H; ++vv) {
          float ry = (vv - hab_cy) / hab_fy;
          for (int uu = 0; uu < HAB_W; ++uu) {
            float R = habDepthRange32.at<float>(vv, uu);
            if (R <= 0.0f) continue;

            float rx = (uu - hab_cx) / hab_fx;
            float inv_norm = 1.0f / std::sqrt(1.0f + rx*rx + ry*ry);
            float Z = R * inv_norm;

            if (Z < (float)minRange || Z > (float)maxRange) continue;
            habDepthZ32.at<float>(vv, uu) = Z;
          }
        }

        // Visualize depth as 8-bit
        cv::Mat habDepth8(HAB_H, HAB_W, CV_8UC1, cv::Scalar(0));
        for (int vv = 0; vv < HAB_H; ++vv) {
          for (int uu = 0; uu < HAB_W; ++uu) {
            float z = habDepthZ32.at<float>(vv, uu);
            if (z <= 0.0f) continue;
            float t = (z - (float)minRange) / ((float)maxRange - (float)minRange);
            t = std::max(0.0f, std::min(1.0f, t));
            habDepth8.at<uchar>(vv, uu) = (uchar)std::lround(t * 255.0f);
          }
        }

        cv::imshow("Habitat RGB", habitatRGB);
        cv::imshow("Habitat Depth (Z)", habDepth8);

        // ---------------- Publish with scan stamp ----------------
        if (!g_have_scan_stamp.load(std::memory_order_acquire)) {
          RCLCPP_WARN_THROTTLE(nh->get_logger(), *nh->get_clock(), 2000,
                               "No /registered_scan yet; skip publish");
        } else {
          std_msgs::msg::Header header;
          header.stamp    = g_last_scan_stamp;
          header.frame_id = "camera_link";

          auto rgb_msg   = cv_bridge::CvImage(header, "bgr8",  habitatRGB).toImageMsg();
          auto depth_msg = cv_bridge::CvImage(header, "32FC1", habDepthZ32).toImageMsg();
          pub_rgb->publish(*rgb_msg);
          pub_depth->publish(*depth_msg);

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

    status = rclcpp::ok();
  }

  delete[] depthArray;
  depthArray = nullptr;

  return 0;
}
