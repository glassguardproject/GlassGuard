// regScanStack.cpp
//
// Materializes the RViz "RegScan" visual into a real point cloud.
//
// In RViz, the RegScan display subscribes to /registered_scan with
// "Decay Time: 5", which simply keeps every sweep received in the last 5
// seconds and renders them overlaid. The points are never actually merged --
// RViz just stacks each incoming message on screen.
//
// This node does the same time-windowed accumulation but produces a genuine
// merged cloud that it republishes, so you really have those points (and can
// save them to PLY). It does NOT do the extrinsicCalib-style 10 m spatial crop,
// voxel downsample, or camera re-centering: points are kept exactly as they
// arrive on /registered_scan (world / map frame), to match the RViz look.
//
// Display it in RViz with Decay Time: 0 on the output topic and it is visually
// identical to the current RegScan (Decay 5) -- except now the points are real.

#include <cstdio>
#include <deque>
#include <string>
#include <sys/stat.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>

using PointType = pcl::PointXYZI;  // registered_scan carries intensity

class RegScanStack : public rclcpp::Node
{
public:
  RegScanStack() : rclcpp::Node("regScanStack")
  {
    // Time window to keep, in seconds -- mirrors RViz "Decay Time".
    stackSeconds_ = this->declare_parameter<double>("stackSeconds", 5.0);
    inputTopic_   = this->declare_parameter<std::string>("inputTopic", "/registered_scan");
    outputTopic_  = this->declare_parameter<std::string>("outputTopic", "/registered_scan_stack");

    // Optional extrinsicCalib-style PLY dumping of the stacked cloud.
    saveOutputs_  = this->declare_parameter<bool>("saveOutputs", false);
    saveInterval_ = this->declare_parameter<int>("saveInterval", 10);  // every Nth published stack
    outputFolder_ = this->declare_parameter<std::string>("outputFolder", "./regscan_stack_output");

    if (stackSeconds_ <= 0.0) stackSeconds_ = 5.0;
    if (saveInterval_ < 1) saveInterval_ = 1;

    if (saveOutputs_) {
      ::mkdir(outputFolder_.c_str(), 0775);  // ok if it already exists
    }

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(outputTopic_, 2);
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      inputTopic_, 5,
      std::bind(&RegScanStack::scanHandler, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
      "regScanStack: stacking last %.2f s of %s -> %s (save=%s, every %d, folder=%s)",
      stackSeconds_, inputTopic_.c_str(), outputTopic_.c_str(),
      saveOutputs_ ? "true" : "false", saveInterval_, outputFolder_.c_str());
  }

private:
  struct Sweep {
    int64_t stamp_ns;
    pcl::PointCloud<PointType>::Ptr cloud;
  };

  void scanHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr scanIn)
  {
    const int64_t stamp_ns = rclcpp::Time(scanIn->header.stamp).nanoseconds();
    frameId_ = scanIn->header.frame_id;  // world / map frame from SLAM

    auto cloud = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::fromROSMsg(*scanIn, *cloud);

    buffer_.push_back(Sweep{stamp_ns, cloud});

    // Drop sweeps older than the window, measured against the newest sweep --
    // robust to bag playback and to clock vs message-stamp differences. This is
    // exactly what RViz decay does: keep messages newer than (latest - decay).
    const int64_t window_ns = static_cast<int64_t>(stackSeconds_ * 1e9);
    while (!buffer_.empty() && (stamp_ns - buffer_.front().stamp_ns) > window_ns) {
      buffer_.pop_front();
    }

    // Merge the windowed sweeps into one real cloud (raw, no crop/downsample).
    pcl::PointCloud<PointType> stacked;
    size_t total = 0;
    for (const auto& s : buffer_) total += s.cloud->points.size();
    stacked.points.reserve(total);
    for (const auto& s : buffer_) stacked += *s.cloud;

    // Publish with the latest sweep's stamp and the SLAM world frame so RViz
    // (Decay Time 0) shows the union identically to the old RegScan (Decay 5).
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(stacked, out);
    out.header.stamp = scanIn->header.stamp;
    out.header.frame_id = frameId_;
    pub_->publish(out);

    if (saveOutputs_ && (publishCounter_ % saveInterval_ == 0)) {
      stacked.width = static_cast<uint32_t>(stacked.points.size());
      stacked.height = 1;
      stacked.is_dense = false;
      char name[64];
      std::snprintf(name, sizeof(name), "/stack_%06d.ply", saveCounter_);
      const std::string file = outputFolder_ + name;
      const int st = pcl::io::savePLYFileBinary(file, stacked);
      RCLCPP_INFO(this->get_logger(),
        "Saved %s (ply=%s, sweeps=%zu, points=%zu)",
        file.c_str(), st == 0 ? "ok" : "failed", buffer_.size(), stacked.points.size());
      saveCounter_++;
    }
    publishCounter_++;

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "stacked %zu sweeps over %.2f s -> %zu points",
      buffer_.size(), stackSeconds_, stacked.points.size());
  }

  double stackSeconds_;
  std::string inputTopic_, outputTopic_, outputFolder_, frameId_{"map"};
  bool saveOutputs_;
  int saveInterval_;
  int publishCounter_ = 0;
  int saveCounter_ = 0;

  std::deque<Sweep> buffer_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RegScanStack>());
  rclcpp::shutdown();
  return 0;
}
