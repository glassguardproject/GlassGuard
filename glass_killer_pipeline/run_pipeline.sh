#!/bin/bash
# 3-node Glass Killer pipeline (throughput build). Launches the provider + the 3 stage
# nodes as SEPARATE processes (3 GILs -> true multicore). The single-process node
# (../run_glass_killer_full.sh) is unchanged and remains the scoring reference.
#
# Usage: ./run_pipeline.sh [bag]     (bag default = bldgD_int)
set -u
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastdds_no_shm.xml
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/.ros/fastdds_no_shm.xml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GK=.
PIPE=$GK/glass_killer_pipeline
BAG=${1:-${GG_DATA_ROOT:-$HOME/glassguard_data}/bldgD_int}
ROS_SETUP=/opt/ros/jazzy/setup.bash
CAM_INSTALL=${GG_ROS_WS:-$HOME/ros_ws}/install/setup.bash

echo "[pipeline] provider + 3 stage nodes; bag=$BAG"

# preflight: the stage bodies must be ported out of ../glass_killer_ros_node.py _process
# before this publishes anything. Fail loud instead of launching silent stubs.
if grep -q "raise NotImplementedError" "$PIPE"/node1_detect.py "$PIPE"/node2_geometry.py "$PIPE"/node3_mapping.py; then
  echo "[pipeline] ABORT: stage bodies not ported yet (run_stage() is a stub in one or more nodes)."
  echo "           This runner will NOT publish until _process spans are lifted into each run_stage()."
  echo "           To publish the same output as before TODAY, use the single-node runner:"
  echo "             cd $GK && METHOD=360 BAG=$BAG ./run_glass_killer_full.sh"
  exit 2
fi
mkdir -p /dev/shm/gkpipe && rm -f /dev/shm/gkpipe/*.pkl 2>/dev/null

# provider (same launch the monolith uses to produce /registered_scan + /camera/image)
( source "$ROS_SETUP"; [ -f "$CAM_INSTALL" ] && source "$CAM_INSTALL"
  exec ros2 launch extrinsic_latency_calib glass_killer.launch ) > /tmp/pipe_provider.log 2>&1 &
sleep 3
( source "$ROS_SETUP"
  exec ros2 run image_transport republish --ros-args -p in_transport:=compressed -p out_transport:=raw \
       --remap in/compressed:=/camera/image/compressed --remap out:=/camera/image ) > /tmp/pipe_repub.log 2>&1 &
sleep 1

# 3 stage nodes -- each its own process
( source "$ROS_SETUP"; exec conda run -n sam3 --no-capture-output python "$PIPE/node1_detect.py" )   > /tmp/pipe_n1.log 2>&1 &
( source "$ROS_SETUP"; exec conda run -n sam3 --no-capture-output python "$PIPE/node2_geometry.py" ) > /tmp/pipe_n2.log 2>&1 &
( source "$ROS_SETUP"; exec conda run -n sam3 --no-capture-output python "$PIPE/node3_mapping.py" )  > /tmp/pipe_n3.log 2>&1 &
sleep 5

# rviz (reuse the monolith's config) + bag
( source "$ROS_SETUP"; exec ros2 run rviz2 rviz2 -d "$GK/glass_killer.rviz" ) > /tmp/pipe_rviz.log 2>&1 &
sleep 2
( source "$ROS_SETUP"
  exec ros2 bag play --disable-keyboard-controls "$BAG" < /dev/null )
