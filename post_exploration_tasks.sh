#!/bin/bash

# ====================================================================
# Script: post_exploration_tasks.sh
# Description:
#   Launches additional ROS 2 nodes after exploration is stopped.
#   Specifically, it launches the camera subscriber and patrol systems.
#   Keeps these nodes running alongside map saving and map merging.
#
# Usage:
#   Ensure this script is executable and is invoked by launch_and_monitor.sh
#   chmod +x post_exploration_tasks.sh
# ====================================================================

# --- Configuration ---

# Ensure NUM_ROBOTS is defined
if [ -z "$NUM_ROBOTS" ]; then
    echo "Error: NUM_ROBOTS is not defined. Exiting."
    exit 1
fi

# Log files for additional nodes
CAMERA_SUBSCRIBER_LOG_FILE="camera_subscriber.log"
PATROL_LOG_FILE="patrol_launch.log"

# --- Functions ---

# Function to cleanup additional processes
cleanup_additional() {
    echo "======= Cleaning Up Additional Processes ======="

    # Terminate camera_subscriber
    if [ ! -z "$CAMERA_SUBSCRIBER_PID" ] && ps -p "$CAMERA_SUBSCRIBER_PID" > /dev/null 2>&1; then
        kill "$CAMERA_SUBSCRIBER_PID" 2>/dev/null
        wait "$CAMERA_SUBSCRIBER_PID" 2>/dev/null
        echo "Terminated camera_subscriber PID: $CAMERA_SUBSCRIBER_PID"
    fi

    # Terminate patrol_launch
    if [ ! -z "$PATROL_PID" ] && ps -p "$PATROL_PID" > /dev/null 2>&1; then
        kill "$PATROL_PID" 2>/dev/null
        wait "$PATROL_PID" 2>/dev/null
        echo "Terminated patrol_launch PID: $PATROL_PID"
    fi

    echo "Additional cleanup complete."
}

# --- Trap Signals for Cleanup ---
trap cleanup_additional SIGINT SIGTERM

# --- Environment Setup ---
export TURTLEBOT3_MODEL='waffle'
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH}:/opt/ros/${ROS_DISTRO}/share/turtlebot3_gazebo/models"

# Ensure ROS 2 environment is sourced
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
    echo "Error: ROS 2 distribution '${ROS_DISTRO}' not found."
    exit 1
fi

# --- Launch Additional ROS 2 Nodes ---

# Start camera_subscriber node
echo "Launching camera_subscriber..."
ros2 launch scripts_package camera_subscriber.launch.py > "$CAMERA_SUBSCRIBER_LOG_FILE" 2>&1 &
CAMERA_SUBSCRIBER_PID=$!
echo "camera_subscriber PID: $CAMERA_SUBSCRIBER_PID"

# Start patrol system
echo "Launching patrol system..."
ros2 launch patroling patrol_launch.py num_robots:=$NUM_ROBOTS > "$PATROL_LOG_FILE" 2>&1 &
PATROL_PID=$!
echo "patrol_launch PID: $PATROL_PID"

# --- Keep the script Running ---
# Wait indefinitely to keep the script active and maintain the background processes
wait