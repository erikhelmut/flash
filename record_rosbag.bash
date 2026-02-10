#!/bin/bash

if [ -z "$1" ]; then
    echo "ERROR: No folder name provided."
    echo "Usage: ./record_rosbag.bash <session_name>"
    exit 1
fi

FOLDER_NAME="$1"
BASE_DIR="/home/erik/flash/data/rosbags"
TASK_DIR="$BASE_DIR/$FOLDER_NAME"
RECORD_DIR="$TASK_DIR/$(date +%Y-%m-%d_%H-%M-%S)"
LAUNCH_FILE="imitator umi_launch.py"

# Create base directory if needed
mkdir -p "$TASK_DIR"

# Start the ROS 2 launch file
echo "Launching: $LAUNCH_FILE"
ros2 launch $LAUNCH_FILE &

# Save the PID of the launch process
LAUNCH_PID=$!

# Start ros2 bag recording in the background
echo "Starting ros2 bag recording in: $RECORD_DIR"
ros2 bag record --all -o "$RECORD_DIR" --max-cache-size 8192 &

# Save the PID of the background process
BAG_PID=$!

# Wait for the launch process to finish
wait $LAUNCH_PID

# When the launch process exits, kill the ros2 bag process
echo "Shutting down ros2 bag recording..."
kill $BAG_PID

echo "Done."