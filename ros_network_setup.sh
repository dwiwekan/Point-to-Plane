#!/bin/bash
# ROS network configuration

# ROS1 machine info
export ROS_MASTER_URI=http://192.168.50.84:11311
export ROS_IP=192.168.50.192

# ROS2 domain ID (must be the same on both machines)
export ROS_DOMAIN_ID=42

# Print network configuration
echo "ROS network configuration:"
echo "ROS_MASTER_URI="
echo "ROS_IP="
echo "ROS_DOMAIN_ID=30"
