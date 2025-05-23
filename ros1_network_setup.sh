#!/bin/bash
# ROS1 network configuration

# ROS1 machine info (this computer)
export ROS_MASTER_URI=http://192.168.50.84:11311
export ROS_IP=192.168.50.84
export ROS_HOSTNAME=rail0070-ASUS-TUF-Gaming-F17-FX706HE-FX706HE

# ROS2 domain ID (must be the same on both machines)
export ROS_DOMAIN_ID=42

# Print network configuration
echo "ROS network configuration:"
echo "ROS_MASTER_URI=http://localhost:11311"
echo "ROS_IP="
echo "ROS_HOSTNAME="
echo "ROS_DOMAIN_ID="
