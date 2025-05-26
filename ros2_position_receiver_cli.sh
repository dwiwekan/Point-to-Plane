#!/bin/bash

# ROS2 Position Receiver using CLI tools
# This script listens for position messages on the /goal_pose topic
# and processes them without requiring rclpy
# Updated to handle PoseStamped messages instead of simple Point messages

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROS2_SETUP="/opt/ros/humble/setup.bash"
TOPIC_NAME="/goal_pose"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to process a position message
process_position() {
    local frame_id=$1
    local secs=$2
    local nsecs=$3
    local x=$4
    local y=$5
    local z=$6
    local qx=$7
    local qy=$8
    local qz=$9
    local qw=${10}
    
    print_success "Received PoseStamped message with structure:"
    echo "----------------------------------------"
    echo "header:"
    echo "  stamp:"
    echo "    secs: $secs"
    echo "    nsecs: $nsecs"
    echo "  frame_id: '$frame_id'"
    echo "pose:"
    echo "  position:"
    echo "    x: $x"
    echo "    y: $y"
    echo "    z: $z"
    echo "  orientation:"
    echo "    x: $qx"
    echo "    y: $qy"
    echo "    z: $qz"
    echo "    w: $qw"
    echo "----------------------------------------"
    
    print_status "Processing position for robot movement..."
    
    # Example: Convert position to robot-specific commands
    # You would implement your robot control logic here
    
    # For demonstration, we'll just log that we're sending it to the robot
    print_status "Position sent to robot controller"
}

# Main function
main() {
    print_status "Starting ROS2 Position Receiver (CLI Version)"
    
    # Source ROS2 setup
    if [ ! -f "$ROS2_SETUP" ]; then
        print_error "ROS2 setup file not found at: $ROS2_SETUP"
        print_error "Please check your ROS2 installation"
        exit 1
    fi
    
    # Source ROS2
    print_status "Sourcing ROS2 setup..."
    source "$ROS2_SETUP"
    
    # Source network configuration if available
    if [ -f "ros_network_setup.sh" ]; then
        print_status "Sourcing network configuration..."
        source ./ros_network_setup.sh
    else
        print_warning "Network configuration not found!"
        print_warning "The script might still work on local machine"
    fi
    
    # Check if ros2 command is available
    if ! command -v ros2 &> /dev/null; then
        print_error "ros2 command not found. Make sure ROS2 is properly installed."
        exit 1
    fi
    
    print_status "Waiting for positions on $TOPIC_NAME topic..."
    print_status "Press Ctrl+C to exit"
    echo ""
    echo "Listening for geometry_msgs/msg/PoseStamped messages..."
    echo "Will display complete message structure when received."
    echo ""
    
    # Listen for messages using ros2 topic echo
    ros2 topic echo "$TOPIC_NAME" | while read line; do
        # Check for start of header section in the PoseStamped message
        # The message structure is:
        # header:
        #   stamp:
        #     sec: ...
        #     nanosec: ...
        #   frame_id: ...
        # pose:
        #   position:
        #     x: ...
        #     y: ...
        #     z: ...
        #   orientation:
        #     x: ...
        #     y: ...
        #     z: ...
        #     w: ...
        
        # Initialize variables to store message fields
        local frame_id=""
        local secs=0
        local nsecs=0
        local pos_x=0.0
        local pos_y=0.0
        local pos_z=0.0
        local orient_x=0.0
        local orient_y=0.0
        local orient_z=0.0
        local orient_w=0.0
        
        if [[ $line == *"header:"* ]]; then
            # We found the header section - read stamp and frame_id
            read line  # stamp:
            read line  # sec:
            if [[ $line == *"sec:"* ]]; then
                secs=$(echo "$line" | sed 's/.*sec: //')
            fi
            read line  # nanosec:
            if [[ $line == *"nanosec:"* ]]; then
                nsecs=$(echo "$line" | sed 's/.*nanosec: //')
            fi
            read line  # frame_id:
            if [[ $line == *"frame_id:"* ]]; then
                frame_id=$(echo "$line" | sed 's/.*frame_id: //' | tr -d "'" | tr -d '"')
            fi
            
            # Continue reading to get the pose data
            while read line; do
                if [[ $line == *"position:"* ]]; then
                    # Process position data
                    read line  # x:
                    if [[ $line == *"x:"* ]]; then
                        pos_x=$(echo "$line" | sed 's/.*x: //')
                    fi
                    read line  # y:
                    if [[ $line == *"y:"* ]]; then
                        pos_y=$(echo "$line" | sed 's/.*y: //')
                    fi
                    read line  # z:
                    if [[ $line == *"z:"* ]]; then
                        pos_z=$(echo "$line" | sed 's/.*z: //')
                    fi
                    # Continue to orientation section
                    read line  # orientation:
                    read line  # x:
                    if [[ $line == *"x:"* ]]; then
                        orient_x=$(echo "$line" | sed 's/.*x: //')
                    fi
                    read line  # y:
                    if [[ $line == *"y:"* ]]; then
                        orient_y=$(echo "$line" | sed 's/.*y: //')
                    fi
                    read line  # z:
                    if [[ $line == *"z:"* ]]; then
                        orient_z=$(echo "$line" | sed 's/.*z: //')
                    fi
                    read line  # w:
                    if [[ $line == *"w:"* ]]; then
                        orient_w=$(echo "$line" | sed 's/.*w: //')
                    fi
                    
                    # Process the complete message
                    process_position "$frame_id" "$secs" "$nsecs" "$pos_x" "$pos_y" "$pos_z" "$orient_x" "$orient_y" "$orient_z" "$orient_w"
                    
                    # Break out of the inner loop after processing a message
                    break
                fi
                
                # Break if we reach the end of a message
                if [[ $line == "---" ]]; then
                    break
                fi
            done
        fi
    done
}

# Run the main function
main "$@"
