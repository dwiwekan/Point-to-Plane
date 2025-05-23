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
    local x=$1
    local y=$2
    local z=$3
    
    print_success "Received position from PoseStamped message: x=$x, y=$y, z=$z"
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
    
    # Listen for messages using ros2 topic echo
    ros2 topic echo "$TOPIC_NAME" | while read line; do
        # Check for start of position section in the PoseStamped message
        # The message structure is:
        # header:
        #   stamp: ...
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
        
        if [[ $line == *"position:"* ]]; then
            # We found the position section, now read the next lines to get x, y, z
            read line  # This should be the line with "x:"
            if [[ $line == *"x:"* ]]; then
                x_val=$(echo "$line" | sed 's/.*x: //')
                
                # Read y and z values
                read line
                y_val=$(echo "$line" | sed 's/.*y: //')
                read line
                z_val=$(echo "$line" | sed 's/.*z: //')
                
                # Process the position
                process_position "$x_val" "$y_val" "$z_val"
            fi
        fi
    done
}

# Run the main function
main "$@"
