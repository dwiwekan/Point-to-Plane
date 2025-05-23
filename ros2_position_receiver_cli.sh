#!/bin/bash

# ROS2 Position Receiver using CLI tools
# This script listens for position messages on the /goal_position topic
# and processes them without requiring rclpy

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROS2_SETUP="/opt/ros/humble/setup.bash"
TOPIC_NAME="/goal_position"

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
    
    print_success "Received position: x=$x, y=$y, z=$z"
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
        # Check for new message (geometry_msgs/Point messages have x, y, z fields)
        if [[ $line == *"x:"* ]]; then
            # Extract x value
            x_val=$(echo "$line" | sed 's/x: //')
            # Read next two lines for y and z
            read line
            y_val=$(echo "$line" | sed 's/y: //')
            read line
            z_val=$(echo "$line" | sed 's/z: //')
            
            # Process the position
            process_position "$x_val" "$y_val" "$z_val"
        fi
    done
}

# Run the main function
main "$@"
