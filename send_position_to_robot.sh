#!/bin/bash

# ROS Bridge Position Publisher Script
# This script sets up ROS Bridge between ROS1 Noetic and ROS2 Foxy
# and publishes object positions to the robot

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
GOAL_TOPIC="/goal_pose"
ROS1_SETUP="/opt/ros/noetic/setup.bash"
ROS2_SETUP="/opt/ros/foxy/setup.bash"  # Note: This was updated to use Foxy instead of Humble
BRIDGE_WAIT_TIME=5  # Seconds to wait for bridge to start

# Python environment variables
PYTHON_ENV_PATH=""  # Will be set dynamically if needed
USE_CONDA=false     # Set to true if using conda environments

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

# Function to check if ROS Bridge is running
check_ros_bridge() {
    if pgrep -f "ros2 run ros1_bridge" > /dev/null; then
        return 0  # ROS Bridge is running
    else
        return 1  # ROS Bridge is not running
    fi
}

# Function to configure network for ROS communication
configure_network() {
    print_status "Configuring network settings for ROS communication..."
    
    # Get hostname and IP
    hostname=$(hostname)
    ip_address=$(hostname -I | awk '{print $1}')
    
    print_status "This computer (ROS1):"
    print_status "- Hostname: $hostname"
    print_status "- IP Address: $ip_address"
    
    # Ask for ROS2 machine info
    echo -n "Enter ROS2 machine IP address: "
    read ros2_ip
    
    # Create network configuration file
    cat > ros1_network_setup.sh << EOF
#!/bin/bash
# ROS1 network configuration

# ROS1 machine info (this computer)
export ROS_MASTER_URI=http://${ip_address}:11311
export ROS_IP=${ip_address}
export ROS_HOSTNAME=${hostname}

# ROS2 domain ID (must be the same on both machines)
export ROS_DOMAIN_ID=42

# Print network configuration
echo "ROS network configuration:"
echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
echo "ROS_HOSTNAME=$ROS_HOSTNAME"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
EOF
    
    chmod +x ros1_network_setup.sh
    
    print_success "Network configuration created in ros1_network_setup.sh"
    print_status "Source this file before running the bridge or publisher:"
    print_status "source ./ros1_network_setup.sh"
    
    # Create a hosts file entry hint
    print_status "You may need to update /etc/hosts on both machines."
    print_status "Run the following commands on each machine (with sudo):"
    print_status "sudo sh -c 'echo \"$ip_address $hostname\" >> /etc/hosts'"
    print_status "sudo sh -c 'echo \"$ros2_ip ros2-hostname\" >> /etc/hosts'"
    print_status "(Replace ros2-hostname with the actual hostname of your ROS2 machine)"
    
    # Source the network setup automatically
    source ./ros1_network_setup.sh
}

# Function to start ROS Bridge
start_ros_bridge() {
    print_status "Starting ROS Bridge..."
    
    # Check if ROS Bridge is already running
    if check_ros_bridge; then
        print_warning "ROS Bridge is already running"
        return 0
    fi
    
    # Check if network settings are configured
    if [ ! -f "ros1_network_setup.sh" ]; then
        print_warning "Network settings not found. Creating them now..."
        configure_network
    else
        # Source the network configuration
        print_status "Using existing network configuration"
        source ./ros1_network_setup.sh
    fi
    
    # Check if ROS1 and ROS2 setup files exist
    if [ ! -f "$ROS1_SETUP" ]; then
        print_error "ROS1 setup file not found: $ROS1_SETUP"
        print_error "Please check your ROS1 Noetic installation"
        return 1
    fi
    
    if [ ! -f "$ROS2_SETUP" ]; then
        print_error "ROS2 setup file not found: $ROS2_SETUP"
        print_error "Please check your ROS2 Foxy installation"
        return 1
    fi
    
    # Check if ros1_bridge package is installed in ROS2
    if ! source "$ROS2_SETUP" > /dev/null 2>&1 && ros2 pkg list | grep -q ros1_bridge; then
        print_error "ros1_bridge package not found in ROS2"
        print_error "Please install it with: sudo apt install ros-humble-ros1-bridge"
        return 1
    fi
    
    # Ensure roscore is running
    if ! pgrep -f rosmaster > /dev/null; then
        print_warning "roscore is not running. Starting it now..."
        gnome-terminal -- bash -c "
            echo 'Starting roscore...';
            source $ROS1_SETUP;
            source ./ros1_network_setup.sh;
            roscore;
            read -p 'Press Enter to close this terminal...'
        " &
        
        # Wait for roscore to start
        print_status "Waiting for roscore to start..."
        sleep 3
    fi
    
    # Start ROS Bridge in background terminal with network settings
    print_status "Launching ROS1-ROS2 Bridge..."
    gnome-terminal -- bash -c "
        echo 'Starting ROS Bridge...';
        source ./ros1_network_setup.sh;
        source $ROS1_SETUP;
        source $ROS2_SETUP;
        echo 'Using ros1_bridge to connect ROS1 Noetic and ROS2 Foxy...';
        ros2 run ros1_bridge dynamic_bridge --bridge-all-topics;
        read -p 'Press Enter to close this terminal...'
    " &
    
    # Wait for the bridge to start
    print_status "Waiting $BRIDGE_WAIT_TIME seconds for ROS Bridge to start..."
    sleep $BRIDGE_WAIT_TIME
    
    # Check if bridge started successfully
    if check_ros_bridge; then
        print_success "ROS Bridge started successfully"
        verify_bridge_topics
        return 0
    else
        print_error "Failed to start ROS Bridge"
        print_error "Check the terminal window for error messages"
        print_warning "Try running with debug info:"
        print_warning "ROS_LOG_DIR=/tmp ros2 run ros1_bridge dynamic_bridge --bridge-all-topics --verbose"
        return 1
    fi
}

# Function to verify that bridge is forwarding the topic
verify_bridge_topics() {
    print_status "Verifying that ROS Bridge is properly forwarding topics..."
    
    # Source ROS1
    source "$ROS1_SETUP" > /dev/null 2>&1
    
    # Get list of ROS1 topics
    local ros1_topics=$(rostopic list 2>/dev/null | sort)
    
    if [ $? -ne 0 ]; then
        print_error "Failed to get ROS1 topics. Is roscore running?"
        print_warning "Make sure to run 'roscore' in another terminal"
        return 1
    fi
    
    # Show current ROS1 topics
    print_status "Current ROS1 topics:"
    echo "$ros1_topics" | sed 's/^/  /'
    
    return 0
}

# Function to detect Python environment with torch
detect_python_env() {
    print_status "Detecting Python environment with torch..."
    
    # First try: check if torch is available in current environment
    if python3 -c "import torch" &>/dev/null; then
        print_success "torch is available in current Python environment"
        return 0
    fi
    
    # Second try: Check for virtual environments in the project directory
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        print_status "Found venv in project directory, using it"
        PYTHON_ENV_PATH="./venv/bin/activate"
        return 0
    fi
    
    # Third try: Check for conda environments
    if command -v conda &>/dev/null; then
        print_status "Conda detected, checking environments..."
        USE_CONDA=true
        
        # List all conda environments with torch
        conda_env=$(conda env list | grep -v "^#" | awk '{print $1}')
        
        for env in $conda_env; do
            if conda run -n $env python -c "import torch" &>/dev/null; then
                print_success "Found torch in conda environment: $env"
                PYTHON_ENV_PATH="$env"
                return 0
            fi
        done
    fi
    
    # Fourth try: Check if pip can install torch
    print_warning "Could not find torch in any Python environment"
    print_warning "Would you like to install torch in the current environment? (y/n)"
    read install_choice
    
    if [[ "$install_choice" == "y" || "$install_choice" == "Y" ]]; then
        print_status "Installing torch using pip..."
        pip install torch
        
        # Verify installation
        if python3 -c "import torch" &>/dev/null; then
            print_success "torch successfully installed"
            return 0
        else
            print_error "Failed to install torch"
            return 1
        fi
    fi
    
    return 1
}

# Function to publish position using the Python script
publish_position() {
    local query="$1"
    local visualize="$2"
    local match_index="$3"
    
    print_status "Publishing position for query: '$query'"
    if [ -n "$match_index" ]; then
        print_status "Using match index: $match_index"
    fi
    
    # Check if network settings are configured
    if [ -f "ros1_network_setup.sh" ]; then
        source ./ros1_network_setup.sh
    else
        print_warning "Network settings not found. Creating them now..."
        configure_network
    fi
    
    # Check if ROS Bridge is running
    if ! check_ros_bridge; then
        print_warning "ROS Bridge is not running"
        print_status "Starting ROS Bridge first..."
        start_ros_bridge
        
        if [ $? -ne 0 ]; then
            print_error "Failed to start ROS Bridge"
            return 1
        fi
    fi
    
    # Check if roscore is running
    if ! pgrep -f "rosmaster" > /dev/null; then
        print_warning "roscore doesn't seem to be running"
        print_status "Starting roscore in another terminal..."
        gnome-terminal -- bash -c "
            echo 'Starting roscore...';
            source $ROS1_SETUP;
            source ./ros1_network_setup.sh;
            roscore;
            read -p 'Press Enter to close this terminal...'
        " &
        
        # Wait for roscore to start
        print_status "Waiting for roscore to start..."
        sleep 3
    fi
    
    # Check if Python script exists
    if [ ! -f "ros_position_publisher.py" ]; then
        print_error "ros_position_publisher.py not found!"
        return 1
    fi
    
    # Make sure the script is executable
    chmod +x ros_position_publisher.py
    
    # Source ROS1 for Python script
    source "$ROS1_SETUP" > /dev/null 2>&1
    
    # Detect Python environment with torch if needed
    if ! python3 -c "import torch" &>/dev/null; then
        print_warning "torch module not found in current Python environment"
        if ! detect_python_env; then
            print_error "Could not find or setup a Python environment with torch"
            print_error "Please activate your environment with torch manually before running this script"
            return 1
        fi
    fi
    
    # Run the position publisher with appropriate environment
    print_status "Running position publisher for query: '$query'"
    
    # Determine visualization flag
    local viz_flag=""
    if [ "$visualize" = "false" ]; then
        viz_flag="--no-viz"
        print_status "Skipping visualization to save time"
    else
        print_status "Visualization enabled - will load mesh and display result"
    fi
    
    # Determine match index flag
    local match_idx_flag=""
    if [ -n "$match_index" ]; then
        match_idx_flag="--match-index $match_index"
        print_status "Using match index: $match_index"
    fi
    
    start_time=$(date +%s.%N)
    
    if [ -z "$PYTHON_ENV_PATH" ]; then
        # Using current environment
        python3 ros_position_publisher.py $viz_flag $match_idx_flag "$query"
    elif [ "$USE_CONDA" = true ]; then
        # Using conda environment
        conda run -n "$PYTHON_ENV_PATH" python3 ros_position_publisher.py $viz_flag $match_idx_flag "$query"
    else
        # Using virtual environment
        (source "$PYTHON_ENV_PATH" && python3 ros_position_publisher.py $viz_flag $match_idx_flag "$query")
    fi
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    print_status "Total processing time: ${elapsed:.2f} seconds"
    
    local status=$?
    if [ $status -eq 0 ]; then
        print_success "Position published successfully"
        
        # Verify the message was published
        print_status "Checking if message was published..."
        if rostopic echo -n 1 "$GOAL_TOPIC" &>/dev/null; then
            print_success "Message verified on topic: $GOAL_TOPIC"
        else
            print_warning "Could not verify message on topic: $GOAL_TOPIC"
            print_warning "This could be because the message was published too quickly"
        fi
    else
        print_error "Failed to publish position (exit code: $status)"
    fi
    
    return $status
}

# Function to monitor the goal position topic
monitor_topic() {
    print_status "Monitoring $GOAL_TOPIC topic..."
    print_status "Press Ctrl+C to stop monitoring"
    
    # Check if ROS1 environment is available
    source "$ROS1_SETUP" > /dev/null 2>&1
    
    if command -v rostopic &> /dev/null; then
        rostopic echo "$GOAL_TOPIC"
    else
        print_error "rostopic command not found. Make sure ROS1 is properly installed."
        return 1
    fi
}

# Function to send a custom position
send_custom_position() {
    print_status "Sending custom position to $GOAL_TOPIC..."
    
    # Get the coordinates from user
    echo -n "Enter X coordinate (default: 0.0): "
    read x_coord
    x_coord=${x_coord:-0.0}
    
    echo -n "Enter Y coordinate (default: 0.0): "
    read y_coord
    y_coord=${y_coord:-0.0}
    
    echo -n "Enter Z coordinate (default: 0.0): "
    read z_coord
    z_coord=${z_coord:-0.0}
    
    # Get optional frame_id
    echo -n "Enter frame_id (default: map): "
    read frame_id
    frame_id=${frame_id:-map}
    
    # Source ROS1
    source "$ROS1_SETUP" > /dev/null 2>&1
    
    # Send a custom position using rostopic pub with PoseStamped format
    rostopic pub "$GOAL_TOPIC" geometry_msgs/PoseStamped "header:
  stamp:
    secs: 0
    nsecs: 0
  frame_id: '$frame_id'
pose:
  position:
    x: $x_coord
    y: $y_coord
    z: $z_coord
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0" --once
    
    print_success "Custom position sent: x=$x_coord, y=$y_coord, z=$z_coord in frame '$frame_id'"
}

# Function to send a test position
send_test_position() {
    print_status "Sending test position to $GOAL_TOPIC..."
    
    # Source ROS1
    source "$ROS1_SETUP" > /dev/null 2>&1
    
    # Send a test position using rostopic pub with PoseStamped format
    rostopic pub "$GOAL_TOPIC" geometry_msgs/PoseStamped "header:
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'map'
pose:
  position:
    x: 1.0
    y: 2.0
    z: 0.5
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0" --once
    
    print_success "Test position sent: x=1.0, y=2.0, z=0.5 in frame 'map'"
}

# Function to check ROS Bridge status with more details
check_bridge_status() {
    if check_ros_bridge; then
        print_success "ROS Bridge is running"
        
        # Show some information about the bridge
        print_status "Bridge process details:"
        pgrep -af "ros2 run ros1_bridge" || true
        
        # Check if we can detect the bridge topics
        local ros1_topics_count=0
        local ros2_topics_count=0
        
        # Check ROS1 topics
        source "$ROS1_SETUP" > /dev/null 2>&1
        if command -v rostopic &> /dev/null; then
            ros1_topics_count=$(rostopic list 2>/dev/null | wc -l)
            print_status "ROS1 topics available: $ros1_topics_count"
        else
            print_warning "Cannot check ROS1 topics (is roscore running?)"
        fi
        
        # Check ROS2 topics
        source "$ROS2_SETUP" > /dev/null 2>&1
        if command -v ros2 &> /dev/null; then
            ros2_topics_count=$(ros2 topic list 2>/dev/null | wc -l)
            print_status "ROS2 topics available: $ros2_topics_count"
        else
            print_warning "Cannot check ROS2 topics"
        fi
        
        # Check if goal topic is listed
        source "$ROS1_SETUP" > /dev/null 2>&1
        if rostopic list 2>/dev/null | grep -q "$GOAL_TOPIC"; then
            print_success "Goal topic is available in ROS1: $GOAL_TOPIC"
            
            # Show topic info
            print_status "Topic information:"
            rostopic info "$GOAL_TOPIC" 2>/dev/null || print_warning "Cannot get topic info"
        else
            print_warning "Goal topic not found in ROS1: $GOAL_TOPIC"
            print_warning "You may need to publish to it first"
        fi
        
    else
        print_warning "ROS Bridge is not running"
        print_status "Start the bridge with option 1 in the menu or run './send_position_to_robot.sh start-bridge'"
    fi
}

# Function to stop ROS Bridge
stop_ros_bridge() {
    print_status "Stopping ROS Bridge..."
    
    # Kill all ros1_bridge processes
    pkill -f "ros2 run ros1_bridge" || true
    pkill -f "dynamic_bridge" || true
    
    # Check if it was successfully stopped
    if ! check_ros_bridge; then
        print_success "ROS Bridge stopped successfully"
    else
        print_error "Failed to stop ROS Bridge"
        print_status "Try stopping it manually with: pkill -f 'ros2 run ros1_bridge'"
    fi
}

# Function to check requirements
check_requirements() {
    print_status "Checking system requirements..."
    local all_ok=true
    
    # Check if ROS1 Noetic is installed
    if [ ! -f "$ROS1_SETUP" ]; then
        print_error "ROS1 Noetic not found at: $ROS1_SETUP"
        all_ok=false
    else
        print_success "ROS1 Noetic found"
    fi
    
    # Check if ROS2 Foxy is installed
    if [ ! -f "$ROS2_SETUP" ]; then
        print_error "ROS2 Foxy not found at: $ROS2_SETUP"
        all_ok=false
    else
        print_success "ROS2 Foxy found"
    fi
    
    # Check if ros1_bridge is installed
    source "$ROS2_SETUP" > /dev/null 2>&1
    if ! ros2 pkg list 2>/dev/null | grep -q ros1_bridge; then
        print_error "ros1_bridge package not found"
        print_error "Please install it with: sudo apt install ros-humble-ros1-bridge"
        all_ok=false
    else
        print_success "ros1_bridge package found"
    fi
    
    # Check Python dependencies
    print_status "Checking Python dependencies..."
    if ! python3 -c "import rospy, geometry_msgs" 2>/dev/null; then
        print_warning "Missing Python ROS dependencies"
        print_warning "Install them with: pip install -r requirement.txt"
        all_ok=false
    else
        # Check specifically for PoseStamped message type
        if ! python3 -c "from geometry_msgs.msg import PoseStamped" 2>/dev/null; then
            print_warning "Missing geometry_msgs PoseStamped message type"
            print_warning "Make sure you have the right version of geometry_msgs"
            all_ok=false
        else
            print_success "Python ROS dependencies found including PoseStamped"
        fi
    fi
    
    # Overall status
    if [ "$all_ok" = true ]; then
        print_success "All requirements satisfied"
    else
        print_warning "Some requirements are missing"
        print_warning "Please address the issues above before using this script"
    fi
}

# Main menu function
show_menu() {
    echo ""
    echo "=========================================="
    echo "  ROS Bridge Position Publisher"
    echo "=========================================="
    echo "1. Configure network settings"
    echo "2. Start ROS Bridge"
    echo "3. Find and publish object position"
    echo "4. Send custom position"
    echo "5. Send test position"
    echo "6. Monitor $GOAL_TOPIC topic"
    echo "7. Check ROS Bridge status"
    echo "8. Check system requirements"
    echo "9. Setup Python environment"
    echo "10. Stop ROS Bridge"
    echo "0. Exit"
    echo "=========================================="
    echo -n "Enter your choice: "
}

# Main script
main() {
    print_status "ROS Bridge Position Publisher Script"
    print_status "This script helps send object positions from ROS1 Noetic to ROS2 Foxy robot"
    
    # Check if we have command line arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            "network")
                configure_network
                ;;
            "start-bridge")
                start_ros_bridge
                ;;
            "publish")
                local visualize=false  # Default to NO visualization to save time
                local query=""
                local match_index=""
                
                # Check for visualization flags
                if [[ "$*" == *"--visualize"* ]]; then
                    visualize=true
                    # Remove --visualize from arguments before extracting query
                    set -- "${@/--visualize/}"
                fi
                
                if [[ "$*" == *"--no-viz"* ]]; then
                    visualize=false
                    # Remove --no-viz from arguments before extracting query
                    set -- "${@/--no-viz/}"
                fi
                
                # Check for match index flag
                if [[ "$*" == *"--match-index"* ]]; then
                    # Extract the match index value
                    local match_idx_pattern="--match-index ([0-9]+)"
                    if [[ "$*" =~ $match_idx_pattern ]]; then
                        match_index="${BASH_REMATCH[1]}"
                        # Remove --match-index and its value from arguments
                        set -- "${@/--match-index $match_index/}"
                    fi
                fi
                
                if [ $# -ge 2 ]; then
                    # Get all arguments from 2nd onward as the query
                    shift
                    query="$*"
                    publish_position "$query" "$visualize" "$match_index"
                else
                    print_error "Please provide a search query"
                    echo "Usage: $0 publish [--visualize | --no-viz] [--match-index INDEX] <search_query>"
                    echo "  --visualize       Enable 3D visualization (loads mesh, may take longer)"
                    echo "  --no-viz          Disable visualization (faster, default)"
                    echo "  --match-index N   Use Nth match instead of the best match (default: 0)"
                fi
                ;;
            "monitor")
                monitor_topic
                ;;
            "test")
                send_test_position
                ;;
            "custom")
                send_custom_position
                ;;
            "status")
                check_bridge_status
                ;;
            "check")
                check_requirements
                ;;
            "stop")
                stop_ros_bridge
                ;;
            "env-setup")
                detect_python_env
                ;;
            *)
                echo "Usage: $0 {network|start-bridge|publish [--visualize|--no-viz] [--match-index INDEX] <query>|monitor|test|custom|status|check|env-setup|stop}"
                echo "Examples:"
                echo "  $0 publish chair                         # Find a chair and send position (no visualization)"
                echo "  $0 publish --visualize chair             # Find a chair, send position and visualize in 3D"
                echo "  $0 publish --no-viz chair                # Find a chair and send position without visualization"
                echo "  $0 publish --match-index 1 chair         # Use the 2nd best match instead of the best match"
                echo "  $0 publish --visualize --match-index 2 chair  # Use the 3rd best match and visualize it"
                echo "  $0 start-bridge                          # Start the ROS Bridge"
                echo "  $0 env-setup                             # Setup Python environment"
                echo "  $0 test                                  # Send a test position"
                echo "  $0 monitor                       # Monitor the goal position topic"
                echo "  $0 custom                        # Send custom coordinates"
                ;;
        esac
        return
    fi
    
    # Interactive mode
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                configure_network
                ;;
            2)
                start_ros_bridge
                ;;
            3)
                echo -n "Enter search query for object: "
                read query
                
                if [ -n "$query" ]; then
                    echo -n "Show visualization? (y/N): "
                    read viz_choice
                    
                    echo -n "Use a specific match index? (default: 0 for best match): "
                    read match_idx_choice
                    
                    if [[ "$viz_choice" == "y" || "$viz_choice" == "Y" ]]; then
                        publish_position "$query" "true" "$match_idx_choice"
                    else
                        publish_position "$query" "false" "$match_idx_choice"
                    fi
                else
                    print_error "No query provided"
                fi
                ;;
            4)
                send_custom_position
                ;;
            5)
                send_test_position
                ;;
            6)
                monitor_topic
                ;;
            7)
                check_bridge_status
                ;;
            8)
                check_requirements
                ;;
            9)
                detect_python_env
                ;;
            10)
                stop_ros_bridge
                ;;
            0)
                print_status "Exiting..."
                break
                ;;
            *)
                print_error "Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function with all arguments
main "$@"