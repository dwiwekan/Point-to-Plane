#!/bin/bash

# Setup script for ROS2 Position Receiver
# This script installs dependencies and sets up the ROS2 side of the bridge

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
ROS2_SETUP="/opt/ros/foxy/setup.bash"

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

# Function to check if ros1_bridge is installed
check_bridge() {
    source "$ROS2_SETUP" > /dev/null 2>&1
    if ros2 pkg list | grep -q ros1_bridge; then
        return 0  # Found
    else
        return 1  # Not found
    fi
}

# Function to install ros1_bridge
install_bridge() {
    print_status "Installing ros1_bridge package..."
    sudo apt update
    sudo apt install -y ros-foxy-ros1-bridge
}

# Function to start ROS2 receiver node
start_receiver() {
    print_status "Starting ROS2 position receiver node..."
    
    # Check if the script exists
    if [ ! -f "ros2_position_receiver.py" ]; then
        print_error "ros2_position_receiver.py not found!"
        return 1
    fi
    
    # Make sure it's executable
    chmod +x ros2_position_receiver.py
    
    # Source ROS2
    source "$ROS2_SETUP"
    
    # Run the receiver
    python3 ros2_position_receiver.py
}

# Function to configure network for ROS communication
configure_network() {
    print_status "Configuring network settings for ROS communication..."
    
    # Get hostname and IP
    hostname=$(hostname)
    ip_address=$(hostname -I | awk '{print $1}')
    
    print_status "This computer:"
    print_status "- Hostname: $hostname"
    print_status "- IP Address: $ip_address"
    
    # Ask for ROS1 machine info
    echo -n "Enter ROS1 machine IP address: "
    read ros1_ip
    
    # Create network configuration file
    cat > ros_network_setup.sh << EOF
#!/bin/bash
# ROS network configuration

# ROS1 machine info
export ROS_MASTER_URI=http://${ros1_ip}:11311
export ROS_IP=${ip_address}

# ROS2 domain ID (must be the same on both machines)
export ROS_DOMAIN_ID=42

# Print network configuration
echo "ROS network configuration:"
echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
EOF
    
    chmod +x ros_network_setup.sh
    
    print_success "Network configuration created in ros_network_setup.sh"
    print_status "Source this file before running the bridge or receiver:"
    print_status "source ./ros_network_setup.sh"
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    local all_ok=true
    
    # Check if ROS2 Foxy is installed
    if [ ! -f "$ROS2_SETUP" ]; then
        print_error "ROS2 Foxy not found at: $ROS2_SETUP"
        all_ok=false
    else
        print_success "ROS2 Foxy found"
    fi
    
    # Check if ros1_bridge is installed
    if ! check_bridge; then
        print_warning "ros1_bridge package not found"
        print_status "Will install it with: sudo apt install ros-foxy-ros1-bridge"
        all_ok=false
    else
        print_success "ros1_bridge package found"
    fi
    
    # Check Python dependencies
    print_status "Checking Python dependencies..."
    if ! python3 -c "import rclpy, geometry_msgs" 2>/dev/null; then
        print_warning "Missing Python ROS2 dependencies"
        print_warning "Will install them with: pip install -r ros2_requirements.txt"
        
        # Create ROS2 requirements file
        echo "rclpy
geometry_msgs" > ros2_requirements.txt
        
        all_ok=false
    else
        print_success "Python ROS2 dependencies found"
    fi
    
    # Overall status
    if [ "$all_ok" = true ]; then
        print_success "All requirements satisfied"
    else
        print_warning "Some requirements are missing and will be installed"
    fi
}

# Function to guide through setup steps
guide_setup() {
    print_status "ROS2 Position Receiver Setup Guide"
    print_status "Follow these steps to set up the ROS2 receiver side:"
    
    print_status "1. Configure network settings"
    configure_network
    
    print_status "2. Install required packages"
    if ! check_bridge; then
        install_bridge
    fi
    
    if [ -f "ros2_requirements.txt" ]; then
        pip install -r ros2_requirements.txt
    fi
    
    print_status "3. Start ROS2 side of bridge (in a separate terminal):"
    print_status "   a. Source network settings: source ./ros_network_setup.sh"
    print_status "   b. Source ROS2: source $ROS2_SETUP"
    print_status "   c. Run bridge: ros2 run ros1_bridge dynamic_bridge"
    
    print_status "4. Start position receiver (in another terminal):"
    print_status "   a. Source network settings: source ./ros_network_setup.sh"
    print_status "   b. Source ROS2: source $ROS2_SETUP"
    print_status "   c. Run receiver: python3 ros2_position_receiver.py"
    
    print_success "Setup guide complete! Please follow the steps above."
}

# Function to start bridge from ROS2 side
start_bridge() {
    print_status "Starting ROS1-ROS2 Bridge from ROS2 side..."
    
    # Source ROS2
    source "$ROS2_SETUP"
    
    # Run bridge
    if [ -f "ros_network_setup.sh" ]; then
        source ./ros_network_setup.sh
    else
        print_warning "Network settings not found. Network configuration may be required."
    fi
    
    print_status "Running bridge..."
    ros2 run ros1_bridge dynamic_bridge
}

# Main menu
show_menu() {
    echo ""
    echo "=========================================="
    echo "  ROS2 Position Receiver Setup"
    echo "=========================================="
    echo "1. Check system requirements"
    echo "2. Configure network settings"
    echo "3. Install required packages"
    echo "4. Start ROS Bridge (from ROS2 side)"
    echo "5. Start Position Receiver"
    echo "6. Guided setup (steps 1-3)"
    echo "0. Exit"
    echo "=========================================="
    echo -n "Enter your choice: "
}

# Main script
main() {
    print_status "ROS2 Position Receiver Setup Script"
    print_status "This script helps set up the ROS2 side to receive positions from ROS1"
    
    # Check if we have command line arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            "check")
                check_requirements
                ;;
            "network")
                configure_network
                ;;
            "install")
                install_bridge
                ;;
            "bridge")
                start_bridge
                ;;
            "receiver")
                start_receiver
                ;;
            "guide")
                guide_setup
                ;;
            *)
                echo "Usage: $0 {check|network|install|bridge|receiver|guide}"
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
                check_requirements
                ;;
            2)
                configure_network
                ;;
            3)
                if ! check_bridge; then
                    install_bridge
                else
                    print_success "ros1_bridge is already installed"
                fi
                ;;
            4)
                start_bridge
                ;;
            5)
                start_receiver
                ;;
            6)
                guide_setup
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