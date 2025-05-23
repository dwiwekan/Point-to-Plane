#!/usr/bin/env python3
"""
ROS2 Python Setup Helper

This script helps locate and configure Python to use ROS2 modules.
It's useful when the standard ROS2 Python packages aren't properly installed.
"""

import os
import sys
import site
import subprocess
from pathlib import Path


def find_ros2_python_path():
    """Find the ROS2 Python path by looking in standard locations."""
    ros_root = Path("/opt/ros/humble")
    
    # Look for the Python site-packages directory in ROS installation
    candidates = list(ros_root.glob("**/site-packages"))
    
    if not candidates:
        print("ERROR: Could not find ROS2 Python packages.")
        return None
    
    # Return the first matching directory
    return candidates[0]


def create_symlinks(src_dir):
    """Create symlinks for ROS2 Python packages in the user's site-packages directory."""
    # Get user site-packages directory
    user_site = site.getusersitepackages()
    os.makedirs(user_site, exist_ok=True)
    
    print(f"Creating symlinks in {user_site} for ROS2 Python modules...")
    
    # List key ROS2 packages to link
    ros_packages = ["rclpy", "geometry_msgs"]
    
    for pkg in ros_packages:
        src = src_dir / pkg
        dst = Path(user_site) / pkg
        
        if src.exists() and not dst.exists():
            try:
                # Create symlink from ROS2 path to user site-packages
                dst.symlink_to(src, target_is_directory=True)
                print(f"Created symlink: {dst} -> {src}")
            except Exception as e:
                print(f"Failed to create symlink for {pkg}: {e}")
        elif dst.exists():
            print(f"Symlink for {pkg} already exists")
        else:
            print(f"Source package {pkg} not found in {src_dir}")


def setup_ros_pythonpath():
    """Create a shell script that sets up PYTHONPATH for ROS2."""
    ros_python_path = find_ros2_python_path()
    if not ros_python_path:
        return False
    
    script_content = f"""#!/bin/bash
# ROS2 Python path setup
export PYTHONPATH={ros_python_path}:$PYTHONPATH

# Echo the configuration
echo "ROS2 Python environment:"
echo "PYTHONPATH includes: {ros_python_path}"
"""
    
    script_path = Path("./ros2_python_setup.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable
    
    print(f"Created {script_path} to set up ROS2 Python environment")
    print("Source this file before running ROS2 Python scripts:")
    print("source ./ros2_python_setup.sh")
    return True


def main():
    """Main function to set up ROS2 Python environment."""
    print("Setting up ROS2 Python environment...")
    
    # Try to import rclpy to see if it's already working
    try:
        import rclpy
        print("ROS2 Python modules are already accessible!")
        return 0
    except ImportError:
        print("ROS2 Python modules not found. Setting up environment...")
    
    # Find ROS2 Python path
    ros_python_path = find_ros2_python_path()
    if not ros_python_path:
        print("ERROR: Could not locate ROS2 Python modules.")
        return 1
    
    print(f"Found ROS2 Python modules at: {ros_python_path}")
    
    # Create symlinks for easy access
    create_symlinks(ros_python_path)
    
    # Create setup script
    setup_ros_pythonpath()
    
    # Try importing again
    try:
        sys.path.append(str(ros_python_path))
        import rclpy
        print("SUCCESS: ROS2 Python modules can now be imported!")
        return 0
    except ImportError:
        print("ERROR: Still unable to import ROS2 Python modules.")
        print("You may need to restart your Python interpreter or install ROS2 properly.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
