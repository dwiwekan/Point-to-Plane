#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import sys
import os
import time

class PositionReceiver(Node):
    """ROS2 Node to receive position data from ROS1 via ROS Bridge"""
    
    def __init__(self):
        super().__init__('position_receiver')
        
        # Create subscription to /goal_position topic
        self.subscription = self.create_subscription(
            Point,
            '/goal_position',
            self.position_callback,
            10)
        
        # Optional: Create a publisher if you need to forward the position to another ROS2 topic
        # self.publisher = self.create_publisher(Point, '/robot_goal', 10)
        
        self.get_logger().info('Position receiver node initialized')
        self.get_logger().info('Waiting for positions on /goal_position topic...')
    
    def position_callback(self, msg):
        """Callback function when a position message is received"""
        self.get_logger().info('Received position:')
        self.get_logger().info(f'  x: {msg.x}')
        self.get_logger().info(f'  y: {msg.y}')
        self.get_logger().info(f'  z: {msg.z}')
        
        # Process the position data (example: you can add your robot control code here)
        self.process_position(msg)
        
        # Optional: Forward the message to another topic if needed
        # self.publisher.publish(msg)
    
    def process_position(self, position):
        """Process the received position data (implement your robot logic here)"""
        # This is where you'd implement code to control your robot to move to the position
        self.get_logger().info('Processing position for robot movement...')
        
        # Example: Convert position to robot-specific commands
        # robot_cmd = convert_to_robot_command(position)
        # send_command_to_robot(robot_cmd)
        
        # For demonstration, we'll just log that we're sending it to the robot
        self.get_logger().info('Position sent to robot controller')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create and run the node
        position_receiver = PositionReceiver()
        
        # Keep the node running
        rclpy.spin(position_receiver)
        
    except KeyboardInterrupt:
        print('Node stopped by keyboard interrupt')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        # Clean shutdown
        if rclpy.ok():
            position_receiver.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()