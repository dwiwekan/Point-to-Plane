#!/usr/bin/env python3

import rospy
import json
import sys
import os
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from give_the_task_efficient import ObjectLocator, Config

class ROSPositionPublisher:
    """Class to publish object positions to ROS topic"""
    
    def __init__(self):
        """Initialize the ROS publisher"""
        # Initialize ROS node
        rospy.init_node('position_publisher', anonymous=True)
        
        # Create publisher for /goal_pose topic
        self.position_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
        
        # Wait for publisher to be ready
        rospy.sleep(0.5)
        
        print("🤖 ROS Position Publisher initialized")
        print(f"📡 Publishing to topic: /goal_pose")
    
    def publish_position(self, x, y, z, frame_id="map"):
        """
        Publish a position to the /goal_pose topic using PoseStamped message
        
        Args:
            x, y, z: Coordinates in the map frame
            frame_id: The coordinate frame (default: "map")
        """
        # Create PoseStamped message
        pose_msg = PoseStamped()
        
        # Set header
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = frame_id
        
        # Set position
        pose_msg.pose = Pose()
        pose_msg.pose.position = Point(float(x), float(y), float(z))
        
        # Set orientation (default to identity quaternion)
        pose_msg.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        
        # Publish the message
        self.position_pub.publish(pose_msg)
        
        print(f"📤 Published position to /goal_pose:")
        print(f"   Frame: {pose_msg.header.frame_id}")
        print(f"   Position: x={pose_msg.pose.position.x}, y={pose_msg.pose.position.y}, z={pose_msg.pose.position.z}")
        print(f"   Orientation: w={pose_msg.pose.orientation.w} (identity quaternion)")
        
        return True
    
    def find_and_publish_object(self, query):
        """
        Find the best matching object and publish its position
        
        Args:
            query: Text query to search for
        """
        print(f"\n🔍 Searching for object: '{query}'")
        
        # Create config for object locator
        config = Config()
        
        # Check file paths and update if needed
        if not os.path.exists(config.dsg_path):
            if os.path.exists("data/dsg_with_mesh.json"):
                config.dsg_path = "data/dsg_with_mesh.json"
            elif os.path.exists("data/dsg.json"):
                config.dsg_path = "data/dsg.json"
            else:
                print(f"❌ DSG file not found!")
                return False
        
        # Initialize object locator
        locator = ObjectLocator(config)
        
        # Set query and load data
        locator.set_query(query)
        
        if not locator.load_dsg():
            print("❌ Failed to load DSG data")
            return False
        
        # Find matching objects
        locator.find_matching_objects()
        
        if not locator.all_matches:
            print("❌ No matching objects found")
            return False
        
        # Get the best match
        best_match = locator.all_matches[0]
        
        print(f"\n✅ Best match found:")
        print(f"   Node ID: {best_match['node_id']}")
        print(f"   Type: {best_match['type']}")
        print(f"   Similarity: {best_match['similarity']:.4f}")
        print(f"   Original Position: {best_match['position']}")
        
        # Extract position
        pos = best_match['position']
        
        # Publish the position
        success = self.publish_position(pos[0], pos[1], pos[2])
        
        if success:
            print(f"✅ Successfully sent position to robot via ROS!")
        
        return success

def main():
    """Main function"""
    try:
        # Create publisher
        publisher = ROSPositionPublisher()
        
        # Get query from command line arguments or prompt
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = input("🔍 Enter search query for object: ")
        
        if not query.strip():
            print("❌ No query provided")
            return
        
        # Find object and publish position
        success = publisher.find_and_publish_object(query.strip())
        
        if success:
            print("\n🚀 Position sent to robot successfully!")
            print("💡 The robot should receive the goal position on /goal_pose topic")
        else:
            print("\n❌ Failed to send position to robot")
            
    except rospy.ROSInterruptException:
        print("\n⚠️ ROS node interrupted")
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()