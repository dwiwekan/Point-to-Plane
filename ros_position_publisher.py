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
    
    def find_and_publish_object(self, query, visualize=True, match_index=0):
        """
        Find the best matching object, publish its position, and optionally visualize it
        
        Args:
            query: Text query to search for
            visualize: Whether to visualize the object in 3D (default: True)
            match_index: Index of the match to use (default: 0 for best match)
        """
        print(f"\n🔍 Searching for object: '{query}'")
        
        # Create config for object locator
        config = Config()
        
        # Set visualization mode for best match
        config.visualization_mode = "best"
        
        # Check file paths and update if needed
        if not os.path.exists(config.dsg_path):
            print(f"⚠️ DSG path not found at: {config.dsg_path}")
            # Try data/ folder (lowercase)
            if os.path.exists("data/dsg_with_mesh.json"):
                config.dsg_path = "data/dsg_with_mesh.json"
                print(f"✅ Found DSG at: {config.dsg_path}")
            elif os.path.exists("data/dsg.json"):
                config.dsg_path = "data/dsg.json"
                print(f"✅ Found DSG at: {config.dsg_path}")
            # Try Data/ folder (uppercase)
            elif os.path.exists("Data/dsg_with_mesh.json"):
                config.dsg_path = "Data/dsg_with_mesh.json"
                print(f"✅ Found DSG at: {config.dsg_path}")
            elif os.path.exists("Data/dsg.json"):
                config.dsg_path = "Data/dsg.json"
                print(f"✅ Found DSG at: {config.dsg_path}")
            else:
                print(f"❌ DSG file not found!")
                return False
        
        # Check mesh path and update if needed
        if not os.path.exists(config.mesh_path):
            print(f"⚠️ Mesh path not found at: {config.mesh_path}")
            # Try data/ folder (lowercase)
            if os.path.exists("data/cloud_aligned_new.ply"):
                config.mesh_path = "data/cloud_aligned_new.ply"
                print(f"✅ Found mesh at: {config.mesh_path}")
            elif os.path.exists("data/cloud_aligned.ply"):
                config.mesh_path = "data/cloud_aligned.ply"
                print(f"✅ Found mesh at: {config.mesh_path}")
            # Try Data/ folder (uppercase)
            elif os.path.exists("Data/cloud_aligned_new.ply"):
                config.mesh_path = "Data/cloud_aligned_new.ply"
                print(f"✅ Found mesh at: {config.mesh_path}")
            elif os.path.exists("Data/cloud_aligned.ply"):
                config.mesh_path = "Data/cloud_aligned.ply"
                print(f"✅ Found mesh at: {config.mesh_path}")
        
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
        
        # Get the match based on the specified index
        if match_index < len(locator.all_matches):
            selected_match = locator.all_matches[match_index]
            match_type = "Selected" if match_index > 0 else "Best"
            
            print(f"\n✅ {match_type} match found (index {match_index}):")
            print(f"   Node ID: {selected_match['node_id']}")
            print(f"   Type: {selected_match['type']}")
            print(f"   Similarity: {selected_match['similarity']:.4f}")
            print(f"   Original Position: {selected_match['position']}")
        else:
            print(f"❌ Match index {match_index} is out of range. Only {len(locator.all_matches)} matches found.")
            return False
        
        # Extract position
        pos = selected_match['position']
        
        # Publish the position
        success = self.publish_position(pos[0], pos[1], pos[2])
        
        if success:
            print(f"✅ Successfully sent position to robot via ROS!")
        
        # Visualize the best match if requested
        if visualize:
            print("\n🌟 Starting visualization of selected match...")
            # Use the visualize_best_match method to show the object in 3D
            locator.visualize_best_match(match_index)
        
        return success

def main():
    """Main function"""
    try:
        # Create publisher
        publisher = ROSPositionPublisher()
        
        # Get query from command line arguments or prompt
        visualize = True  # Default to visualize
        query = ""
        match_index = 0  # Default to best match (index 0)
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            # Check for --no-viz flag
            if "--no-viz" in sys.argv:
                visualize = False
                # Remove the flag from arguments
                sys.argv.remove("--no-viz")
            
            # Check for --match-index flag
            match_index_flag = "--match-index"
            if match_index_flag in sys.argv:
                idx = sys.argv.index(match_index_flag)
                if idx + 1 < len(sys.argv):
                    try:
                        match_index = int(sys.argv[idx + 1])
                        # Remove the flag and its value from arguments
                        sys.argv.pop(idx)  # Remove flag
                        sys.argv.pop(idx)  # Remove value
                    except ValueError:
                        print(f"⚠️ Invalid match index: {sys.argv[idx + 1]}. Using default (0).")
                else:
                    print(f"⚠️ No value provided for {match_index_flag}. Using default (0).")
            
            # The rest is the query
            if len(sys.argv) > 1:
                query = " ".join(sys.argv[1:])
        
        # If no query provided via command line, prompt user
        if not query:
            query = input("🔍 Enter search query for object: ")
            viz_choice = input("🖼️ Visualize object? (Y/n): ").lower()
            visualize = viz_choice != "n"
            
            try:
                match_idx_input = input("🔢 Match index to use (default: 0 for best match): ")
                if match_idx_input.strip():
                    match_index = int(match_idx_input)
            except ValueError:
                print(f"⚠️ Invalid match index. Using default (0).")
        
        if not query.strip():
            print("❌ No query provided")
            return
        
        # Find object, publish position, and visualize if requested
        success = publisher.find_and_publish_object(query.strip(), visualize, match_index)
        
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