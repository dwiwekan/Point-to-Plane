import json
import torch
import open3d as o3d
import numpy as np
import colorsys
import argparse
import time
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from open_clip import create_model_and_transforms, tokenize

# Default configuration
@dataclass
class Config:
    # Visualization settings
    visualization_mode: str = "best"  # "multiple", "best", or "both"
    similarity_threshold: float = 0.15  # Show objects with at least 15% similarity
    max_objects: int = 10  # Maximum number of objects to display
    show_connection_lines: bool = False  # Show lines to original positions
    
    # Position offset (in meters) for highlighted objects
    sphere_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z offsets
    
    # Paths
    dsg_path: str = "Data/dsg_with_mesh.json"
    mesh_path: str = "Data/cloud_aligned_new.ply"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"

class ObjectLocator:
    """Class for locating objects in 3D scenes using semantic search"""
    
    def __init__(self, config: Config):
        """
        Initialize the object locator
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.start_time = time.time()
        self.debug_timing("Initializing")
        
        # Initialize model
        self.model, _, self.preprocess = self._load_clip_model()
        
        # Initialize variables for DSG processing
        self.dsg_raw = None
        self.nodes_raw = None
        self.objects = {}
        self.all_matches = []
        
        # Initialize query variables
        self.query = None
        self.text_feature = None
        
    def _load_clip_model(self) -> Tuple:
        """
        Load the CLIP model
        
        Returns:
            Tuple of (model, _, preprocess)
        """
        print(f"\n🔍 Loading CLIP model ({self.config.clip_model}) on {self.config.device}...")
        model_start = time.time()
        
        model, _, preprocess = create_model_and_transforms(
            self.config.clip_model, 
            pretrained=self.config.clip_pretrained
        )
        model = model.to(self.config.device).eval()
        
        print(f"✅ Model loaded in {time.time() - model_start:.2f} seconds")
        return model, _, preprocess
    
    def set_query(self, query: str) -> None:
        """
        Set the query and compute its text embedding
        
        Args:
            query: Text query
        """
        print(f"\n🔍 Processing query: '{query}'")
        self.query = query
        
        # Tokenize and encode the query
        query_start = time.time()
        text_tokens = tokenize([query]).to(self.config.device)
        
        with torch.no_grad():
            self.text_feature = self.model.encode_text(text_tokens)
            self.text_feature /= self.text_feature.norm(dim=-1, keepdim=True)
        
        print(f"✅ Query encoded in {time.time() - query_start:.2f} seconds")
    
    def load_dsg(self) -> bool:
        """
        Load and parse the Dynamic Scene Graph
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n📦 Loading DSG from {self.config.dsg_path}")
        dsg_start = time.time()
        
        try:
            with open(self.config.dsg_path, 'r') as f:
                self.dsg_raw = json.load(f)
            
            print(f"✅ DSG loaded in {time.time() - dsg_start:.2f} seconds")
            
            # Extract nodes
            self.nodes_raw = self.dsg_raw.get("nodes", [])
            print(f"📊 Total nodes in DSG: {len(self.nodes_raw)}")
            
            # Process the DSG to extract objects
            self._process_dsg()
            
            return True
        except Exception as e:
            print(f"❌ Failed to load DSG: {e}")
            print(f"  Check if the file exists at: {self.config.dsg_path}")
            return False
    
    def _process_dsg(self) -> None:
        """
        Process the DSG to extract objects with semantic features and positions
        """
        process_start = time.time()
        print("\n📦 Processing DSG to extract objects...")
        
        # Analyze node types
        node_types = {}
        feature_counts = 0
        position_counts = 0
        
        # Extract valid objects (nodes with both semantic features and positions)
        for i, node in enumerate(self.nodes_raw):
            if not isinstance(node, dict) or "attributes" not in node:
                continue
            
            attributes = node["attributes"]
            has_semantic = False
            has_position = False
            
            # Check for semantic features
            if (isinstance(attributes, dict) and 
                "semantic_feature" in attributes and 
                isinstance(attributes["semantic_feature"], dict) and 
                "data" in attributes["semantic_feature"] and 
                isinstance(attributes["semantic_feature"]["data"], list) and 
                len(attributes["semantic_feature"]["data"]) == 768):
                has_semantic = True
                feature_counts += 1
            
            # Check for position
            if "position" in attributes:
                has_position = True
                position_counts += 1
            
            # Track node types
            if isinstance(node, dict) and "type" in node:
                node_type = node["type"]
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Only use nodes with both features
            if has_semantic and has_position:
                node_id = str(node.get("id", f"node_{i}"))
                self.objects[node_id] = {
                    "position": attributes["position"],
                    "semantic_feature": attributes["semantic_feature"]["data"],
                    "type": node.get("type", "unknown"),
                }
        
        # Debug output
        print(f"✅ DSG processing completed in {time.time() - process_start:.2f} seconds")
        print(f"📊 Valid objects found: {len(self.objects)}")
        print(f"📊 Nodes with semantic features: {feature_counts}")
        print(f"📊 Nodes with position data: {position_counts}")
        
        # Print node type distribution
        print("\n📊 Node type distribution:")
        for node_type, count in node_types.items():
            print(f" - {node_type}: {count} nodes")
        
        # Show sample of objects
        if self.objects:
            print(f"\n📦 Sample of extracted objects:")
            for i, (node_id, node) in enumerate(list(self.objects.items())[:3]):
                print(f" - Node ID: {node_id}")
                print(f"   Type: {node['type']}")
                print(f"   Position: {node['position']}")
                print(f"   Feature length: {len(node['semantic_feature'])}")
                print("   ------")
    
    def find_matching_objects(self) -> None:
        """
        Find objects matching the query based on semantic similarity
        """
        if self.text_feature is None or not self.objects:
            print("❌ Cannot find matching objects: text feature or objects missing")
            return
        
        print(f"\n🔍 Finding objects matching query: '{self.query}'")
        search_start = time.time()
        
        # Calculate similarity for all objects
        self.all_matches = []
        
        for node_id, node in self.objects.items():
            # Get the semantic feature vector
            feat = node["semantic_feature"]
            node_feat = torch.tensor(feat, dtype=torch.float32).to(self.config.device).view(1, -1)
            node_feat /= node_feat.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity with query
            similarity = torch.nn.functional.cosine_similarity(
                self.text_feature, node_feat).item()
            
            # Normalize position data
            position_array = None
            if isinstance(node["position"], list) and len(node["position"]) == 3:
                position_array = node["position"]
            elif isinstance(node["position"], dict) and all(k in node["position"] for k in ["x", "y", "z"]):
                position_array = [node["position"]["x"], node["position"]["y"], node["position"]["z"]]
            
            if position_array:
                self.all_matches.append({
                    "node_id": node_id,
                    "position": position_array,
                    "type": node.get("type", "unknown"),
                    "similarity": similarity
                })
        
        # Sort by similarity (highest first)
        self.all_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        print(f"✅ Found {len(self.all_matches)} potential matches in {time.time() - search_start:.2f} seconds")
        
        # Show top matches
        if self.all_matches:
            print(f"\n🏆 Top matches:")
            for i, match in enumerate(self.all_matches[:5]):
                print(f" {i+1}. {match['type']} (similarity: {match['similarity']:.4f})")
    
    def load_mesh(self) -> Optional[Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]]:
        """
        Load the mesh or point cloud for visualization
        
        Returns:
            Loaded mesh or point cloud, or None if loading fails
        """
        try:
            print(f"\n📂 Loading mesh from: {self.config.mesh_path}")
            mesh_start = time.time()
            
            # Try loading as a mesh first
            mesh = o3d.io.read_triangle_mesh(self.config.mesh_path)
            
            if mesh.has_triangles():
                print(f"✅ Successfully loaded mesh with {len(mesh.triangles)} triangles")
                mesh.compute_vertex_normals()
                print(f"   Mesh loaded in {time.time() - mesh_start:.2f} seconds")
                return mesh
            else:
                print(f"⚠️ File contains no triangles. Loading as point cloud...")
                pcd = o3d.io.read_point_cloud(self.config.mesh_path)
                
                if pcd.has_points():
                    print(f"✅ Successfully loaded point cloud with {len(pcd.points)} points")
                    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # Gray color
                    print(f"   Point cloud loaded in {time.time() - mesh_start:.2f} seconds")
                    return pcd
                else:
                    raise ValueError("File contains neither mesh triangles nor points")
        except Exception as e:
            print(f"❌ Failed to load mesh: {e}")
            print(f"  Check the file path: {self.config.mesh_path}")
            return None
    
    def visualize_best_match(self, match_index: int = 0) -> None:
        """
        Visualize a specific matching object in the 3D scene
        
        Args:
            match_index: Index of the match to visualize (default: 0 for best match)
        """
        if not self.all_matches:
            print("❌ No matching objects found.")
            return
        
        # Verify that the match index is valid
        if match_index < 0 or match_index >= len(self.all_matches):
            print(f"❌ Invalid match index: {match_index}. Using best match (index 0).")
            match_index = 0
        
        # Load mesh for visualization
        geometry = self.load_mesh()
        if geometry is None:
            return
        
        # Get the selected match
        selected_match = self.all_matches[match_index]
        match_ordinal = "best" if match_index == 0 else f"{match_index+1}{'st' if match_index == 0 else 'nd' if match_index == 1 else 'rd' if match_index == 2 else 'th'}"
        print(f"\n✅ {match_ordinal.capitalize()} match object:")
        print(f" - Query: '{self.query}'")
        print(f" - Node ID: {selected_match['node_id']}")
        print(f" - Type: {selected_match['type']}")
        print(f" - 3D Position: {selected_match['position']}")
        print(f" - Similarity: {selected_match['similarity']:.4f}")
        
        # Get position with offsets
        pos = selected_match['position']
        adjusted_pos = [
            pos[0] + self.config.sphere_offset[0],
            pos[1] + self.config.sphere_offset[1],
            pos[2] + self.config.sphere_offset[2]
        ]
        
        # Create sphere to highlight the selected match
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.paint_uniform_color([1, 0, 0])  # Red
        sphere.compute_vertex_normals()
        sphere.translate(adjusted_pos)
        
        # List of geometries to visualize
        geometries = [geometry, sphere]
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        geometries.append(coordinate_frame)
        
        # Show a line connecting the actual position to the sphere (if enabled)
        if self.config.show_connection_lines:
            line_points = [pos, adjusted_pos]
            line_indices = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red line
            geometries.append(line_set)
        
        # Visualization
        print("\n🌟 Starting 3D visualization (selected match object)...")
        print(f"📌 Original position: {pos}")
        print(f"📌 Adjusted position: {adjusted_pos}")
        
        # Use simple draw_geometries function as in original code
        match_label = "Best Match" if match_index == 0 else f"Match #{match_index+1}"
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"{match_label}: {self.query}",
            width=1024, 
            height=768
        )
        
        print("✅ Visualization complete")
    
    def visualize_multiple_matches(self) -> None:
        """
        Visualize multiple matching objects in the 3D scene
        """
        # Only show objects above threshold
        filtered_matches = [match for match in self.all_matches 
                           if match["similarity"] >= self.config.similarity_threshold]
        filtered_matches = filtered_matches[:self.config.max_objects]
        
        if not filtered_matches:
            print("❌ No objects found above the similarity threshold.")
            return
        
        # Load mesh for visualization
        geometry = self.load_mesh()
        if geometry is None:
            return
        
        print(f"\n✅ Similar objects ({len(filtered_matches)}):")
        print(f" - Query: '{self.query}'")
        
        for i, match in enumerate(filtered_matches):
            print(f"\n👉 Match #{i+1}:")
            print(f" - Node ID: {match['node_id']}")
            print(f" - Type: {match['type']}")
            print(f" - 3D Position: {match['position']}")
            print(f" - Similarity: {match['similarity']:.4f}")
        
        # Create list of geometries
        geometries = [geometry]
        
        # Create spheres with different colors based on similarity
        for i, match in enumerate(filtered_matches):
            pos = match["position"]
            adjusted_pos = [
                pos[0] + self.config.sphere_offset[0],
                pos[1] + self.config.sphere_offset[1],
                pos[2] + self.config.sphere_offset[2]
            ]
            
            # Generate color (evenly distributed around color wheel)
            hue = i / len(filtered_matches)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = [r, g, b]
            
            # Size based on similarity (more similar = larger sphere)
            radius = 0.1 + match["similarity"] * 0.2
            
            # Create sphere with offset
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            sphere.translate(adjusted_pos)
            geometries.append(sphere)
            
            # Create connection line between original position and sphere (if enabled)
            if self.config.show_connection_lines:
                line_points = [pos, adjusted_pos]
                line_indices = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                line_set.colors = o3d.utility.Vector3dVector([color])
                geometries.append(line_set)
            
            print(f"📍 Object #{i+1}: original={pos}, adjusted={adjusted_pos}")
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        geometries.append(coordinate_frame)
        
        print("\n🌟 Starting 3D visualization...")
        
        # Use simple draw_geometries function as in original code
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Multiple Objects: {self.query}",
            width=1024, 
            height=768
        )
        
        print("✅ Visualization complete")
    
    def create_wireframe_box(self, center, size=(0.3, 0.3, 0.3), color=(0, 1, 0)):
        """Create a wireframe bounding box."""
        # Calculate the 8 vertices of the box
        x, y, z = center
        w, h, d = size[0]/2, size[1]/2, size[2]/2
        
        # Define the 8 vertices of the box
        points = [
            [x-w, y-h, z-d], [x+w, y-h, z-d], [x+w, y+h, z-d], [x-w, y+h, z-d],
            [x-w, y-h, z+d], [x+w, y-h, z+d], [x+w, y+h, z+d], [x-w, y+h, z+d]
        ]
        
        # Define the 12 lines (edges) of the box
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
        ]
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Assign colors (same color for all lines)
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def visualize_objects_on_mesh(
        self,
        matches: List[dict],
        create_wireframe: bool = True,
        show_labels: bool = True,
        mesh_color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    ):
        """
        Visualize objects on a mesh with enhanced features
        
        Args:
            matches: List of object matches
            create_wireframe: Whether to use wireframe boxes
            show_labels: Whether to show text labels
            mesh_color: Color for the mesh
        """
        # Load the mesh
        geometry = self.load_mesh()
        if geometry is None:
            return
        
        # Set a uniform color for the mesh for better visibility
        geometry.paint_uniform_color(mesh_color)
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        
        # Create visualization objects
        geometries = [geometry, coordinate_frame]
        labels_info = []
        
        # Add bounding boxes for each object
        for i, match in enumerate(matches):
            position = match["position"]
            similarity = match["similarity"]
            
            # Generate color based on similarity (green = high, red = low)
            hue = similarity * 0.3  # 0.3 = green, 0 = red
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = (r, g, b)
            
            # Size proportional to similarity, with a minimum size
            size_scale = 0.2 + similarity * 0.2
            box_size = (size_scale, size_scale, size_scale)
            
            # Create either wireframe or solid box
            if create_wireframe:
                box = self.create_wireframe_box(position, box_size, color)
            else:
                box = o3d.geometry.TriangleMesh.create_box(
                    width=box_size[0], height=box_size[1], depth=box_size[2])
                box.paint_uniform_color(color)
                box.compute_vertex_normals()
                
                # Center the box at the given position
                box.translate([-box_size[0]/2, -box_size[1]/2, -box_size[2]/2])  # Center at origin
                box.translate(position)  # Move to desired position
            
            geometries.append(box)
            
            # Add a label marker if requested
            if show_labels:
                # Position the label marker above the object
                label_pos = [position[0], position[1], position[2] + box_size[2]/2 + 0.05]
                label_text = f"{match['type']} ({similarity:.2f})"
                
                # Create a small sphere as a marker
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                marker.paint_uniform_color(color)
                marker.translate(label_pos)
                
                geometries.append(marker)
                labels_info.append((label_text, label_pos))
        
        # If we have labels, print them (since we can't render text directly in 3D)
        if show_labels and labels_info:
            print("\n📝 Object labels (shown as colored markers in visualization):")
            for i, (text, pos) in enumerate(labels_info):
                print(f" - Marker {i+1}: {text} at {pos}")
        
        print("\n🌟 Starting enhanced 3D visualization...")
        
        # Use simple draw_geometries function as in original code
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Query: {self.query}",
            width=1024, 
            height=768
        )
        
        print("✅ Visualization complete")
    
    def visualize_results(self, match_index: int = 0) -> None:
        """
        Visualize search results based on the configured visualization mode
        
        Args:
            match_index: Index of the match to visualize when using 'best' mode (default: 0)
        """
        visualization_start = time.time()
        
        if self.config.visualization_mode == "multiple":
            self.visualize_multiple_matches()
        elif self.config.visualization_mode == "best":
            self.visualize_best_match(match_index)
        elif self.config.visualization_mode == "both":
            self.visualize_best_match(match_index)
            self.visualize_multiple_matches()
        elif self.config.visualization_mode == "advanced":
            # Filter matches based on threshold
            filtered_matches = [match for match in self.all_matches 
                              if match["similarity"] >= self.config.similarity_threshold]
            filtered_matches = filtered_matches[:self.config.max_objects]
            
            if filtered_matches:
                self.visualize_objects_on_mesh(filtered_matches, create_wireframe=True, show_labels=True)
            else:
                print("❌ No objects found above the similarity threshold.")
        else:
            # Interactive selection
            print("\n🔍 Select visualization mode:")
            print(" 1. View multiple objects (above similarity threshold)")
            print(" 2. View only a specific match")
            print(" 3. View both specific match and multiple objects")
            print(" 4. Advanced visualization with wireframe boxes")
            choice = input("Choice (1-4): ")
            
            if choice == "1":
                self.visualize_multiple_matches()
            elif choice == "2":
                if len(self.all_matches) > 1:
                    print(f"\nFound {len(self.all_matches)} matches:")
                    for i, match in enumerate(self.all_matches[:10]):  # Show top 10 matches
                        print(f" {i}. {match['type']} (similarity: {match['similarity']:.4f})")
                    
                    match_idx_input = input(f"Enter match index to visualize (0-{len(self.all_matches)-1}, default: {match_index}): ")
                    try:
                        match_idx = int(match_idx_input)
                        self.visualize_best_match(match_idx)
                    except ValueError:
                        self.visualize_best_match(match_index)  # Use provided default if invalid input
                else:
                    self.visualize_best_match(match_index)
            elif choice == "3":
                if len(self.all_matches) > 1:
                    print(f"\nFound {len(self.all_matches)} matches:")
                    for i, match in enumerate(self.all_matches[:10]):  # Show top 10 matches
                        print(f" {i}. {match['type']} (similarity: {match['similarity']:.4f})")
                    
                    match_idx_input = input(f"Enter match index to visualize (0-{len(self.all_matches)-1}, default: {match_index}): ")
                    try:
                        match_idx = int(match_idx_input)
                        self.visualize_best_match(match_idx)
                    except ValueError:
                        self.visualize_best_match(match_index)  # Use provided default if invalid input
                else:
                    self.visualize_best_match(match_index)
                self.visualize_multiple_matches()
            elif choice == "4":
                filtered_matches = [match for match in self.all_matches 
                                  if match["similarity"] >= self.config.similarity_threshold]
                filtered_matches = filtered_matches[:self.config.max_objects]
                
                if filtered_matches:
                    self.visualize_objects_on_mesh(filtered_matches)
                else:
                    print("❌ No objects found above the similarity threshold.")
            else:
                print("❌ Invalid choice. Defaulting to multiple objects.")
                self.visualize_multiple_matches()
        
        print(f"✅ Visualization completed in {time.time() - visualization_start:.2f} seconds")
    
    def run(self, query: str, match_index: int = 0) -> None:
        """
        Run the complete object location process with the given query
        
        Args:
            query: Text query to search for
            match_index: Index of the match to visualize when using 'best' mode (default: 0)
        """
        self.set_query(query)
        
        if not self.load_dsg():
            return
        
        self.find_matching_objects()
        self.visualize_results(match_index)
        
        total_time = time.time() - self.start_time
        print(f"\n✅ Total execution time: {total_time:.2f} seconds")
    
    def debug_timing(self, message: str) -> None:
        """
        Print debug timing information
        
        Args:
            message: Message to print
        """
        elapsed = time.time() - self.start_time
        print(f"DEBUG: {message} (elapsed: {elapsed:.2f}s)")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Locate objects in 3D scenes using semantic search")
    parser.add_argument("--query", type=str, help="Text query to search for objects")
    parser.add_argument("--dsg", type=str, help="Path to DSG JSON file")
    parser.add_argument("--mesh", type=str, help="Path to mesh or point cloud file")
    parser.add_argument("--mode", type=str, choices=["best", "multiple", "both", "advanced", "interactive"],
                      help="Visualization mode")
    parser.add_argument("--threshold", type=float, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--max-objects", type=int, help="Maximum number of objects to display")
    parser.add_argument("--show-lines", action="store_true", help="Show connection lines to original positions")
    parser.add_argument("--match-index", type=int, default=0, help="Index of the match to visualize when using 'best' mode")
    
    args = parser.parse_args()
    
    # Create default config
    config = Config()
    
    # Override with command line arguments if provided
    if args.dsg:
        config.dsg_path = args.dsg
    if args.mesh:
        config.mesh_path = args.mesh
    if args.mode:
        config.visualization_mode = args.mode
    if args.threshold is not None:
        config.similarity_threshold = args.threshold
    if args.max_objects is not None:
        config.max_objects = args.max_objects
    if args.show_lines:
        config.show_connection_lines = True
    
    # Verify file paths
    if not os.path.exists(config.dsg_path):
        print(f"❌ DSG file not found at: {config.dsg_path}")
        print("Checking Data directory...")
        
        # Try to find in Data directory
        if os.path.exists("Data/dsg.json"):
            config.dsg_path = "Data/dsg.json"
            print(f"✅ Found DSG file at: {config.dsg_path}")
        elif os.path.exists("Data/dsg_with_mesh.json"):
            config.dsg_path = "Data/dsg_with_mesh.json"
            print(f"✅ Found DSG file at: {config.dsg_path}")
    
    if not os.path.exists(config.mesh_path):
        print(f"❌ Mesh file not found at: {config.mesh_path}")
        print("Checking Data directory...")
        
        # Try to find in Data directory
        if os.path.exists("Data/mesh.ply"):
            config.mesh_path = "Data/mesh.ply"
            print(f"✅ Found mesh file at: {config.mesh_path}")
        elif os.path.exists("Data/cloud_aligned.ply"):
            config.mesh_path = "Data/cloud_aligned.ply"
            print(f"✅ Found point cloud file at: {config.mesh_path}")
    
    # Initialize object locator
    locator = ObjectLocator(config)
    
    # Get query from command line or prompt
    query = args.query
    if not query:
        query = input("\n🔍 Enter your search query: ")
    
    # Get match index
    match_index = args.match_index
    
    # Run the locator
    locator.run(query, match_index)

if __name__ == "__main__":
    main()