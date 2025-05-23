import json
import torch
import open3d as o3d
import numpy as np
import colorsys
from open_clip import create_model_and_transforms, tokenize
# Add these imports at the top
import math
from typing import List, Tuple, Optional

# Visualization settings
VISUALIZATION_MODE = "best"  # "multiple" or "best"
SIMILARITY_THRESHOLD = 0.15  # Show objects with at least 15% similarity
MAX_OBJECTS = 10  # Maximum number of objects to display
SHOW_CONNECTION_LINES = False  # Set to True to show lines to original positions

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms("ViT-L-14", pretrained="openai")
model = model.to(device).eval()

# Define natural language query
print("\n🔍 Enter your query: ", end='')
query = input()
# query = "go to the trash bin"
text_tokens = tokenize([query]).to(device)

with torch.no_grad():
    text_feature = model.encode_text(text_tokens)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)

# DSG JSON path
json_path = "backend/dsg_with_mesh.json"
# mesh_path = "backend/mesh.ply"  # Mesh file path
mesh_path = "cloud_aligned.ply"  # Original

# Sphere position offsets in meters
SPHERE_OFFSET_X = 0  # -5cm to the left 
SPHERE_OFFSET_Y = 0   # 2cm forward
SPHERE_OFFSET_Z = 0   # 3cm up

# Add this after your existing functions
def create_bounding_box(center, size=(0.3, 0.3, 0.3), color=(0, 1, 0)):
    """Create a bounding box centered at the given position with specified size."""
    box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    
    # Center the box at the given position
    box.translate([-size[0]/2, -size[1]/2, -size[2]/2])  # Center at origin
    box.translate(center)  # Move to desired position
    
    return box

def create_wireframe_box(center, size=(0.3, 0.3, 0.3), color=(0, 1, 0)):
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

def create_text_label(text, position, font_size=16, color=(1, 1, 1)):
    """Create a text label at the specified position."""
    # In reality, Open3D doesn't directly support 3D text rendering
    # So we'll use a small sphere as a marker with a unique color
    # In a real application, you would need to implement 3D text properly
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    marker.paint_uniform_color(color)
    marker.translate(position)
    return marker, text

# Load JSON
try:
    with open(json_path, 'r') as f:
        dsg_raw = json.load(f)
    print(f"\n📦 DEBUG: Successfully loaded DSG from {json_path}")
except Exception as e:
    print(f"\n❌ ERROR: Failed to load DSG JSON: {e}")
    print(f"Check if the file exists at: {json_path}")
    exit(1)

print("\n📦 DEBUG: json type =", type(dsg_raw))
print("\n🧪 dsg.keys():", list(dsg_raw.keys()))
print("🧪 dsg['nodes'] type:", type(dsg_raw.get("nodes")))

# Extract nodes from DSG (confirmed as list)
nodes_raw = dsg_raw.get("nodes", [])
print(f"\n📊 Total nodes in DSG: {len(nodes_raw)}")

# Advanced DSG debugging
node_types = {}
for node in nodes_raw:
    if isinstance(node, dict) and "type" in node:
        node_type = node["type"]
        node_types[node_type] = node_types.get(node_type, 0) + 1

print("\n📊 Node type distribution:")
for node_type, count in node_types.items():
    print(f" - {node_type}: {count} nodes")

# Extract potential objects
objects = {}
feature_counts = 0
position_counts = 0

for i, node in enumerate(nodes_raw):
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
    
    # Only use nodes with both features
    if has_semantic and has_position:
        node_id = str(node.get("id", f"node_{i}"))
        objects[node_id] = {
            "position": attributes["position"],
            "semantic_feature": attributes["semantic_feature"]["data"],
            "type": node.get("type", "unknown"),
        }

# Debug output
print(f"\n📦 Valid objects found: {len(objects)}")
print(f"📊 Nodes with semantic features: {feature_counts}")
print(f"📊 Nodes with position data: {position_counts}")

for i, (node_id, node) in enumerate(objects.items()):
    print(f"\n - node_id: {node_id}")
    print(f" - type: {node['type']}")
    print(f" - position: {node['position']}")
    print(f" - feature length: {len(node['semantic_feature'])}")
    print("------")
    if i >= 4:
        print("(showing only first 5 objects)")
        break

# Calculate similarity and store all objects
all_matches = []

for node_id, node in objects.items():
    feat = node["semantic_feature"]
    node_feat = torch.tensor(feat, dtype=torch.float32).to(device).view(1, -1)
    node_feat /= node_feat.norm(dim=-1, keepdim=True)

    similarity = torch.nn.functional.cosine_similarity(text_feature, node_feat).item()
    
    # Normalize position data
    position_array = None
    if isinstance(node["position"], list) and len(node["position"]) == 3:
        position_array = node["position"]
    elif isinstance(node["position"], dict) and all(k in node["position"] for k in ["x", "y", "z"]):
        position_array = [node["position"]["x"], node["position"]["y"], node["position"]["z"]]
    
    if position_array:
        all_matches.append({
            "node_id": node_id,
            "position": position_array,
            "type": node.get("type", "unknown"),
            "similarity": similarity
        })

# Sort by similarity
all_matches.sort(key=lambda x: x["similarity"], reverse=True)

# Load mesh function
def load_mesh(path):
    try:
        print(f"\n📂 Attempting to load mesh from: {path}")
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_triangles():
            print(f"⚠️ Warning: Mesh has no triangles. Loading as point cloud.")
            pcd = o3d.io.read_point_cloud(path)
            if not pcd.has_points():
                raise ValueError("Point cloud has no points.")
            geometry = pcd
            geometry.paint_uniform_color([0.8, 0.8, 0.8])  # Gray color
        else:
            print(f"✅ Successfully loaded mesh with {len(mesh.triangles)} triangles")
            mesh.compute_vertex_normals()
            geometry = mesh
        return geometry
    except Exception as e:
        print(f"❌ Mesh loading failed: {e}")
        print(f"👉 Check the file path: {path}")
        exit(1)

def visualize_objects_on_mesh(
    mesh_path: str,
    matches: List[dict],
    query: str,
    create_wireframe: bool = True,
    show_labels: bool = True,
    mesh_color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
    custom_camera_params: Optional[dict] = None
):
    """
    Visualize objects on a mesh with enhanced visual features.
    
    Args:
        mesh_path: Path to the mesh file
        matches: List of object matches with positions and similarities
        query: The text query that was used
        create_wireframe: Whether to use wireframe boxes instead of solid boxes
        show_labels: Whether to show text labels
        mesh_color: Color for the mesh
        custom_camera_params: Optional custom camera parameters
    """
    # Load the mesh
    print("\n🔍 Loading mesh and preparing visualization...")
    geometry = load_mesh(mesh_path)
    
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
            box = create_wireframe_box(position, box_size, color)
        else:
            box = create_bounding_box(position, box_size, color)
        
        geometries.append(box)
        
        # Add a label if requested
        if show_labels:
            # Position the label above the object
            label_pos = [position[0], position[1], position[2] + box_size[2]/2 + 0.05]
            label_text = f"{match['type']} ({similarity:.2f})"
            marker, text = create_text_label(label_text, label_pos, color=color)
            geometries.append(marker)
            labels_info.append((text, label_pos))
    
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Query: {query}", width=1280, height=960)
    
    # Add all geometries to the scene
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Configure the renderer
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 3.0
    opt.line_width = 2.0  # For wireframe rendering
    opt.light_on = True
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.FLAT
    
    # Configure the camera view
    ctr = vis.get_view_control()
    if custom_camera_params:
        # Use custom camera parameters if provided
        ctr.set_zoom(custom_camera_params.get('zoom', 0.8))
        ctr.set_front(custom_camera_params.get('front', [0, 0, -1]))
        ctr.set_lookat(custom_camera_params.get('lookat', [0, 0, 0]))
        ctr.set_up(custom_camera_params.get('up', [0, -1, 0]))
    else:
        # Default camera position - adjust as needed
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
    
    # Print help message
    print("\n🌟 Starting enhanced 3D visualization...")
    print("💡 TIP: Use mouse to rotate, Ctrl+mouse to pan, mouse wheel to zoom")
    print("💡 Press 'H' to see keyboard shortcuts")
    
    # If we have labels, print them (since we can't render text directly in 3D)
    if show_labels:
        print("\n📝 Object labels (shown as colored markers in visualization):")
        for i, (text, pos) in enumerate(labels_info):
            print(f" - Marker {i+1}: {text} at {pos}")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Visualize best match function
def visualize_best_match():
    if not all_matches:
        print("❌ No similar objects found.")
        return
        
    best_match = all_matches[0]
    print(f"\n✅ Highest similarity object:")
    print(f" - Query: '{query}'")
    print(f" - Node ID: {best_match['node_id']}")
    print(f" - Type: {best_match['type']}")
    print(f" - 3D Position: {best_match['position']}")
    print(f" - Similarity: {best_match['similarity']:.4f}")
    
    # Load mesh
    print("\n🔍 Preparing 3D visualization...")
    geometry = load_mesh(mesh_path)
    
    # Get position with offsets
    pos = best_match['position']
    adjusted_pos = [
        pos[0] + SPHERE_OFFSET_X,
        pos[1] + SPHERE_OFFSET_Y,
        pos[2] + SPHERE_OFFSET_Z
    ]
    
    # Highlight best match object with offset position
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color([1, 0, 0])  # Red
    sphere.compute_vertex_normals()
    sphere.translate(adjusted_pos)
    
    # List of geometries to visualize
    geometries = [geometry, sphere]
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coordinate_frame)
    
    # Show a thin line connecting the actual position to the sphere (if enabled)
    if SHOW_CONNECTION_LINES:
        line_points = [pos, adjusted_pos]
        line_indices = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red line
        geometries.append(line_set)
    
    # Visualization
    print("\n🌟 Starting 3D visualization (best match object)...")
    print(f"📌 Original position: {pos}")
    # print(f"📌 Adjusted position: {adjusted_pos}")
    o3d.visualization.draw_geometries(geometries, 
                                   window_name=f"Best Match: {query}", 
                                   width=1024, height=768)

# Visualize multiple matches function
def visualize_multiple_matches():
    # Only show objects above threshold
    filtered_matches = [match for match in all_matches if match["similarity"] >= SIMILARITY_THRESHOLD]
    filtered_matches = filtered_matches[:MAX_OBJECTS]
    
    if not filtered_matches:
        print("❌ No objects found above the similarity threshold.")
        return
        
    print(f"\n✅ Similar objects ({len(filtered_matches)}):")
    print(f" - Query: '{query}'")
    
    for i, match in enumerate(filtered_matches):
        print(f"\n👉 Match #{i+1}:")
        print(f" - Node ID: {match['node_id']}")
        print(f" - Type: {match['type']}")
        print(f" - 3D Position: {match['position']}")
        print(f" - Similarity: {match['similarity']:.4f}")
    
    # Load mesh
    print("\n🔍 Preparing 3D visualization...")
    geometry = load_mesh(mesh_path)
    
    # Create spheres for objects
    geometries = [geometry]
    
    # Create spheres with different colors based on similarity
    for i, match in enumerate(filtered_matches):
        pos = match["position"]
        adjusted_pos = [
            pos[0] + SPHERE_OFFSET_X,
            pos[1] + SPHERE_OFFSET_Y,
            pos[2] + SPHERE_OFFSET_Z
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
        if SHOW_CONNECTION_LINES:
            line_points = [pos, adjusted_pos]
            line_indices = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.colors = o3d.utility.Vector3dVector([color])
            geometries.append(line_set)
        
        print(f"📍 Object #{i+1}: original={pos}, adjusted={adjusted_pos}")
    
    # Create coordinate frame (for orientation reference)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coordinate_frame)
    
    # Visualization
    print("\n🌟 Starting 3D visualization...")
    o3d.visualization.draw_geometries(geometries, 
                                   window_name=f"Multiple Objects: {query}", 
                                   width=1024, height=768)

# Run visualization based on selected mode
if VISUALIZATION_MODE == "multiple":
    visualize_multiple_matches()
elif VISUALIZATION_MODE == "best":
    visualize_best_match()
elif VISUALIZATION_MODE == "both":
    visualize_best_match()
    visualize_multiple_matches()
else:
    # Interactive selection
    print("\n🔍 Select visualization mode:")
    print(" 1. View multiple objects (above similarity threshold)")
    print(" 2. View only the best match")
    choice = input("Choice (1 or 2): ")
    
    if choice == "1":
        visualize_multiple_matches()
    elif choice == "2":
        visualize_best_match()
    elif choice == "3":
        visualize_best_match()
        visualize_multiple_matches()
    else:
        print("❌ Invalid choice. Defaulting to multiple objects.")
        visualize_multiple_matches()