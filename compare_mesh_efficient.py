"""
Efficient point cloud comparison and visualization tool
"""
import open3d as o3d
import numpy as np
import os
import sys
import argparse
import time
from typing import Tuple, List, Optional, Union, Dict, Any

def load_point_clouds(file1: str, file2: str, verbose: bool = True) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Load two point cloud files and return them
    
    Args:
        file1: Path to first point cloud file
        file2: Path to second point cloud file
        verbose: Whether to print information
        
    Returns:
        Tuple of (point_cloud1, point_cloud2)
    """
    try:
        if verbose:
            print(f"Loading first point cloud from: {file1}")
        pcd1 = o3d.io.read_point_cloud(file1)
        if not pcd1.has_points():
            raise ValueError(f"No points found in {file1}")
            
        if verbose:
            print(f"Loading second point cloud from: {file2}")
        pcd2 = o3d.io.read_point_cloud(file2)
        if not pcd2.has_points():
            raise ValueError(f"No points found in {file2}")
            
        if verbose:
            print(f"Point cloud 1: {len(pcd1.points)} points, has colors: {pcd1.has_colors()}")
            print(f"Point cloud 2: {len(pcd2.points)} points, has colors: {pcd2.has_colors()}")
        
        return pcd1, pcd2
    except Exception as e:
        print(f"Error loading point clouds: {e}")
        sys.exit(1)

def compare_point_clouds(
    pcd1: o3d.geometry.PointCloud, 
    pcd2: o3d.geometry.PointCloud,
    distance_threshold: float = 0.001,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two point clouds and return detailed statistics
    
    Args:
        pcd1: First point cloud
        pcd2: Second point cloud
        distance_threshold: Threshold for counting points (meters)
        verbose: Whether to print results
        
    Returns:
        Dictionary with comparison statistics
    """
    start_time = time.time()
    
    # Get point counts
    num_points1 = len(pcd1.points)
    num_points2 = len(pcd2.points)
    
    if verbose:
        print(f"\nComparing point clouds:")
        print(f"Point cloud 1 has {num_points1:,} points")
        print(f"Point cloud 2 has {num_points2:,} points")
    
    # Check if clouds are the same size for direct comparison
    same_size = num_points1 == num_points2
    
    # Calculate distances based on cloud sizes
    if same_size:
        # Direct point-to-point comparison for same-sized clouds
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        # Compute distances efficiently
        distances = np.linalg.norm(points1 - points2, axis=1)
        distance_method = "point-to-point"
    else:
        # Use nearest neighbor for differently sized clouds
        distances = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
        distance_method = "nearest-neighbor"
    
    # Calculate statistics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    
    # Count points exceeding threshold
    diff_points = np.sum(distances > distance_threshold)
    diff_percentage = (diff_points / len(distances)) * 100
    
    # Prepare histograms for distance distribution
    hist, bin_edges = np.histogram(distances, bins=20)
    
    # Create result dictionary
    results = {
        "distance_method": distance_method,
        "points1": num_points1,
        "points2": num_points2,
        "min_distance": min_dist,
        "max_distance": max_dist,
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "std_distance": std_dist,
        "points_above_threshold": diff_points,
        "percentage_above_threshold": diff_percentage,
        "threshold": distance_threshold,
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        },
        "distances": distances
    }
    
    # Print results if verbose
    if verbose:
        print(f"\nDistance calculation method: {distance_method}")
        print(f"Min distance: {min_dist:.6f} m")
        print(f"Max distance: {max_dist:.6f} m")
        print(f"Mean distance: {mean_dist:.6f} m")
        print(f"Median distance: {median_dist:.6f} m")
        print(f"Standard deviation: {std_dist:.6f} m")
        print(f"Points with distance > {distance_threshold}m: {diff_points:,} ({diff_percentage:.2f}%)")
        print(f"Comparison completed in {time.time() - start_time:.2f} seconds")
    
    return results

def visualize_point_clouds(
    pcd1: o3d.geometry.PointCloud, 
    pcd2: o3d.geometry.PointCloud,
    name1: str = "Point Cloud 1",
    name2: str = "Point Cloud 2",
    window_name: str = "Point Cloud Comparison"
) -> None:
    """
    Visualize two point clouds with different colors
    
    Args:
        pcd1: First point cloud
        pcd2: Second point cloud
        name1: Name of first point cloud
        name2: Name of second point cloud
        window_name: Visualization window name
    """
    # Create copies to preserve original point clouds
    pcd1_vis = o3d.geometry.PointCloud(pcd1)
    pcd2_vis = o3d.geometry.PointCloud(pcd2)
    
    # Set different colors for the two point clouds
    pcd1_vis.paint_uniform_color([1, 0, 0])  # Red for first point cloud
    pcd2_vis.paint_uniform_color([0, 0, 1])  # Blue for second point cloud
    
    # Create a visualization window
    print(f"\nVisualizing point clouds:")
    print(f"Red: {name1}")
    print(f"Blue: {name2}")
    print("Close the window to continue...")
    
    # Create custom visualizer for better experience
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=960)
    
    # Add point clouds
    vis.add_geometry(pcd1_vis)
    vis.add_geometry(pcd2_vis)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Improve visualization settings
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.light_on = True
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_color_mapped_distances(
    pcd: o3d.geometry.PointCloud, 
    distances: np.ndarray,
    min_dist: Optional[float] = None,
    max_dist: Optional[float] = None,
    window_name: str = "Distance Visualization"
) -> None:
    """
    Visualize point cloud with color mapped to distances
    
    Args:
        pcd: Point cloud to visualize
        distances: Array of distances for each point
        min_dist: Minimum distance for color mapping (auto if None)
        max_dist: Maximum distance for color mapping (auto if None)
        window_name: Visualization window name
    """
    # Use provided min/max or calculate from data
    if min_dist is None:
        min_dist = np.min(distances)
    if max_dist is None:
        max_dist = np.max(distances)
    
    # Create a copy for visualization
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = pcd.points
    
    # Normalize distances for color mapping
    normalized_distances = np.clip((distances - min_dist) / (max_dist - min_dist + 1e-10), 0, 1)
    
    # Apply a colormap (blue->green->red)
    colors = np.zeros((len(normalized_distances), 3))
    for i, d in enumerate(normalized_distances):
        if d < 0.5:  # blue to green
            colors[i] = [0, d*2, 1-d*2]
        else:        # green to red
            colors[i] = [(d-0.5)*2, 1-(d-0.5)*2, 0]
    
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    
    # Create custom visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=960)
    
    # Add point cloud
    vis.add_geometry(pcd_colored)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Improve visualization settings
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.light_on = True
    
    # Create color scale visualization in the console
    print("\nDistance color scale:")
    print(f"Blue (min): {min_dist:.6f} m")
    print(f"Green (mid): {(min_dist + max_dist) / 2:.6f} m")
    print(f"Red (max): {max_dist:.6f} m")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def align_point_clouds(
    source: o3d.geometry.PointCloud, 
    target: o3d.geometry.PointCloud,
    verbose: bool = True
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Align source point cloud to target point cloud using multi-stage ICP
    
    Args:
        source: Source point cloud to be transformed
        target: Target point cloud (reference)
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (transformed_source, transformation_matrix)
    """
    # Make a copy of the source to transform
    source_copy = o3d.geometry.PointCloud(source)
    
    if verbose:
        print("\nAligning point clouds using multi-stage ICP...")
    
    # Check if clouds have normals and estimate if needed
    if not source_copy.has_normals():
        if verbose:
            print("Estimating normals for source point cloud...")
        source_copy.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    if not target.has_normals():
        if verbose:
            print("Estimating normals for target point cloud...")
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Stage 1: Coarse alignment with larger threshold (point-to-point)
    threshold1 = 0.05  # 5cm for initial alignment
    if verbose:
        print(f"Stage 1: Coarse alignment (threshold: {threshold1}m)...")
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_copy, target, threshold1, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    
    if verbose:
        print(f"  Stage 1 fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.6f}")
    
    # Apply initial transformation
    transformation = reg_p2p.transformation
    source_copy.transform(transformation)
    
    # Stage 2: Fine alignment with tighter threshold (point-to-plane)
    threshold2 = 0.02  # 2cm for fine alignment
    if verbose:
        print(f"Stage 2: Fine alignment (threshold: {threshold2}m)...")
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_copy, target, threshold2, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=100)
    )
    
    # Apply final transformation
    source_copy.transform(reg_p2l.transformation)
    
    # Calculate combined transformation
    final_transformation = np.matmul(reg_p2l.transformation, transformation)
    
    if verbose:
        print(f"Alignment complete")
        print(f"Final fitness score: {reg_p2l.fitness:.4f}")
        print(f"Final inlier RMSE: {reg_p2l.inlier_rmse:.6f}")
        print("Transformation matrix:")
        print(np.array2string(final_transformation, precision=6, suppress_small=True))
    
    # Create a fresh transformed copy of the original source
    source_transformed = o3d.geometry.PointCloud(source)
    source_transformed.transform(final_transformation)
    
    return source_transformed, final_transformation

def save_transformed_point_cloud(
    point_cloud: o3d.geometry.PointCloud, 
    filepath: str, 
    verbose: bool = True
) -> bool:
    """
    Save a point cloud to a file
    
    Args:
        point_cloud: Point cloud to save
        filepath: Path to save to
        verbose: Whether to print information
        
    Returns:
        True if successful, False otherwise
    """
    try:
        o3d.io.write_point_cloud(filepath, point_cloud)
        if verbose:
            print(f"Transformed point cloud saved to: {filepath}")
        return True
    except Exception as e:
        if verbose:
            print(f"Error saving point cloud: {e}")
        return False

def visualize_transformation(
    original: o3d.geometry.PointCloud, 
    transformed: o3d.geometry.PointCloud, 
    reference: o3d.geometry.PointCloud
) -> None:
    """
    Visualize original vs transformed vs reference point clouds
    
    Args:
        original: Original source point cloud
        transformed: Transformed source point cloud
        reference: Reference (target) point cloud
    """
    # First visualization: original vs reference
    original_vis = o3d.geometry.PointCloud(original)
    reference_vis = o3d.geometry.PointCloud(reference)
    original_vis.paint_uniform_color([1, 0, 0])     # Red
    reference_vis.paint_uniform_color([0, 0, 1])    # Blue
    
    # Create custom visualizer for first view
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="Before Alignment", width=1280, height=960)
    
    # Add point clouds
    vis1.add_geometry(original_vis)
    vis1.add_geometry(reference_vis)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis1.add_geometry(coordinate_frame)
    
    # Improve visualization settings
    opt = vis1.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.light_on = True
    
    print("\nVisualizing BEFORE alignment")
    print("Red: Original source point cloud")
    print("Blue: Reference point cloud")
    print("Close the window to continue...")
    
    # Run first visualizer
    vis1.run()
    vis1.destroy_window()
    
    # Second visualization: transformed vs reference
    transformed_vis = o3d.geometry.PointCloud(transformed)
    transformed_vis.paint_uniform_color([0, 1, 0])  # Green
    
    # Create custom visualizer for second view
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="After Alignment", width=1280, height=960)
    
    # Add point clouds
    vis2.add_geometry(transformed_vis)
    vis2.add_geometry(reference_vis)
    
    # Add coordinate frame
    vis2.add_geometry(coordinate_frame)
    
    # Improve visualization settings
    opt = vis2.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.light_on = True
    
    print("\nVisualizing AFTER alignment")
    print("Green: Transformed source point cloud")
    print("Blue: Reference point cloud")
    print("Close the window to continue...")
    
    # Run second visualizer
    vis2.run()
    vis2.destroy_window()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Compare and align point clouds")
    parser.add_argument("file1", help="Path to first point cloud file (reference/target)")
    parser.add_argument("file2", help="Path to second point cloud file (to be aligned)")
    parser.add_argument("--threshold", type=float, default=0.001, 
                       help="Distance threshold for comparison (meters, default: 0.001)")
    parser.add_argument("--align", action="store_true", 
                       help="Perform alignment of the second point cloud to the first")
    parser.add_argument("--output", type=str, default="aligned_output.ply",
                       help="Output path for aligned point cloud (default: aligned_output.ply)")
    parser.add_argument("--no-visualization", action="store_true", 
                       help="Skip visualization")
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.file1, args.file2]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Print header
    print(f"=== Point Cloud Comparison Tool ===")
    print(f"Reference: {args.file1}")
    print(f"Source: {args.file2}")
    print(f"Threshold: {args.threshold} meters")
    
    # Load point clouds
    pcd1, pcd2 = load_point_clouds(args.file1, args.file2)
    
    # Compare point clouds
    comparison_results = compare_point_clouds(pcd1, pcd2, args.threshold)
    
    # Visualize the two point clouds together
    if not args.no_visualization:
        visualize_point_clouds(pcd1, pcd2, 
                             name1=os.path.basename(args.file1), 
                             name2=os.path.basename(args.file2))
    
    # Visualize with color map representing distances
    if not args.no_visualization:
        visualize_color_mapped_distances(pcd1, comparison_results["distances"])
    
    # Perform alignment if requested
    if args.align:
        transformed_cloud, transformation = align_point_clouds(pcd2, pcd1)
        
        # Save the transformed point cloud
        save_transformed_point_cloud(transformed_cloud, args.output)
        
        # Visualize before and after transformation
        if not args.no_visualization:
            visualize_transformation(pcd2, transformed_cloud, pcd1)
        
        # Compare the aligned point cloud with the reference
        print("\nComparing ALIGNED point clouds:")
        aligned_comparison = compare_point_clouds(pcd1, transformed_cloud, args.threshold)
        
        # Visualize with color map representing distances after alignment
        if not args.no_visualization:
            visualize_color_mapped_distances(pcd1, aligned_comparison["distances"])
        
        # Print improvement summary
        original_mean = comparison_results["mean_distance"]
        aligned_mean = aligned_comparison["mean_distance"]
        improvement = (original_mean - aligned_mean) / original_mean * 100
        
        print("\nAlignment Improvement Summary:")
        print(f"Original mean distance: {original_mean:.6f} meters")
        print(f"Aligned mean distance: {aligned_mean:.6f} meters")
        print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()