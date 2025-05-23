#!/usr/bin/env python3
# filepath: /home/nara/PointCloud/Folder Efficieny/plane_to_point_efficient.py
import open3d as o3d
import numpy as np
import time
import argparse
from typing import Tuple, Optional

def get_global_registration(source, target, voxel_size=0.7):
    """
    Perform global registration using FPFH features and RANSAC
    with constraints to prevent axis flipping/mirroring
    
    Args:
        source: Source point cloud
        target: Target point cloud
        voxel_size: Voxel size for downsampling
        
    Returns:
        4x4 transformation matrix
    """
    start_time = time.time()
    print("Starting global registration...")
    
    # Downsample point clouds
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Estimate normals
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    # Compute FPFH features
    print("Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    # Apply RANSAC for global registration
    print("RANSAC registration...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, 
        voxel_size*1.5,  # slightly larger than voxel_size
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # 3 points to determine a rigid transformation
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*1.5)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    # Get the transformation matrix
    transformation = result.transformation
    
    # Now check if the transformation flips/mirrors any axes
    # We'll check if the determinant of the rotation part is negative
    rotation_matrix = transformation[:3, :3]
    det = np.linalg.det(rotation_matrix)
    
    print(f"Determinant of rotation matrix: {det}")
    
    # If determinant is negative, we have a reflection (improper rotation)
    if det < 0:
        print("WARNING: Global registration produced a mirroring transformation!")
        print("Original transformation:")
        print(transformation)
        
        # Fix by negating the column corresponding to the axis most affected by reflection
        # This is a heuristic approach - find which axis has changed direction the most
        original_axes = np.eye(3)
        transformed_axes = rotation_matrix @ original_axes
        
        # Calculate how much each axis has been flipped (dot product with original)
        dot_products = [np.dot(original_axes[:, i], transformed_axes[:, i]) for i in range(3)]
        most_flipped_axis = np.argmin(dot_products)
        
        print(f"Most flipped axis is {['X', 'Y', 'Z'][most_flipped_axis]} with dot product {dot_products[most_flipped_axis]}")
        
        # Create corrected transformation matrix (flip the most affected axis)
        corrected_transform = transformation.copy()
        corrected_transform[:3, most_flipped_axis] *= -1
        
        # Also adjust translation to compensate for the flip
        # This is a simplified approach and might need refinement for your specific case
        centroid = np.mean(np.asarray(source_down.points), axis=0)
        translation_correction = 2 * centroid[most_flipped_axis] * corrected_transform[:3, most_flipped_axis]
        corrected_transform[:3, 3] -= translation_correction
        
        print("Corrected transformation:")
        print(corrected_transform)
        
        # Verify the determinant is now positive
        new_det = np.linalg.det(corrected_transform[:3, :3])
        print(f"Corrected determinant: {new_det}")
        
        transformation = corrected_transform
    
    print(f"Global registration completed in {time.time() - start_time:.2f} seconds")
    return transformation

def align_point_clouds(
    source_path: str,
    target_path: str,
    output_path: str,
    voxel_size: float = 0.7,
    threshold: float = 0.7,
    max_iterations: int = 200,
    visualize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    Aligns a source point cloud to a target point cloud using Point-to-Plane ICP
    and saves the aligned source cloud.
    
    Args:
        source_path: Path to the source point cloud file
        target_path: Path to the target point cloud file
        output_path: Path to save the aligned source point cloud
        voxel_size: Voxel size for global registration (default: 0.7)
        threshold: ICP correspondence distance threshold (default: 0.7)
        max_iterations: Maximum number of ICP iterations
        visualize: Whether to visualize the result
        verbose: Whether to print detailed information
        
    Returns:
        Tuple containing:
        - transformation matrix (4x4 numpy array)
        - fitness score (higher is better)
        - inlier RMSE (lower is better)
    """
    # Start timing
    start_time = time.time()
    
    # Load point clouds
    if verbose:
        print(f"Loading source point cloud: {source_path}")
    source_cloud = o3d.io.read_point_cloud(source_path)
    if not source_cloud.has_points():
        raise ValueError(f"Could not read source point cloud from {source_path}")

    if verbose:
        print(f"Loading target point cloud: {target_path}")
    target_cloud = o3d.io.read_point_cloud(target_path)
    if not target_cloud.has_points():
        raise ValueError(f"Could not read target point cloud from {target_path}")

    # Set source cloud color to blue if it has no colors (for visualization)
    if not source_cloud.has_colors():
        if verbose:
            print("Source cloud has no colors. Setting blue color for visualization.")
        source_cloud.paint_uniform_color([0, 0.651, 0.929])  # Blue
    
    # Preprocessing: Ensure clouds have normals (required for Point-to-Plane ICP)
    prepare_point_cloud_for_icp(source_cloud, "source", verbose)
    prepare_point_cloud_for_icp(target_cloud, "target", verbose)
    
    # First perform global registration for initial alignment
    if verbose:
        print("\nPerforming global registration...")
    init_trans = get_global_registration(source_cloud, target_cloud, voxel_size)
    
    # Apply the initial transformation from global registration
    source_cloud.transform(init_trans)
    
    if verbose:
        print("Global registration matrix:")
        print(np.array2string(init_trans, precision=6, suppress_small=True))
    
    # Multi-stage ICP for refinement
    if verbose:
        print("\nRefining alignment with multi-stage ICP...")
    transformation = run_multistage_icp(source_cloud, target_cloud, threshold, max_iterations, verbose)
    
    # Apply the refinement transformation to source cloud
    source_cloud.transform(transformation)
    
    # Final transformation is the combination of global registration and ICP refinement
    final_transformation = np.matmul(transformation, init_trans)
    
    # Process mesh if available or create one
    has_output_mesh = process_mesh(source_path, source_cloud, output_path, final_transformation, verbose)
    
    # If no mesh was created or found, save the aligned point cloud
    if not has_output_mesh:
        if verbose:
            print(f"Saving aligned source cloud to: {output_path}")
        o3d.io.write_point_cloud(output_path, source_cloud, write_ascii=False)
    
    # Get metadata about the alignment
    fitness, inlier_rmse = calculate_alignment_metrics(source_cloud, target_cloud, threshold)
    
    # Report total time
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\nOperation completed in {elapsed_time:.2f} seconds")
        print(f"Final alignment: Fitness={fitness:.4f}, RMSE={inlier_rmse:.6f}")
    
    # Visualize if requested
    if visualize:
        if has_output_mesh:
            # Try to load the mesh for visualization
            try:
                output_mesh = o3d.io.read_triangle_mesh(output_path)
                if len(output_mesh.triangles) > 0:
                    visualize_alignment(source_cloud, target_cloud, final_transformation, mesh=output_mesh)
                    return final_transformation, fitness, inlier_rmse
            except Exception as e:
                if verbose:
                    print(f"Could not load mesh for visualization: {e}")
        
        # Fall back to point cloud visualization if mesh isn't available
        visualize_alignment(source_cloud, target_cloud, final_transformation)
    
    return final_transformation, fitness, inlier_rmse

def prepare_point_cloud_for_icp(cloud: o3d.geometry.PointCloud, name: str, verbose: bool = True) -> None:
    """
    Prepares a point cloud for ICP by estimating normals if needed and optimizing data
    
    Args:
        cloud: The point cloud to prepare
        name: Name of the cloud (for logging)
        verbose: Whether to print status messages
    """
    # Downsample to improve performance if the cloud is very large (>1 million points)
    orig_points = len(cloud.points)
    if orig_points > 1000000:
        if verbose:
            print(f"Downsampling {name} cloud ({orig_points} points)...")
        cloud = cloud.voxel_down_sample(voxel_size=0.01)  # Adjust voxel size as needed
        if verbose:
            print(f"  Downsampled to {len(cloud.points)} points")
    
    # Estimate normals if needed
    if not cloud.has_normals():
        if verbose:
            print(f"Estimating normals for {name} cloud...")
        
        # Calculate appropriate radius parameter based on point cloud density
        points = np.asarray(cloud.points)
        if len(points) > 0:
            # Use KDTree to estimate point density
            pcd_tree = o3d.geometry.KDTreeFlann(cloud)
            # Sample 100 random points to estimate average distance to nearest neighbors
            num_samples = min(100, len(points))
            indices = np.random.choice(len(points), num_samples, replace=False)
            avg_distance = 0
            
            for idx in indices:
                [_, idx_neighbors, _] = pcd_tree.search_knn_vector_3d(cloud.points[idx], 30)
                if len(idx_neighbors) > 1:  # Ensure we have neighbors
                    # Calculate average distance to these neighbors
                    neighbors = np.asarray(cloud.points)[idx_neighbors[1:]]  # Exclude self
                    point = np.asarray(cloud.points)[idx]
                    distances = np.linalg.norm(neighbors - point, axis=1)
                    if len(distances) > 0:
                        avg_distance += np.mean(distances)
            
            if num_samples > 0:
                avg_distance /= num_samples
                # Use this to set a good radius for normal estimation
                radius = max(0.05, avg_distance * 5)  # At least 0.05 units
            else:
                radius = 0.1  # Default
        else:
            radius = 0.1  # Default
        
        # Estimate normals with adaptive radius
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        
        # Orient normals consistently
        cloud.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

def run_multistage_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float,
    max_iterations: int,
    verbose: bool = True
) -> np.ndarray:
    """
    Performs multi-stage ICP for better alignment
    
    Args:
        source: Source point cloud
        target: Target point cloud
        threshold: Final ICP threshold to use
        max_iterations: Maximum iterations for ICP
        verbose: Whether to print status information
        
    Returns:
        The final transformation matrix
    """
    # Create copies to avoid modifying original clouds
    source_copy = o3d.geometry.PointCloud(source)
    
    # Stage 1: Coarse alignment with higher threshold (point-to-point)
    if verbose:
        print("Stage 1: Coarse alignment...")
    stage1_threshold = threshold * 3
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_copy, target, stage1_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    transformation = reg_p2p.transformation
    source_copy.transform(transformation)
    
    if verbose:
        print(f"  Stage 1 fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.6f}")
    
    # Stage 2: Medium-precision alignment (point-to-plane)
    if verbose:
        print("Stage 2: Medium-precision alignment...")
    stage2_threshold = threshold * 1.5
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_copy, target, stage2_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60)
    )
    stage2_trans = reg_p2l.transformation
    transformation = np.matmul(stage2_trans, transformation)  # Combine transformations
    source_copy.transform(stage2_trans)
    
    if verbose:
        print(f"  Stage 2 fitness: {reg_p2l.fitness:.4f}, RMSE: {reg_p2l.inlier_rmse:.6f}")
    
    # Stage 3: Fine alignment with specified threshold (point-to-plane)
    if verbose:
        print("Stage 3: Fine alignment...")
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_copy, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=max_iterations)
    )
    stage3_trans = reg_p2l.transformation
    transformation = np.matmul(stage3_trans, transformation)  # Combine transformations
    
    if verbose:
        print(f"  Stage 3 fitness: {reg_p2l.fitness:.4f}, RMSE: {reg_p2l.inlier_rmse:.6f}")
        print("Transformation matrix:")
        print(np.array2string(transformation, precision=6, suppress_small=True))
    
    return transformation

def calculate_alignment_metrics(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float
) -> Tuple[float, float]:
    """
    Calculates fitness and RMSE metrics for the alignment
    
    Args:
        source: Source point cloud (already transformed)
        target: Target point cloud
        threshold: Distance threshold for correspondence pairs
        
    Returns:
        Tuple of (fitness, inlier_rmse)
    """
    # Compute distances between the two point clouds
    distances = np.asarray(source.compute_point_cloud_distance(target))
    
    # Compute fitness (ratio of inlier correspondences to total points)
    num_inliers = np.sum(distances < threshold)
    fitness = num_inliers / len(distances)
    
    # Compute inlier RMSE (root mean square of distances for inlier correspondences)
    inlier_distances = distances[distances < threshold]
    if len(inlier_distances) > 0:
        inlier_rmse = np.sqrt(np.mean(np.square(inlier_distances)))
    else:
        inlier_rmse = np.inf
    
    return fitness, inlier_rmse

def process_mesh(
    source_path: str,
    source_cloud: o3d.geometry.PointCloud,
    output_path: str,
    transformation: np.ndarray,
    verbose: bool = True
) -> bool:
    """
    Extract mesh from source file or create a new mesh from the point cloud and save it
    
    Args:
        source_path: Path to the source file that might contain a mesh
        source_cloud: Transformed source point cloud
        output_path: Path to save the mesh or point cloud
        transformation: Transformation matrix applied to the source
        verbose: Whether to print detailed information
        
    Returns:
        Boolean indicating whether a mesh was successfully processed
    """
    # Check if source file contains a mesh
    if verbose:
        print("\nChecking if source file contains a mesh...")
    try:
        # Try to read the source file as a triangle mesh
        source_mesh = o3d.io.read_triangle_mesh(source_path)
        has_source_mesh = len(source_mesh.triangles) > 0
    except Exception:
        has_source_mesh = False
    
    if has_source_mesh:
        if verbose:
            print(f"Source file has mesh with {len(source_mesh.triangles)} triangles, transforming it...")
        # Apply the same transformation to the mesh as was applied to the point cloud
        source_mesh.transform(transformation)
        
        # Ensure the mesh has colors, copy from the point cloud if needed
        if not source_mesh.has_vertex_colors() and source_cloud.has_colors():
            if verbose:
                print("Transferring colors from point cloud to mesh...")
            if len(source_mesh.vertices) == len(source_cloud.points):
                source_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(source_cloud.colors))
            else:
                if verbose:
                    print(f"Warning: Mesh vertices ({len(source_mesh.vertices)}) don't match point cloud points ({len(source_cloud.points)})")
                source_mesh.paint_uniform_color([0, 0.651, 0.929])  # Blue
        
        if verbose:
            print(f"Saving mesh to: {output_path}")
        # Save as PLY to preserve both mesh and point cloud data
        o3d.io.write_triangle_mesh(output_path, source_mesh, write_vertex_colors=True)
        return True
    else:
        if verbose:
            print("Source file doesn't contain mesh data. Creating mesh using Poisson reconstruction...")
        
        # Ensure we have normals for Poisson reconstruction
        if not source_cloud.has_normals():
            if verbose:
                print("Estimating normals for Poisson mesh...")
            source_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            source_cloud, depth=9, width=0, scale=1.1, linear_fit=False
        )
        
        if len(mesh.triangles) == 0:
            if verbose:
                print("Poisson reconstruction failed to create a mesh")
            return False
        
        if verbose:
            print(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Transfer color from point cloud to mesh if possible
        if source_cloud.has_colors():
            if len(mesh.vertices) == len(source_cloud.points):
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(source_cloud.colors))
                if verbose:
                    print("Transferred vertex colors from point cloud to mesh.")
            else:
                if verbose:
                    print("Mesh and point cloud vertex count mismatch; applying uniform color to mesh...")
                mesh.paint_uniform_color([0, 0.651, 0.929])  # Blue
        else:
            mesh.paint_uniform_color([0, 0.651, 0.929])  # Blue
        
        if verbose:
            print(f"Saving reconstructed mesh to: {output_path}")
        # Save the mesh with vertex colors
        o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
        return True

def visualize_alignment(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: Optional[np.ndarray] = None,
    mesh: Optional[o3d.geometry.TriangleMesh] = None
) -> None:
    """
    Visualizes alignment between source and target point clouds, and mesh if available
    
    Args:
        source: Source point cloud (already transformed)
        target: Target point cloud
        transformation: Transformation matrix (for display purposes)
        mesh: Optional triangle mesh to visualize
    """
    """
    Visualizes alignment between source and target point clouds
    
    Args:
        source: Source point cloud (already transformed)
        target: Target point cloud
        transformation: Transformation matrix (for display purposes)
    """
    # Create copies for visualization
    target_vis = o3d.geometry.PointCloud(target)
    
    # Set target color if needed
    if not target_vis.has_colors():
        target_vis.paint_uniform_color([1, 0.706, 0])  # Yellow
    
    print("\nVisualizing alignment result...")
    
    # Create custom visualizer for better experience
    vis = o3d.visualization.Visualizer()
    window_name = "Point Cloud Alignment with Mesh" if mesh is not None else "Point Cloud Alignment"
    vis.create_window(window_name=window_name, width=1280, height=960)
    
    # Add geometries based on what's available
    if mesh is not None and len(mesh.triangles) > 0:
        # If we have a mesh, use it instead of the source point cloud
        print("Visualizing with mesh...")
        vis.add_geometry(mesh)
        vis.add_geometry(target_vis)
    else:
        # Otherwise just use the point clouds
        source_vis = o3d.geometry.PointCloud(source)
        if not source_vis.has_colors():
            source_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue
        vis.add_geometry(source_vis)
        vis.add_geometry(target_vis)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Improve visualization settings
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.light_on = True
    
    # If we have a mesh, add mesh-specific settings
    if mesh is not None:
        opt.mesh_show_wireframe = True  # Show mesh wireframe
        opt.mesh_show_back_face = True  # Show back faces of mesh
        opt.line_width = 2.0  # Thicker lines for wireframe
    
    # Display transformation matrix in console
    if transformation is not None:
        print("\nTransformation matrix:")
        print(np.array2string(transformation, precision=6, suppress_small=True))
    
    print("\nVisualization controls:")
    print("- Left mouse: Rotate")
    print("- Ctrl + left mouse: Pan")
    print("- Mouse wheel: Zoom")
    print("- Press 'H' for more options")
    
    vis.run()
    vis.destroy_window()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Align point clouds using Point-to-Plane ICP")
    parser.add_argument("source", help="Path to source point cloud file")
    parser.add_argument("target", help="Path to target point cloud file")
    parser.add_argument("output", help="Path to save aligned point cloud")
    parser.add_argument("--voxel-size", type=float, default=0.7, help="Voxel size for global registration (default: 0.7)")
    parser.add_argument("--threshold", type=float, default=0.7, help="ICP threshold (default: 0.7)")
    parser.add_argument("--max-iterations", type=int, default=200, help="Max ICP iterations (default: 200)")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    align_point_clouds(
        args.source, 
        args.target, 
        args.output,
        voxel_size=args.voxel_size,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
        visualize=not args.no_visualize,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()