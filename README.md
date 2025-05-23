# Point Cloud Processing Tools

This repository contains a set of tools for working with 3D point clouds and meshes. The tools have been optimized for better performance, code organization, and user experience.

## Tools Overview

### 1. Point-to-Plane ICP (`plane_to_point_efficient.py`)

Aligns one point cloud to another using point-to-plane Iterative Closest Point (ICP) algorithm. This tool is useful for registering and aligning 3D scans.

**Improvements:**
- Multi-stage ICP for better and faster convergence
- Adaptive normal estimation based on point cloud density
- Proper error handling and input validation
- Performance optimizations (downsampling for large point clouds)
- Better visualization options and controls
- Command-line parameter support
- Progress reporting and timing information

### 2. Point Cloud Comparison (`compare_mesh_efficient.py`)

Compares two point clouds or meshes to analyze differences. This tool can also perform alignment before comparison.

**Improvements:**
- More comprehensive distance metrics and statistics
- Better visualization of distance distributions with color mapping
- Support for both same-sized and differently sized point clouds
- Enhanced visualization with better controls and settings
- Command-line parameter support
- Performance optimizations for large point clouds
- Detailed statistics and histogram generation

### 3. Object Localization (`give_the_task_efficient.py`)

Locates objects in a 3D scene based on natural language queries using semantic features.

**Improvements:**
- Object-oriented design for better code organization
- Enhanced visualizations with multiple display modes
- Better handling of positions and offsets
- Performance optimizations for large scenes
- Improved error handling and file validation
- Command-line parameter support
- Progress reporting and timing information
- Support for custom visualization settings

## Usage

### Shell Script

The easiest way to use these tools is through the provided shell script:

```bash
./run_tools.sh
```

This interactive script will guide you through using each tool with default options, or you can customize parameters as needed.

### Command Line Usage

Alternatively, you can run each tool directly from the command line:

**1. Point-to-Plane ICP:**
```bash
./plane_to_point_efficient.py source.ply target.ply output.ply --threshold 0.05
```

**2. Point Cloud Comparison:**
```bash
./compare_mesh_efficient.py mesh.ply cloud.ply --threshold 0.001 --align
```

**3. Object Localization:**
```bash
./give_the_task_efficient.py --query "trash bin" --dsg Data/dsg_with_mesh.json --mesh Data/cloud_aligned.ply
```

## Requirements

- Python 3.6+
- Open3D
- NumPy
- Open-CLIP (for object localization)
- PyTorch (for object localization)

## Performance Considerations

- For large point clouds (>1 million points), downsampling is automatically applied to improve performance
- Multi-stage ICP provides better alignment with fewer iterations
- Visualization is optimized for better performance and user experience

## License

This software is provided as-is under the MIT License.