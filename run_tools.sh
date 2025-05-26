# Script to run the efficient versions of the point cloud tools

# Set text colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "\n${BLUE}===== Point Cloud Processing Tool Suite =====${NC}"
echo -e "This script lets you run the optimized versions of the point cloud tools"

# Function to check file existence
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}ERROR: File not found: $1${NC}"
        return 1
    fi
    return 0
}

# Make sure efficient scripts are executable
chmod +x plane_to_point_efficient.py
chmod +x compare_mesh_efficient.py
chmod +x give_the_task_efficient.py

# Define Python interpreter (using python3 to ensure we use the right version)
PYTHON="python3"

# Display menu
echo -e "\n${YELLOW}Choose an operation:${NC}"
echo -e "${GREEN}1${NC}. Align Point Clouds (Point-to-Plane ICP)"
echo -e "${GREEN}2${NC}. Compare Point Clouds/Meshes"
echo -e "${GREEN}3${NC}. Locate Objects in Scene"
echo -e "${GREEN}4${NC}. Run all tools in sequence"
echo -e "${GREEN}q${NC}. Quit"

# Get user choice
read -p "Enter your choice [1-4 or q]: " choice

case $choice in
    1)
        echo -e "\n${BLUE}==== Running Point-to-Plane ICP Alignment ====${NC}"
        
        # Ask for source and target files
        read -p "Enter source point cloud file path [default: realsense/nuri.ply]: " source_file
        source_file=${source_file:-"realsense/nuri.ply"}
        
        read -p "Enter target point cloud file path [default: data/mesh.ply]: " target_file
        target_file=${target_file:-"data/mesh.ply"}
        
        read -p "Enter output aligned file path [default: realsense/nuri_new.ply]: " output_file
        output_file=${output_file:-"realsense/nuri_new.ply"}
        
        # Check if files exist
        check_file "$source_file" && check_file "$target_file" || exit 1
        
        # Ask for threshold
        read -p "Enter alignment threshold [default: 0.7]: " threshold
        threshold=${threshold:-0.7}
        
        # Run the ICP alignment
        echo -e "\n${YELLOW}Running point-to-plane ICP alignment...${NC}"
        $PYTHON plane_to_point_efficient.py "$source_file" "$target_file" "$output_file" --threshold "$threshold"
        
        echo -e "\n${GREEN}Alignment complete!${NC}"
        echo -e "Original source: $source_file"
        echo -e "Target: $target_file"
        echo -e "Aligned output: $output_file"
        ;;
        
    2)
        echo -e "\n${BLUE}==== Running Point Cloud/Mesh Comparison ====${NC}"
        
        # Ask for file paths
        read -p "Enter first point cloud/mesh file path [default: data/mesh.ply]: " file1
        file1=${file1:-"data/mesh.ply"}
        
        read -p "Enter second point cloud/mesh file path [default: data/cloud_aligned.ply]: " file2
        file2=${file2:-"data/cloud_aligned.ply"}
        
        # Check if files exist
        check_file "$file1" && check_file "$file2" || exit 1
        
        # Ask for comparison threshold
        read -p "Enter comparison threshold in meters [default: 0.001]: " threshold
        threshold=${threshold:-0.001}
        
        # Ask if alignment should be performed
        read -p "Perform alignment before comparison? [y/N]: " do_align
        
        if [[ $do_align == "y" || $do_align == "Y" ]]; then
            read -p "Enter output path for aligned file [default: data/aligned_for_comparison.ply]: " aligned_output
            aligned_output=${aligned_output:-"data/aligned_for_comparison.ply"}
            
            # Run comparison with alignment
            echo -e "\n${YELLOW}Running comparison with alignment...${NC}"
            $PYTHON compare_mesh_efficient.py "$file1" "$file2" --threshold "$threshold" --align --output "$aligned_output"
        else
            # Run comparison without alignment
            echo -e "\n${YELLOW}Running comparison without alignment...${NC}"
            $PYTHON compare_mesh_efficient.py "$file1" "$file2" --threshold "$threshold"
        fi
        
        echo -e "\n${GREEN}Comparison complete!${NC}"
        ;;
        
    3)
        echo -e "\n${BLUE}==== Running Object Localization in Scene ====${NC}"
        
        # Ask for DSG and mesh paths
        read -p "Enter DSG JSON file path [default: data/dsg_with_mesh.json]: " dsg_path
        dsg_path=${dsg_path:-"data/dsg_with_mesh.json"}
        
        read -p "Enter mesh/point cloud file path [default: realsense/nuri_new.ply]: " mesh_path
        mesh_path=${mesh_path:-"realsense/nuri_new.ply"}
        
        # Check if files exist
        check_file "$dsg_path" && check_file "$mesh_path" || exit 1
        
        # Ask for query
        read -p "Enter search query (e.g., 'trash bin', 'chair'): " query
        
        if [ -z "$query" ]; then
            echo -e "${RED}Error: Search query cannot be empty${NC}"
            exit 1
        fi
        
        # Ask for visualization mode
        echo -e "\n${YELLOW}Visualization modes:${NC}"
        echo "1. Best match only"
        echo "2. Multiple matches"
        echo "3. Both best and multiple"
        echo "4. Advanced visualization"
        read -p "Choose visualization mode [1-4, default: 1]: " vis_mode
        
        case $vis_mode in
            1) 
                mode="best"
                read -p "Enter match index to visualize (0 for best match, default: 0): " match_index
                match_index=${match_index:-0}
                ;;
            2) mode="multiple" ;;
            3) mode="both" ;;
            4) mode="advanced" ;;
            *) 
                mode="best"
                match_index=0
                ;;
        esac
        
        # Ask for similarity threshold
        read -p "Enter similarity threshold (0.0-1.0) [default: 0.15]: " threshold
        threshold=${threshold:-0.15}
        
        # Run the object locator
        echo -e "\n${YELLOW}Searching for objects matching '$query'...${NC}"
        if [ "$mode" = "best" ]; then
            $PYTHON give_the_task_efficient.py --query "$query" --dsg "$dsg_path" --mesh "$mesh_path" --mode "$mode" --threshold "$threshold" --match-index "$match_index"
        else
            $PYTHON give_the_task_efficient.py --query "$query" --dsg "$dsg_path" --mesh "$mesh_path" --mode "$mode" --threshold "$threshold"
        fi
        
        echo -e "\n${GREEN}Object search complete!${NC}"
        ;;
        
    4)
        echo -e "\n${BLUE}==== Running Complete Workflow ====${NC}"
        echo -e "${YELLOW}This will run all three tools in sequence${NC}"
        
        # First run alignment
        echo -e "\n${BLUE}Step 1: Point-to-Plane ICP Alignment${NC}"
        source_file="data/cloud_aligned.ply"
        target_file="data/mesh.ply"
        aligned_output="data/cloud_aligned_workflow.ply"
        
        if check_file "$source_file" && check_file "$target_file"; then
            echo -e "${YELLOW}Running point-to-plane ICP alignment...${NC}"
            $PYTHON plane_to_point_efficient.py "$source_file" "$target_file" "$aligned_output" --threshold 0.05
            echo -e "${GREEN}Alignment complete!${NC}"
        else
            echo -e "${RED}Skipping alignment due to missing files${NC}"
        fi
        
        # Then run comparison
        echo -e "\n${BLUE}Step 2: Point Cloud Comparison${NC}"
        if [ -f "$aligned_output" ] && [ -f "$target_file" ]; then
            echo -e "${YELLOW}Running comparison between aligned cloud and target...${NC}"
            $PYTHON compare_mesh_efficient.py "$target_file" "$aligned_output" --threshold 0.001
            echo -e "${GREEN}Comparison complete!${NC}"
        else
            echo -e "${RED}Skipping comparison due to missing files${NC}"
        fi
        
        # Finally run object localization
        echo -e "\n${BLUE}Step 3: Object Localization${NC}"
        dsg_path="data/dsg_with_mesh.json"
        
        if check_file "$dsg_path" && check_file "$target_file"; then
            read -p "Enter search query for object localization: " query
            
            if [ -n "$query" ]; then
                echo -e "${YELLOW}Searching for objects matching '$query'...${NC}"
                $PYTHON give_the_task_efficient.py --query "$query" --dsg "$dsg_path" --mesh "$target_file" --mode "advanced"
                echo -e "${GREEN}Object search complete!${NC}"
            else
                echo -e "${RED}Skipping object localization due to empty query${NC}"
            fi
        else
            echo -e "${RED}Skipping object localization due to missing files${NC}"
        fi
        
        echo -e "\n${GREEN}Complete workflow finished!${NC}"
        ;;
        
    q|Q)
        echo -e "\n${GREEN}Exiting. Goodbye!${NC}"
        exit 0
        ;;
        
    *)
        echo -e "\n${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac