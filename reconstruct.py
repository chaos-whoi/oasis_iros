import os
project_dir = os.path.dirname(os.path.realpath(__file__))
import time
import numpy as np
import pickle
import mcubes
import open3d as o3d

from argparse import ArgumentParser
from tqdm import tqdm

from sonar_reconstruction.sonar_voxel_grid import SonarVoxelGrid
from utils.config_utils import read_config
from utils.dataset_utils import load_data

if __name__ == "__main__":
    # Parse command line args
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default="configs/tank.yaml")
    args = parser.parse_args()
    config_path = args.config
    config = read_config(config_path)
    cache_path = os.path.join(config.cache_dir, config.dataset_name)
    output_path = os.path.join(config.output_dir, config.dataset_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True) 
    
    if os.path.exists(cache_path) and config.use_cache:
        with open(os.path.join(cache_path, "unprocessed_data.pkl"), "rb") as f:
            data = pickle.load(f)
        sonar_imgs = data["sonar_imgs"]
        cam_imgs = data["cam_imgs"]
        son_poses = data["son_poses"]
        cam_poses = data["cam_poses"]
    else:
        sonar_imgs, cam_imgs, son_poses, cam_poses = load_data(project_dir, config)
        
        # Save data to pickle file
        data = {
            "sonar_imgs": sonar_imgs,
            "cam_imgs": cam_imgs,
            "son_poses": son_poses, 
            "cam_poses": cam_poses
        }
        os.makedirs(cache_path, exist_ok=True)
        with open(os.path.join(cache_path, "unprocessed_data.pkl"), "wb") as f:
            pickle.dump(data, f)

    # Create reconstruction using sonar data
    grid = SonarVoxelGrid(config)

    if config.use_cache and os.path.exists(os.path.join(output_path, "voxel_grid.ply")):
        print("Loading saved reconstruction")

        grid.load_voxel_grid(os.path.join(output_path, "voxel_grid.ply"))
        with open(os.path.join(output_path, "voxel_occ_ratio.pkl"), "rb") as f:
            grid.occ_ratio = pickle.load(f)
    else:
        print("Computing sonar reconstruction")

        loop_times = []

        num_frames = len(sonar_imgs)
        for img_idx in tqdm(range(num_frames)): # Idx used instead of enumerate to avoid issues with tqdm
            son_img = sonar_imgs[img_idx]

            if img_idx != 0:
                # Compute difference between previous pose and current pose
                pose_diff = np.linalg.norm(son_poses[img_idx][:3, 3] - prev_pose[:3, 3])
                if pose_diff < config.min_pose_delta: # Avoid "stacks" of identical frames
                    # print(f"Skipping frame {img_idx}")
                    continue
            
            prev_pose = son_poses[img_idx]

            # Record start time (after data loading)
            lt = time.time()

            # Process update
            grid.update_voxel_grid(son_img, son_poses[img_idx], disp=False)
            loop_times.append(time.time() - lt)

        avg_loop_time = sum(loop_times) / len(loop_times)
        print(f"Average sonar processing loop time: {avg_loop_time:.4f} seconds")

        # Save grid
        voxel_grid_filepath = os.path.join(output_path, "voxel_grid.ply")
        grid.save_voxel_grid(voxel_grid_filepath)
        
        # Save voxel occupancy ratio to cache
        voxel_occ_ratio_filepath = os.path.join(output_path, "voxel_occ_ratio.pkl")
        with open(voxel_occ_ratio_filepath, "wb") as f:
            pickle.dump(grid.occ_ratio, f)

    # View results
    grid.visualize_voxel_grid()
    occupancy_grid = grid.occ_ratio > grid.occ_ratio_thresh

    mesh_filepath = os.path.join(output_path, "voxel_grid_meshed.obj")
    smoothed_mesh_filepath = os.path.join(output_path, "voxel_grid_smoothed.obj")

    # Save occupancy grid as smoothed mesh    
    vertices, triangles = mcubes.marching_cubes(occupancy_grid, 0)
    vertices = vertices * grid.voxel_grid_params['voxel_size'] + grid.voxel_grid_params['grid_origin']  # convert to grid reference frame
    mcubes.export_obj(vertices, triangles, mesh_filepath)
    print(f"Mesh saved to {mesh_filepath}")

    smoothed_grid = mcubes.smooth(occupancy_grid)
    vertices, triangles = mcubes.marching_cubes(smoothed_grid, 0)
    vertices = vertices * grid.voxel_grid_params['voxel_size'] + grid.voxel_grid_params['grid_origin']  # convert to grid reference frame
    mcubes.export_obj(vertices, triangles, smoothed_mesh_filepath)
    print(f"Smoothed mesh saved to {smoothed_mesh_filepath}")

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(smoothed_mesh_filepath)
    grid.initialize_smoothed_renderer(mesh)