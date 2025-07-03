import os
project_dir = os.path.dirname(os.path.realpath(__file__))
import time
import numpy as np
import pickle
import mcubes
import open3d as o3d

from argparse import ArgumentParser
from tqdm import tqdm

# from sonar_reconstruction.sonar_voxel_grid import SonarVoxelGrid
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

    # # Create reconstruction using sonar data
    # grid = SonarVoxelGrid(config)

    # if args.use_cache and os.path.exists(os.path.join(config.cache_path, "voxel_grid.ply")):
    # # Load grid from cache
    #     grid.load_voxel_grid(os.path.join(config.cache_path, "voxel_grid.ply"))
    #     with open(os.path.join(config.cache_path, "voxel_occ_ratio.pkl"), "rb") as f:
    #         grid.occ_ratio = pickle.load(f)
    
    # # Construct grid
    # else:
    #     # Keep track of time
    #     loop_times = []
    #     # Process sonar data first
    #     print("Generating sonar map")

    #     prev_pose = son_poses[0]

    #     img_idx_list = list(range(len(sonar_imgs)))

    #     for img_idx, son_img in tqdm(zip(img_idx_list, sonar_imgs)):
    #         # Compute difference between previous pose and current pose
    #         pose_diff = np.linalg.norm(son_poses[img_idx][:3, 3] - prev_pose[:3, 3])
    #         if pose_diff < 0.01: # if we've move less than a centimeter ignore change
    #             # print(f"Skipping frame {img_idx}")
    #             continue

    #         prev_pose = son_poses[img_idx]


    #         # TODO: Temp = only process every 10th image
    #         # if img_idx % 100 != 0:
    #         #     continue
    #         # if img_idx <= 100:
    #         #     continue
    #         # if img_idx >= 200:
    #         #     break

    #         # Record start time (after data loading)
    #         lt = time.time()

    #         disp = False
    #         # if img_idx == 90:
    #         #     disp = True

    #         # Process update
    #         gs.grid.update_voxel_grid(son_img, son_poses[img_idx], disp=disp)
    #         loop_times.append(time.time() - lt)

    #         # if img_idx >= 800:
    #         #     break

    #     avg_loop_time = sum(loop_times) / len(loop_times)
    #     print(f"Average sonar processing loop time: {avg_loop_time:.4f} seconds")

    #     # Save grid
    #     voxel_grid_filepath = os.path.join(config.cache_path, "voxel_grid.ply")
    #     gs.grid.save_voxel_grid(voxel_grid_filepath)
        
    #     # Save voxel occupancy ratio to cache
    #     voxel_occ_ratio_filepath = os.path.join(config.cache_path, "voxel_occ_ratio.pkl")
    #     with open(voxel_occ_ratio_filepath, "wb") as f:
    #         pickle.dump(gs.grid.occ_ratio, f)
    
    # # View results
    # gs.grid.visualize_voxel_grid()
    # occupancy_grid = gs.grid.occ_ratio>gs.grid.occ_ratio_thresh

    # mesh_filepath = os.path.join(config.cache_path, "voxel_grid_meshed.obj")
    # smoothed_mesh_filepath = os.path.join(config.cache_path, "voxel_grid_smoothed.obj")

    # # if not args.use_cache or not os.path.exists(os.path.join(config.cache_path, "voxel_grid_meshed.obj")):
    # # Save occupancy grid as smoothed mesh    
    # vertices, triangles = mcubes.marching_cubes(occupancy_grid, 0)
    # vertices = vertices*gs.grid.voxel_grid_params['voxel_size'] + gs.grid.voxel_grid_params['grid_origin'] # convert to grid reference frame
    # mcubes.export_obj(vertices, triangles, mesh_filepath)
    # print(f"Mesh saved to {mesh_filepath}")

    # smoothed_grid = mcubes.smooth(occupancy_grid)
    # vertices, triangles = mcubes.marching_cubes(smoothed_grid, 0)
    # vertices = vertices*gs.grid.voxel_grid_params['voxel_size'] + gs.grid.voxel_grid_params['grid_origin'] # convert to grid reference frame
    # mcubes.export_obj(vertices, triangles, smoothed_mesh_filepath)
    # print(f"Smoothed mesh saved to {smoothed_mesh_filepath}")

    # # Load mesh
    # mesh = o3d.io.read_triangle_mesh(smoothed_mesh_filepath)
    # gs.grid.initialize_smoothed_renderer(mesh)

    # # Update splatting model
    # print("Updating splatting model")
    # start_time = time.time()
    # for img_idx, cam_img in enumerate(tqdm(cam_imgs)):
    #     # if img_idx == 5: #TODO: temp - choose one frame
    #     #     break

    #     # Render depth image from sonar data
    #     depth_img = gs.grid.compute_render_smoothed(cam_poses[img_idx])
        
    #     # # TEMP: Visualize results - checking camera pose---------------------------
    #     # import open3d as o3d
    #     # vis = o3d.visualization.Visualizer()
    #     # vis.create_window(width=600, height=600, visible=True)
    #     # # vis.add_geometry(gs.grid.voxel_grid)
    #     # vis.add_geometry(mesh)
    #     # # Visualize origin
    #     # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    #     # origin.transform(  [[1, 0, 0, 0],
    #     #                     [0, 1, 0, 0],
    #     #                     [0, 0, 1, 0],
    #     #                     [0, 0, 0, 1]])   
    #     # vis.add_geometry(origin)
        
    #     # # Visualize camera pose
    #     # ctr = vis.get_view_control()
    #     # vis_param = ctr.convert_to_pinhole_camera_parameters()
    #     # vis_param.intrinsic.set_intrinsics(
    #     #     width=gs.grid.camera_params['width'], height=gs.grid.camera_params['height'], 
    #     #     fx=gs.grid.camera_params['fx'], fy=gs.grid.camera_params['fy'],
    #     #     cx=gs.grid.camera_params['cx'], cy=gs.grid.camera_params['cy'])

    #     # cameraLines = o3d.geometry.LineSet.create_camera_visualization(
    #     #         view_width_px=gs.grid.camera_params['width'], view_height_px=gs.grid.camera_params['height'], 
    #     #         intrinsic=vis_param.intrinsic.intrinsic_matrix, extrinsic=cam_poses[img_idx])
    #     # vis.add_geometry(cameraLines)    
    #     # vis.run()
    #     # # -----------------------------------------------------------------------------

    #     # # For debugging - Display depth and camera image side-by-side -----------------
    #     # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #     # axs[0].imshow(depth_img, cmap='gray')
    #     # axs[0].set_title(f"Depth Image {img_idx}")
    #     # axs[1].imshow(cam_img)
    #     # axs[1].set_title(f"Camera Image {img_idx}")
    #     # plt.show()
    #     # # ------------------------------------------------------------------------------

    #     # Process frame
    #     # TODO: FIGURE OUT IF TIMESTAMP IS NEEDED

    #     # TODO: get rid of this try/except, figure out what the real problem is
    # #     try:
    # #         gs.rtg_slam.process_frame(cam_img, depth_img, cam_poses[img_idx], 
    # #             img_idx)
    # #     except:
    # #         print("Skipping frame")

    # # gs.rtg_slam.save_output()
 

    