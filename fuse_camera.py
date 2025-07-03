"""Rendering color image on mesh"""

import os
project_dir = os.path.dirname(os.path.realpath(__file__))
import sys
from argparse import ArgumentParser
from sonar_reconstruction.sonar_voxel_grid import SonarVoxelGrid

from utils.config_utils import read_config
from utils.dataset_utils import load_data

import time
import numpy as np
import cv2
import open3d as o3d
import rembg


# Increase brightness of cam_img, for rendering
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# Parse command line args
parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str, default="configs/tank.yaml")
args = parser.parse_args()
config_path = args.config
config = read_config(config_path)
cache_path = os.path.join(config.cache_dir, config.dataset_name)

# Initialize voxel grid
grid = SonarVoxelGrid(config)

# Load reconstruction from cache
mesh_filepath = os.path.join(cache_path, "voxel_grid_meshed.obj")
smoothed_mesh_filepath = os.path.join(cache_path, "voxel_grid_smoothed.obj")

if not (os.path.exists(mesh_filepath) and os.path.exists(smoothed_mesh_filepath)):
    print(f"Warning: Mesh files not found in {cache_path}. Please run reconstruct.py first to generate the required mesh files.")
    sys.exit(1)

mesh = o3d.io.read_triangle_mesh(mesh_filepath)
grid.initialize_smoothed_renderer(mesh)

# Load camera data
_, cam_imgs, _, cam_poses = load_data(project_dir, config, cam=True, sonar=False)

# Initialize background remover
model_name = "isnet-general-use"
rembg_session = rembg.new_session(model_name)

pcd_list = []

for idx, cam_img in enumerate(cam_imgs):
    cam_img = increase_brightness(cam_img, value=30)

    # Apply background removal to optical image using rembg  
    log_time = time.time()
    cam_img_mask = rembg.remove(cam_img, session=rembg_session, only_mask=True)
    # print(f"Time to remove background: {time.time() - log_time}")

    # Render depth image
    depth_img = grid.compute_render_smoothed(cam_poses[idx])
    depth_img_gray = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save the depth image
    depth_img_gray_path = os.path.join(cache_path, f"depth_image_gray_{idx}.png")
    cv2.imwrite(depth_img_gray_path, depth_img_gray)

    # Save the camera image
    cam_img_path = os.path.join(cache_path, f"cam_image_{idx}.png")
    cv2.imwrite(cam_img_path, cam_img)

    # Remove background in depth image
    ret, son_img_mask = cv2.threshold(depth_img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Add masks
    mask = cv2.bitwise_and(cam_img_mask, son_img_mask)

    # Save the masks
    cam_img_mask_path = os.path.join(cache_path, f"cam_img_mask_{idx}.png")
    cv2.imwrite(cam_img_mask_path, cam_img_mask)

    son_img_mask_path = os.path.join(cache_path, f"son_img_mask_{idx}.png")
    cv2.imwrite(son_img_mask_path, son_img_mask)

    combined_mask_path = os.path.join(cache_path, f"combined_mask_{idx}.png")
    cv2.imwrite(combined_mask_path, mask)

    # Mask out depth
    depth_img[mask < 10] = 0

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cam_img), 
        o3d.geometry.Image(depth_img), 
        convert_rgb_to_intensity=False
    )

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=grid.camera_params['width'], height=grid.camera_params['height'], 
            fx=grid.camera_params['fx'], fy=grid.camera_params['fy'],
            cx=grid.camera_params['cx'], cy=grid.camera_params['cy']),
        extrinsic = cam_poses[idx]
    )

    points = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_list.append(pcd)

# Initialize visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720)

# Load mesh from file
mesh = o3d.io.read_triangle_mesh(smoothed_mesh_filepath)
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

# Add points
for pcd in pcd_list:
    vis.add_geometry(pcd)

vis.run()
vis.destroy_window()