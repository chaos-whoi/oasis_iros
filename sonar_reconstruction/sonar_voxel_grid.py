"""
Implementation concept based on OCEANS 24 Halifax paper:
'Sonar-Aided Manipulation in Low-Visibility Conditions by Novice Users'
https://doi.org/10.1109/OCEANS55160.2024.10753694

Original implementaiton modified to use Open3D for voxel grid representation 
instead of 3D numpy arrays
"""

import os

import open3d as o3d
import numpy as np
import math
import PIL.Image
import pickle
import time
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

from sonar_reconstruction.polar_remapper import PolarRemapper
from sonar_reconstruction.utils import max_image_downsample, preprocess_sonar_image

class SonarVoxelGrid():
    def __init__(self, args):
        self.sonar_params = {
            'min_range': args.sonar_min_range,
            'max_range': args.sonar_max_range,
            'h_fov': args.sonar_h_fov, # radians
            'v_fov': args.sonar_v_fov, # radians
            'num_beams': args.sonar_num_beams,
            'num_ranges': args.sonar_num_ranges,
            'range_resolution': args.sonar_max_range/args.sonar_num_ranges, 
            'downsample_factor': args.sonar_downsample,
            'polar_img': args.sonar_polar
        }
        
        self.voxel_grid_params = {
            'grid_size': np.array(args.voxel_grid_size),
            'grid_origin': np.array(args.voxel_grid_origin),
            'default_color': np.array(args.voxel_default_color)
        }

        self.filtering_params = {
            'range_window': args.range_window,
            'empty_ranges': args.empty_ranges,
            'occupancy_threshold': args.occupancy_threshold,
            'false_negative_rate': args.false_negative_rate
        }

        self.camera_params = {
            'width': args.width,
            'height': args.height,
            'fx': args.fx,
            'fy': args.fy,
            'cx': args.cx,
            'cy': args.cy,
            'depth_scale': args.depth_scale
        }

        # For creating final voxel grid
        self.occ_ratio_thresh = 1-self.filtering_params['false_negative_rate']

        # Compute voxel size
        # Voxel size = 1 pixel in downsampled image
        self.voxel_grid_params['voxel_size'] = self.sonar_params['range_resolution'] * \
            self.sonar_params['downsample_factor'] 
        print(f"Sonar Downsample Factor: {self.sonar_params['downsample_factor']}, Voxel size: {self.voxel_grid_params['voxel_size']}")
        self.voxel_grid_params['voxel_radius'] = self.voxel_grid_params['voxel_size'] * math.sqrt(2)

        # Compute voxel template
        self.voxel_template = self.compute_voxel_template()

        # Initialize Voxel Grid
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.voxel_grid.origin = self.voxel_grid_params['grid_origin']
        self.voxel_grid.voxel_size = self.voxel_grid_params['voxel_size']

        # Track voxels we've previously observed
        shape = self.voxel_grid_params['grid_size']/self.voxel_grid_params['voxel_size'] + 1
        self.obs_count = np.zeros(shape.astype(int))
        self.occ_count = np.zeros(shape.astype(int))
        
        # Track intensities
        self.occ_ratio = -np.ones(shape.astype(int))
        
        # Initialize remapper for converting from polar to cartesian image
        self.remapper = PolarRemapper(self.sonar_params)

        # Setup visualizer to render depthmaps.
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.camera_params['width'], 
            height=self.camera_params['height'], visible=False)
        self.vis.add_geometry(self.voxel_grid)
        self.ctr = self.vis.get_view_control()
        self.vis_param = self.ctr.convert_to_pinhole_camera_parameters()
        self.vis_param.intrinsic.set_intrinsics(
            width=self.camera_params['width'], height=self.camera_params['height'], 
            fx=self.camera_params['fx'], fy=self.camera_params['fy'],
            cx=self.camera_params['cx'], cy=self.camera_params['cy'])

        # Setup visualizer to render smoothed depthmaps.
        self.mesh_vis = o3d.visualization.Visualizer()
        self.mesh_vis.create_window(width=self.camera_params['width'], 
            height=self.camera_params['height'], visible=False)
        self.mesh_ctr = self.mesh_vis.get_view_control()
        self.mesh_vis_param = self.mesh_ctr.convert_to_pinhole_camera_parameters()
        self.mesh_vis_param.intrinsic.set_intrinsics(
            width=self.camera_params['width'], height=self.camera_params['height'], 
            fx=self.camera_params['fx'], fy=self.camera_params['fy'],
            cx=self.camera_params['cx'], cy=self.camera_params['cy'])

    def compute_voxel_template(self):
        # r = self.sonar_params['range_resolution']*self.sonar_params['num_ranges']
        # Floor div is needed here since downsample factor works on kernel of image
        #self.voxel_grid_params['voxel_size']*((self.sonar_params['num_ranges']//self.sonar_params['downsample_factor']))

        r_ref = self.voxel_grid_params['voxel_size']*(self.sonar_params['num_ranges']//self.sonar_params['downsample_factor'])
        y_ref = r_ref*np.cos(np.pi/2-self.sonar_params['h_fov']/2)
        z_ref = r_ref*np.sin(np.pi/2-self.sonar_params['v_fov']/2)

        x_range = np.arange(0, r_ref, self.voxel_grid_params['voxel_size'])
        y_range = np.arange(0, y_ref, self.voxel_grid_params['voxel_size'])
        z_range = np.arange(0, z_ref, self.voxel_grid_params['voxel_size'])

        # Add flipped side of y an z - ensure gridding is always centered at 0
        neg_y_range = -np.flip(y_range[1:])
        neg_z_range = -np.flip(z_range[1:])
        y_range = np.concatenate((neg_y_range, y_range))
        z_range = np.concatenate((neg_z_range, z_range))

        # Notation: 
        # p = point
        # XY = Y point is in X frame
        # x = x coordinate
        p_OV_x, p_OV_y, p_OV_z = np.meshgrid(x_range, y_range, z_range, indexing='xy') # TODO: this might need to be ij
        p_ranges = np.sqrt(p_OV_x**2 + p_OV_y**2)

        mask_h_fov = (abs(np.arctan2(p_OV_y, p_OV_x)) < self.sonar_params['h_fov']/2)
        mask_v_fov = (abs(np.arctan2(p_OV_z, p_ranges)) < self.sonar_params['v_fov']/2)
        d = np.sqrt(p_OV_x**2 + p_OV_y**2 + p_OV_z**2)
        
        mask_r = (d < r_ref) & (d > self.sonar_params['min_range'])

        total_mask = mask_h_fov & mask_v_fov & mask_r

        # For verification - 2D
        # plt.plot(p_OV_x[total_mask], p_OV_z[total_mask], '.')
        # plt.show()
        # plt.plot(p_OV_x[total_mask], p_OV_y[total_mask], '.')
        # plt.show()

        # For verification - 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(p_OV_x[total_mask], p_OV_y[total_mask], p_OV_z[total_mask], s=1)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('Voxel Template Points')
        # plt.show()

        # Extract overlap 
        p_OV = np.vstack((p_OV_x[total_mask], p_OV_y[total_mask], p_OV_z[total_mask], np.ones(p_OV_x[total_mask].shape[0])))

        return p_OV

    def update_voxel_grid(self, son_img, sonar_pose, disp=False):
        """Update voxel grid with new data frame
        
        Args:
            son_img (np.array): Grayscale polar sonar image
            sonar_pose (np.array): Sonar pose in world frame (4x4 matrix)    
        """
        # Convert to grayscale cartesian image
        if self.sonar_params['polar_img']:
            gray_img = self.remapper.pol2cart_gray(son_img)
        else:
            gray_img = son_img
            gray_img = cv2.flip(gray_img, 1) # Oculus driver records cartesian images mirrored
    
        orig_img = gray_img.copy()

        # Preprocess sonar image
        gray_img = preprocess_sonar_image(gray_img, self.remapper, self.filtering_params)
        
        if disp:
            plt.figure()
            plt.imshow(orig_img, cmap='gray')
            plt.title('Unprocessed Sonar Image')
            plt.axis('off')

            plt.figure()
            plt.imshow(gray_img, cmap='gray')
            plt.title('Processed Sonar Image')
            plt.axis('off')
            plt.show()

        # Downsample image
        gray_img = max_image_downsample(gray_img, self.sonar_params['downsample_factor'])
        
        gray_img_og = gray_img.copy()

        # Pose of Sonar (S) in World (W) frame
        X_WS = sonar_pose

        # Compute template Voxels (V) in World frame (W)      
        p_WV = np.matmul(X_WS, self.voxel_template) # 4xN matrix

        # Find nearest global voxel index in grid for each template voxel 
        # (using m to denote map) # Nx3 matrix
        p_bar_WV_m = np.floor(
            (p_WV[0:3,:].T - self.voxel_grid_params['grid_origin'])/  \
            self.voxel_grid_params['voxel_size']).astype(int)

        # Extract voxels contained inside map
        x_mask = (p_bar_WV_m[:,0] >= 0) & (p_bar_WV_m[:,0] < self.obs_count.shape[0])
        y_mask = (p_bar_WV_m[:,1] >= 0) & (p_bar_WV_m[:,1] < self.obs_count.shape[1])
        z_mask = (p_bar_WV_m[:,2] >= 0) & (p_bar_WV_m[:,2] < self.obs_count.shape[2])
        p_bar_WV_m = p_bar_WV_m[x_mask & y_mask & z_mask]

        # Compute position of gridded voxels (V) in world coordinates (W)
        p_bar_WV = ((p_bar_WV_m + 0.5)*self.voxel_grid_params['voxel_size']) + self.voxel_grid_params['grid_origin']

        # Add ones column for homogeneous coordinates
        p_bar_WV = np.hstack((p_bar_WV, np.ones((p_bar_WV.shape[0], 1))))

        # Compute position of gridded voxels (V) in Oculus coordinates (O)
        p_bar_SV = np.matmul(np.linalg.inv(X_WS), p_bar_WV.T).T

        # Compute projection of each voxel on image - i->x, j->y
        img_proj = cv2.flip(gray_img, 0)
        img_proj = cv2.flip(img_proj, 1)

        r = np.sqrt(p_bar_SV[:,0]**2 + p_bar_SV[:,1]**2 + p_bar_SV[:,2]**2)
        theta = np.arctan2(p_bar_SV[:,1],p_bar_SV[:,0]) # Not negative due to coordinates - theta=0: x=1, y=0; theta=pi/2: x=0, y=1 

        # For verification: Plot all r, theta as a polar plot
        # plt.figure(figsize=(10, 5))
        # ax = plt.subplot(111, projection='polar')
        # ax.scatter(theta, r, s=1)
        # ax.set_xlabel('theta')
        # ax.set_ylabel('r')
        # ax.set_title('r vs theta (Polar Plot)')
        # ax.grid(True)
        # plt.show()
        # i = np.round(r*np.cos(theta)/self.sonar_params['range_resolution']).astype(int)
        # j = np.round(r*np.sin(theta)/self.sonar_params['range_resolution']).astype(int) + int(img_proj.shape[1]/2)

        # Floor is used here instead of round because pixels are defined by corner points
        i = np.floor(r*np.cos(theta)/self.voxel_grid_params['voxel_size']).astype(int)
        j = np.floor(r*np.sin(theta)/self.voxel_grid_params['voxel_size'] + img_proj.shape[1]/2).astype(int)

        # Update voxel grid
        for idx, voxel_idx in enumerate(p_bar_WV_m):
            # Update observation counter
            self.obs_count[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]] += 1

            # Update intensity value
            if i[idx] >= img_proj.shape[0] or j[idx] >= img_proj.shape[1]:
                print(f"WARNING: Something is weird with the indexing. Attempted to index ({i[idx]},{j[idx]}). Image size: {img_proj.shape}")
                continue

            # Occupancy counter
            if img_proj[i[idx], j[idx]] == 255:
                self.occ_count[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]] += 1
                
            # Update voxel grid
            obs_count = self.obs_count[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]]
            occ_count = self.occ_count[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]]

            occ_ratio = occ_count/obs_count
            self.occ_ratio[voxel_idx[0]][voxel_idx[1]][voxel_idx[2]] = occ_ratio

            if (occ_ratio > self.occ_ratio_thresh):
                color=[occ_ratio*0.6]*3 # 0.6 ensures rendered voxels won't be white
                self.voxel_grid.add_voxel(o3d.geometry.Voxel(grid_index=voxel_idx, color=color))
            else:
                self.voxel_grid.remove_voxel(voxel_idx)

    def compute_render(self, pose):
        # self.vis.update_geometry(self.voxel_grid) # For some reason this doesn't work
        self.vis.remove_geometry(self.voxel_grid)
        self.vis.add_geometry(self.voxel_grid)

        self.vis_param.extrinsic = pose
        self.ctr.convert_from_pinhole_camera_parameters(self.vis_param, allow_arbitrary=True)

        # Capture depth image and make a point cloud.
        self.vis.poll_events()
        self.vis.update_renderer()

        depth = self.vis.capture_depth_float_buffer(False)
        depth_np = np.asarray(depth)*self.camera_params['depth_scale']
        depth_np = depth_np.astype(np.uint16)
        return depth_np

    def initialize_smoothed_renderer(self, mesh):
        self.mesh_vis.clear_geometries()
        self.mesh_vis.add_geometry(mesh)

    def compute_render_smoothed(self, pose):
        self.mesh_vis_param.extrinsic = pose
        self.mesh_ctr.convert_from_pinhole_camera_parameters(self.mesh_vis_param, allow_arbitrary=True)

        # Capture depth image and make a point cloud.
        self.mesh_vis.poll_events()
        self.mesh_vis.update_renderer()

        depth = self.mesh_vis.capture_depth_float_buffer(False)
        depth_np = np.asarray(depth)*self.camera_params['depth_scale']
        depth_np = depth_np.astype(np.uint16)
        return depth_np

    def visualize_voxel_grid(self):
        # For Verification: Visualize results
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=600, height=600, visible=True)
        vis.add_geometry(self.voxel_grid)
        vis.run()

    def save_voxel_grid(self, output_path):
        o3d.io.write_voxel_grid(output_path, self.voxel_grid)
        print(f"Voxel grid saved at {output_path}")

    def load_voxel_grid(self, input_path):
        self.voxel_grid = o3d.io.read_voxel_grid(input_path)
        print(f"Voxel grid loaded from {input_path}")