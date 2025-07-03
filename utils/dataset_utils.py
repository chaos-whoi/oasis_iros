import os
from tqdm import tqdm
import pickle
import PIL.Image
import numpy as np

def load_data(project_dir, args, cam=True, sonar=True):
    sonar_imgs = []
    cam_imgs = []
    son_poses = []
    cam_poses = []

    dataset_dir = f"{project_dir}/{args.dataset_dir}/{args.dataset_name}"

    if sonar:
        sonar_image_dir = f"{dataset_dir}/sonar/"
        sonar_img_files = [file for file in sorted(os.listdir(sonar_image_dir)) if file.endswith('.png')]

        with open(f"{dataset_dir}/son_poses.pkl", 'rb') as f:
            son_poses_saved = pickle.load(f)
        
        if args.frame_num == -1:
            args.frame_num = len(sonar_img_files)

        son_poses_saved = son_poses_saved[args.frame_start:args.frame_num:args.frame_step]
        sonar_img_files = sonar_img_files[args.frame_start:args.frame_num:args.frame_step]

        print("Loading sonar data...")
        for s_filename in tqdm(sonar_img_files):
            # Load data
            img = PIL.Image.open(sonar_image_dir + s_filename).convert("L")
            son_img_np = np.array(img)
            sonar_imgs.append(son_img_np)

        for son_pose in son_poses_saved:
            son_poses.append(np.array(son_pose))

    if cam:
        cam_image_dir = f"{dataset_dir}/color/"
        cam_img_files = [file for file in sorted(os.listdir(cam_image_dir)) if file.endswith('.png')]

        with open(f"{dataset_dir}/cam_poses.pkl", 'rb') as f:
            cam_poses_saved = pickle.load(f)

        if args.frame_num == -1:
            args.frame_num = len(cam_img_files)

        cam_poses_saved = cam_poses_saved[args.frame_start:args.frame_num:args.frame_step]
        cam_img_files = cam_img_files[args.frame_start:args.frame_num:args.frame_step]

        print("Loading camera data...")
        for c_filename in tqdm(cam_img_files):
            # Load data
            img = PIL.Image.open(cam_image_dir + c_filename)
            cam_img_np = np.array(img)
            cam_imgs.append(cam_img_np)

        for cam_pose in cam_poses_saved:
            # Convert cam pose to correct frame (dataset has camera 90 degrees rotated)
            R = np.array([[0, -1, 0, 0], 
                        [1, 0, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])
            
            # Uncomment for conventional camera orientation
            # R = np.array([[1, 0, 0, 0], 
            #     [0, 1, 0, 0], 
            #     [0, 0, 1, 0], 
            #     [0, 0, 0, 1]])
            cam_pose = np.linalg.inv(cam_pose) # Inverse is necessary here for open3D rendering
            cam_pose = R @ cam_pose
            cam_poses.append(cam_pose)


    return sonar_imgs, cam_imgs, son_poses, cam_poses