#!/usr/bin/env python3

"""
This script remaps a polar sonar image to a Cartesian image using OpenCV's remap function.
"""

import os
import cv2
import numpy as np

class PolarRemapper():
    def __init__(self, sonar_params):
        self.params = sonar_params

        max_range = sonar_params['max_range']
        min_angle = -sonar_params['h_fov'] / 2
        max_angle = sonar_params['h_fov'] / 2

        # Create a grid of ranges and angles
        ranges = np.linspace(0, max_range, sonar_params['num_ranges'])
        angles = np.linspace(min_angle, max_angle, sonar_params['num_beams'])

        # Create a meshgrid for ranges and angles
        A, R = np.meshgrid(angles, ranges)

        pol_height = sonar_params['num_ranges']
        pol_width = sonar_params['num_beams']
        pol_center_px = (pol_width-1)/2

        range_resolution = (max_range) / sonar_params['num_ranges']
        cart_width_m = 2*max_range*np.sin(max_angle)
        cart_height = pol_height
        cart_width = int(np.round(cart_width_m/range_resolution)) + 1
        cart_center_px = (cart_width-1)/2

        # Initialize map
        self.pol2cart_x = np.zeros((cart_height, cart_width), dtype=np.float32)
        self.pol2cart_y = np.zeros((cart_height, cart_width), dtype=np.float32)

        for i in range(cart_height):
            for j in range(cart_width):
                x = j - cart_center_px
                y = cart_height - i

                # Arctan is x/y because 0 degrees points towards +y
                self.pol2cart_x[i,j] = pol_center_px + (np.arctan(x/y) * pol_width / sonar_params['h_fov'])
                self.pol2cart_y[i,j] = np.sqrt(x**2 + y**2)

        self.cart2pol_x = np.zeros((pol_height, pol_width), dtype=np.float32)
        self.cart2pol_y = np.zeros((pol_height, pol_width), dtype=np.float32)

        for i in range(pol_height):
            for j in range(pol_width):
                r = pol_height-i
                theta = (j+0.5) * (sonar_params['h_fov'] / pol_width) - (sonar_params['h_fov'] / 2)

                x = r * np.sin(theta)
                y = r * np.cos(theta)

                self.cart2pol_x[i,j] = np.round(x + cart_width / 2)
                self.cart2pol_y[i,j] = np.round(y)

        self.pol_width = pol_width
        self.pol_height = pol_height
        self.cart_width = cart_width
        self.cart_height = cart_height
        
    def pol2cart(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.pol2cart_gray(gray_image)

    def pol2cart_gray(self, image):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # print(f"Image dimensions: {width} x {height}")

        if height != self.pol_height or width != self.pol_width:
            print(f"Error: Image dimensions do not match the specified parameters.")
            exit()    

        # Remap the image from polar to Cartesian coordinates
        dst = cv2.remap(image, self.pol2cart_x, self.pol2cart_y, cv2.INTER_LINEAR)
        return dst

    def cart2pol(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.cart2pol_gray(gray_image)

    def cart2pol_gray(self, image):
        image = cv2.flip(image, 0)
        
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # print(f"Image dimensions: {width} x {height}")

        if height != self.cart_height or width != self.cart_width:
            print(f"Error: Image dimensions do not match the specified parameters.")
            exit()    

        # Remap the image from polar to Cartesian coordinates
        dst = cv2.remap(image, self.cart2pol_x, self.cart2pol_y, cv2.INTER_LINEAR)
        dst = cv2.flip(dst, 0)
        return dst