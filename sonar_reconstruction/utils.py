import numpy as np
import time
import matplotlib.pyplot as plt

def max_image_downsample(img, factor):
    """Downsample image by a factor. 
    
    Args:
        img (np.array): 2D grayscale image
        factor (int): Factor by which to downsample the image
    """
    factor = int(factor)

    # Pad the image to ensure the dimensions are divisible by the kernel size
    pad_height = (factor - img.shape[0] % factor) % factor
    pad_width = (factor - img.shape[1] % factor) % factor
    img_padded = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Downsample by keeping the maximum value in each section
    downsampled_img = img_padded.reshape(
        img_padded.shape[0] // factor, factor,
        img_padded.shape[1] // factor, factor
    ).max(axis=(1, 3))

    return downsampled_img

def compute_std_row(row_data, row_mean, row_std, occupancy_threshold):
    std_row_data = (row_data - row_mean)/row_std
    
    # Use the occupancy threshold to determine if a pixel is occupied
    std_row_data[std_row_data >= occupancy_threshold] = 255
    std_row_data[std_row_data < occupancy_threshold] = 0
    return std_row_data

# Standard deviation-based sonar preprocessing
def preprocess_sonar_image(son_img, remapper, params):
    polar_image = remapper.cart2pol_gray(son_img)

    # Calculate mean and standard deviation for the background
    bg_mean = np.mean(polar_image[:params['empty_ranges'], :])
    bg_std = np.std(polar_image[:params['empty_ranges'], :])

    std_img = np.zeros_like(polar_image)

    for r in range(polar_image.shape[0]):
        row_data = polar_image[r, :]

        # If there is a return greater than background mean + 2*std then it is likely a feature
        if np.max(row_data) > bg_mean + 2*bg_std:
            window_data = polar_image[max(0, r-params['range_window']):min(polar_image.shape[0], r+params['range_window']), :]
            row_mean = np.mean(window_data) # we observed other ranges can be affected by nearby ringing
            row_std = np.std(window_data)

            # enforce the background as the minimum mean/std
            row_mean = max(row_mean, bg_mean)
            row_std = max(row_std, bg_std)
            std_img[r, :] = compute_std_row(row_data, row_mean, row_std, params['occupancy_threshold'])
    
    cart_std_img = remapper.pol2cart_gray(std_img)
    
    return cart_std_img