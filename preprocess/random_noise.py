# Random noise for protected images in #2088 
import os
import cv2
import numpy as np

# Define input and output folder paths
input_folder = "./protected_images/gap_diff_per16"
output_folder = "./protected_images/gap_diff_per16_random_noise"

# Ensure the output folder exists; create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Noise intensity
noise_scale = 0.05

# Traverse all image files in the input folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the input image path
            input_path = os.path.join(root, file)
            
            # Read the image
            img = cv2.imread(input_path)
            
            # Convert the image to float type to add noise
            img = img.astype(np.float32) / 255.0
            
            # Generate random noise with the same size as the image
            noise = np.random.normal(loc=0, scale=noise_scale, size=img.shape).astype(np.float32)
            
            # Add the noise to the image
            noisy_img = img + noise
            
            # Clip the image to the 0-1 range and convert back to 8-bit type
            noisy_img = np.clip(noisy_img, 0, 1) * 255
            noisy_img = noisy_img.astype(np.uint8)
            
            # Construct the output image path
            output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
            
            # Ensure the output folder exists; create it if it doesn't
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the processed image
            cv2.imwrite(output_path, noisy_img)

print("Random noise added to images successfully!")