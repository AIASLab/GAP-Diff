# Gaussian blur for protected images in #2088 
import os
import cv2

# Define input and output folder paths
input_folder = "./protected_images/gap_diff_per16"
output_folder = "./protected_images/gap_diff_per16_gb"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Specify Gaussian kernel size and standard deviation
kernel = 3
kernel_size = (3, 3)
sigma = 0.05 

# Traverse all image files in the input folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(root, file)
            img = cv2.imread(input_path)
            
            # Apply Gaussian blur to the image
            blurred_img = cv2.GaussianBlur(img, kernel_size, sigmaX=0.1, sigmaY=2.0)
            
            # Construct the output image path
            output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, blurred_img)

print("Image processing with Gaussian blur completed successfully!")