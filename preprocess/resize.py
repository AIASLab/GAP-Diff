# Resize for protected images in #2088 
import os
import cv2

# Define input and output folder paths
input_folder = "./protected_images/gap_diff_per16"
output_folder = "./protected_images/gap_diff_per16_resize"

# Ensure the output folder exists; if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define target sizes
size_large = (256, 256)
#size_small = (512, 512)

# Traverse through all image files in the input folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the input image path
            input_path = os.path.join(root, file)
            
            # Read the image
            img = cv2.imread(input_path)
            
            # Resize the image to 1024x1024
            resized_large = cv2.resize(img, size_large, interpolation=cv2.INTER_LINEAR)
            
            # Resize the image back to 512x512
            #resized_small = cv2.resize(resized_large, size_small, interpolation=cv2.INTER_LINEAR)
            
            # Construct the output image path
            output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
            
            # Ensure the output directory exists; if not, create it
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the processed image
            cv2.imwrite(output_path, resized_large)

print("Resize images successfully!")