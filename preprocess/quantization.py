# Quantization for protected images in #2088 
import os
import cv2
import numpy as np

# Define input and output folder paths
input_folder = "./protected_images/gap_diff_per16"
output_folder = "./protected_images/gap_diff_per16_quantization"

# Ensure the output folder exists; create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Quantization levels
quantization_levels = 64  # 6-bit

# Traverse all image files in the input folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the input image path
            input_path = os.path.join(root, file)
            
            # Read the image
            img = cv2.imread(input_path)
            
            # Convert the image to float type for quantization
            img = img.astype(np.float32) / 255.0
            
            # Quantize the image (dividing it into 64 levels)
            quantized_img = np.floor(img * quantization_levels) / quantization_levels
            
            # Scale the image back to the 0-255 range and convert to 8-bit type
            quantized_img = np.clip(quantized_img * 255, 0, 255).astype(np.uint8)
            
            # Construct the output image path
            output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
            
            # Ensure the output folder exists; create it if it doesn't
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the processed image
            cv2.imwrite(output_path, quantized_img)

print("Image quantization completed successfully!")