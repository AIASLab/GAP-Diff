# Real jpeg for protected images in #2088 
import os
from PIL import Image

import argparse
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="do real jpeg compression")
    parser.add_argument(
        "--quality",
        type=int,
        help="the quality of jpeg compression",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        required=True,
        help="Path to source images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to preprocessed images",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Traverse all subfolders in the source folder
    for folder_name in os.listdir(args.source_path):
        input_subfolder = os.path.join(args.source_path, folder_name)
        output_subfolder = os.path.join(args.output_path, folder_name)
        
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        
        # Get the paths of all image files in the current subfolder
        image_files = [f for f in os.listdir(input_subfolder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Define JPEG compression quality
        quality = args.quality  
        
        # Loop through each image file in the current subfolder
        for image_file in image_files:
            with Image.open(os.path.join(input_subfolder, image_file)) as img:
                output_path = os.path.join(output_subfolder, os.path.splitext(image_file)[0] + "_compressed.jpg")
                
                # Save the image in JPEG format with compression
                img.save(output_path, format='JPEG', quality=quality)

    print("Image compression completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    main(args)