"""
Author: Nick Kottek
Date: 14.05.2024
"""

import os
from PIL import Image

# Define directory paths
input_directory = "../dataset"
output_directory = "../dataset_compressed"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate subdirectorys in input directory
for root, dirs, files in os.walk(input_directory):
    # Iterate files in subdirectorys
    for file in files:
        if file.endswith(".jpg"):
            # Inputpath of image
            input_path = os.path.join(root, file)

            # Output path of image in corresponding directory of the output directory
            relative_path = os.path.relpath(input_path, input_directory)
            output_path = os.path.join(output_directory, relative_path)

            # Create output directory of subdirectory if it doesn't exit
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Get Image and resize to 64x64 px
            with Image.open(input_path) as img:
                img = img.resize((64, 64))

                # Save compressed image to output directory
                print(output_path)
                img.save(output_path)
