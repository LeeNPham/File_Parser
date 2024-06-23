# Photo Extraction Script

This script processes scanned images containing multiple photos and extracts each photo as an individual image file. It applies various image processing techniques to identify and separate the photos, ensuring that they are correctly oriented and free from white margins.

## Features

- Adds white padding around the input image to improve contour detection.
- Converts the image to grayscale and applies Gaussian blur to reduce noise.
- Uses adaptive thresholding, dilation, and erosion to enhance the edges of the photos.
- Finds contours and identifies rectangular and non-rectangular (skewed) photos.
- Applies perspective transform to obtain a top-down view of each photo.
- Crops white margins and removes a specified number of pixels from the edges to ensure clean borders.
- Saves each extracted photo as an individual image file.
- Generates debug images at various stages of processing for troubleshooting.

## Requirements

- Python 3.x
- OpenCV library

## Installation

1. Install Python from [python.org](https://www.python.org/).
2. Install OpenCV library using pip:
    ```bash
    pip install opencv-python-headless
    ```

## Usage

1. Place your input images in the specified input folder.
2. Update the `input_folder`, `output_folder`, and `debug_folder` variables in the script to match your input folder path, desired output folder, and debug folder.
3. Run the script:
    ```bash
    python extract_photos.py
    ```
4. The extracted photos will be saved in the specified output folder. Debug images will be saved in the specified debug folder.

## Script Overview

### Functions

- **rotate_image(image, angle)**: Rotates the image by a specified angle.
- **four_point_transform(image, pts)**: Performs a perspective transform to obtain a top-down view of the image.
- **crop_white_margins(image)**: Crops white margins from the image.
- **add_padding(image, padding_size=100)**: Adds white padding around the image.
- **crop_edges(image, crop_size=5)**: Crops a specified number of pixels from the edges of the image.
- **extract_photos(image_path, output_folder, debug_folder)**: Main function to extract photos from a scanned image file.
- **process_folder(input_folder, output_folder, debug_folder)**: Processes all images in the input folder.

### Parameters

- `image_path` (str): Path to the input image file.
- `output_folder` (str): Folder where the extracted photos will be saved.
- `debug_folder` (str): Folder where the debug images will be saved.
- `input_folder` (str): Folder containing the input images.
- `angle` (float): Angle by which to rotate the image.
- `pts` (numpy.ndarray): Four points specifying the region to transform.
- `padding_size` (int): Size of the padding to add.
- `crop_size` (int): Number of pixels to crop from each edge.

### Example

```python
import cv2
import numpy as np
import os

# Define paths
input_folder = "./mnt/data/input_photos"
output_folder = "./mnt/data/output_photos"
debug_folder = "./mnt/data/output_debug_photos"

# Create output and debug folders if they do not exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# Process all images in the input folder
process_folder(input_folder, output_folder, debug_folder)
```

### Debugging
The script generates debug images at various stages of processing to help with troubleshooting:

- **debug_all_contours.png:**
>Shows all detected contours in the padded image.
- **debug_transformed_{photo_count}.png:**
>Shows the transformed photo before cropping white margins.
- **debug_transformed_skew_{photo_count}.png:**
>Shows the transformed skewed photo before cropping white margins.
These debug images are saved in the specified debug folder.




### License

> This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) with an additional profit-sharing clause. Any entity using this software for commercial purposes is required to share 10% of the profits generated from the use of this software with the original author(s).
