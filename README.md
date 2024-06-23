# Photo Extraction Script

This script processes a scanned image containing multiple photos and extracts each photo as an individual image file. It applies various image processing techniques to identify and separate the photos, ensuring that they are correctly oriented and free from white margins.

## Features

- Adds white padding around the input image to improve contour detection.
- Converts the image to grayscale and applies Gaussian blur to reduce noise.
- Uses adaptive thresholding, dilation, and erosion to enhance the edges of the photos.
- Finds contours and identifies rectangular and non-rectangular (skewed) photos.
- Applies perspective transform to obtain a top-down view of each photo.
- Crops white margins and removes a specified number of pixels from the edges to ensure clean borders.
- Saves each extracted photo as an individual image file.

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

1. Place your input image in the specified directory.
2. Update the `image_path` and `output_folder` variables in the script to match your input image path and desired output folder.
3. Run the script:
    ```bash
    python extract_photos.py
    ```
4. The extracted photos will be saved in the specified output folder.

## Script Overview

### Functions

- **rotate_image(image, angle)**: Rotates the image by a specified angle.
- **four_point_transform(image, pts)**: Performs a perspective transform to obtain a top-down view of the image.
- **crop_white_margins(image)**: Crops white margins from the image.
- **add_padding(image, padding_size=100)**: Adds white padding around the image.
- **crop_edges(image, crop_size=5)**: Crops a specified number of pixels from the edges of the image.
- **extract_photos(image_path, output_folder)**: Main function to extract photos from a scanned image file.

### Parameters

- `image_path` (str): Path to the input image file.
- `output_folder` (str): Folder where the extracted photos will be saved.
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
image_path = "./mnt/data/test-1.jpg"
output_folder = "./mnt/data/output_photos"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Extract photos
extract_photos(image_path, output_folder)
```

### Debugging
The script generates debug images at various stages of processing to help with troubleshooting:

- **debug_all_contours.png:**
> Shows all detected contours in the padded image.

- **debug_transformed_{photo_count}.png:**
> Shows the transformed photo before cropping white margins.

- **debug_transformed_skew_{photo_count}.png:**
> Shows the transformed skewed photo before cropping white margins.
>
These debug images are saved in the specified output folder.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
