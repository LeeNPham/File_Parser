import cv2
import numpy as np
import os


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def crop_white_margins(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y : y + h, x : x + w]
    return cropped_image


def add_padding(image, padding_size=100):
    padded_image = cv2.copyMakeBorder(
        image,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    return padded_image


def crop_edges(image, crop_size=5):
    return image[crop_size:-crop_size, crop_size:-crop_size]


def extract_photos(image_path, output_folder, debug_folder):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from path: {image_path}")

    # Add white padding to the image
    padded_image = add_padding(image)

    # Convert to grayscale
    gray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Invert the thresholded image
    thresh = cv2.bitwise_not(thresh)

    # Use dilation and erosion to close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    thresh = cv2.erode(thresh, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Draw all contours
    debug_image = padded_image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_folder, "debug_all_contours.png"), debug_image)

    # Loop through contours and extract rectangular photos
    photo_count = 0
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.01 * cv2.arcLength(
            contour, True
        )  # Further reduced epsilon for finer approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter out small or irrelevant contours
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if (
                area > 5000
            ):  # Adjusted filter to exclude very small contours based on area
                # Extract the photo
                photo = four_point_transform(padded_image, approx.reshape(4, 2))

                # Debug: Save the transformed photo before cropping white margins
                cv2.imwrite(
                    os.path.join(debug_folder, f"debug_transformed_{photo_count}.png"),
                    photo,
                )

                # Remove white margins
                photo = crop_white_margins(photo)

                # Crop edges by 5 pixels
                photo = crop_edges(photo, 1)

                # Save the photo
                output_path = os.path.join(output_folder, f"photo_{photo_count}.png")
                cv2.imwrite(output_path, photo)
                photo_count += 1
        else:
            # Handle non-rectangular contours (like the skewed image)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            photo = four_point_transform(padded_image, box)
            area = cv2.contourArea(contour)
            if area > 5000:
                # Debug: Save the transformed photo before cropping white margins
                cv2.imwrite(
                    os.path.join(
                        debug_folder, f"debug_transformed_skew_{photo_count}.png"
                    ),
                    photo,
                )

                # Remove white margins
                photo = crop_white_margins(photo)

                # Crop edges by 5 pixels
                photo = crop_edges(photo, 2)

                # Save the photo
                output_path = os.path.join(output_folder, f"photo_{photo_count}.png")
                cv2.imwrite(output_path, photo)
                photo_count += 1

    print(f"Extracted {photo_count} photos from {os.path.basename(image_path)}.")


def process_folder(input_folder, output_folder, debug_folder):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            try:
                extract_photos(file_path, output_folder, debug_folder)
            except ValueError as e:
                print(e)


# Define paths
input_folder = "./mnt/data/input_folder"
output_folder = "./mnt/data/output_photos"
debug_folder = "./mnt/data//output_photos/output_debug_photos"

# Create output and debug folders if they do not exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# Process all images in the input folder
process_folder(input_folder, output_folder, debug_folder)
