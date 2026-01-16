import random
import cv2 as cv
import numpy as np

IMAGE_SIZE = 256
GEN_SIZE = 512


def random_crop(image, size):
    # Handle both 2D and 3D arrays
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return image[y:y + size, x:x + size]


def generate_random_mask(height, width, num_strokes=4, thickness=8):
    """
    Generate a random mask with black strokes on white background.

    Args:
        height: Image height
        width: Image width
        num_strokes: Number of random strokes to draw
        thickness: thickness

    Returns:
        Binary mask (1 = valid pixel, 0 = missing/hole)
    """
    # Create white mask (all ones)
    mask = np.ones((height, width), dtype=np.uint8)

    # Draw random strokes
    pt1 = (np.random.randint(0, width), np.random.randint(0, height))
    for _ in range(num_strokes):
        # Random start and end points
        pt2 = (np.random.randint(0, width), np.random.randint(0, height))

        # Draw black line (value 0)
        cv.line(mask, pt1, pt2, color=0, thickness=thickness)
        pt1 = pt2

    return mask


def apply_mask(image, mask):
    masked_img = image.copy()
    masked_img[mask == 0] = 0
    return masked_img