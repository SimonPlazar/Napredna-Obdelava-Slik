import numpy as np
import random
import cv2


# Shape generation functions

def generate_triangle(width, height):
    points = np.array(
        [[random.randint(int(-width * 0.1), int(width * 1.1)), random.randint(int(-height * 0.1), int(height * 1.1))]
         for _ in range(3)], dtype=np.int32)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return "polygon", (points, color)


def generate_rectangle(width, height):
    points = np.array(
        [[random.randint(int(-width * 0.1), int(width * 1.1)), random.randint(int(-height * 0.1), int(height * 1.1))]
         for _ in range(4)], dtype=np.int32)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return "polygon", (points, color)


def generate_ellipse(width, height):
    center = (random.randint(int(-width * 0.1), int(width * 1.1)),
              random.randint(int(-height * 0.1), int(height * 1.1)))
    axes = (random.randint(50, width // 2), random.randint(50, height // 2))
    angle = random.randint(0, 360)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return "ellipse", (center, axes, angle, color)


def generate_star(width, height):
    center = (random.randint(0, width), random.randint(0, height))
    points = np.array(
        [[random.randint(int(-width * 0.1), int(width * 1.1)), random.randint(int(-height * 0.1), int(height * 1.1))]
         for _ in range(random.choice([4, 5]))], dtype=np.int32)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return "star", (center, points, color)


def generate_image(width, height, num_triangles, num_rectangles, num_ellipses, num_stars):
    bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    shapes = []

    for _ in range(num_triangles):
        shapes.append(generate_triangle(width, height))
    for _ in range(num_rectangles):
        shapes.append(generate_rectangle(width, height))
    for _ in range(num_ellipses):
        shapes.append(generate_ellipse(width, height))
    for _ in range(num_stars):
        shapes.append(generate_star(width, height))

    random.shuffle(shapes)

    for shape_type, params in shapes:
        if shape_type == "polygon":
            points, color = params
            cv2.fillPoly(img, [points], color, lineType=cv2.LINE_AA)
        elif shape_type == "ellipse":
            center, axes, angle, color = params
            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
        elif shape_type == "star":
            center, points, color = params
            for point in points:
                cv2.line(img, center, tuple(point), color, 1, lineType=cv2.LINE_AA)

    return img


# NOISE FUNCTIONS

def noise_add(image, sigma=None):
    if sigma is None:
        sigma = random.uniform(0.1, 0.3)
    noisy = image.astype(np.float32) + np.random.normal(0, sigma * 255, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def noise_mul(image, sigma=None):
    if sigma is None:
        sigma = random.uniform(0.1, 0.3)
    noisy = image.astype(np.float32) * np.random.normal(1, sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy
