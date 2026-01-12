import random
import string
import cv2 as cv
import numpy as np

IMAGE_SIZE = 256
GEN_SIZE = 512

FONTS = [
    cv.FONT_HERSHEY_SIMPLEX,
    cv.FONT_HERSHEY_COMPLEX,
    cv.FONT_HERSHEY_DUPLEX,
    cv.FONT_HERSHEY_TRIPLEX
]

def random_text(min_len=5, max_len=15):
    chars = string.ascii_letters + string.digits + " " * 10
    return "".join(random.choice(chars) for _ in range(random.randint(min_len, max_len)))


def random_crop(image, size):
    # Handle both 2D and 3D arrays
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return image[y:y + size, x:x + size]


#%%
def generate_text_mask(size, min_len=5, max_len=15):
    mask = np.zeros((size, size, 1), dtype=np.float32)

    text = random_text(min_len=min_len, max_len=max_len)
    # print("Text:", text)
    font = random.choice(FONTS)
    scale = random.uniform(0.8, 2.5)
    thickness = random.randint(1, 4)

    # Split text into lines
    words = text.split()
    lines = []
    words_per_line = random.randint(2, 4)
    for i in range(0, len(words), words_per_line):
        lines.append(' '.join(words[i:i + words_per_line]))

    # Calculate line height with spacing
    line_height = cv.getTextSize('A', font, scale, thickness)[0][1] + 10
    total_height = len(lines) * line_height

    # Starting y position (centered vertically)
    start_y = (size - total_height) // 2 + line_height

    # Create RGB mask for drawing
    mask_rgb = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw each line
    for i, line in enumerate(lines):
        (tw, th), _ = cv.getTextSize(line, font, scale, thickness)

        # Center horizontally
        x = (size - tw) // 2
        y = start_y + i * line_height

        # Ensure within bounds
        if y > 0 and y < size and x < size:
            cv.putText(
                mask_rgb, line, (max(0, x), y),
                font, scale, (255, 255, 255),
                thickness, cv.LINE_AA
            )

    # Convert to single channel mask
    mask = mask_rgb[:, :, 0:1].astype(np.float32) / 255.0
    return mask
#%%
def augment_affine(image, angle, scale, flip_horizontal=False, flip_vertical=False):
    h, w = image.shape[:2]
    M = cv.getRotationMatrix2D((w // 2, h // 2), angle, scale)

    interp = cv.INTER_NEAREST if image.shape[-1] == 1 else cv.INTER_LINEAR

    out = cv.warpAffine(
        image, M, (w, h),
        flags=interp,
        # borderMode=cv.BORDER_REFLECT_101
    )

    # Horizontal flip (flip along vertical axis)
    if flip_horizontal:
        out = cv.flip(out, 1)

    # Vertical flip (flip along horizontal axis)
    if flip_vertical:
        out = cv.flip(out, 0)

    if out.ndim == 2:
        out = out[:, :, None]

    return out
#%%
def apply_text_texture(bg, text_img, mask):
    bg = bg.astype(np.float32)
    text_img = text_img.astype(np.float32)

    result = (1.0 - mask) * bg + mask * text_img
    return result

#%%
def generate_sample(bg_path, text_path):
    bg = cv.imread(bg_path)
    tx = cv.imread(text_path)

    bg = random_crop(bg, GEN_SIZE)
    tx = random_crop(tx, GEN_SIZE)

    bg = bg.astype(np.float32)
    tx = tx.astype(np.float32)

    # 1️⃣ maska
    mask = generate_text_mask(GEN_SIZE, min_len=50, max_len=100)

    # 2️⃣ skupni augmentacijski parametri
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.9, 1.1)
    flip_h = random.random() > 0.5
    flip_v = random.random() > 0.5

    mask = augment_affine(mask, angle, scale, flip_h, flip_v)
    bg = augment_affine(bg, angle, scale, flip_h, flip_v)
    tx = augment_affine(tx, angle, scale, flip_h, flip_v)

    # 3️⃣ aplikacija maske
    image = apply_text_texture(bg, tx, mask)

    # 4️⃣ center crop + normalizacija
    start = (GEN_SIZE - IMAGE_SIZE) // 2
    end = start + IMAGE_SIZE

    image = image[start:end, start:end] / 255.0
    mask = mask[start:end, start:end]
    return image, mask