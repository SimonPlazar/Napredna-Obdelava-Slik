import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path


# Utility function

def make_low_resolution(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Pretvori RGB sliko v low-resolution verzijo z danim faktorjem.
    Pomanjšanje in povečanje sta izvedena z bilinearno interpolacijo.

    Args:
        img: BGR slika kot numpy array (OpenCV format)
        scale: Faktor pomanjšanja (2 ali 3)

    Returns:
        BGR slika kot numpy array
    """
    assert scale in (2, 3), "Scale mora biti 2 ali 3"

    h, w = img.shape[:2]
    low_w, low_h = w // scale, h // scale

    # pomanjšanje
    img_down = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_LINEAR)

    # povečanje nazaj
    img_up = cv2.resize(img_down, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_up


def reconstruct_rgb_from_ycbcr(y_channel, cb_channel, cr_channel):
    """
    Reconstructs RGB image from Y, Cb, and Cr channels.

    Args:
        y_channel: Enhanced Y channel from the model (high resolution)
        cb_channel: Cb channel from low-res image (will be interpolated)
        cr_channel: Cr channel from low-res image (will be interpolated)

    Returns:
        RGB image
    """
    # Get target dimensions from Y channel
    target_height, target_width = y_channel.shape[:2]

    # Resize Cb and Cr channels to match Y channel dimensions
    if cb_channel.shape[:2] != (target_height, target_width):
        cb_channel = cv2.resize(cb_channel, (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR)

    if cr_channel.shape[:2] != (target_height, target_width):
        cr_channel = cv2.resize(cr_channel, (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR)

    # Ensure all channels have the same dtype
    cb_channel = cb_channel.astype(y_channel.dtype)
    cr_channel = cr_channel.astype(y_channel.dtype)

    # Merge channels back to YCbCr
    ycbcr = cv2.merge([y_channel, cb_channel, cr_channel])

    # Convert YCbCr to RGB
    rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

    return rgb


# Train Dataset for T91 images

class T91TrainDataset(Dataset):
    def __init__(self, dataset_dir, scale_factor=2, patch_size=33, min_patches=1000):
        """
        Args:
            dataset_dir: Path to directory containing images
            scale_factor: Downscaling factor for creating LR images
            patch_size: Size of patches to extract (33x33)
            min_patches: Minimum number of patches per epoch (default 1000)
        """
        # Get all image paths from directory
        dataset_path = Path(dataset_dir)
        self.image_paths = list(dataset_path.glob('*.png'))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {dataset_dir}")

        self.scale_factor = scale_factor
        self.patch_size = patch_size

        # Calculate patches per image to reach min_patches
        self.patches_per_image = max(1, min_patches // len(self.image_paths) + 1)
        self.total_patches = len(self.image_paths) * self.patches_per_image

        # Preload and preprocess all images ONCE (only 91 images in RAM)
        self.hr_images_ycbcr = []
        self.lr_images_ycbcr = []

        print(f"Preprocessing {len(self.image_paths)} images...")
        for img_path in self.image_paths:
            # Load original image
            img_rgb = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # Create low-res version
            h, w = img_rgb.shape[:2]
            img_lr = cv2.resize(img_rgb, (w // scale_factor, h // scale_factor),
                                interpolation=cv2.INTER_LINEAR)
            img_lr = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_LINEAR)

            # Convert to YCbCr and store ONCE
            hr_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            lr_ycbcr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2YCrCb)

            self.hr_images_ycbcr.append(hr_ycbcr)
            self.lr_images_ycbcr.append(lr_ycbcr)

        print(f"Loaded {len(self.image_paths)} images")
        print(f"Patches per image: {self.patches_per_image}")
        print(f"Total patches per epoch: {self.total_patches}")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Map logical index to actual image index
        img_idx = idx % len(self.hr_images_ycbcr)

        # Get corresponding image
        hr_ycbcr = self.hr_images_ycbcr[img_idx]
        lr_ycbcr = self.lr_images_ycbcr[img_idx]

        # Extract Y channel only
        hr_y = hr_ycbcr[:, :, 0]
        lr_y = lr_ycbcr[:, :, 0]

        # Random crop at same location
        h, w = hr_y.shape
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        hr_patch = hr_y[top:top + self.patch_size, left:left + self.patch_size]
        lr_patch = lr_y[top:top + self.patch_size, left:left + self.patch_size]

        # Normalize to [0, 1]
        hr_patch = hr_patch.astype(np.float32) / 255.0
        lr_patch = lr_patch.astype(np.float32) / 255.0

        # Convert to torch tensors [C, H, W]
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)
        lr_patch = torch.from_numpy(lr_patch).unsqueeze(0)

        return lr_patch, hr_patch

    def get_original(self, idx):
        """Get full YCbCr images for a given index (uses same mapping as __getitem__)"""
        img_idx = idx % len(self.hr_images_ycbcr)

        lr_ycbcr = self.lr_images_ycbcr[img_idx].copy()
        hr_ycbcr = self.hr_images_ycbcr[img_idx].copy()

        return lr_ycbcr, hr_ycbcr


# Test Dataset for Set5/Set14 images

class TestDataset:
    def __init__(self, test_dir, scale_factor=2):
        """
        Args:
            test_dir: Path to test images (Set5/Set14)
            scale_factor: Downscaling factor (2 or 3)
        """
        self.test_dir = Path(test_dir)
        self.scale_factor = scale_factor

        # Load and preprocess all images
        self.images = []
        image_paths = sorted(self.test_dir.glob('*.png'))

        for img_path in image_paths:
            # Load original
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Create LR version
            h, w = img_rgb.shape[:2]
            img_lr = cv2.resize(img_rgb, (w // scale_factor, h // scale_factor),
                                interpolation=cv2.INTER_LINEAR)
            img_lr = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_LINEAR)

            # Convert to YCbCr
            hr_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
            lr_ycbcr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0

            self.images.append({
                'lr': lr_ycbcr,
                'hr': hr_ycbcr,
                'filename': img_path.name
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            lr_ycbcr: Low-res image in YCbCr (H, W, 3), float32, range [0, 1]
            hr_ycbcr: High-res image in YCbCr (H, W, 3), float32, range [0, 1]
        """
        item = self.images[idx]
        return item['lr'], item['hr']

    def get_filename(self, idx):
        """Get filename for given index."""
        return self.images[idx]['filename']


# Model

import torch.nn as nn


class SuperResolution(nn.Module):
    def __init__(self):
        """
        Architecture:
        - Conv2D (64 filters, 9x9 kernel)
        - ReLU
        - Conv2D (32 filters, 5x5 kernel)
        - ReLU
        - Conv2D (1 filter, 5x5 kernel)
        """
        super(SuperResolution, self).__init__()

        # padding = (kernel_size - 1) // 2

        # Input: 1 channel (Y), Output: 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        # Input: 64 channels, Output: 32 channels
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        # Input: 32 channels, Output: 1 channel (Y)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor (low-res Y channel), shape [B, 1, H, W]

        Returns:
            Output tensor (high-res Y channel), shape [B, 1, H, W]
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


# Model testing functions

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.numpy()

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Clip to [0, 1] range
    y_true = np.clip(y_true, 0.0, 1.0)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    return peak_signal_noise_ratio(y_true, y_pred, data_range=1.0)


def calculate_ssim(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.numpy()

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Clip to [0, 1] range
    y_true = np.clip(y_true, 0.0, 1.0)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    return structural_similarity(y_true, y_pred, data_range=1.0)
