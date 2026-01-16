import json
import pickle
from pathlib import Path

from torch.utils.data import Dataset
import numpy as np
import torch
import cv2 as cv
import random

from Generator import *

IMAGE_SIZE = 256


class InpaintingDataset(Dataset):
    """
    Dataset for image inpainting with three modes:
    - on_the_fly: Uses fixed images with random masks generated each time
    - random: Picks completely random images each time
    - pregenerated: Uses pre-saved images/masks
    """

    def __init__(
            self,
            image_dir,
            checkpoint_dir,
            length,
            image_size=256,
            num_strokes=(4, 5),
            thickness=(7, 9),
            mode='on_the_fly',
            force_regenerate=False
    ):
        """
        Args:
            image_dir: Directory with all images
            checkpoint_dir: Directory for saving/loading dataset state
            length: Number of samples in dataset
            image_size: Target image size
            num_strokes: Tuple (min, max) number of strokes
            thickness: Tuple (min, max) stroke thickness
            mode: 'on_the_fly', 'random', or 'pregenerated'
            force_regenerate: If True, ignore existing checkpoint
        """
        self.image_dir = Path(image_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.length = length
        self.image_size = image_size
        self.num_strokes = num_strokes
        self.thickness = thickness
        self.mode = mode

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load all available images
        self.image_paths = sorted([str(p) for p in self.image_dir.glob('*.jpg')] +
                                  [str(p) for p in self.image_dir.glob('*.png')])

        if len(self.image_paths) == 0:
            raise ValueError("No images found in image_dir")

        # Load or create dataset state
        self.state_file = self.checkpoint_dir / 'dataset_state.json'
        self.data_file = self.checkpoint_dir / 'pregenerated_data.pkl'

        if mode == 'random':
            print(f"✓ Random mode: picking from {len(self.image_paths)} images")
        elif not force_regenerate and self._load_checkpoint():
            print(f"✓ Loaded existing dataset checkpoint ({self.mode} mode)")
        else:
            print(f"✗ Creating new dataset ({self.mode} mode)")
            self._initialize_dataset()

    def _initialize_dataset(self):
        """Initialize dataset: assign images or pregenerate samples"""
        if self.mode == 'random':
            return

        # Create deterministic image assignments
        np.random.seed(42)
        self.assigned_images = []
        for i in range(self.length):
            img_idx = np.random.randint(0, len(self.image_paths))
            self.assigned_images.append(self.image_paths[img_idx])

        # Save state
        state = {
            'mode': self.mode,
            'length': self.length,
            'image_size': self.image_size,
            'num_strokes': self.num_strokes,
            'thickness': self.thickness,
            'assigned_images': self.assigned_images
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Pregenerate if needed
        if self.mode == 'pregenerated':
            print(f"Pregenerating {self.length} samples...")
            self._pregenerate_samples()

    def _pregenerate_samples(self):
        """Generate and save all samples to disk"""
        samples = []
        np.random.seed(42)  # Fix seed for reproducible masks

        for idx, img_path in enumerate(self.assigned_images):
            if idx % 100 == 0:
                print(f"  Generated {idx}/{self.length}")

            image, mask, masked_image = self._generate_sample(img_path, fixed_seed=idx)
            samples.append((image, mask, masked_image))

        with open(self.data_file, 'wb') as f:
            pickle.dump(samples, f)

        self.pregenerated_data = samples
        print(f"✓ Saved {self.length} samples to {self.data_file}")

    def _load_checkpoint(self):
        """Load existing dataset state"""
        if not self.state_file.exists():
            return False

        with open(self.state_file, 'r') as f:
            state = json.load(f)

        # Verify state matches
        if (state['mode'] != self.mode or
                state['length'] != self.length or
                state['image_size'] != self.image_size):
            print(f"  State mismatch")
            return False

        self.assigned_images = [str(p) for p in state['assigned_images']]

        # For pregenerated mode, load data
        if self.mode == 'pregenerated':
            if not self.data_file.exists():
                print(f"  Pregenerated data file not found")
                return False

            with open(self.data_file, 'rb') as f:
                self.pregenerated_data = pickle.load(f)

            if len(self.pregenerated_data) != self.length:
                print(f"  Data length mismatch")
                return False

        return True

    def _generate_sample(self, img_path, fixed_seed=None):
        """
        Generate a single inpainting sample.

        Args:
            img_path: Path to image
            fixed_seed: Optional seed for reproducible masks (used in pregenerated mode)

        Returns:
            image: np.ndarray [H, W, 3] normalized to [0, 1]
            mask: np.ndarray [H, W, 1] binary {0, 1}
            masked_image: np.ndarray [H, W, 3] image with mask applied
        """
        # Load image
        img = cv.imread(img_path)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Random crop
        img_rgb = random_crop(img_rgb, self.image_size)

        # Generate mask (with optional fixed seed)
        if fixed_seed is not None:
            np.random.seed(fixed_seed)

        num_strokes = np.random.randint(self.num_strokes[0], self.num_strokes[1])
        thickness = np.random.randint(self.thickness[0], self.thickness[1])
        mask = generate_random_mask(self.image_size, self.image_size,
                                    num_strokes=num_strokes,
                                    thickness=thickness)

        # Apply mask
        masked_img = apply_mask(img_rgb, mask)

        # Normalize
        image = img_rgb.astype(np.float32) / 255.0
        masked_image = masked_img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        # Add channel dimension to mask
        mask = mask[:, :, np.newaxis]

        return image, mask, masked_image

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            masked_image: torch.Tensor [3, H, W] - image with holes
            mask: torch.Tensor [1, H, W] - binary mask (0=hole, 1=valid)
            image: torch.Tensor [3, H, W] - original image (ground truth)
        """
        if self.mode == 'pregenerated':
            image, mask, masked_image = self.pregenerated_data[idx]
        elif self.mode == 'random':
            img_path = random.choice(self.image_paths)
            image, mask, masked_image = self._generate_sample(img_path)
        elif self.mode == 'on_the_fly':
            img_path = self.assigned_images[idx]
            image, mask, masked_image = self._generate_sample(img_path)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Convert to torch tensors [H,W,C] -> [C,H,W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        masked_image = torch.from_numpy(masked_image).permute(2, 0, 1).float()

        return masked_image, mask, image


# NEVRONSKA MREŽA
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """
    Partial Convolution layer as described in:
    Liu et al., Image Inpainting for Irregular Holes using Partial Convolutions
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=True
        )

        # konvolucijski operator z enicami (kot buffer!)
        self.register_buffer(
            "conv_all_ones",
            torch.ones(1, 1, kernel_size, kernel_size)
        )

        self.kernel_area = kernel_size * kernel_size

    def forward(self, x, mask):
        """
        x    : [B, C, H, W]
        mask : [B, 1, H, W]
        """

        # 1️⃣ maskiranje vhoda
        x = x * mask

        # 2️⃣ klasična konvolucija
        out = self.conv(x)

        # 3️⃣ posodobitev maske (brez gradientov)
        with torch.no_grad():
            mask_sum = F.conv2d(
                mask,
                self.conv_all_ones,
                stride=self.conv.stride,
                padding=self.conv.padding
            )
            mask_new = (mask_sum > 0).float()

        # 4️⃣ normalizacija
        out = out * self.kernel_area / (mask_sum + 1e-8)
        out = out * mask_new

        return out, mask_new


class PartialBlock(nn.Module):
    """
    PartialConv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.pconv = PartialConv2d(
            in_channels, out_channels,
            kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        x = self.relu(x)
        return x, mask


class UNetInpainting(nn.Module):
    """
    Reduced U-Net with Partial Convolutions

    Input :
        x    : [B, 3, 256, 256]
        mask : [B, 1, 256, 256]

    Output:
        out  : [B, 3, 256, 256]
    """

    def __init__(self):
        super().__init__()

        # ================= ENCODER =================
        # [B, 3, 256, 256] → [B, 32, 256, 256]
        self.enc1 = PartialBlock(3, 32, stride=1)

        # [B, 32, 256, 256] → [B, 64, 128, 128]
        self.enc2 = PartialBlock(32, 64, stride=2)

        # [B, 64, 128, 128] → [B, 128, 64, 64]
        self.enc3 = PartialBlock(64, 128, stride=2)

        # [B, 128, 64, 64] → [B, 128, 32, 32]
        self.enc4 = PartialBlock(128, 128, stride=2)

        # ================= DECODER =================
        # [B, 128, 32, 32] → [B, 128, 64, 64]
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        # concat with enc3 → [B, 256, 64, 64] → [B, 64, 64, 64]
        self.dec1 = PartialBlock(128 + 128, 64)

        # [B, 64, 64, 64] → [B, 64, 128, 128]
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        # concat with enc2 → [B, 128, 128, 128] → [B, 32, 128, 128]
        self.dec2 = PartialBlock(64 + 64, 32)

        # ================= OUTPUT =================
        # [B, 32, 128, 128] → [B, 32, 256, 256]
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        # [B, 32, 256, 256] → [B, 3, 256, 256]
        self.out_conv = PartialConv2d(32, 3)

    def forward(self, x, mask):
        """
        x    : [B, 3, 256, 256]
        mask : [B, 1, 256, 256]
        """

        # =============== ENCODER =================
        e1, m1 = self.enc1(x, mask)
        # e1 : [B, 32, 256, 256]
        # m1 : [B, 1, 256, 256]

        e2, m2 = self.enc2(e1, m1)
        # e2 : [B, 64, 128, 128]
        # m2 : [B, 1, 128, 128]

        e3, m3 = self.enc3(e2, m2)
        # e3 : [B, 128, 64, 64]
        # m3 : [B, 1, 64, 64]

        e4, m4 = self.enc4(e3, m3)
        # e4 : [B, 128, 32, 32]
        # m4 : [B, 1, 32, 32]

        # =============== DECODER =================
        d = self.up1(e4)  # [B, 128, 64, 64]
        m = self.up1(m4)  # [B, 1, 64, 64]
        d = torch.cat([d, e3], dim=1)  # [B, 256, 64, 64]
        m = torch.max(m, m3)  # [B, 1, 64, 64]
        d, m = self.dec1(d, m)  # [B, 64, 64, 64]

        d = self.up2(d)  # [B, 64, 128, 128]
        m = self.up2(m)  # [B, 1, 128, 128]
        d = torch.cat([d, e2], dim=1)  # [B, 128, 128, 128]
        m = torch.max(m, m2)  # [B, 1, 128, 128]
        d, m = self.dec2(d, m)  # [B, 32, 128, 128]

        # =============== OUTPUT =================
        d = self.up3(d)  # [B, 32, 256, 256]
        m = self.up3(m)  # [B, 1, 256, 256]
        out, _ = self.out_conv(d, m)  # [B, 3, 256, 256]
        out = torch.sigmoid(out)

        return out


def loss_fn(predicted, image, mask):
    hole = 1 - mask
    return ((predicted - image).abs() * hole).sum() / (hole.sum() + 1e-8)
