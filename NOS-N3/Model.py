import json
import pickle
from pathlib import Path

from torch.utils.data import Dataset

from Generator import *

IMAGE_SIZE = 256
GEN_SIZE = 512


class TextSegmentationDataset(Dataset):
    """
    Dataset for text segmentation with three modes:
    - on_the_fly: Generates samples from fixed pairs with random augmentations
    - random: Picks completely random image pairs each time
    - pregenerated: Uses pre-saved images/masks
    """

    def __init__(
            self,
            image_dir,
            checkpoint_dir,
            length,
            mode='on_the_fly',  # 'on_the_fly', 'random', or 'pregenerated'
            force_regenerate=False
    ):
        """
        Args:
            image_dir: Directory with all images (backgrounds and textures)
            checkpoint_dir: Directory for saving/loading dataset state
            length: Number of samples in dataset (fixed epoch length)
            mode: 'on_the_fly', 'random', or 'pregenerated'
            force_regenerate: If True, ignore existing checkpoint
        """
        self.image_dir = Path(image_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.length = length
        self.mode = mode

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load all available images
        self.image_paths = sorted([str(p) for p in self.image_dir.glob('*.jpg')] +
                                  [str(p) for p in self.image_dir.glob('*.png')])

        if len(self.image_paths) < 2:
            raise ValueError("Need at least 2 images in image_dir")

        # Load or create dataset state
        self.state_file = self.checkpoint_dir / 'dataset_state.json'
        self.data_file = self.checkpoint_dir / 'pregenerated_data.pkl'

        if mode == 'random':
            # Random mode doesn't need checkpoint for pairs
            print(f"✓ Random mode: picking from {len(self.image_paths)} images")
        elif not force_regenerate and self._load_checkpoint():
            print(f"✓ Loaded existing dataset checkpoint ({self.mode} mode)")
        else:
            print(f"✗ Creating new dataset ({self.mode} mode)")
            self._initialize_dataset()

    def _initialize_dataset(self):
        """Initialize dataset: create image pairs or pregenerate samples"""
        if self.mode == 'random':
            # No need to save pairs for random mode
            return

        # Create deterministic but different image pairs
        np.random.seed(42)
        self.image_pairs = []
        for i in range(self.length):
            # Pick two different random images
            bg_idx = np.random.randint(0, len(self.image_paths))
            text_idx = np.random.randint(0, len(self.image_paths))
            while text_idx == bg_idx:  # Ensure different images
                text_idx = np.random.randint(0, len(self.image_paths))

            self.image_pairs.append((self.image_paths[bg_idx], self.image_paths[text_idx]))

        # Save state
        state = {
            'mode': self.mode,
            'length': self.length,
            'image_pairs': self.image_pairs
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
        for idx, (bg_path, text_path) in enumerate(self.image_pairs):
            if idx % 100 == 0:
                print(f"  Generated {idx}/{self.length}")

            image, mask = self._generate_sample(bg_path, text_path)
            samples.append((image, mask))

        with open(self.data_file, 'wb') as f:
            pickle.dump(samples, f)

        # Load the data into memory after saving
        self.pregenerated_data = samples

        print(f"✓ Saved {self.length} samples to {self.data_file}")

    def _load_checkpoint(self):
        """Load existing dataset state"""
        if not self.state_file.exists():
            return False

        with open(self.state_file, 'r') as f:
            state = json.load(f)

        # Verify state matches
        if state['mode'] != self.mode or state['length'] != self.length:
            print(f"  State mismatch: saved mode={state['mode']}, length={state['length']}")
            return False

        self.image_pairs = [(str(bg), str(tx)) for bg, tx in state['image_pairs']]

        # For pregenerated mode, verify data file exists
        if self.mode == 'pregenerated':
            if not self.data_file.exists():
                print(f"  Pregenerated data file not found: {self.data_file}")
                return False

            with open(self.data_file, 'rb') as f:
                self.pregenerated_data = pickle.load(f)

            if len(self.pregenerated_data) != self.length:
                print(f"  Data length mismatch: {len(self.pregenerated_data)} != {self.length}")
                return False

        return True

    def _get_random_pair(self):
        """Pick two different random images from available images"""
        bg_idx = random.randint(0, len(self.image_paths) - 1)
        text_idx = random.randint(0, len(self.image_paths) - 1)
        while text_idx == bg_idx:
            text_idx = random.randint(0, len(self.image_paths) - 1)

        return self.image_paths[bg_idx], self.image_paths[text_idx]

    @staticmethod
    def get_random_augmentation_params():
        """
        Generate random augmentation parameters.

        Returns:
            dict: Random augmentation parameters
        """
        return {
            'angle': random.uniform(-15, 15),
            'scale': random.uniform(0.9, 1.1),
            'flip_horizontal': random.random() > 0.5,
            'flip_vertical': random.random() > 0.5
        }

    def _generate_sample(self, bg_path, text_path):
        """
        Generate a single sample with independent random augmentations.

        Args:
            bg_path: Path to background image
            text_path: Path to texture image

        Returns:
            image: np.ndarray [H, W, 3] normalized to [0, 1]
            mask: np.ndarray [H, W, 1] binary {0, 1}
        """
        # Load and initial crop
        bg = cv.imread(bg_path)
        tx = cv.imread(text_path)

        # Convert BGR to RGB
        bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)
        tx = cv.cvtColor(tx, cv.COLOR_BGR2RGB)

        bg = random_crop(bg, GEN_SIZE)
        tx = random_crop(tx, GEN_SIZE)

        bg = bg.astype(np.float32)
        tx = tx.astype(np.float32)

        # 1️⃣ Generate mask with its own augmentation
        mask = generate_text_mask(GEN_SIZE, min_len=50, max_len=100)
        mask_params = self.get_random_augmentation_params()
        mask = augment_affine(mask, **mask_params)

        # 2️⃣ Augment background with different params
        bg_params = self.get_random_augmentation_params()
        bg = augment_affine(bg, **bg_params)

        # 3️⃣ Augment texture with different params
        tx_params = self.get_random_augmentation_params()
        tx = augment_affine(tx, **tx_params)

        # 4️⃣ Apply mask to compose image
        image = apply_text_texture(bg, tx, mask)

        # 5️⃣ Center crop to final size
        start = (GEN_SIZE - IMAGE_SIZE) // 2
        end = start + IMAGE_SIZE

        image = image[start:end, start:end] / 255.0
        mask = mask[start:end, start:end]

        return image, mask

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            image: torch.Tensor [3, H, W] normalized to [0, 1]
            mask: torch.Tensor [1, H, W] binary {0, 1}
        """
        if self.mode == 'pregenerated':
            image, mask = self.pregenerated_data[idx]
        elif self.mode == 'random':
            # Pick completely random pair every time
            bg_path, text_path = self._get_random_pair()
            image, mask = self._generate_sample(bg_path, text_path)
        elif self.mode == 'on_the_fly':
            # Use fixed pair but random augmentation
            bg_path, text_path = self.image_pairs[idx]
            image, mask = self._generate_sample(bg_path, text_path)
        else:
            print("Unknown mode")
            return None, None

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # [H,W,C] -> [C,H,W]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # [H,W,1] -> [1,H,W]

        return image, mask

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    2x [Conv(3x3) -> BatchNorm -> ReLU]
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    2xConv + MaxPool
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)     # skip connection
        down = self.pool(skip)
        return skip, down


class UpBlock(nn.Module):
    """
    ConvTranspose2d + Concat + 2xConv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# ============================================================
# U-Net po priloženi shemi
# ============================================================

class UNetTextSegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------- Encoder ----------------
        self.down1 = DownBlock(3, 32)      # 256 -> 128
        self.down2 = DownBlock(32, 64)     # 128 -> 64
        self.down3 = DownBlock(64, 128)    # 64 -> 32

        # ---------------- Bottleneck ----------------
        self.bottleneck = DoubleConv(128, 256)  # 32x32

        # ---------------- Decoder ----------------
        self.up3 = UpBlock(256, 128)   # 32 -> 64
        self.up2 = UpBlock(128, 64)    # 64 -> 128
        self.up1 = UpBlock(64, 32)     # 128 -> 256

        # ---------------- Output ----------------
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip1, x = self.down1(x)   # (32, 256, 256)
        skip2, x = self.down2(x)   # (64, 128, 128)
        skip3, x = self.down3(x)   # (128, 64, 64)

        # Bottleneck
        x = self.bottleneck(x)     # (256, 32, 32)

        # Decoder
        x = self.up3(x, skip3)     # (128, 64, 64)
        x = self.up2(x, skip2)     # (64, 128, 128)
        x = self.up1(x, skip1)     # (32, 256, 256)

        # Output (brez sigmoide)
        x = self.out_conv(x)       # (1, 256, 256)
        return x
