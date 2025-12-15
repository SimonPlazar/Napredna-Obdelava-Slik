import torch
import torch.nn as nn


class FilterBranch(nn.Module):
    def __init__(self, use_bias: bool = True):
        super(FilterBranch, self).__init__()
        self.R_conv = nn.Conv2d(1, 8, kernel_size=11, padding=5, bias=use_bias)
        self.G_conv = nn.Conv2d(1, 8, kernel_size=11, padding=5, bias=use_bias)
        self.B_conv = nn.Conv2d(1, 8, kernel_size=11, padding=5, bias=use_bias)

    def forward(self, x):
        # x: (B, 3, H, W)
        R = x[:, 0:1, :, :]  # (B, 1, H, W)
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]

        Rf = self.R_conv(R)  # (B, 8, H, W)
        Gf = self.G_conv(G)
        Bf = self.B_conv(B)

        # Združimo rezultate v (B, 3, 8, H, W)
        filtered = torch.stack([Rf, Gf, Bf], dim=1)
        return filtered  # (B, 3, 8, H, W)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1,
                 dropout_rate=0.1):
        super(ResNetBlock, self).__init__()
        out_channels = out_channels or in_channels  # če ni določeno, ohrani enako št. kanalov

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout(out)
        out += residual

        out = self.relu(out)
        return out


class WeightBranch(nn.Module):
    def __init__(self, ):
        super(WeightBranch, self).__init__()

        self.resnet1 = ResNetBlock(3, 32)
        self.resnet2 = ResNetBlock(32, 32)
        self.resnet3 = ResNetBlock(32, 32)

        self.final_conv = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        logints = self.final_conv(x)  # -> (B, 8, H, W)
        weights = self.softmax(logints)  # -> (B, 8, H, W)
        return weights


class EdgePreservingDenoiser(nn.Module):
    def __init__(self):
        super(EdgePreservingDenoiser, self).__init__()
        self.filter_branch = FilterBranch()
        self.weight_branch = WeightBranch()

    def forward(self, x):
        # x shape: (B, 3, H, W)
        filtered = self.filter_branch(x)  # (B, 3, 8, H, W)
        weights = self.weight_branch(x)  # (B, 8, H, W)

        # Združi veje za vsak barvni kanal
        # Razširi weights za 3 kanale: (B, 1, 8, H, W)
        weights = weights.unsqueeze(1)  # (B, 1, 8, H, W)

        # Pomnožimo in seštejemo po 8 filtrih
        output = (filtered * weights).sum(dim=2)  # (B, 3, H, W)
        output = torch.clamp(output, 0.0, 1.0)

        return output


# Dataset
from torch.utils.data import Dataset
import torch
import numpy as np
from Generator import generate_image, noise_add


class SyntheticNoiseDataset(Dataset):
    """
    In-memory dataset optimized for fast DataLoader batching.
    - If `clean_images`/`noisy_images` are provided they can be:
        * stacked torch.Tensor shape [N, C, H, W]
        * numpy array shape [N, C, H, W] or [N, H, W, C]
        * iterable of per-sample arrays/tensors
    - Generated samples are stacked once and normalized to 0..1.
    """

    def __init__(self, clean_images=None, noisy_images=None,
                 num_samples=1000, noise_fn=noise_add, noise_range=(0.1, 0.3),
                 normalize=True, max_value=255.0, to_channels_last=False):
        self.normalize = normalize
        self.max_value = float(max_value)
        self.to_channels_last = to_channels_last

        if clean_images is not None and noisy_images is not None:
            self.clean = self._stack_and_prepare(clean_images)
            self.noisy = self._stack_and_prepare(noisy_images)
        else:
            clean_list = []
            noisy_list = []
            for _ in range(num_samples):
                img = generate_image(256, 256, 5, 5, 5, 5)  # HWC numpy
                noisy = noise_fn(img, np.random.uniform(*noise_range))
                clean_list.append(img)
                noisy_list.append(noisy)
            self.clean = self._stack_and_prepare(clean_list)
            self.noisy = self._stack_and_prepare(noisy_list)

        assert self.clean.shape[0] == self.noisy.shape[0], "Mismatched dataset sizes"

    def __len__(self):
        return self.clean.shape[0]

    def __getitem__(self, idx):
        # Return direct indexed tensors (fast). Do not clone here.
        return self.noisy[idx], self.clean[idx]

    def _as_tensor(self, img):
        t = torch.as_tensor(img)
        # convert HWC -> CHW if needed
        if t.ndim == 3 and t.shape[2] in (1, 3):
            t = t.permute(2, 0, 1).contiguous()
        elif t.ndim == 2:
            t = t.unsqueeze(0).contiguous()
        return t.float()

    def _stack_and_prepare(self, arr):
        # If input already stacked tensor
        if isinstance(arr, torch.Tensor) and arr.ndim == 4:
            t = arr.float().contiguous()
        else:
            # handle numpy stacked array shape [N, H, W, C] or iterable
            if isinstance(arr, np.ndarray) and arr.ndim == 4:
                items = [self._as_tensor(arr[i]) for i in range(arr.shape[0])]
            else:
                items = [self._as_tensor(x) for x in arr]
            t = torch.stack(items, dim=0)  # [N, C, H, W]

        if self.normalize:
            # detect 0..255-like data
            if t.max().item() > 1.1:
                t = t / self.max_value
            t = t.clamp(0.0, 1.0)

        # optional memory format for faster conv on GPU
        if self.to_channels_last:
            t = t.contiguous(memory_format=torch.channels_last)

        return t.contiguous()


# =============================================================================
# DATASET UTILITIES
# =============================================================================
from pathlib import Path
import pickle


def load_dataset_arrays(checkpoint_dir, dataset_name):
    """
    Naloži dataset arrays iz pickle file.

    Args:
        checkpoint_dir: Direktorij s checkpointi
        dataset_name: Ime dataseta ('train' ali 'val')

    Returns:
        (clean_images, noisy_images) ali (None, None) če ne obstaja
    """
    checkpoint_dir = Path(checkpoint_dir)
    dataset_path = checkpoint_dir / f"{dataset_name}_dataset.pkl"

    if dataset_path.exists():
        print(f"📂 Loading {dataset_name} dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        clean_images = data['clean']
        noisy_images = data['noisy']
        print(f"✅ Loaded {len(clean_images)} samples")
        return clean_images, noisy_images
    else:
        print(f"⚠️ No {dataset_name} dataset found at {dataset_path}")
        return None, None


def save_dataset_arrays(checkpoint_dir, dataset_name, dataset, overwrite=False):
    """
    Save dataset tensors to `checkpoint_dir/{dataset_name}_dataset.pkl`.
    Accepts `dataset.clean`/`dataset.noisy` as either:
      - a stacked torch.Tensor shape [N, C, H, W], or
      - an iterable/list of per-sample tensors/arrays.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    dataset_path = checkpoint_dir / f"{dataset_name}_dataset.pkl"

    if dataset_path.exists() and not overwrite:
        print(f"⚠️ Dataset already exists at {dataset_path}. Set overwrite=True to replace.")
        return

    def _ensure_stacked(tensors):
        # If already a tensor, use it; otherwise try to stack an iterable
        if isinstance(tensors, torch.Tensor):
            t = tensors
        else:
            t = torch.stack(list(tensors), dim=0)
        # detach, move to cpu and make contiguous for safe pickling
        return t.detach().cpu().contiguous()

    clean_t = _ensure_stacked(dataset.clean)
    noisy_t = _ensure_stacked(dataset.noisy)

    data = {
        'clean': clean_t,
        'noisy': noisy_t
    }

    print(f"💾 Saving {dataset_name} dataset to {dataset_path}")
    with open(dataset_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Saved {clean_t.shape[0]} samples")


# Eval model utility

def calculate_psnr(clean, denoised, max_value=1.0, eps=1e-10):
    """Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((clean - denoised) ** 2)
    mse = torch.clamp(mse, min=eps)
    max_val_tensor = torch.tensor(max_value, device=mse.device, dtype=mse.dtype)
    return 20.0 * torch.log10(max_val_tensor) - 10.0 * torch.log10(mse)


def calculate_snr(clean, noisy, eps=1e-10):
    """Signal-to-Noise Ratio (SNR) in dB."""
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((clean - noisy) ** 2)
    noise_power = torch.clamp(noise_power, min=eps)
    return 10.0 * torch.log10(signal_power / noise_power)


def evaluate_model(model, dataloader, device, max_batches=None):
    """
    Evalvacija modela na validation setu.
    Vrne: avg_loss, avg_snr, avg_psnr
    """
    model.eval()
    total_loss = 0.0
    total_snr = 0.0
    total_psnr = 0.0
    num_batches = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            denoised = model(noisy)
            loss = criterion(denoised, clean)

            # Metrike
            total_loss += loss.item()

            # SNR in PSNR za vsak sample v batch
            for i in range(clean.size(0)):
                snr = calculate_snr(clean[i], denoised[i]).item()
                psnr = calculate_psnr(clean[i], denoised[i]).item()
                total_snr += snr
                total_psnr += psnr

            num_batches += 1
            if max_batches and num_batches >= max_batches:
                break

    num_samples = num_batches * dataloader.batch_size
    avg_loss = total_loss / num_batches
    avg_snr = total_snr / num_samples
    avg_psnr = total_psnr / num_samples

    model.train()
    return avg_loss, avg_snr, avg_psnr
