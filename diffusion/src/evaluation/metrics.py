import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import scipy.linalg


def compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                             mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class InceptionI3D(nn.Module):

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 5, 5),
                               stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3),
                               stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3),
                               stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),
                               stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


_fvd_model: Optional[InceptionI3D] = None
_fvd_device: Optional[str] = None


def _get_fvd_model(device: str, in_channels: int = 1) -> InceptionI3D:
    global _fvd_model, _fvd_device

    if _fvd_model is None or _fvd_device != device:
        _fvd_model = InceptionI3D(in_channels=in_channels)
        _fvd_model = _fvd_model.to(device)
        _fvd_model.eval()
        _fvd_device = device

        torch.manual_seed(42)
        for m in _fvd_model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return _fvd_model


def _extract_features(videos: torch.Tensor, model: nn.Module,
                      batch_size: int = 16) -> np.ndarray:
    features = []

    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            feat = model(batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_fvd(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    device: str = "cuda"
) -> float:
    def to_ncthw(x):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {x.dim()}D")
        if x.shape[2] <= 4 and x.shape[1] > 4:
            return x.permute(0, 2, 1, 3, 4)
        return x

    real_videos = to_ncthw(real_videos)
    generated_videos = to_ncthw(generated_videos)

    real_videos = real_videos.to(device)
    generated_videos = generated_videos.to(device)

    in_channels = real_videos.shape[1]
    model = _get_fvd_model(device, in_channels=in_channels)

    real_features = _extract_features(real_videos, model)
    gen_features = _extract_features(generated_videos, model)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_gen.ndim == 0:
        sigma_gen = np.array([[sigma_gen]])

    eps = 1e-6
    sigma_real = sigma_real + eps * np.eye(sigma_real.shape[0])
    sigma_gen = sigma_gen + eps * np.eye(sigma_gen.shape[0])

    fvd = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fvd


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cuda"
) -> float:
    try:
        from cleanfid import fid as cleanfid
        import tempfile
        import os
        from PIL import Image

        with tempfile.TemporaryDirectory() as real_dir, \
             tempfile.TemporaryDirectory() as gen_dir:

            real_np = real_images.cpu().numpy()
            gen_np = generated_images.cpu().numpy()

            if real_np.shape[1] == 1:
                real_np = np.repeat(real_np, 3, axis=1)
            if gen_np.shape[1] == 1:
                gen_np = np.repeat(gen_np, 3, axis=1)

            real_np = ((real_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            gen_np = ((gen_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            for i, img in enumerate(real_np):
                img = img.transpose(1, 2, 0)
                Image.fromarray(img).save(os.path.join(real_dir, f"{i:05d}.png"))

            for i, img in enumerate(gen_np):
                img = img.transpose(1, 2, 0)
                Image.fromarray(img).save(os.path.join(gen_dir, f"{i:05d}.png"))

            score = cleanfid.compute_fid(real_dir, gen_dir, device=device)

        return float(score)

    except ImportError:
        return _compute_fid_fallback(real_images, generated_images, device)


def _compute_fid_fallback(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str
) -> float:

    class SimpleCNN(nn.Module):
        def __init__(self, in_channels: int = 1):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 192, 3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(192, 384, 3, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        def forward(self, x):
            x = self.features(x)
            return x.view(x.size(0), -1)

    model = SimpleCNN(in_channels=real_images.shape[1]).to(device)
    model.eval()

    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    with torch.no_grad():
        real_feat = model(real_images.to(device)).cpu().numpy()
        gen_feat = model(generated_images.to(device)).cpu().numpy()

    mu_real, sigma_real = np.mean(real_feat, axis=0), np.cov(real_feat, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_feat, axis=0), np.cov(gen_feat, rowvar=False)

    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_gen.ndim == 0:
        sigma_gen = np.array([[sigma_gen]])

    return compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_lpips(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    net: str = "alex",
) -> float:
    try:
        import lpips

        loss_fn = lpips.LPIPS(net=net)

        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)

        with torch.no_grad():
            scores = loss_fn(real_images, generated_images)

        return float(scores.mean())

    except ImportError:
        raise NotImplementedError(
            "LPIPS computation requires lpips library. "
            "Install with: pip install lpips"
        )


def compute_ssim(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    from torch.nn.functional import conv2d

    def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g)

    def _ssim_single(img1: torch.Tensor, img2: torch.Tensor,
                     kernel: torch.Tensor, data_range: float) -> torch.Tensor:
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(img1.shape[1], 1, -1, -1)

        mu1 = conv2d(img1, kernel, padding=kernel.shape[-1]//2, groups=img1.shape[1])
        mu2 = conv2d(img2, kernel, padding=kernel.shape[-1]//2, groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(img1 ** 2, kernel, padding=kernel.shape[-1]//2,
                           groups=img1.shape[1]) - mu1_sq
        sigma2_sq = conv2d(img2 ** 2, kernel, padding=kernel.shape[-1]//2,
                           groups=img2.shape[1]) - mu2_sq
        sigma12 = conv2d(img1 * img2, kernel, padding=kernel.shape[-1]//2,
                         groups=img1.shape[1]) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    device = real_images.device
    kernel = _gaussian_kernel().to(device)

    ssim_scores = []
    for i in range(len(real_images)):
        score = _ssim_single(
            real_images[i:i+1],
            generated_images[i:i+1],
            kernel,
            data_range
        )
        ssim_scores.append(score.item())

    return float(np.mean(ssim_scores))


def compute_psnr(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    mse = F.mse_loss(generated_images, real_images, reduction='none')
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return float(psnr.mean().item())


def compute_temporal_consistency(
    videos: torch.Tensor,
    method: str = "flow_diff",
) -> float:
    if videos.dim() == 5:
        if videos.shape[2] == videos.shape[3]:
            pass
        else:
            videos = videos.permute(0, 2, 1, 3, 4)

    N, T, C, H, W = videos.shape

    if method == "frame_diff":
        first_diff = videos[:, 1:] - videos[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        consistency = torch.abs(second_diff).mean().item()

    elif method == "flow_diff":
        dt = videos[:, 1:] - videos[:, :-1]
        dx = videos[:, :, :, :, 2:] - videos[:, :, :, :, :-2]
        dy = videos[:, :, :, 2:, :] - videos[:, :, :, :-2, :]

        dx_t = dx[:, 1:] - dx[:, :-1]
        dy_t = dy[:, 1:] - dy[:, :-1]

        consistency = (torch.abs(dx_t).mean() + torch.abs(dy_t).mean()).item() / 2

    else:
        raise ValueError(f"Unknown method: {method}")

    return consistency


def compute_video_ssim(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    if real_videos.dim() == 5:
        if real_videos.shape[2] != real_videos.shape[3]:
            real_videos = real_videos.permute(0, 2, 1, 3, 4)
        if generated_videos.shape[2] != generated_videos.shape[3]:
            generated_videos = generated_videos.permute(0, 2, 1, 3, 4)

    N, T, C, H, W = real_videos.shape

    real_flat = real_videos.reshape(N * T, C, H, W)
    gen_flat = generated_videos.reshape(N * T, C, H, W)

    return compute_ssim(real_flat, gen_flat, data_range)


def compute_video_psnr(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    if real_videos.dim() == 5:
        if real_videos.shape[2] != real_videos.shape[3]:
            real_videos = real_videos.permute(0, 2, 1, 3, 4)
        if generated_videos.shape[2] != generated_videos.shape[3]:
            generated_videos = generated_videos.permute(0, 2, 1, 3, 4)

    N, T, C, H, W = real_videos.shape

    real_flat = real_videos.reshape(N * T, C, H, W)
    gen_flat = generated_videos.reshape(N * T, C, H, W)

    return compute_psnr(real_flat, gen_flat, data_range)
