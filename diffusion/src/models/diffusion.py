import torch
import torch.nn as nn
from typing import Optional

try:
    from video_diffusion_pytorch import Unet3D, GaussianDiffusion
    VIDEO_DIFFUSION_AVAILABLE = True
except ImportError:
    VIDEO_DIFFUSION_AVAILABLE = False
    Unet3D = None
    GaussianDiffusion = None


class VideoDiffusionModel(nn.Module):

    def __init__(
        self,
        image_size: int = 64,
        channels: int = 1,
        num_frames: int = 16,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4, 8),
        timesteps: int = 1000,
        use_kae: bool = False,
        kae_model: Optional[nn.Module] = None,
    ):
        super().__init__()

        if not VIDEO_DIFFUSION_AVAILABLE:
            raise ImportError(
                "video-diffusion-pytorch not installed. "
                "Install with: pip install video-diffusion-pytorch"
            )

        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.use_kae = use_kae
        self.kae_model = kae_model

        self.register_buffer('latent_min', torch.tensor(-0.5))
        self.register_buffer('latent_max', torch.tensor(0.5))

        if self.kae_model is not None:
            self.kae_model.eval()
            for param in self.kae_model.parameters():
                param.requires_grad = False

        if use_kae and kae_model is not None:
            latent_dim = kae_model.latent_dim
            self.latent_C = 4
            self.latent_H = self.latent_W = int((latent_dim // self.latent_C) ** 0.5)
            diffusion_channels = self.latent_C
            diffusion_size = self.latent_H
        else:
            diffusion_channels = channels
            diffusion_size = image_size

        self.unet = Unet3D(
            dim=dim,
            dim_mults=dim_mults,
            channels=diffusion_channels,
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=diffusion_size,
            num_frames=num_frames,
            channels=diffusion_channels,
            timesteps=timesteps,
        )

    def set_latent_normalization(self, latent_min: float, latent_max: float):
        self.latent_min = torch.tensor(latent_min, device=self.latent_min.device)
        self.latent_max = torch.tensor(latent_max, device=self.latent_max.device)

    def _normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        z_clamped = torch.clamp(z, self.latent_min, self.latent_max)
        z_normalized = (z_clamped - self.latent_min) / (self.latent_max - self.latent_min + 1e-8)
        return z_normalized

    def _denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        z_denormalized = z * (self.latent_max - self.latent_min) + self.latent_min
        return z_denormalized

    def _encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.shape[2] == self.num_frames:
            x = x.permute(0, 2, 1, 3, 4)

        B, T, C, H, W = x.shape

        with torch.no_grad():
            x_flat = x.reshape(B * T, C, H, W)
            encoded = self.kae_model.encode(x_flat)
            if isinstance(encoded, tuple):
                z_flat = encoded[0]
            else:
                z_flat = encoded
            z = z_flat.view(B, T, -1)

        latent_dim = z.shape[-1]
        C_latent = 4
        H_latent = W_latent = int((latent_dim // C_latent) ** 0.5)
        assert C_latent * H_latent * W_latent == latent_dim, f"Latent dim {latent_dim} not factorizable"

        z = z.view(B, T, C_latent, H_latent, W_latent)
        z = z.permute(0, 2, 1, 3, 4)

        z = self._normalize_latent(z)
        return z

    def _decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        z = self._denormalize_latent(z)

        B, C_latent, T, H_latent, W_latent = z.shape
        latent_dim = C_latent * H_latent * W_latent

        z = z.permute(0, 2, 1, 3, 4)
        z = z.reshape(B, T, latent_dim)

        with torch.no_grad():
            z_flat = z.reshape(B * T, latent_dim)
            x_flat = self.kae_model.decode(z_flat)
            C, H, W = x_flat.shape[1:]
            x = x_flat.view(B, T, C, H, W)

        x = x.permute(0, 2, 1, 3, 4)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_kae and self.kae_model is not None:
            x = self._encode_to_latent(x)

        return self.diffusion(x)

    @torch.no_grad()
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        samples = self.diffusion.sample(batch_size=batch_size)

        if self.use_kae and self.kae_model is not None:
            samples = self._decode_from_latent(samples)

        return samples


VideoDiffusion = VideoDiffusionModel
