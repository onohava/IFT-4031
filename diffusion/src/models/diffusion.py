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
    """
    Video diffusion model with optional KAE latent encoding.

    Supports both single-frame and multi-frame KAE encoding:
    - Single-frame (input_frames=1): Each frame encoded independently
    - Multi-frame (input_frames>1): Windows of frames encoded together

    For multi-frame encoding with input_frames=5 and num_frames=15:
    - 15 video frames -> 3 latent "frames" (each encoding 5 video frames)
    - Diffusion operates on the compact latent sequence
    """

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

        # Determine diffusion parameters based on KAE config
        if use_kae and kae_model is not None:
            latent_dim = kae_model.latent_dim
            self.input_frames = getattr(kae_model, 'input_frames', 1)
            self.is_multiframe = self.input_frames > 1

            # Calculate latent spatial dimensions
            # For latent_dim=64: [4, 4, 4]
            # For latent_dim=512: [8, 8, 8]
            if latent_dim == 64:
                self.latent_C = 4
                self.latent_H = self.latent_W = 4
            elif latent_dim == 512:
                self.latent_C = 8
                self.latent_H = self.latent_W = 8
            else:
                # General case: try to factorize
                self.latent_C = 4
                self.latent_H = self.latent_W = int((latent_dim // self.latent_C) ** 0.5)
                if self.latent_C * self.latent_H * self.latent_W != latent_dim:
                    # Try with 8 channels
                    self.latent_C = 8
                    self.latent_H = self.latent_W = int((latent_dim // self.latent_C) ** 0.5)

            diffusion_channels = self.latent_C
            diffusion_size = self.latent_H

            # For multi-frame KAE, diffusion operates on fewer temporal "frames"
            if self.is_multiframe:
                # Number of latent frames = video_frames // input_frames
                self.num_latent_frames = num_frames // self.input_frames
            else:
                self.num_latent_frames = num_frames
        else:
            diffusion_channels = channels
            diffusion_size = image_size
            self.num_latent_frames = num_frames
            self.is_multiframe = False
            self.input_frames = 1

        self.unet = Unet3D(
            dim=dim,
            dim_mults=dim_mults,
            channels=diffusion_channels,
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=diffusion_size,
            num_frames=self.num_latent_frames,
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
        """
        Encode video frames to latent space.

        For single-frame KAE: each frame encoded independently
        For multi-frame KAE: windows of input_frames encoded together

        Args:
            x: Video tensor [B, T, C, H, W] or [B, C, T, H, W]

        Returns:
            Latent tensor [B, C_lat, T_lat, H_lat, W_lat]
        """
        # Handle [B, C, T, H, W] format
        if x.dim() == 5 and x.shape[2] == self.num_frames:
            x = x.permute(0, 2, 1, 3, 4)

        B, T, C, H, W = x.shape

        with torch.no_grad():
            if self.is_multiframe:
                # Multi-frame encoding: encode windows of input_frames
                num_windows = T // self.input_frames
                # Trim to exact multiple
                x_trimmed = x[:, :num_windows * self.input_frames]

                latents = []
                for i in range(num_windows):
                    start = i * self.input_frames
                    end = start + self.input_frames
                    window = x_trimmed[:, start:end]  # [B, input_frames, C, H, W]
                    z_window = self.kae_model.encode(window)  # [B, latent_dim]
                    latents.append(z_window)

                z = torch.stack(latents, dim=1)  # [B, num_windows, latent_dim]
            else:
                # Single-frame encoding: each frame encoded independently
                x_flat = x.reshape(B * T, C, H, W)
                encoded = self.kae_model.encode(x_flat)
                if isinstance(encoded, tuple):
                    z_flat = encoded[0]
                else:
                    z_flat = encoded
                z = z_flat.view(B, T, -1)  # [B, T, latent_dim]

        # Reshape for 3D diffusion: [B, C_lat, T_lat, H_lat, W_lat]
        T_lat = z.shape[1]
        z = z.view(B, T_lat, self.latent_C, self.latent_H, self.latent_W)
        z = z.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        z = self._normalize_latent(z)
        return z

    def _decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent space back to video frames.

        For single-frame KAE: each latent decoded to one frame
        For multi-frame KAE: each latent decoded to input_frames frames using Koopman dynamics

        Args:
            z: Latent tensor [B, C_lat, T_lat, H_lat, W_lat]

        Returns:
            Video tensor [B, C, T, H, W]
        """
        z = self._denormalize_latent(z)

        B, C_latent, T_lat, H_latent, W_latent = z.shape
        latent_dim = C_latent * H_latent * W_latent

        z = z.permute(0, 2, 1, 3, 4)  # [B, T_lat, C, H, W]
        z = z.reshape(B, T_lat, latent_dim)  # [B, T_lat, latent_dim]

        with torch.no_grad():
            if self.is_multiframe and hasattr(self.kae_model, 'decode_with_dynamics'):
                # Multi-frame decoding: use Koopman dynamics to generate multiple frames per latent
                all_frames = []
                for t in range(T_lat):
                    z_t = z[:, t]  # [B, latent_dim]
                    # Generate input_frames frames using Koopman dynamics
                    frames = self.kae_model.decode_with_dynamics(z_t, num_frames=self.input_frames)
                    all_frames.append(frames)  # [B, input_frames, C, H, W]

                # Concatenate all frame sequences
                x = torch.cat(all_frames, dim=1)  # [B, T_lat * input_frames, C, H, W]
            else:
                # Single-frame decoding: each latent -> one frame
                z_flat = z.reshape(B * T_lat, latent_dim)
                x_flat = self.kae_model.decode(z_flat)
                C, H, W = x_flat.shape[1:]
                x = x_flat.view(B, T_lat, C, H, W)

        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
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
