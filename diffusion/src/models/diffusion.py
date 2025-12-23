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

    For multi-frame encoding with input_frames=5 and num_frames=16:
    - Sliding window (stride=1): 16 frames -> 12 latent "frames"
    - Non-overlapping (stride=5): 16 frames -> 3 latent "frames"
    - Diffusion operates on the latent sequence
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
        window_stride: int = 1,  # Stride for sliding window (1 = maximum overlap)
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
        self.window_stride = window_stride

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

            # For multi-frame KAE, calculate number of latent frames based on sliding window
            if self.is_multiframe:
                # Sliding window: num_latent_frames = (num_frames - input_frames) // stride + 1
                # E.g., 16 frames, input_frames=5, stride=1 -> (16-5)//1 + 1 = 12 latents
                # E.g., 16 frames, input_frames=5, stride=5 -> (16-5)//5 + 1 = 3 latents (non-overlapping)
                self.num_latent_frames = (num_frames - self.input_frames) // self.window_stride + 1
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
        For multi-frame KAE: sliding window of input_frames encoded together
            - stride=1: maximum overlap, most latent frames
            - stride=input_frames: non-overlapping windows

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
                # Multi-frame encoding with sliding window
                # Calculate number of windows: (T - input_frames) // stride + 1
                num_windows = (T - self.input_frames) // self.window_stride + 1

                latents = []
                for i in range(num_windows):
                    start = i * self.window_stride
                    end = start + self.input_frames
                    window = x[:, start:end]  # [B, input_frames, C, H, W]
                    z_window = self.kae_model.encode(window)  # [B, latent_dim]
                    latents.append(z_window)

                z = torch.stack(latents, dim=1)  # [B, num_windows, latent_dim]
            else:
                # Single-frame encoding: each frame encoded independently
                x_flat = x.reshape(B * T, C, H, W)
                encoded = self.kae_model.encode(x_flat)
                if isinstance(encoded, tuple):
                    # VAE returns (mu, logvar) - use reparameterization if available
                    mu, logvar = encoded
                    if hasattr(self.kae_model, 'reparameterize'):
                        z_flat = self.kae_model.reparameterize(mu, logvar)
                    else:
                        z_flat = mu  # Fallback to mu
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
        For multi-frame KAE with sliding window:
            - Each latent generates input_frames using Koopman dynamics
            - Overlapping regions are averaged for smooth transitions
            - stride=1: maximum smoothness, output frames = (T_lat - 1) * stride + input_frames

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
                # Multi-frame decoding with overlap averaging
                # Output frames = (T_lat - 1) * stride + input_frames
                output_frames = (T_lat - 1) * self.window_stride + self.input_frames

                # Get image dimensions from first decode
                z_0 = z[:, 0]
                first_frames = self.kae_model.decode_with_dynamics(z_0, num_frames=self.input_frames)
                _, _, C, H, W = first_frames.shape

                # Accumulator for averaging overlapping frames
                frame_sum = torch.zeros(B, output_frames, C, H, W, device=z.device)
                frame_count = torch.zeros(B, output_frames, 1, 1, 1, device=z.device)

                # Add first window's contribution
                frame_sum[:, :self.input_frames] += first_frames
                frame_count[:, :self.input_frames] += 1

                # Process remaining windows
                for t in range(1, T_lat):
                    z_t = z[:, t]  # [B, latent_dim]
                    frames = self.kae_model.decode_with_dynamics(z_t, num_frames=self.input_frames)

                    # Position in output where this window starts
                    start_pos = t * self.window_stride
                    end_pos = start_pos + self.input_frames

                    frame_sum[:, start_pos:end_pos] += frames
                    frame_count[:, start_pos:end_pos] += 1

                # Average overlapping regions
                x = frame_sum / frame_count.clamp(min=1)
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
