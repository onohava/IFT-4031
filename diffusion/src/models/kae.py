"""
Wrapper for KoopmanAE to provide encode/decode interface for diffusion training.

This module adapts the KoopmanAE architecture from koopmanAE/models.py to work
with the VideoDiffusionModel which expects encode() and decode() methods.

Supports both single-frame (INPUT_FRAMES=1) and multi-frame (INPUT_FRAMES>1) encoding.
For multi-frame, the encoder takes INPUT_FRAMES frames and produces 1 latent vector.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

# Add koopmanAE to path for imports
KOOPMAN_AE_PATH = Path(__file__).parent.parent.parent.parent / "koopmanAE"
if str(KOOPMAN_AE_PATH) not in sys.path:
    sys.path.insert(0, str(KOOPMAN_AE_PATH))

# Import from koopmanAE/models.py (using importlib to avoid naming conflict)
import importlib.util
_spec = importlib.util.spec_from_file_location("koopman_models", KOOPMAN_AE_PATH / "models.py")
_koopman_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_koopman_module)
KoopmanAE = _koopman_module.KoopmanAE
ConvEncoder = _koopman_module.ConvEncoder
ConvDecoder = _koopman_module.ConvDecoder


@dataclass
class KAEConfig:
    """Configuration for KoopmanAE matching the original Config class interface."""
    CHANNELS: int = 1
    INPUT_FRAMES: int = 1
    PRED_FRAMES: int = 5
    LATENT_DIM: int = 64
    DATASET_NAME: str = "MovingMNIST"
    IMG_SIZE: int = 64


class KoopmanAutoencoder(nn.Module):
    """
    Wrapper around KoopmanAE that provides encode/decode interface
    expected by VideoDiffusionModel.

    Key responsibilities:
    1. Provide encode(x) -> z and decode(z) -> x methods
    2. Handle data normalization: diffusion uses [-1, 1], KAE uses [0, 1]
    3. Expose latent_dim and input_frames attributes
    4. Support both single-frame and multi-frame encoding
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,  # Kept for API compatibility
        image_size: int = 64,
        dataset_name: str = "MovingMNIST",
        input_frames: int = 1,
        pred_frames: int = 5,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.input_frames = input_frames  # Track for multi-frame encoding

        # Create config for KoopmanAE
        self.cfg = KAEConfig(
            CHANNELS=input_channels,
            INPUT_FRAMES=input_frames,
            PRED_FRAMES=pred_frames,
            LATENT_DIM=latent_dim,
            DATASET_NAME=dataset_name,
            IMG_SIZE=image_size,
        )

        # Initialize the underlying KoopmanAE
        self.kae = KoopmanAE(self.cfg)

        # Store encoder and decoder references for direct access
        self.encoder_net = self.kae.encoder
        self.decoder_net = self.kae.decoder
        self.dynamics = self.kae.dynamics  # Koopman dynamics matrix K

    def _normalize_for_kae(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from diffusion space [-1, 1] to KAE space [0, 1]."""
        return (x + 1.0) / 2.0

    def _denormalize_from_kae(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from KAE space [0, 1] to diffusion space [-1, 1]."""
        return x * 2.0 - 1.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.

        For single-frame (input_frames=1):
            x: [B, C, H, W] -> z: [B, latent_dim]

        For multi-frame (input_frames>1):
            x: [B, input_frames, C, H, W] -> z: [B, latent_dim]

        Args:
            x: Input tensor in range [-1, 1] (diffusion convention)

        Returns:
            z: Latent tensor of shape [B, latent_dim]
        """
        # Convert from [-1, 1] to [0, 1] for KAE
        x = self._normalize_for_kae(x)

        # KoopmanAE encoder expects [B, T, C, H, W]
        if x.dim() == 4:
            # Single frame case: [B, C, H, W] -> [B, 1, C, H, W]
            x = x.unsqueeze(1)
        # For multi-frame: x is already [B, T, C, H, W]

        # Use the encoder from koopmanAE/models.py
        z = self.encoder_net(x)  # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Latent tensor of shape [B, latent_dim]

        Returns:
            x: Reconstructed tensor of shape [B, C, H, W]
               in range [-1, 1] (diffusion convention)
        """
        # Use the decoder from koopmanAE/models.py
        x = self.decoder_net(z)  # [B, C, H, W]

        # KoopmanAE outputs logits for MNIST, apply sigmoid
        x = torch.sigmoid(x)

        # Convert from [0, 1] to [-1, 1] for diffusion
        x = self._denormalize_from_kae(x)
        return x

    def decode_with_dynamics(self, z: torch.Tensor, num_frames: int = 5) -> torch.Tensor:
        """
        Decode latent and use Koopman dynamics K to generate multiple frames.

        This uses the learned Koopman operator: z_{t+1} = K @ z_t

        Args:
            z: Initial latent tensor [B, latent_dim]
            num_frames: Number of frames to generate

        Returns:
            Video tensor [B, num_frames, C, H, W] in range [-1, 1]
        """
        frames = []
        q = z.clone()

        for _ in range(num_frames):
            # Decode current latent to frame
            frame = self.decoder_net(q)
            frame = torch.sigmoid(frame)
            frame = self._denormalize_from_kae(frame)
            frames.append(frame)

            # Apply Koopman dynamics to get next latent
            q = self.dynamics(q)

        return torch.stack(frames, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        z = self.encode(x)
        return self.decode(z)


def load_koopman_ae_from_checkpoint(
    checkpoint_path: str,
    latent_dim: int = 64,
    input_channels: int = 1,
    image_size: int = 64,
    dataset_name: str = "MovingMNIST",
    input_frames: int = 1,
    pred_frames: int = 5,
    device: str = "cpu",
) -> KoopmanAutoencoder:
    """
    Load KoopmanAutoencoder wrapper from a koopmanAE checkpoint.

    The checkpoint format from koopmanAE/main.py is a raw state_dict
    with keys like 'encoder.net.0.weight', 'decoder.fc.weight', etc.

    Args:
        checkpoint_path: Path to the .pth file
        latent_dim: Latent dimension (must match checkpoint)
        input_channels: Number of input channels
        image_size: Input image size
        dataset_name: Dataset name for architecture selection
        input_frames: Number of input frames for encoder
        pred_frames: Number of prediction frames
        device: Device to load model on

    Returns:
        Loaded KoopmanAutoencoder wrapper
    """
    # Create the wrapper
    model = KoopmanAutoencoder(
        input_channels=input_channels,
        latent_dim=latent_dim,
        image_size=image_size,
        dataset_name=dataset_name,
        input_frames=input_frames,
        pred_frames=pred_frames,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix if present
            state_dict = {
                k.replace("model.", "").replace("kae.", ""): v
                for k, v in state_dict.items()
            }
        else:
            # Raw state_dict from torch.save(model.state_dict(), ...)
            state_dict = checkpoint
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    # Load into the underlying KoopmanAE
    model.kae.load_state_dict(state_dict, strict=True)

    return model
