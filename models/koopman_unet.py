import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard DDPM Time Embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class KoopmanOperatorBlock(nn.Module):
    """
    The heart of the Koopman Autoencoder.
    Instead of standard non-linear convolutions or Attention, this block approximates
    the linear evolution K(t) * z in the latent space.
    """

    def __init__(self, channels, time_emb_dim, norm_groups=32):
        super().__init__()
        self.channels = channels

        # Rank of the evolution matrix.
        # We ensure rank is at least 4 but scales with channels.
        self.rank = max(4, channels // 4)

        self.to_U = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * self.rank)
        )

        self.to_V = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * self.rank)
        )

        self.to_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )

        self.norm = nn.GroupNorm(norm_groups, channels)

    def forward(self, x, t_emb):
        b, c, h, w = x.shape

        # 1. Generate Dynamic Koopman Operator K(t) = U * V^T
        U = self.to_U(t_emb).view(b, c, self.rank)
        V = self.to_V(t_emb).view(b, self.rank, c)
        bias = self.to_bias(t_emb).view(b, c, 1, 1)

        # 2. Apply Linear Evolution K(t) * x
        # Treat spatial pixels as independent trajectories
        x_flat = x.view(b, c, -1)  # (B, C, N)

        # Low rank associativity: U * (V^T * x)
        projected = torch.matmul(V, x_flat)  # (B, Rank, N)
        evolved = torch.matmul(U, projected)  # (B, C, N)

        evolved = evolved.view(b, c, h, w)

        # 3. Add dynamic bias and residual
        return self.norm(x + evolved + bias)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        # Ensure groups doesn't exceed channels
        groups = min(groups, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class KoopmanUnet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32, 64, 128, 128),
            norm_num_groups=32,
            layers_per_block=2,
            down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D"
            ),
            up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D"
            ),
    ):
        super().__init__()

        self.channels = in_channels
        self.out_channels = out_channels

        # Initial projection
        init_dim = block_out_channels[0]
        self.init_conv = nn.Conv2d(in_channels, init_dim, 3, padding=1)

        # Time Embeddings
        time_dim = init_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # --- Channel Tracking for Skips ---
        # We must simulate the Down-Pass to know exactly what skip dimensions
        # will be available for the Up-Pass.
        skip_channel_counts = [init_dim]  # Result of init_conv
        current_trace_channel = init_dim

        # --- DOWN BLOCKS ---
        output_channel = init_dim
        current_channel = init_dim

        for i, out_channel in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            block_type = down_block_types[i]
            use_koopman = "Attn" in block_type

            resnet_blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                resnet_blocks.append(ResnetBlock(
                    current_channel,
                    out_channel,
                    time_emb_dim=time_dim,
                    groups=norm_num_groups
                ))
                current_channel = out_channel
                # Track: Every ResNet output is stored as a skip
                skip_channel_counts.append(out_channel)

            koopman_layer = None
            if use_koopman:
                koopman_layer = KoopmanOperatorBlock(current_channel, time_dim, norm_groups=norm_num_groups)

            downsample = None
            if not is_last:
                downsample = nn.Conv2d(current_channel, current_channel, 3, stride=2, padding=1)
                # Track: Downsample output is stored as a skip
                skip_channel_counts.append(current_channel)

            self.downs.append(nn.ModuleList([resnet_blocks, koopman_layer, downsample]))

        # --- MID BLOCK (Bottleneck) ---
        mid_dim = block_out_channels[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=norm_num_groups)
        self.mid_koopman = KoopmanOperatorBlock(mid_dim, time_dim, norm_groups=norm_num_groups)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=norm_num_groups)

        # --- UP BLOCKS ---
        reversed_block_out_channels = list(reversed(block_out_channels))

        # current_channel is now at the bottleneck size

        for i, out_channel in enumerate(reversed_block_out_channels):
            is_last = i == len(reversed_block_out_channels) - 1
            block_type = up_block_types[i]
            use_koopman = "Attn" in block_type

            resnet_blocks = nn.ModuleList()

            # Diffusers UpBlock has (layers_per_block + 1) ResNets
            for _ in range(layers_per_block + 1):
                # We pop the matching skip channel from our simulation stack
                # This fixes the mismatch error by using exact history
                skip_ch = skip_channel_counts.pop()

                res_in = current_channel + skip_ch

                resnet_blocks.append(ResnetBlock(
                    res_in,
                    out_channel,
                    time_emb_dim=time_dim,
                    groups=norm_num_groups
                ))
                current_channel = out_channel

            koopman_layer = None
            if use_koopman:
                koopman_layer = KoopmanOperatorBlock(current_channel, time_dim, norm_groups=norm_num_groups)

            upsample = None
            if not is_last:
                upsample = nn.ConvTranspose2d(current_channel, current_channel, 4, 2, 1)

            self.ups.append(nn.ModuleList([resnet_blocks, koopman_layer, upsample]))

        self.final_norm = nn.GroupNorm(norm_num_groups, block_out_channels[0])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        # Store skip connections
        skips = [x]

        # --- DOWN ---
        for blocks, koopman, downsample in self.downs:
            for block in blocks:
                x = block(x, t)
                skips.append(x)  # Store output of every ResNet for skip

            if koopman is not None:
                x = koopman(x, t)

            if downsample is not None:
                x = downsample(x)
                skips.append(x)  # Store downsampled version

        # --- MID ---
        x = self.mid_block1(x, t)
        x = self.mid_koopman(x, t)
        x = self.mid_block2(x, t)

        # --- UP ---
        for blocks, koopman, upsample in self.ups:
            for block in blocks:
                # Retrieve the skip connection
                skip = skips.pop()

                # Handle spatial dimension mismatch (common in U-Net with odd dims)
                if skip.shape != x.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='nearest')

                x = torch.cat((x, skip), dim=1)
                x = block(x, t)

            if koopman is not None:
                x = koopman(x, t)

            if upsample is not None:
                x = upsample(x)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)


# --- Verification ---
if __name__ == "__main__":
    print("Initializing Config-Aligned Koopman U-Net...")

    # Configuration from User JSON
    config = {
        "in_channels": 1,
        "out_channels": 1,
        "block_out_channels": (32, 64, 128, 128),
        "norm_num_groups": 32,
        "layers_per_block": 2,
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    }

    model = KoopmanUnet(**config)

    # Test Input (Batch 2, 1 Channel, 32x32)
    dummy_x = torch.randn(2, 1, 32, 32)
    dummy_t = torch.randint(0, 1000, (2,))

    output = model(dummy_x, dummy_t)

    print(f"Input shape: {dummy_x.shape}")
    print(f"Output shape: {output.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")

    assert output.shape == dummy_x.shape, "Output shape mismatch!"
    print("Success.")
