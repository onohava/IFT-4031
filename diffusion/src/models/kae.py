import torch
import torch.nn as nn


class KoopmanAutoencoder(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: list = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        encoder_layers = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.flatten_size = hidden_dims[-1] * (image_size // (2 ** len(hidden_dims))) ** 2
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)

        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        for i in range(len(hidden_dims_rev) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_rev[i],
                        hidden_dims_rev[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims_rev[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims_rev[-1],
                    input_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self._hidden_dims = hidden_dims

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), self._hidden_dims[-1],
                   self.image_size // (2 ** len(self._hidden_dims)),
                   self.image_size // (2 ** len(self._hidden_dims)))
        x_recon = self.decoder(h)
        return x_recon

    def koopman_step(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        x_recon = self.decode(z)
        return {"z": z, "x_recon": x_recon}

    def forward_sequence(self, x_seq: torch.Tensor) -> dict:
        B, T, C, H, W = x_seq.shape

        x_flat = x_seq.view(B * T, C, H, W)
        z_flat = self.encode(x_flat)
        z_seq = z_flat.view(B, T, -1)

        z_pred = []
        z_t = z_seq[:, 0]
        for t in range(T):
            z_pred.append(z_t)
            z_t = self.koopman_step(z_t)
        z_pred = torch.stack(z_pred, dim=1)

        x_recon_flat = self.decode(z_flat)
        x_recon = x_recon_flat.view(B, T, C, H, W)

        return {
            "z_seq": z_seq,
            "z_pred": z_pred,
            "x_recon": x_recon,
        }
