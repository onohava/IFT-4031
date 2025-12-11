import torch
import torch.nn as nn


class VideoVAE(nn.Module):

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

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

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

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), self._hidden_dims[-1],
                   self.image_size // (2 ** len(self._hidden_dims)),
                   self.image_size // (2 ** len(self._hidden_dims)))
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return {"z": z, "x_recon": x_recon, "mu": mu, "logvar": logvar}

    def forward_sequence(self, x_seq: torch.Tensor) -> dict:
        B, T, C, H, W = x_seq.shape

        x_flat = x_seq.view(B * T, C, H, W)
        mu_flat, logvar_flat = self.encode(x_flat)
        z_flat = self.reparameterize(mu_flat, logvar_flat)

        z_seq = z_flat.view(B, T, -1)
        mu_seq = mu_flat.view(B, T, -1)
        logvar_seq = logvar_flat.view(B, T, -1)

        x_recon_flat = self.decode(z_flat)
        x_recon = x_recon_flat.view(B, T, C, H, W)

        return {
            "z_seq": z_seq,
            "x_recon": x_recon,
            "mu": mu_seq,
            "logvar": logvar_seq,
        }


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 1e-4) -> dict:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }
