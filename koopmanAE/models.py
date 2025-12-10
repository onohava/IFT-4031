import torch
import torch.nn as nn


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class ConvEncoder(nn.Module):
    def __init__(self, channels, input_frames, latent_dim):
        super(ConvEncoder, self).__init__()
        self.effective_channels = channels * input_frames

        self.net = nn.Sequential(
            nn.Conv2d(self.effective_channels, 32, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        # x shape: [Batch, T, C, H, W] -> Flatten T and C
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w)

        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        # Decoder outputs 1 frame
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),  # 64x64
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.net(x)


class Dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(Dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)
        U, _, V = torch.linalg.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

    def forward(self, x):
        return self.dynamics(x)


class DynamicsBack(nn.Module):
    def __init__(self, b, omega):
        super(DynamicsBack, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        # Initialize as pseudo-inverse of forward
        self.dynamics.weight.data = torch.linalg.pinv(omega.dynamics.weight.data.t())

    def forward(self, x):
        return self.dynamics(x)


class KoopmanAE(nn.Module):
    def __init__(self, cfg):
        super(KoopmanAE, self).__init__()
        self.steps = cfg.PRED_FRAMES

        self.encoder = ConvEncoder(cfg.CHANNELS, cfg.INPUT_FRAMES, cfg.LATENT_DIM)
        self.decoder = ConvDecoder(cfg.CHANNELS, cfg.LATENT_DIM)

        self.dynamics = Dynamics(cfg.LATENT_DIM, init_scale=0.99)
        self.backdynamics = DynamicsBack(cfg.LATENT_DIM, self.dynamics)

    def forward(self, x_stack, mode='forward'):
        # x_stack: [B, Input_Frames, C, H, W]
        z = self.encoder(x_stack)

        out = []
        q = z.clone()

        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))
            out.append(self.decoder(z))
            return out

        if mode == 'backward':
            for _ in range(self.steps):
                q = self.backdynamics(q)
                out.append(self.decoder(q))
            out.append(self.decoder(z))
            return out