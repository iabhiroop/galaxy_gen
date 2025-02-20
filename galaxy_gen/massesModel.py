import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Dataset


class Encoder2D(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # Output: 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: 8x8
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder2D(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder2D, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.convt1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # Output: 16x16
        self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Output: 32x32
        self.convt3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)   # Output: 64x64

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(-1, 128, 8, 8)
        z = F.relu(self.convt1(z))
        z = F.relu(self.convt2(z))
        x_hat = torch.sigmoid(self.convt3(z))  # Sigmoid for [0, 1] range
        return x_hat

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2], dim=1)
        net_output = self.net(x1)
        log_s, t = torch.split(net_output, [x2.shape[1], x2.shape[1]], dim=1)
        s = torch.sigmoid(log_s + 2)
        y1 = x1
        y2 = s * x2 + t
        return torch.cat([y1, y2], dim=1), torch.sum(torch.log(s), dim=1)

    def inverse(self, y):
        y1, y2 = torch.split(y, [y.shape[1] // 2, y.shape[1] - y.shape[1] // 2], dim=1)
        net_output = self.net(y1)
        log_s, t = torch.split(net_output, [y2.shape[1], y2.shape[1]], dim=1)
        s = torch.sigmoid(log_s + 2)
        x1 = y1
        x2 = (y2 - t) / s
        return torch.cat([x1, x2], dim=1), -torch.sum(torch.log(s), dim=1)

class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden_dim) for _ in range(n_layers)])
        self.register_buffer('prior_mean', torch.zeros(dim))
        self.register_buffer('prior_std', torch.ones(dim))

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det += ld
        return z, log_det

class VAEFlow(nn.Module):
    def __init__(self, latent_dim, flow_hidden_dim, n_flow_layers):
        super(VAEFlow, self).__init__()
        self.encoder = Encoder2D(latent_dim)
        self.decoder = Decoder2D(latent_dim)
        self.flow = RealNVP(latent_dim, flow_hidden_dim, n_flow_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z_flow, log_det = self.flow(z)
        recon_x = self.decoder(z_flow)
        return recon_x, mu, logvar, log_det

    def loss_function(self, recon_x, x, mu, logvar, log_det, beta=1.0):
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
        mse = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        flow_loss = -log_det.sum()  # Negative log determinant

        total_loss = bce + mse + beta * kld + flow_loss
        return total_loss, bce, kld, flow_loss