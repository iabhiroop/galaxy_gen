import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class Encoder2D_Gray(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2D_Gray, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Adjusted fully connected layer
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder2D_Gray(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder2D_Gray, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 64 * 16 * 16)
        self.convt1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.convt2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.view(-1, 64, 16, 16)
        z = F.relu(self.convt1(z))
        x_hat = torch.sigmoid(self.convt2(z))  # Output is now 64x64
        return x_hat

class VAEWithResidualFlow_Gray(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_flows):
        super(VAEWithResidualFlow_Gray, self).__init__()
        self.encoder = Encoder2D_Gray(latent_dim)
        self.decoder = Decoder2D_Gray(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        loss = recon_loss + kld
        return loss