import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class Encoder2D(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2D, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(32 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(32 * 32 * 32, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder2D(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder2D, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 32 * 32)
        self.convt = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(-1, 32, 32, 32)
        x_hat = torch.sigmoid(self.convt(z))
        return x_hat

class AutoregressiveFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoregressiveFlow, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        h = self.fc(x)
        scale = torch.tanh(h)
        translate = h 
        y = x * torch.exp(scale) + translate
        return y, scale.sum(dim=-1)

    def inverse(self, y):
        h = self.fc(y)
        scale = torch.tanh(h)
        translate = h
        x = (y - translate) * torch.exp(-scale)
        return x, -scale.sum(dim=-1)

class AutoregressiveVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim,num_flows):
        super(AutoregressiveVAE, self).__init__()
        self.encoder = Encoder2D(latent_dim)
        self.autoregressive_flow = AutoregressiveFlow(latent_dim, hidden_dim)
        self.decoder = Decoder2D(latent_dim)
        self.base_distribution = MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z_flow, log_det_jacobian = self.autoregressive_flow(z)
        recon_x = self.decoder(z_flow)
        return recon_x, mu, logvar, log_det_jacobian

    def loss_function(self, recon_x, x, mu, logvar, log_det_jacobian, beta=0.1):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = 0.5 * F.binary_cross_entropy(recon_x, x, reduction='sum') + 0.5 * F.mse_loss(recon_x, x, reduction='sum')
        loss = recon_loss + beta * kld - log_det_jacobian.sum()
        return loss

    def log_prob(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z_flow, log_det_jacobian = self.autoregressive_flow(z)
        log_prob_base = self.base_distribution.log_prob(z_flow)
        return log_prob_base + log_det_jacobian