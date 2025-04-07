import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super().__init__()
        self.mask = mask
        
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, reverse=False):
        masked_x = x * self.mask
        
        if not reverse:
            scale = self.scale_net(masked_x) * (1 - self.mask)
            translation = self.translation_net(masked_x) * (1 - self.mask)
            z = masked_x + (1 - self.mask) * (x * torch.exp(scale) + translation)
            log_det = torch.sum(scale * (1 - self.mask), dim=1)
        else:
            scale = self.scale_net(masked_x) * (1 - self.mask)
            translation = self.translation_net(masked_x) * (1 - self.mask)
            z = masked_x + (1 - self.mask) * ((x - translation) * torch.exp(-scale))
            log_det = -torch.sum(scale * (1 - self.mask), dim=1)
            
        return z, log_det

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.flows = nn.ModuleList()
        
        for i in range(num_layers):
            mask = torch.ones(input_dim)
            # alternate which dimensions are masked
            mask[::2] = 0 if i % 2 == 0 else 1
            self.flows.append(CouplingLayer(input_dim, hidden_dim, mask))
    
    def forward(self, x, reverse=False):
        log_det_sum = 0
        
        if not reverse:
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_sum += log_det
        else:
            for flow in reversed(self.flows):
                x, log_det = flow(x, reverse=True)
                log_det_sum += log_det
                
        return x, log_det_sum

# --- Modified Encoder and Decoder with minimal convolution ---
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            # Single conv layer: 64x64x1 -> 32x32x16 using a smaller 3x3 kernel.
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        # The feature map is now 16 x 32 x 32 = 16*1024 features.
        self.fc_mu = nn.Linear(16 * 32 * 32, latent_dim)
        self.fc_var = nn.Linear(16 * 32 * 32, latent_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Map latent vector back to 16 x 32 x 32 feature map.
        self.fc = nn.Linear(latent_dim, 16 * 32 * 32)
        self.deconv = nn.Sequential(
            # Single deconvolution: 32x32x16 -> 64x64x1 using a matching 3x3 kernel.
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 16, 32, 32)
        x = self.deconv(x)
        return x

class VAEFlow(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, num_flows=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.flow = RealNVP(latent_dim, hidden_dim, num_flows)
        self.normal = Normal(0, 1)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode to get latent parameters.
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # Apply normalizing flow.
        z_flow, log_det = self.flow(z)
        
        # Decode.
        x_recon = self.decoder(z_flow)
        
        # Reconstruction loss.
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # KL divergence (note: log_det from flow is subtracted).
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) - log_det.sum()
        
        return x_recon, recon_loss, kl_loss
    
    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.flow.flows[0].mask.size(0))
            z_inv, _ = self.flow(z, reverse=True)
            samples = self.decoder(z_inv)
        return samples

def train_step(model, optimizer, x):
    optimizer.zero_grad()
    x_recon, recon_loss, kl_loss = model(x)
    total_loss = recon_loss + kl_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), recon_loss.item(), kl_loss.item()
