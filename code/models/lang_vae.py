import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageVAE(nn.Module):
    """
    VAE to compress language latent from 768 to 64 dimensions
    """
    def __init__(self, input_dim=768, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Mean and log variance for latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        """
        Encode input to latent space
        Args:
            x: (B, 1, input_dim) or (B, input_dim)
        Returns:
            mu: mean of latent distribution
            logvar: log variance of latent distribution
        """
        if len(x.shape) == 3:
            x = x.squeeze(1)  # (B, input_dim)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent to original space
        Args:
            z: (B, latent_dim)
        Returns:
            x_recon: (B, input_dim)
        """
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (B, 1, input_dim) or (B, input_dim)
        Returns:
            x_recon: reconstructed input
            mu: mean of latent distribution
            logvar: log variance of latent distribution
            z: sampled latent
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        if len(x.shape) == 3:
            x_recon = x_recon.unsqueeze(1)  # (B, 1, input_dim)
        
        return x_recon, mu, logvar, z
    
    def encode_to_latent(self, x):
        """
        Encode to latent without sampling (deterministic)
        Args:
            x: (B, 1, input_dim) or (B, input_dim)
        Returns:
            z: (B, latent_dim) - mean of latent distribution
        """
        mu, _ = self.encode(x)
        return mu
    
    def decode_from_latent(self, z):
        """
        Decode from latent
        Args:
            z: (B, latent_dim)
        Returns:
            x_recon: (B, input_dim)
        """
        return self.decode(z)


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss: reconstruction loss + KL divergence
    Args:
        x_recon: reconstructed input
        x: original input
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weight for KL divergence term
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

