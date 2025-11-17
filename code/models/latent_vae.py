import torch
from torch import nn
import torch.nn.functional as F
import hydra


class MotionVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, dropout_p=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, input_dim),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # Clamp logvar to prevent extreme values that can cause NaN
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_sequence(self, x, deterministic=True):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        mu, logvar = self.encode(flat)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return z.reshape(b, t, self.latent_dim)

    @torch.no_grad()
    def decode_sequence(self, z):
        b, t, d = z.shape
        flat = z.reshape(-1, d)
        recon = self.decode(flat)
        return recon.reshape(b, t, self.input_dim)


def load_motion_vae(vae_cfg, device):
    vae = hydra.utils.instantiate(vae_cfg.model)
    state = torch.load(vae_cfg.ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    vae.load_state_dict(state)
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae

