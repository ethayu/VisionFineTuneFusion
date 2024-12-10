import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=768):
        """
        Autoencoder for fine-tuning the CLS token.

        Args:
            input_dim (int): Input dimension of the CLS token.
            latent_dim (int): Latent space dimension.
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 896),
            nn.ReLU(),
            nn.Linear(896, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 896),
            nn.ReLU(),
            nn.Linear(896, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z
