import torch
import torch.nn as nn
from models.dino_model import load_dino_model
from models.autoencoder import Autoencoder

class DepthEstimationModel(nn.Module):
    def __init__(self, model_type="dino", dino_model_name="facebook/dino-v2-large", autoencoder_path=None, device="cuda"):
        """
        Depth estimation model using DiNO features.

        Args:
            model_type (str): Model type ("dino" or "custom").
            dino_model_name (str): Pre-trained DiNO model name.
            autoencoder_path (str): Path to trained autoencoder checkpoint (if using custom model).
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        super(DepthEstimationModel, self).__init__()
        self.dino = load_dino_model(dino_model_name).to(device).eval()

        if model_type == "custom":
            assert autoencoder_path, "Autoencoder path must be provided for custom model."
            self.autoencoder = Autoencoder(input_dim=1024, latent_dim=512).to(device)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
            self.autoencoder.eval()
        else:
            self.autoencoder = None

        # Regression head
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=1)  # Single channel for depth map
        )

    def forward(self, x):
        """
        Forward pass for depth estimation.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predicted depth map.
        """
        with torch.no_grad():
            features = self.dino(x).last_hidden_state[:, 0, :]  # CLS token features
            if self.autoencoder:
                _, features = self.autoencoder(features)

        features = features.view(features.size(0), 1024, 1, 1)  # Reshape for convolution
        depth_map = self.head(features)
        return depth_map
