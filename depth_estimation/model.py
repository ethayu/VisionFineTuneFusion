import torch
import torch.nn as nn
from models.dino_model import load_dino_model
from models.autoencoder import Autoencoder, PatchAutoencoder

class DepthEstimationModel(nn.Module):
    def __init__(
        self,
        model_type="dino",
        dino_model_name="facebook/dino-v2-large",
        cls_autoencoder_path=None,
        patch_autoencoder_path=None,
        device="cuda"
    ):
        """
        Depth estimation model using DiNO features.

        Args:
            model_type (str): Model type ("dino", "custom_cls", or "custom_patch").
            dino_model_name (str): Pre-trained DiNO model name.
            cls_autoencoder_path (str): Path to trained CLS autoencoder checkpoint (if using custom CLS model).
            patch_autoencoder_path (str): Path to trained Patch autoencoder checkpoint (if using custom Patch model).
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        super(DepthEstimationModel, self).__init__()
        self.dino = load_dino_model(dino_model_name).to(device).eval()

        if model_type == "custom_cls" or model_type == "custom_patch":
            assert cls_autoencoder_path or patch_autoencoder_path, "Autoencoder path(s) must be provided for custom models."

            if cls_autoencoder_path:
                self.cls_autoencoder = Autoencoder(input_dim=1024, latent_dim=512).to(device)
                self.cls_autoencoder.load_state_dict(torch.load(cls_autoencoder_path, map_location=device))
                self.cls_autoencoder.eval()
            else:
                self.cls_autoencoder = None

            if patch_autoencoder_path:
                self.patch_autoencoder = PatchAutoencoder(input_dim=768, latent_dim=256).to(device)
                self.patch_autoencoder.load_state_dict(torch.load(patch_autoencoder_path, map_location=device))
                self.patch_autoencoder.eval()
            else:
                self.patch_autoencoder = None
        else:
            self.cls_autoencoder = None
            self.patch_autoencoder = None

        self.model_type = model_type

        # Regression head for CLS and Patch models
        self.cls_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=1)
        )
        self.patch_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
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
            dino_outputs = self.dino(x)
            cls_features = dino_outputs.last_hidden_state[:, 0, :]
            patch_features = dino_outputs.last_hidden_state[:, 1:, :]

            if self.cls_autoencoder:
                _, cls_features = self.cls_autoencoder(cls_features)

            if self.patch_autoencoder:
                patch_reconstructed = []
                for patch in patch_features.unbind(dim=1):
                    _, latent_patch = self.patch_autoencoder(patch)
                    patch_reconstructed.append(latent_patch)
                patch_features = torch.stack(patch_reconstructed, dim=1)

        if self.model_type == "custom_cls":
            cls_features = cls_features.view(cls_features.size(0), 1024, 1, 1)
            return self.cls_head(cls_features)
        elif self.model_type == "custom_patch":
            patch_features = patch_features.view(patch_features.size(0), 768, 1, 1)
            return self.patch_head(patch_features)
        else:
            raise ValueError("Unsupported model type for Depth Estimation.")
