import torch
import torch.nn as nn
from models.dino_model import load_dino_model
from models.autoencoder import Autoencoder, PatchAutoencoder
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthEstimationModel(nn.Module):
    """
    Depth estimation model using DiNO features.
    
    Attributes:
        dino: DiNO model for feature extraction
        cls_autoencoder: Optional autoencoder for CLS token features
        patch_autoencoder: Optional autoencoder for patch features
        model_type: Type of model ("dino", "custom_cls", or "custom_patch")
        cls_head: Regression head for CLS token features
        patch_head: Regression head for patch features
    """
    
    def __init__(
        self,
        model_type: str = "dino",
        dino_model_name: str = "facebook/dino-v2-large",
        cls_autoencoder_path: Optional[str] = None,
        patch_autoencoder_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super(DepthEstimationModel, self).__init__()
        
        if model_type not in ["dino", "custom_cls", "custom_patch"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load DiNO model
        logger.info(f"Loading DiNO model: {dino_model_name}")
        self.dino = load_dino_model(dino_model_name).to(device)
        
        # Initialize autoencoders if needed
        self.cls_autoencoder = None
        self.patch_autoencoder = None
        
        if model_type in ["custom_cls", "custom_patch"]:
            if cls_autoencoder_path and model_type == "custom_cls":
                logger.info("Loading CLS autoencoder")
                self.cls_autoencoder = Autoencoder(input_dim=1024, latent_dim=512).to(device)
                try:
                    self.cls_autoencoder.load_state_dict(
                        torch.load(cls_autoencoder_path, map_location=device)
                    )
                    self.cls_autoencoder.eval()
                except Exception as e:
                    raise RuntimeError(f"Failed to load CLS autoencoder: {e}")
            
            if patch_autoencoder_path and model_type == "custom_patch":
                logger.info("Loading Patch autoencoder")
                self.patch_autoencoder = PatchAutoencoder(input_dim=768, latent_dim=256).to(device)
                try:
                    self.patch_autoencoder.load_state_dict(
                        torch.load(patch_autoencoder_path, map_location=device)
                    )
                    self.patch_autoencoder.eval()
                except Exception as e:
                    raise RuntimeError(f"Failed to load Patch autoencoder: {e}")
        
        self.model_type = model_type
        
        # Initialize regression heads
        self._init_regression_heads()
    
    def _init_regression_heads(self):
        """Initialize the regression heads with proper weight initialization."""
        self.cls_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1)
        )
        
        self.patch_head = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 1, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features from the input using DiNO."""
        if self.training:
            dino_outputs = self.dino(x)
        else:
            with torch.no_grad():
                dino_outputs = self.dino(x)
        
        cls_features = dino_outputs.last_hidden_state[:, 0, :]
        patch_features = dino_outputs.last_hidden_state[:, 1:, :]
        
        return cls_features, patch_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for depth estimation.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Predicted depth map of shape (B, 1, H, W)
            
        Raises:
            ValueError: If model_type is not supported
        """
        cls_features, patch_features = self.extract_features(x)
        
        if self.model_type == "custom_cls":
            if self.cls_autoencoder:
                _, cls_features = self.cls_autoencoder(cls_features)
            
            # Reshape for convolutional head
            B = cls_features.size(0)
            cls_features = cls_features.view(B, -1, 1, 1)
            cls_features = cls_features.expand(-1, -1, x.size(2)//16, x.size(3)//16)
            return self.cls_head(cls_features)
        
        elif self.model_type == "custom_patch" or self.model_type == "dino":
            if self.patch_autoencoder and self.model_type == "custom_patch":
                patch_reconstructed = []
                for patch in patch_features.unbind(dim=1):
                    _, latent_patch = self.patch_autoencoder(patch)
                    patch_reconstructed.append(latent_patch)
                patch_features = torch.stack(patch_reconstructed, dim=1)
            
            # Reshape patches to spatial dimensions
            B, N, C = patch_features.shape
            H = W = int(np.sqrt(N))
            patch_features = patch_features.transpose(1, 2).view(B, C, H, W)
            
            return self.patch_head(patch_features)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")