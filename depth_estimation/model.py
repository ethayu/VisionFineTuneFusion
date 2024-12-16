import torch
import torch.nn as nn
import logging
from typing import Optional
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Standard library imports
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Project imports
from models import load_dino_model, CLSAutoencoder, PatchAutoencoder  # Import from models/__init__.py
from utils import load_checkpoint, compute_cosine_similarity  # Import from utils/__init__.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DepthEstimationModel(nn.Module):
    def __init__(
        self,
        model_type: str = "dino",
        dino_model_name: str = "facebook/dinov2-large",
        cls_autoencoder_path: Optional[str] = None,
        patch_autoencoder_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super(DepthEstimationModel, self).__init__()
        
        if model_type not in ["dino", "custom_cls", "custom_patch"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Loading DiNO model: {dino_model_name}")
        self.dino = load_dino_model(dino_model_name).to(device)
        
        self.cls_autoencoder = None
        self.patch_autoencoder = None
        
        if model_type in ["custom_cls", "custom_patch"]:
            if cls_autoencoder_path and model_type == "custom_cls":
                logger.info("Loading CLS autoencoder")
                self.cls_autoencoder = CLSAutoencoder(
                    input_dim=1024,
                    latent_dim=512
                ).to(device)
                state_dict = torch.load(cls_autoencoder_path, map_location=device)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.cls_autoencoder.load_state_dict(state_dict)
                self.cls_autoencoder.eval()
            
            if patch_autoencoder_path and model_type == "custom_patch":
                logger.info("Loading Patch autoencoder")
                self.patch_autoencoder = PatchAutoencoder(
                    input_dim=1024,
                    latent_dim=512
                ).to(device)
                state_dict = torch.load(patch_autoencoder_path, map_location=device)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.patch_autoencoder.load_state_dict(state_dict)
                self.patch_autoencoder.eval()
        
        self.model_type = model_type
        self._init_regression_heads()
    
    def _init_regression_heads(self):
        self.num_upsamples = 4  # To go from 14x14 to 224x224
        
        # Initialize both heads with appropriate input channels
        if self.model_type in ["custom_cls", "dino"]:
            cls_layers = []
            current_channels = 1024 if self.model_type == "dino" else 512
            
            for i in range(self.num_upsamples):
                out_channels = current_channels // 2 if i < self.num_upsamples - 1 else 64
                cls_layers.extend([
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                ])
                current_channels = out_channels
            
            cls_layers.extend([
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1)
            ])
            
            self.cls_head = nn.Sequential(*cls_layers)
        
        if self.model_type in ["custom_patch", "dino"]:
            patch_layers = []
            current_channels = 1024 if self.model_type == "dino" else 512
            
            for i in range(self.num_upsamples):
                out_channels = current_channels // 2 if i < self.num_upsamples - 1 else 64
                patch_layers.extend([
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                ])
                current_channels = out_channels
            
            patch_layers.extend([
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1)
            ])
            
            self.patch_head = nn.Sequential(*patch_layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor):
        with torch.no_grad():
            dino_outputs = self.dino(x)
        
        cls_features = dino_outputs.last_hidden_state[:, 0, :]  # [B, 1024]
        patch_features = dino_outputs.last_hidden_state[:, 1:, :]  # [B, N, 1024]
        
        return cls_features, patch_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_features, patch_features = self.extract_features(x)
        
        if self.model_type == "custom_cls":
            if self.cls_autoencoder:
                _, cls_features = self.cls_autoencoder(cls_features)
            
            B = cls_features.size(0)
            cls_features = cls_features.view(B, -1, 1, 1)
            cls_features = cls_features.expand(-1, -1, 14, 14)
            out = self.cls_head(cls_features)
            
        elif self.model_type == "custom_patch":
            if self.patch_autoencoder:
                patch_reconstructed = []
                for patch in patch_features.unbind(dim=1):
                    _, latent_patch = self.patch_autoencoder(patch)
                    patch_reconstructed.append(latent_patch)
                patch_features = torch.stack(patch_reconstructed, dim=1)
            
            B, N, C = patch_features.shape
            H = W = int(N ** 0.5)
            patch_features = patch_features.transpose(1, 2).view(B, C, H, W)
            out = self.patch_head(patch_features)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Ensure output size matches target size
        if out.size(-1) != 224 or out.size(-2) != 224:
            out = nn.functional.interpolate(
                out, size=(224, 224), mode='bilinear', align_corners=True
            )
        return out
