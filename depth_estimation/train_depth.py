import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from depth_estimation.model import DepthEstimationModel
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image

class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.depth_filenames = sorted(os.listdir(depth_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            depth = self.target_transform(depth)

        return image, depth


def train_depth_model(
    data_loader,
    model,
    epochs,
    device,
    lr=0.0001,
    checkpoint_path="depth_estimation/checkpoints/model.pth"
):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, depth_maps in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, depth_maps = images.to(device), depth_maps.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, depth_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = DepthDataset(
        image_dir="data/depth/images",
        depth_dir="data/depth/depth_maps",
        transform=transform,
        target_transform=target_transform
    )

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DepthEstimationModel(
        model_type="custom_patch",
        dino_model_name="facebook/dino-v2-large",
        patch_autoencoder_path="checkpoints/patch_autoencoder.pth",
        device="cuda"
    )

    train_depth_model(data_loader, model, epochs=10, device="cuda")