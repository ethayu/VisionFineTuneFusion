import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models.autoencoder import Autoencoder, PatchAutoencoder
from models.dino import load_dino_model, extract_cls_and_patches
from models.clip import load_clip_model, get_clip_embeddings
from models.losses import reconstruction_loss, clip_loss
from data.dataset_loader import get_data_loader
import yaml
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


def prepare_patches_for_clip(images, target_size=(224, 224), device='cuda'):
    batch_size = images.shape[0]
    patch_size = 16

    num_patches = 256

    patches = F.unfold(images, kernel_size=(patch_size, patch_size), stride=patch_size)

    patches = patches.reshape(batch_size, 3, 256, 14, 14) 

    patches = patches.reshape(batch_size * 256, 3, 14, 14)   

    upsampled_patches = F.interpolate(patches.view(-1, 3, 14, 14), size=(224, 224), mode='bilinear')
    return upsampled_patches

def train(config):
    debug = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load models
    dino = load_dino_model(config["dino_model_name"]).eval().to(device)
    clip, processor = load_clip_model(config["clip_model_name"])
    clip.eval().to(device)

    # Initialize autoencoders
    autoencoder_cls = Autoencoder(input_dim=config["cls_dim"], latent_dim=config["latent_dim"]).to(device)
    autoencoder_patch = PatchAutoencoder(input_dim=config["patch_dim"], latent_dim=config["latent_dim_patch"]).to(device)

    # Load dataset
    train_loader = get_data_loader(
        "data/coco/train2017", 
        batch_size=config["batch_size"], 
        annotation_file="data/coco/annotations/captions_train2017.json"
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        list(autoencoder_cls.parameters()) + list(autoencoder_patch.parameters()),
        lr=config["learning_rate"]
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

    # Loss storage for plotting (per weight update)
    update_cls_rec_losses = []
    update_clip_image_losses = []
    update_patch_rec_losses = []
    update_patch_clip_losses = []
    update_clip_text_losses = []

    # Training loop
    for epoch in range(config["epochs"]):
        autoencoder_cls.train()
        autoencoder_patch.train()
        cnt = 0
        epoch_total_loss = 0  # Accumulator for total epoch loss


        for images, text in tqdm(train_loader):
            images = images.to(device)

            # Extract CLS and patch tokens
            with torch.no_grad():
                cls_tokens, patch_tokens = extract_cls_and_patches(dino, images)
                cls_tokens, patch_tokens = cls_tokens.to(device), patch_tokens.to(device)
        
            # Pass CLS tokens through CLS autoencoder
            cls_reconstructed, cls_latent = autoencoder_cls(cls_tokens)

            patch_tokens = patch_tokens.reshape(-1, patch_tokens.size(-1)).to(device)  # Flatten along the patch dimension

            # Pass the flattened patches through the Patch Autoencoder
            patch_reconstructed, patch_latent = autoencoder_patch(patch_tokens)

            # Reshape the outputs back to their original shapes
            patch_reconstructed = patch_reconstructed.view(config['batch_size'], config['num_patches'], config['cls_dim'])  # [batch_size, num_patches, reconstructed_dim]
            patch_latent = patch_latent.view(config['batch_size'], config['num_patches'], config['latent_dim'])  # [batch_size, num_patches, latent_dim]
            patch_tokens = patch_tokens.view(config['batch_size'], config['num_patches'], config['patch_dim'])

            # Compute CLIP embeddings
            patch_image_embeds, _ = get_clip_embeddings(
                clip, processor, prepare_patches_for_clip(images, device=device), text, device=device
            )

            patch_image_embeds = patch_image_embeds.view(config['batch_size'], config['num_patches'], -1).to(device)
            
            # Compute losses for CLS
            cls_rec_loss = reconstruction_loss(cls_tokens, cls_reconstructed)
            clip_image_embedding, clip_text_embedding = get_clip_embeddings(clip, processor, images, text, device=device)
            cls_image_loss = clip_loss(cls_latent, clip_image_embedding)
            cls_text_loss = clip_loss(cls_latent,clip_text_embedding)

            # Compute losses for patches
            patch_rec_loss = reconstruction_loss(patch_tokens, patch_reconstructed).mean()
            patch_clip_loss = clip_loss(patch_latent, patch_image_embeds).mean()

            # Total loss
            batch_loss = (
                config['lambda_clip_text'] * cls_text_loss +
                config["lambda_cls_reconstruction"] * cls_rec_loss
                + config["lambda_clip_image"] * cls_image_loss
                + config["lambda_patch_reconstruction"] * patch_rec_loss
                + config["lambda_patch_clip"] * patch_clip_loss
            )

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Accumulate total loss for the epoch
            epoch_total_loss += batch_loss.item()

            # Store the individual losses for each batch
            update_cls_rec_losses.append(cls_rec_loss.item())
            update_clip_image_losses.append(cls_image_loss.item())
            update_clip_text_losses.append(cls_text_loss.item())
            update_patch_rec_losses.append(patch_rec_loss.item())
            update_patch_clip_losses.append(patch_clip_loss.item())

        steps_now = len(update_cls_rec_losses)

        torch.save(autoencoder_cls.state_dict(), f'{config["save_path_cls"]}_{steps_now}.pth')
        torch.save(autoencoder_patch.state_dict(), f'{config["save_path_patch"]}_{steps_now}.pth')
        print("Models saved!")

        # Scheduler step
        scheduler.step()

        # Average loss for the epoch
        average_loss = epoch_total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Average Loss: {average_loss:.4f}")

        # Plot all losses after training
        plt.figure(figsize=(10, 5))
        plt.plot(update_cls_rec_losses, label="CLS Reconstruction Loss")
        plt.plot(update_clip_text_losses, label="CLS Text Loss")
        plt.plot(update_clip_image_losses, label="CLS Image Loss")
        plt.plot(update_patch_rec_losses, label="Patch Reconstruction Loss")
        plt.plot(update_patch_clip_losses, label="Patch CLIP Loss")
        plt.xlabel("Weight Updates")
        plt.ylabel("Loss")
        plt.title("Training Losses Over Weight Updates")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_losses_weight_updates.png")  # Save the plot for weight updates
        plt.show()


if __name__ == "__main__":
    with open("training/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train(config)
