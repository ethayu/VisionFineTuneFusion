import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder, PatchAutoencoder
from models.dino import load_dino_model, extract_cls_and_patches
from models.clip import load_clip_model, get_clip_embeddings
from models.losses import reconstruction_loss, clip_loss
from data.dataset_loader import get_data_loader
import yaml
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


def prepare_patches_for_clip(patches, target_size=(224, 224)):
    batch_size, num_patches, patch_dim = patches.shape

    patches = patches.reshape(-1, patch_dim)  # [batch_size * num_patches, patch_dim]

    patch_images = patches.reshape(-1, 32, 32)  # Assuming patch_dim is 32x32
    
    targ = 32
    
    patch_images = patch_images.unsqueeze(1).repeat(1, 3, 1, 1)  # [batch_size * num_patches, 3, 32, 32]
    # patch_images = F.interpolate(patch_images, size=target_size, mode="bilinear", align_corners=False)
    # if interpolate targ = 224
    
    # Apply sigmoid to the image (assuming this is for normalization)
    patch_images = torch.sigmoid(patch_images)
    patch_images = patch_images.view(batch_size * num_patches, 3, *(targ, targ))
    return patch_images

def train(config):
    debug = False

    # Load models
    dino = load_dino_model(config["dino_model_name"]).eval()
    clip, processor = load_clip_model(config["clip_model_name"])
    clip.eval()

    # Initialize autoencoders
    autoencoder_cls = Autoencoder(input_dim=config["cls_dim"], latent_dim=config["latent_dim"]).to(config["device"])
    autoencoder_patch = PatchAutoencoder(input_dim=config["patch_dim"], latent_dim=config["latent_dim_patch"]).to(config["device"])

    # Load dataset
    train_loader = get_data_loader(
        "data/coco/val2017", 
        batch_size=config["batch_size"], 
        annotation_file="data/coco/annotations/captions_val2017.json"
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        list(autoencoder_cls.parameters()) + list(autoencoder_patch.parameters()),
        lr=config["learning_rate"]
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

    # Loss storage for plotting
    epoch_losses = []

    # Training loop
    for epoch in range(config["epochs"]):
        autoencoder_cls.train()
        autoencoder_patch.train()
        
        epoch_total_loss = 0  # Accumulator for total epoch loss

        for images, text in tqdm(train_loader):
            images = images.to(config["device"])

            # Extract CLS and patch tokens
            with torch.no_grad():
                cls_tokens, patch_tokens = extract_cls_and_patches(dino, images)
        
            # Pass CLS tokens through CLS autoencoder
            cls_reconstructed, cls_latent = autoencoder_cls(cls_tokens)

            patch_tokens = patch_tokens.reshape(-1, patch_tokens.size(-1))  # Flatten along the patch dimension

            # Pass the flattened patches through the Patch Autoencoder
            patch_reconstructed, patch_latent = autoencoder_patch(patch_tokens)

            # Reshape the outputs back to their original shapes
            patch_reconstructed = patch_reconstructed.view(config['batch_size'], config['num_patches'], config['cls_dim'])  # [batch_size, num_patches, reconstructed_dim]
            patch_latent = patch_latent.view(config['batch_size'], config['num_patches'], config['latent_dim'])  # [batch_size, num_patches, latent_dim]
            patch_tokens = patch_tokens.view(config['batch_size'], config['num_patches'], config['patch_dim'])

            # Compute CLIP embeddings
            patch_image_embeds, _ = get_clip_embeddings(clip, processor, prepare_patches_for_clip(patch_tokens), text)
            patch_image_embeds = patch_image_embeds.view(config['batch_size'], config['num_patches'], -1)
            
            # Compute losses for CLS
            cls_rec_loss = reconstruction_loss(cls_tokens, cls_reconstructed)
            clip_image_embedding, clip_text_embedding = get_clip_embeddings(clip, processor, images, text)
            cls_clip_loss = clip_loss(cls_latent, clip_image_embedding)

            # Compute losses for patches
            patch_rec_loss = reconstruction_loss(patch_tokens, patch_reconstructed).mean()
            patch_clip_loss = clip_loss(patch_latent, patch_image_embeds).mean()

            # Total loss
            batch_loss = (
                config["lambda_cls_reconstruction"] * cls_rec_loss
                + config["lambda_cls_clip"] * cls_clip_loss
                + config["lambda_patch_reconstruction"] * patch_rec_loss
                + config["lambda_patch_clip"] * patch_clip_loss
            )

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Accumulate total loss for the epoch
            epoch_total_loss += batch_loss.item()

        # Scheduler step
        scheduler.step()

        # Average loss for the epoch
        average_loss = epoch_total_loss / len(train_loader)
        epoch_losses.append(average_loss)  # Store the epoch loss

        # Print loss
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Average Loss: {average_loss:.4f}")

        # Plot losses after each epoch
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"training_loss_epoch_{epoch + 1}.png")  # Save the plot for each epoch
        plt.show()

    # Save model
    torch.save(autoencoder_cls.state_dict(), config["save_path_cls"])
    torch.save(autoencoder_patch.state_dict(), config["save_path_patch"])
    print("Models saved!")