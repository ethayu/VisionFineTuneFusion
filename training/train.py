import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder, PatchAutoencoder
from models.dino_model import load_dino_model, extract_cls_and_patches
from models.clip_model import load_clip_model, get_clip_embeddings
from models.losses import reconstruction_loss, clip_loss
from data.dataset_loader import get_data_loader
import yaml

def train(config):
    # Load models
    dino = load_dino_model(config["dino_model_name"]).eval()
    clip, processor = load_clip_model(config["clip_model_name"])
    clip.eval()

    # Initialize autoencoders
    autoencoder_cls = Autoencoder(input_dim=config["cls_dim"], latent_dim=config["latent_dim"]).to(config["device"])
    autoencoder_patch = PatchAutoencoder(input_dim=config["patch_dim"], latent_dim=config["latent_dim_patch"]).to(config["device"])

    # Load dataset
    train_loader = get_data_loader(config["data_dir"], batch_size=config["batch_size"])

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        list(autoencoder_cls.parameters()) + list(autoencoder_patch.parameters()),
        lr=config["learning_rate"]
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

    # Training loop
    for epoch in range(config["epochs"]):
        autoencoder_cls.train()
        autoencoder_patch.train()
        total_loss = 0

        for images, texts in train_loader:
            images = images.to(config["device"])

            # Extract CLS and patch tokens
            with torch.no_grad():
                cls_tokens, patch_tokens = extract_cls_and_patches(dino, images)

            # Pass CLS tokens through CLS autoencoder
            cls_reconstructed, cls_latent = autoencoder_cls(cls_tokens)

            # Pass patch tokens through Patch autoencoder
            patch_reconstructed_list = []
            patch_latent_list = []
            for patch in patch_tokens.unbind(dim=1):  # Unbind along the patch dimension
                reconstructed, latent = autoencoder_patch(patch)
                patch_reconstructed_list.append(reconstructed)
                patch_latent_list.append(latent)

            # Compute CLIP embeddings
            patch_image_embeds = []
            for patch in patch_tokens.unbind(dim=1):
                patch_image = patch.view_as(images)  # Reshape each patch if needed
                patch_image_embeds.append(get_clip_embeddings(clip, processor, patch_image, None)[0])

            # Compute losses for CLS
            cls_rec_loss = reconstruction_loss(cls_tokens, cls_reconstructed)
            cls_clip_loss = clip_loss(cls_latent, image_embeds)

            # Compute losses for patches
            patch_rec_loss = sum(
                reconstruction_loss(patch, rec) for patch, rec in zip(patch_tokens.unbind(dim=1), patch_reconstructed_list)
            ) / len(patch_reconstructed_list)
            patch_clip_loss = sum(
                clip_loss(latent, patch_clip) for latent, patch_clip in zip(patch_latent_list, patch_image_embeds)
            ) / len(patch_latent_list)

            # Total loss
            total_loss = (
                config["lambda_cls_reconstruction"] * cls_rec_loss
                + config["lambda_cls_clip"] * cls_clip_loss
                + config["lambda_patch_reconstruction"] * patch_rec_loss
                + config["lambda_patch_clip"] * patch_clip_loss
            )

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss += total_loss.item()

        # Scheduler step
        scheduler.step()

        # Logging
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {total_loss:.4f}")

    # Save model
    torch.save(autoencoder_cls.state_dict(), config["save_path_cls"])
    torch.save(autoencoder_patch.state_dict(), config["save_path_patch"])
    print("Models saved!")

if __name__ == "__main__":
    with open("training/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train(config)
