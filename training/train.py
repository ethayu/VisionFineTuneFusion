import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder
from models.dino_model import load_dino_model, extract_cls_token
from models.clip_model import load_clip_model, get_clip_embeddings
from models.losses import reconstruction_loss, clip_loss
from data.dataset_loader import get_data_loader
import yaml

def train(config):
    # Load models
    dino = load_dino_model(config["dino_model_name"]).eval()
    clip, processor = load_clip_model(config["clip_model_name"])
    clip.eval()
    autoencoder = Autoencoder(input_dim=1024, latent_dim=config["latent_dim"]).to(config["device"])

    # Load dataset
    train_loader = get_data_loader(config["data_dir"], batch_size=config["batch_size"])

    # Optimizer and scheduler
    optimizer = optim.AdamW(autoencoder.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

    # Training loop
    for epoch in range(config["epochs"]):
        autoencoder.train()
        total_loss = 0

        for images, texts in train_loader:
            images = images.to(config["device"])

            # Extract CLS tokens
            with torch.no_grad():
                cls_tokens = extract_cls_token(dino, images)

            # Pass through autoencoder
            reconstructed, latent = autoencoder(cls_tokens)

            # Compute CLIP embeddings
            image_embeds, text_embeds = get_clip_embeddings(clip, processor, images, texts)
            image_embeds, text_embeds = image_embeds.to(config["device"]), text_embeds.to(config["device"])

            # Loss calculation
            rec_loss = reconstruction_loss(cls_tokens, reconstructed)
            clip_image_loss = clip_loss(latent, image_embeds)
            clip_text_loss = clip_loss(latent, text_embeds)

            total_batch_loss = (
                config["lambda_reconstruction"] * rec_loss
                + config["lambda_clip_image"] * clip_image_loss
                + config["lambda_clip_text"] * clip_text_loss
            )

            # Backpropagation
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        # Scheduler step
        scheduler.step()

        # Logging
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {total_loss:.4f}")

    # Save model
    torch.save(autoencoder.state_dict(), config["save_path"])
    print("Model saved!")

if __name__ == "__main__":
    with open("training/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train(config)
