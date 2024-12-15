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

def prepare_patch_for_clip(patch, target_size=(224, 224)):
    # Assuming patch is of shape (batch_size, 1024)
    # Step 1: Reshape the patch to (batch_size, 32, 32) or any other reasonable 2D shape
    patch_image = patch.view(patch.size(0), 32, 32)  # Example reshaping into 32x32
    # Step 2: Expand dimensions to include the 3 color channels (RGB)
    patch_image = patch_image.unsqueeze(1).repeat(1, 3, 1, 1)  # Now shape is (batch_size, 3, 32, 32)
    
    # Step 3: Upsample to 224x224 for CLIP input
    # patch_image = F.interpolate(patch_image, size=target_size, mode="bilinear", align_corners=False)
    patch_image = torch.sigmoid(patch_image) 
    return patch_image

def train(config):
    # Load models
    dino = load_dino_model(config["dino_model_name"]).eval()
    clip, processor = load_clip_model(config["clip_model_name"])
    clip.eval()

    # Initialize autoencoders
    autoencoder_cls = Autoencoder(input_dim=config["cls_dim"], latent_dim=config["latent_dim"]).to(config["device"])
    autoencoder_patch = PatchAutoencoder(input_dim=config["patch_dim"], latent_dim=config["latent_dim_patch"]).to(config["device"])

    # Load dataset
    # train_loader = get_data_loader(config["data_dir"], batch_size=config["batch_size"], annotation_file="data/coco/annotations/captions_train2017.json")

    train_loader = get_data_loader("data/coco/val2017", batch_size=config["batch_size"], annotation_file="data/coco/annotations/captions_val2017.json")

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

        for images, text in tqdm(train_loader):
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
            for patch in tqdm(patch_tokens.unbind(dim=1)[:8]):
                # print(f'Patch tokens shape {patch_tokens.shape}, {images.shape}')
                patch_image = prepare_patch_for_clip(patch)
                patch_image_embeds.append(get_clip_embeddings(clip, processor, patch_image, text)[0])
                # print(patch.shape)
                # patch_image_embeds.append(get_clip_embeddings(clip, processor, patch, None)[0])
            patch_image_embeds =patch_image_embeds * 32


            # Compute losses for CLS
            cls_rec_loss = reconstruction_loss(cls_tokens, cls_reconstructed)

            # print(f'CLS Latent: {cls_latent.shape}, patch_image_embeds: {patch_image_embeds[0].shape}')

            cls_clip_loss = clip_loss(cls_latent, patch_image_embeds[0])

            # print(f'Cls Clip loss {cls_clip_loss}')

            # print(f'Patch Latent List 0 shape {patch_latent_list[0].shape}, Patch image embeds 0 shape {patch_image_embeds[0].shape}')

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
    print('config loaded...')

    train(config)
