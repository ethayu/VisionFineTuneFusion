import torch
from models.autoencoder import Autoencoder
from models.dino_model import load_dino_model, extract_cls_token
from models.clip_model import load_clip_model, get_clip_embeddings

def evaluate(autoencoder, dino, clip, processor, dataloader, device):
    autoencoder.eval()
    dino.eval()
    clip.eval()
    total_similarity = 0
    count = 0

    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(device)

            # Extract CLS tokens and pass through autoencoder
            cls_tokens = extract_cls_token(dino, images)
            _, latent = autoencoder(cls_tokens)

            # Compute CLIP embeddings
            image_embeds, text_embeds = get_clip_embeddings(clip, processor, images, texts)
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)

            # Measure similarity
            image_similarity = torch.cosine_similarity(latent, image_embeds, dim=-1).mean().item()
            text_similarity = torch.cosine_similarity(latent, text_embeds, dim=-1).mean().item()

            total_similarity += (image_similarity + text_similarity) / 2
            count += 1

    print(f"Average Similarity: {total_similarity / count:.4f}")

if __name__ == "__main__":
    # Load models
    autoencoder = Autoencoder(input_dim=1024, latent_dim=768)
    autoencoder.load_state_dict(torch.load("results/checkpoints/autoencoder.pth"))
    autoencoder.eval()

    dino = load_dino_model("facebook/dino-v2-large").eval()
    clip, processor = load_clip_model("openai/clip-vit-base-patch32")

    # Load data
    from data.dataset_loader import get_data_loader
    dataloader = get_data_loader("data/processed", batch_size=32, shuffle=False)

    evaluate(autoencoder, dino, clip, processor, dataloader, device="cuda")
