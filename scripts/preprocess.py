import os
from PIL import Image
from tqdm import tqdm

def preprocess_data(input_dir, output_dir, image_size=(224, 224)):
    """
    Preprocess raw image-text data.
    
    Args:
        input_dir (str): Directory containing raw images and text files.
        output_dir (str): Directory to save processed data.
        image_size (tuple): Desired size of output images (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(input_dir, file_name)
            text_path = image_path.replace(".jpg", ".txt").replace(".png", ".txt")
            
            if not os.path.exists(text_path):
                print(f"Skipping {file_name}: No corresponding text file found.")
                continue
            
            # Resize image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(image_size)
                img.save(os.path.join(output_dir, file_name))
            
            # Copy text
            with open(text_path, "r") as f_in:
                with open(os.path.join(output_dir, os.path.basename(text_path)), "w") as f_out:
                    f_out.write(f_in.read())

if __name__ == "__main__":
    preprocess_data("data/raw", "data/processed", image_size=(224, 224))
