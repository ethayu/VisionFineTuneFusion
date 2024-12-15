import os
from PIL import Image
from tqdm import tqdm

def preprocess_data(input_dir, captions_subdir, output_dir, image_size=(224, 224)):
    """
    Preprocess raw image-text data.

    Args:
        input_dir (str): Directory containing raw images.
        captions_subdir (str): Subdirectory under input_dir containing text files.
        output_dir (str): Directory to save processed data.
        image_size (tuple): Desired size of output images (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Full paths to images and captions directories
    images_dir = input_dir
    captions_dir = os.path.join(input_dir, captions_subdir)

    skipped = 0

    for file_name in tqdm(os.listdir(images_dir)):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(images_dir, file_name)
            id = str(int(os.path.splitext(file_name)[0]))

            text_path = os.path.join(captions_dir, id + ".txt")

            if not os.path.exists(text_path):
                skipped += 1
                # Skip if no corresponding text file is found
                continue

            # Resize and save image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(image_size)
                img.save(os.path.join(output_dir, file_name))

            # Copy text file content
            with open(text_path, "r") as f_in:
                with open(os.path.join(output_dir, os.path.basename(text_path)), "w") as f_out:
                    f_out.write(f_in.read())

if __name__ == "__main__":
    preprocess_data(
        input_dir="data/coco/train2017",
        captions_subdir="captions",
        output_dir="data/coco/processed",
        image_size=(224, 224)
    )
    
