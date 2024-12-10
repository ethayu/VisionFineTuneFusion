import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class COCODataset(Dataset):
    """
    A PyTorch Dataset class for loading the COCO dataset.
    Supports both training and validation splits with image and annotation pairs.
    """

    def __init__(self, image_dir, annotation_file, transform=None, task="retrieval"):
        """
        Args:
            image_dir (str): Directory containing the COCO images (train2017 or val2017).
            annotation_file (str): Path to the COCO annotation JSON file.
            transform (callable, optional): Transformation to apply to the images.
            task (str): Task type ('retrieval', 'depth', etc.).
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.task = task
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        Load the COCO annotation file and filter the data based on the task.
        Returns:
            list: A list of dictionaries, each containing image and annotation information.
        """
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        annotations = []
        for img in data["images"]:
            entry = {"image_id": img["id"], "file_name": img["file_name"]}
            if self.task == "retrieval":
                entry["caption"] = next(
                    (ann["caption"] for ann in data["annotations"] if ann["image_id"] == img["id"]),
                    None,
                )
            annotations.append(entry)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the image, annotations, and additional information.
        """
        item = self.annotations[idx]
        image_path = os.path.join(self.image_dir, item["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        data = {"image": image, "image_id": item["image_id"]}
        if self.task == "retrieval" and "caption" in item:
            data["caption"] = item["caption"]

        return data


def get_data_loader(
    image_dir, annotation_file, batch_size=32, shuffle=True, transform=None, task="retrieval"
):
    """
    Create a PyTorch DataLoader for the COCO dataset.

    Args:
        image_dir (str): Directory containing the COCO images.
        annotation_file (str): Path to the COCO annotation JSON file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable, optional): Transformations to apply to the images.
        task (str): Task type ('retrieval', 'depth', etc.).

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = COCODataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=transform,
        task=task,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    image_dir = "data/coco/train2017/"
    annotation_file = "data/coco/annotations/captions_train2017.json"

    loader = get_data_loader(
        image_dir=image_dir,
        annotation_file=annotation_file,
        batch_size=8,
        shuffle=True,
        transform=transform,
        task="retrieval",
    )

    for batch in loader:
        print(batch["image"].shape, batch["caption"])
