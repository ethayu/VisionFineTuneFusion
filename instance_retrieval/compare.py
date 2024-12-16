import numpy as np
import faiss
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_image(image_path: str, size: int = 224) -> torch.Tensor:
    """Load and preprocess an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
        ])
        return transform(img)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def create_retrieval_grid(query_path: str, 
                         retrieved_paths: List[str],
                         n_results: int = 5) -> torch.Tensor:
    """Create a grid of query image and its retrieved results."""
    # Load query image
    query_tensor = load_and_preprocess_image(query_path)
    if query_tensor is None:
        return None
    
    # Load retrieved images
    retrieved_tensors = []
    for path in retrieved_paths[:n_results]:
        tensor = load_and_preprocess_image(path)
        if tensor is not None:
            retrieved_tensors.append(tensor)
    
    if not retrieved_tensors:
        return None
    
    # Combine query and retrieved images
    all_tensors = [query_tensor] + retrieved_tensors
    grid = make_grid(all_tensors, nrow=len(all_tensors), padding=2)
    return grid

def get_image_label(image_path: str, labels_dir: str) -> str:
    """Get label for an image by reading its corresponding label file."""
    image_num = int(image_path.split('_')[-1].split('.')[0])
    label_filename = f"image_{image_num-1:05d}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    
    try:
        with open(label_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading label file {label_path}: {e}")
        return None

def compute_precision_at_k(retrieved_paths: List[str], 
                         query_label: str,
                         labels_dir: str,
                         k: int = 10) -> float:
    """Compute Precision@K for a single query."""
    retrieved_paths = retrieved_paths[:k]
    correct = sum(1 for path in retrieved_paths 
                 if get_image_label(os.path.basename(path), labels_dir) == query_label)
    return correct / k

def get_retrievals(query_idx: int,
                  features: np.ndarray,
                  image_paths: np.ndarray,
                  index: faiss.Index,
                  k: int = 10) -> List[str]:
    """Get retrieved images for a query."""
    query = features[query_idx:query_idx+1]
    D, I = index.search(query, k + 1)  # +1 to account for self-match
    retrieved_paths = [image_paths[i] for i in I[0] if i != query_idx][:k]
    return retrieved_paths

def evaluate_single_query(query_idx: int,
                        features: np.ndarray,
                        image_paths: np.ndarray,
                        index: faiss.Index,
                        labels_dir: str,
                        k: int = 10) -> Tuple[float, List[str]]:
    """Evaluate precision@k for a single query and return retrieved paths."""
    query_path = image_paths[query_idx]
    query_label = get_image_label(os.path.basename(query_path), labels_dir)
    
    if query_label is None:
        return 0.0, []
    
    retrieved_paths = get_retrievals(query_idx, features, image_paths, index, k)
    precision = compute_precision_at_k(retrieved_paths, query_label, labels_dir, k)
    
    return precision, retrieved_paths

def visualize_contrasting_pair(result: Dict,
                             default_retrievals: List[str],
                             cls_retrievals: List[str],
                             output_dir: str,
                             pair_idx: int):
    """Create and save visualization for a contrasting pair."""
    query_path = result['image_path']
    
    # Create visualization for default DiNO results
    default_grid = create_retrieval_grid(query_path, default_retrievals)
    cls_grid = create_retrieval_grid(query_path, cls_retrievals)
    
    if default_grid is None or cls_grid is None:
        return
    
    plt.figure(figsize=(15, 8))
    
    # Plot default DiNO results
    plt.subplot(2, 1, 1)
    plt.imshow(default_grid.permute(1, 2, 0))
    plt.title(f"Default DiNO Results (Precision@10: {result['default_precision']:.3f})")
    plt.axis('off')
    
    # Plot CLS results
    plt.subplot(2, 1, 2)
    plt.imshow(cls_grid.permute(1, 2, 0))
    plt.title(f"CLS Results (Precision@10: {result['cls_precision']:.3f})")
    plt.axis('off')
    
    # Save plot
    output_path = os.path.join(output_dir, f'pair_{pair_idx}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def find_contrasting_pairs(default_index_file: str,
                         cls_index_file: str,
                         default_features_file: str,
                         cls_features_file: str,
                         labels_dir: str,
                         output_dir: str,
                         k: int = 10,
                         top_n: int = 10) -> List[Dict]:
    """Find and visualize image pairs with high performance on CLS but low on default DiNO."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load indices and features
    default_index = faiss.read_index(default_index_file)
    cls_index = faiss.read_index(cls_index_file)
    
    default_data = np.load(default_features_file)
    cls_data = np.load(cls_features_file)
    
    default_features = default_data["features"]
    cls_features = cls_data["features"]
    image_paths = default_data["image_paths"]
    
    # Store performance differences
    results = []
    
    # Evaluate each image as a query
    logger.info("Evaluating queries...")
    for idx in tqdm(range(len(image_paths))):
        default_precision, default_retrievals = evaluate_single_query(
            idx, default_features, image_paths, default_index, labels_dir, k)
        cls_precision, cls_retrievals = evaluate_single_query(
            idx, cls_features, image_paths, cls_index, labels_dir, k)
        
        # Calculate performance difference (CLS - default)
        difference = cls_precision - default_precision
        
        results.append({
            'image_path': image_paths[idx],
            'default_precision': float(default_precision),
            'cls_precision': float(cls_precision),
            'difference': float(difference),
            'default_retrievals': default_retrievals,
            'cls_retrievals': cls_retrievals
        })
    
    # Sort by largest positive difference (high CLS, low default)
    results.sort(key=lambda x: x['difference'], reverse=True)
    top_results = results[:top_n]
    
    # Create visualizations for top pairs
    logger.info("Creating visualizations...")
    for i, result in enumerate(top_results):
        visualize_contrasting_pair(
            result,
            result['default_retrievals'],
            result['cls_retrievals'],
            output_dir,
            i + 1
        )
    
    return top_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and visualize contrasting performance pairs")
    parser.add_argument("--default_index", required=True, help="Path to default DiNO FAISS index")
    parser.add_argument("--cls_index", required=True, help="Path to CLS FAISS index")
    parser.add_argument("--default_features", required=True, help="Path to default features file")
    parser.add_argument("--cls_features", required=True, help="Path to CLS features file")
    parser.add_argument("--labels_dir", required=True, help="Directory containing image labels")
    parser.add_argument("--output_dir", required=True, help="Directory to save results and visualizations")
    parser.add_argument("--k", type=int, default=10, help="K for Precision@K calculation")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top contrasting pairs to find")
    
    args = parser.parse_args()
    
    try:
        # Find contrasting pairs and create visualizations
        results = find_contrasting_pairs(
            args.default_index,
            args.cls_index,
            args.default_features,
            args.cls_features,
            args.labels_dir,
            args.output_dir,
            args.k,
            args.top_n
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, 'contrasting_pairs.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print top results
        print("\nTop contrasting pairs (high CLS, low default performance):")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Image: {result['image_path']}")
            print(f"   CLS Precision@{args.k}: {result['cls_precision']:.3f}")
            print(f"   Default Precision@{args.k}: {result['default_precision']:.3f}")
            print(f"   Difference: {result['difference']:.3f}")
            
        print(f"\nResults saved to {args.output_dir}")
        print("Visualizations have been created for each pair")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
