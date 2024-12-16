import numpy as np
import faiss
import torch
from pathlib import Path
import logging
from typing import List, Dict, Union, Tuple
from collections import defaultdict
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_class_sizes(labels_dir: str) -> Dict[str, int]:
   """Count number of images in each class by reading label files"""
   class_sizes = defaultdict(int)
   
   for label_file in Path(labels_dir).glob('image_*.txt'):
       try:
           with open(label_file, 'r') as f:
               label = f.read().strip()
               class_sizes[label] += 1
       except Exception as e:
           logger.error(f"Error reading {label_file}: {e}")
   return dict(class_sizes)

def get_image_label(image_path: str, labels_dir: str) -> str:
   """
   Get label for an image by reading its corresponding label file.
   
   Args:
       image_path: Path to image file (e.g., 'image_00001.jpg')
       labels_dir: Directory containing label files
   """
   # Extract image number and create label filename
   image_num = int(image_path.split('_')[-1].split('.')[0])
   label_filename = f"image_{image_num-1:05d}.txt"
   label_path = os.path.join(labels_dir, label_filename)
   
   try:
       with open(label_path, 'r') as f:
           return f.read().strip()
   except Exception as e:
       logger.error(f"Error reading label file {label_path}: {e}")
       return None

def compute_recall_at_k(retrieved_paths: List[str], 
                      query_label: str,
                      labels_dir: str,
                      k: int) -> float:
   """
   Compute Recall@K metric based on class labels.
   
   Args:
       retrieved_paths: List of retrieved image paths
       query_label: Label of the query image
       labels_dir: Directory containing label files
       k: Number of top retrievals to consider
   """
   if k <= 0:
       raise ValueError("k must be positive")
   
   retrieved_paths = retrieved_paths[:k]
   correct = 0
   
   for path in retrieved_paths:
       retrieved_label = get_image_label(os.path.basename(path), labels_dir)
       if retrieved_label == query_label:
           correct += 1
   
   return correct / k

def compute_precision_at_k(retrieved_paths: List[str], 
                           query_label: str,
                           labels_dir: str,
                           k: int) -> float:
    """
    Compute Precision@K metric based on class labels.
    
    Args:
        retrieved_paths: List of retrieved image paths
        query_label: Label of the query image
        labels_dir: Directory containing label files
        k: Number of top retrievals to consider
    
    Returns:
        Precision@K: Ratio of correctly retrieved items in the top k to k.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    # Only consider the top-k retrieved paths
    retrieved_paths = retrieved_paths[:k]
    correct = 0
    
    for path in retrieved_paths:
        # Get the label of the retrieved image
        retrieved_label = get_image_label(os.path.basename(path), labels_dir)
        if retrieved_label == query_label:
            correct += 1
    
    # Precision@K is the number of correct items divided by K
    return correct / k



def compute_average_precision(retrieved_paths: List[str],
                           query_label: str,
                           labels_dir: str) -> float:
   """
   Compute Average Precision for a single query.
   """
   correct = 0
   precisions = []
   
   for i, path in enumerate(retrieved_paths, 1):
       retrieved_label = get_image_label(os.path.basename(path), labels_dir)
       if retrieved_label == query_label:
           correct += 1
           precision = correct / i
           precisions.append(precision)
   
   return np.mean(precisions) if precisions else 0.0

def evaluate_retrieval(index_file: str,
                     features_file: str,
                     query_paths: List[str],
                     labels_dir: str,
                     top_k: int = 100) -> Dict[str, float]:
   """
   Evaluate retrieval performance using multiple metrics.
   """
   # Load index and features
   index = faiss.read_index(index_file)
   data = np.load(features_file)
   features = data["features"]
   image_paths = data["image_paths"]
   
   # Load class sizes
   class_sizes = load_class_sizes(labels_dir)
   
   metrics = defaultdict(list)
   
   for query_path in query_paths:
       query_label = get_image_label(os.path.basename(query_path), labels_dir)
       if query_label is None:
           continue
           
       # Get max k for this class
       max_k = class_sizes.get(query_label, top_k)
       
       # Find query index
       try:
           query_idx = list(image_paths).index(query_path)
       except ValueError:
           logger.error(f"Query image {query_path} not found in features file")
           continue
           
       # Get query features and search
       query = features[query_idx:query_idx+1]
       D, I = index.search(query, min(top_k, max_k))
       
       # Get retrieved paths
       retrieved_paths = [image_paths[i] for i in I[0]]
       
       # Compute metrics for standard k values that don't exceed class size
       for k in [1, 5, 10, 20, 50, 100]:
           if k <= min(top_k, max_k):
               recall = compute_recall_at_k(retrieved_paths, query_label, labels_dir, k)
               precision = compute_precision_at_k(retrieved_paths, query_label, labels_dir, k)
               metrics[f'recall@{k}'].append(recall)
               metrics[f'precision@{k}'].append(precision)
       
       ap = compute_average_precision(retrieved_paths, query_label, labels_dir)
       metrics['mAP'].append(ap)
   
   # Average all metrics
   return {k: np.mean(v) for k, v in metrics.items()}

if __name__ == "__main__":
   import argparse
   
   parser = argparse.ArgumentParser(description="Evaluate image retrieval performance")
   parser.add_argument("--index_file", required=True, help="Path to FAISS index")
   parser.add_argument("--features_file", required=True, help="Path to features file")
   parser.add_argument("--labels_dir", required=True, help="Directory containing image labels")
   parser.add_argument("--output_file", help="Path to save evaluation results")
   parser.add_argument("--top_k", type=int, default=100, help="Maximum number of retrievals to consider")
   parser.add_argument("--num_queries", type=int, default=50, help="Number of queries to evaluate")
   
   args = parser.parse_args()
   
   try:
       # Load features and select random query images
       data = np.load(args.features_file)
       image_paths = data["image_paths"]
       query_paths = np.random.choice(image_paths, size=args.num_queries, replace=False)
    
       
       # Evaluate retrieval
       results = evaluate_retrieval(
           args.index_file,
           args.features_file,
           query_paths,
           args.labels_dir,
           args.top_k
       )
       
       # Print results
       print("\nEvaluation Results:")
       print(json.dumps(results, indent=2))
    
       # Save results if output file specified
       if args.output_file:
           with open(args.output_file, 'w') as f:
               json.dump(results, f, indent=2)
           print(f"\nResults saved to {args.output_file}")
           
   except Exception as e:
       logger.error(f"Error during evaluation: {e}")
       raise
