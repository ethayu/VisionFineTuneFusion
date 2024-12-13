import numpy as np
import faiss
import torch
from pathlib import Path
import logging
from typing import List, Dict, Union, Tuple
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_recall_at_k(retrieved_indices: np.ndarray, 
                       ground_truth_indices: List[np.ndarray],
                       k: int) -> float:
    """
    Compute Recall@K metric.
    
    Args:
        retrieved_indices: Array of shape (n_queries, max_k) containing retrieved indices
        ground_truth_indices: List of arrays containing ground truth indices for each query
        k: Number of top retrievals to consider
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    # Ensure we only consider up to k retrievals
    retrieved_indices = retrieved_indices[:, :k]
    
    recalls = []
    for i in range(len(retrieved_indices)):
        retrieved_set = set(retrieved_indices[i])
        ground_truth_set = set(ground_truth_indices[i])
        if len(ground_truth_set) > 0:
            recall = len(retrieved_set.intersection(ground_truth_set)) / len(ground_truth_set)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0

def compute_precision_at_k(retrieved_indices: np.ndarray,
                          ground_truth_indices: List[np.ndarray],
                          k: int) -> float:
    """
    Compute Precision@K metric.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    retrieved_indices = retrieved_indices[:, :k]
    
    precisions = []
    for i in range(len(retrieved_indices)):
        retrieved_set = set(retrieved_indices[i])
        ground_truth_set = set(ground_truth_indices[i])
        if k > 0:
            precision = len(retrieved_set.intersection(ground_truth_set)) / k
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0

def compute_average_precision(retrieved_indices: np.ndarray,
                            ground_truth_indices: np.ndarray) -> float:
    """
    Compute Average Precision for a single query.
    """
    ground_truth_set = set(ground_truth_indices)
    relevant_retrievals = [idx in ground_truth_set for idx in retrieved_indices]
    
    if not any(relevant_retrievals):
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, is_relevant in enumerate(relevant_retrievals, 1):
        if is_relevant:
            num_relevant += 1
            precision = num_relevant / i
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0

def create_semantic_ground_truth(features: np.ndarray, 
                               similarity_threshold: float = 0.8) -> Dict[int, np.ndarray]:
    """
    Create ground truth based on feature similarity.
    """
    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    features = features / norms
    
    # Compute pairwise similarities
    similarities = features @ features.T
    
    # Create ground truth dictionary
    ground_truth = {}
    for i in range(len(features)):
        # Find indices of similar images (excluding self)
        related = np.where(similarities[i] > similarity_threshold)[0]
        related = related[related != i]  # Remove self
        ground_truth[i] = related
    
    return ground_truth

def evaluate_retrieval(index_file: str,
                      features_file: str,
                      query_indices: List[int],
                      top_k: int = 100,
                      similarity_threshold: float = 0.8) -> Dict[str, float]:
    """
    Evaluate retrieval performance using multiple metrics.
    """
    # Load index and features
    index = faiss.read_index(index_file)
    data = np.load(features_file)
    features = data["features"]
    
    # Create meaningful ground truth based on feature similarity
    ground_truth = create_semantic_ground_truth(features, similarity_threshold)
    
    # Perform retrieval for each query
    retrieved_indices = []
    gt_indices = []
    
    for query_idx in query_indices:
        # Get query features and search
        query = features[query_idx:query_idx+1]
        D, I = index.search(query, top_k)
        
        retrieved_indices.append(I[0])
        gt_indices.append(ground_truth[query_idx])
    
    # Convert retrieved indices to array
    retrieved_indices = np.array(retrieved_indices)
    
    # Keep gt_indices as a list of arrays
    
    # Compute metrics
    metrics = {}
    
    # Compute Recall@K and Precision@K for various K
    for k in [1, 5, 10, 20, 50, 100]:
        if k <= top_k:
            metrics[f'recall@{k}'] = compute_recall_at_k(retrieved_indices, gt_indices, k)
            metrics[f'precision@{k}'] = compute_precision_at_k(retrieved_indices, gt_indices, k)
    
    # Compute Mean Average Precision
    aps = []
    for i in range(len(retrieved_indices)):
        ap = compute_average_precision(retrieved_indices[i], gt_indices[i])
        aps.append(ap)
    metrics['mAP'] = np.mean(aps)
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate image retrieval performance")
    parser.add_argument("--index_file", required=True, help="Path to FAISS index")
    parser.add_argument("--features_file", required=True, help="Path to features file")
    parser.add_argument("--output_file", help="Path to save evaluation results")
    parser.add_argument("--top_k", type=int, default=100, help="Number of retrievals to consider")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries to evaluate")
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                       help="Threshold for considering images as related")
    
    args = parser.parse_args()
    
    try:
        # Load features and select random query indices
        num_features = np.load(args.features_file)["features"].shape[0]
        query_indices = np.random.choice(num_features, size=args.num_queries, replace=False)
        
        # Evaluate retrieval
        results = evaluate_retrieval(
            args.index_file,
            args.features_file,
            query_indices,
            args.top_k,
            args.similarity_threshold
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