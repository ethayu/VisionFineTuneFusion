import numpy as np
from typing import List, Dict, Union, Tuple
import faiss
import torch
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_recall_at_k(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k: int
) -> float:
    """
    Compute Recall@K metric.
    
    Args:
        retrieved_indices: Array of retrieved image indices for each query
        ground_truth_indices: Array of ground truth relevant image indices for each query
        k: Number of top retrievals to consider
        
    Returns:
        Recall@K score
    """
    if k <= 0:
        raise ValueError("k must be positive")
        
    # Ensure we only consider up to k retrievals
    retrieved_indices = retrieved_indices[:, :k]
    
    # Compute recall for each query
    recalls = []
    for i in range(len(retrieved_indices)):
        relevant_retrievals = np.intersect1d(retrieved_indices[i], ground_truth_indices[i])
        recall = len(relevant_retrievals) / len(ground_truth_indices[i])
        recalls.append(recall)
    
    return np.mean(recalls)

def compute_average_precision(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray
) -> float:
    """
    Compute Average Precision for a single query.
    
    Args:
        retrieved_indices: Array of retrieved image indices
        ground_truth_indices: Array of ground truth relevant image indices
        
    Returns:
        Average Precision score
    """
    relevant_retrievals = np.isin(retrieved_indices, ground_truth_indices)
    
    if not relevant_retrievals.any():
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, is_relevant in enumerate(relevant_retrievals, 1):
        if is_relevant:
            num_relevant += 1
            precision = num_relevant / i
            precisions.append(precision)
    
    return np.mean(precisions)

def compute_mean_average_precision(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray
) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Args:
        retrieved_indices: Array of retrieved image indices for each query
        ground_truth_indices: Array of ground truth relevant image indices for each query
        
    Returns:
        mAP score
    """
    aps = []
    for i in range(len(retrieved_indices)):
        ap = compute_average_precision(retrieved_indices[i], ground_truth_indices[i])
        aps.append(ap)
    
    return np.mean(aps)

def evaluate_retrieval(
    index_file: str,
    features_file: str,
    query_indices: List[int],
    ground_truth: Dict[int, List[int]],
    top_k: int = 100
) -> Dict[str, float]:
    """
    Evaluate retrieval performance using multiple metrics.
    
    Args:
        index_file: Path to FAISS index
        features_file: Path to features file
        query_indices: List of indices to use as queries
        ground_truth: Dictionary mapping query indices to lists of relevant image indices
        top_k: Number of retrievals to consider
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load index and features
    index = faiss.read_index(index_file)
    data = np.load(features_file)
    features = data["features"]
    
    # Perform retrieval for each query
    retrieved_indices = []
    gt_indices = []
    
    for query_idx in query_indices:
        # Get query features
        query = features[query_idx:query_idx+1]
        
        # Search index
        D, I = index.search(query, top_k)
        
        retrieved_indices.append(I[0])
        gt_indices.append(ground_truth[query_idx])
    
    # Convert to arrays
    retrieved_indices = np.array(retrieved_indices)
    gt_indices = np.array(gt_indices)
    
    # Compute metrics
    metrics = {}
    
    # Compute Recall@K for different K values
    for k in [1, 5, 10, 100]:
        if k <= top_k:
            metrics[f'recall@{k}'] = compute_recall_at_k(retrieved_indices, gt_indices, k)
    
    # Compute mAP
    metrics['mAP'] = compute_mean_average_precision(retrieved_indices, gt_indices)
    
    return metrics

def evaluate_class_based(
    index_file: str,
    features_file: str,
    class_labels: np.ndarray,
    num_queries_per_class: int = 5,
    top_k: int = 100
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluate retrieval performance using class labels.
    
    Args:
        index_file: Path to FAISS index
        features_file: Path to features file
        class_labels: Array of class labels for each image
        num_queries_per_class: Number of queries to use per class
        top_k: Number of retrievals to consider
        
    Returns:
        Tuple of (overall metrics, per-class metrics)
    """
    # Get unique classes
    unique_classes = np.unique(class_labels)
    
    # Create ground truth dictionary
    ground_truth = {}
    query_indices = []
    
    for class_label in unique_classes:
        # Get indices for this class
        class_indices = np.where(class_labels == class_label)[0]
        
        # Select query indices for this class
        class_queries = np.random.choice(
            class_indices,
            size=min(num_queries_per_class, len(class_indices)),
            replace=False
        )
        
        # Add to query indices and ground truth
        for query_idx in class_queries:
            query_indices.append(query_idx)
            # All images of same class are relevant (except query itself)
            ground_truth[query_idx] = [idx for idx in class_indices if idx != query_idx]
    
    # Compute overall metrics
    overall_metrics = evaluate_retrieval(
        index_file,
        features_file,
        query_indices,
        ground_truth,
        top_k
    )
    
    # Compute per-class metrics
    per_class_metrics = {}
    for class_label in unique_classes:
        class_queries = [
            idx for idx in query_indices
            if class_labels[idx] == class_label
        ]
        
        if class_queries:
            class_gt = {idx: ground_truth[idx] for idx in class_queries}
            per_class_metrics[str(class_label)] = evaluate_retrieval(
                index_file,
                features_file,
                class_queries,
                class_gt,
                top_k
            )
    
    return overall_metrics, per_class_metrics

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Evaluate image retrieval performance")
    parser.add_argument("--index_file", required=True, help="Path to FAISS index")
    parser.add_argument("--features_file", required=True, help="Path to features file")
    parser.add_argument("--class_labels_file", help="Path to class labels file (optional)")
    parser.add_argument("--output_file", help="Path to save evaluation results")
    parser.add_argument("--top_k", type=int, default=100, help="Number of retrievals to consider")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries to evaluate")
    
    args = parser.parse_args()
    
    try:
        results = {}
        
        if args.class_labels_file:
            # Load class labels
            class_labels = np.load(args.class_labels_file)
            
            # Evaluate using class labels
            overall_metrics, per_class_metrics = evaluate_class_based(
                args.index_file,
                args.features_file,
                class_labels,
                num_queries_per_class=args.num_queries // len(np.unique(class_labels)),
                top_k=args.top_k
            )
            
            results = {
                "overall": overall_metrics,
                "per_class": per_class_metrics
            }
        else:
            # Simple random evaluation
            num_features = np.load(args.features_file)["features"].shape[0]
            query_indices = np.random.choice(num_features, size=args.num_queries, replace=False)
            
            # Create simple ground truth (just the query itself)
            ground_truth = {idx: [idx] for idx in query_indices}
            
            results = evaluate_retrieval(
                args.index_file,
                args.features_file,
                query_indices,
                ground_truth,
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