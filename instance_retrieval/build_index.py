import os
import faiss
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
import torch
from torchvision import transforms
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_features(features_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted features and image paths from .npz file.
    
    Args:
        features_file: Path to the .npz file containing features and metadata
        
    Returns:
        Tuple of (features array, image paths array)
    """
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
        
    logger.info(f"Loading features from {features_file}")
    data = np.load(features_file)
    
    features = data["features"].astype(np.float32)
    image_paths = data["image_paths"]
    
    logger.info(f"Loaded {len(features)} feature vectors of dimension {features.shape[1]}")
    
    return features, image_paths

def build_index(
    features: np.ndarray,
    index_type: str = "L2",
    n_lists: Optional[int] = None
) -> faiss.Index:
    """
    Build a FAISS index from the feature vectors.
    
    Args:
        features: Array of feature vectors
        index_type: Type of index to build ("L2" or "IP" for inner product)
        n_lists: Number of Voronoi cells for IVF index (if None, uses sqrt(n))
        
    Returns:
        Built FAISS index
    """
    n_vectors, dim = features.shape
    
    if n_lists is None:
        n_lists = int(np.sqrt(n_vectors))
    
    logger.info(f"Building index for {n_vectors} vectors of dimension {dim}")
    
    # Create appropriate index based on type
    if index_type == "L2":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_L2)
    else:  # Inner product similarity
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(features)
    
    # Train the index
    logger.info("Training index...")
    index.train(features)
    
    # Add vectors to the index
    logger.info("Adding vectors to index...")
    index.add(features)
    
    return index

def build_and_save_index(
    features_file: str,
    index_file: str,
    index_type: str = "L2",
    n_lists: Optional[int] = None
) -> None:
    """
    Build a FAISS index from extracted features and save it to disk.
    
    Args:
        features_file: Path to .npz file containing features
        index_file: Path to save the FAISS index
        index_type: Type of index to build ("L2" or "IP" for inner product)
        n_lists: Number of Voronoi cells for IVF index
    """
    # Create output directory if needed
    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load features
    features, image_paths = load_features(features_file)
    
    # Build the index
    index = build_index(features, index_type, n_lists)
    
    # Verify index size matches feature count
    if index.ntotal != len(features):
        raise ValueError(
            f"Index size ({index.ntotal}) doesn't match feature count ({len(features)})"
        )
    
    # Save the index
    logger.info(f"Saving index to {index_file}")
    faiss.write_index(index, index_file)
    
    # Quick verification test
    logger.info("Running verification test...")
    test_index = faiss.read_index(index_file)
    if test_index.ntotal != index.ntotal:
        raise ValueError("Verification failed: saved index doesn't match original")
    
    logger.info(f"Successfully built and saved index with {index.ntotal} vectors")

def verify_index(
    index_file: str,
    features_file: str,
    n_test: int = 5
) -> None:
    """
    Verify the built index by running some test queries.
    
    Args:
        index_file: Path to the FAISS index file
        features_file: Path to the original features file
        n_test: Number of test queries to run
    """
    logger.info("Running index verification...")
    
    # Load features and index
    features, image_paths = load_features(features_file)
    index = faiss.read_index(index_file)
    
    # Run test queries
    for i in range(min(n_test, len(features))):
        query = features[i:i+1]
        D, I = index.search(query, 1)
        
        if I[0][0] != i:
            logger.warning(
                f"Verification warning: Self-retrieval failed for vector {i}"
            )
    
    logger.info("Index verification complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from extracted features")
    parser.add_argument(
        "--features_file",
        required=True,
        help="Path to .npz file containing extracted features"
    )
    parser.add_argument(
        "--index_file",
        required=True,
        help="Path to save the FAISS index"
    )
    parser.add_argument(
        "--index_type",
        default="L2",
        choices=["L2", "IP"],
        help="Type of index to build (L2 distance or Inner Product similarity)"
    )
    parser.add_argument(
        "--n_lists",
        type=int,
        help="Number of Voronoi cells for IVF index (default: sqrt(n_vectors))"
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip index verification step"
    )
    
    args = parser.parse_args()
    
    try:
        # Build and save the index
        build_and_save_index(
            features_file=args.features_file,
            index_file=args.index_file,
            index_type=args.index_type,
            n_lists=args.n_lists
        )
        
        # Verify the index
        if not args.skip_verification:
            verify_index(args.index_file, args.features_file)
            
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise