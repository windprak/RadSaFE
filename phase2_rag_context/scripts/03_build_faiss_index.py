#!/usr/bin/env python3
"""
Step 3: Build FAISS index for efficient retrieval
"""

import json
import sys
import numpy as np
from pathlib import Path
import faiss
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


class FAISSIndexBuilder:
    """Build and manage FAISS indices"""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
    
    def build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a flat (brute-force) FAISS index
        Best for smaller corpora (<1M vectors)
        
        Args:
            embeddings: Numpy array of embeddings (n x d)
            
        Returns:
            FAISS index
        """
        print(f"Building Flat index for {embeddings.shape[0]} vectors...")
        
        # Create index
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity with normalized vectors)
        
        # Add vectors
        index.add(embeddings.astype(np.float32))
        
        print(f"Index built: {index.ntotal} vectors")
        return index
    
    def build_ivf_index(self, embeddings: np.ndarray, nlist: int = 100) -> faiss.Index:
        """
        Build an IVF (Inverted File) index for faster search on large corpora
        
        Args:
            embeddings: Numpy array of embeddings (n x d)
            nlist: Number of clusters
            
        Returns:
            FAISS index
        """
        print(f"Building IVF index for {embeddings.shape[0]} vectors...")
        print(f"Number of clusters: {nlist}")
        
        # Create quantizer
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        # Create IVF index
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train index
        print("Training index...")
        index.train(embeddings.astype(np.float32))
        
        # Add vectors
        print("Adding vectors...")
        index.add(embeddings.astype(np.float32))
        
        print(f"Index built: {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, output_path: Path, metadata: dict):
        """
        Save FAISS index and metadata
        
        Args:
            index: FAISS index
            output_path: Path to save index
            metadata: Metadata dictionary
        """
        print(f"\nSaving index to: {output_path}")
        faiss.write_index(index, str(output_path))
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def test_index(self, index: faiss.Index, embeddings: np.ndarray, k: int = 10):
        """
        Test index with a sample query
        
        Args:
            index: FAISS index
            embeddings: Original embeddings
            k: Number of neighbors to retrieve
        """
        print(f"\nTesting index with k={k}...")
        
        # Use first embedding as query
        query = embeddings[0:1].astype(np.float32)
        
        # Search
        distances, indices = index.search(query, k)
        
        print(f"Query vector index: 0")
        print(f"Top-{k} results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"  {i+1}. Index: {idx}, Distance: {dist:.4f}")


def main():
    """Main execution"""
    
    # Create output directory
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    embeddings_path = EMBEDDINGS_DIR / "radiopaedia_embeddings.npy"
    
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found: {embeddings_path}")
        print("Please run 02_build_embeddings.py first")
        sys.exit(1)
    
    print(f"Loading embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Load metadata
    metadata_path = embeddings_path.with_suffix('.json')
    with open(metadata_path, 'r') as f:
        embedding_metadata = json.load(f)
    
    # Initialize builder
    builder = FAISSIndexBuilder(embedding_dim=embeddings.shape[1])
    
    # Build index (use Flat for smaller corpora, IVF for larger)
    if embeddings.shape[0] < 100000:
        index = builder.build_flat_index(embeddings)
        index_type = "Flat"
    else:
        nlist = min(int(np.sqrt(embeddings.shape[0])), 1000)
        index = builder.build_ivf_index(embeddings, nlist=nlist)
        index_type = "IVF"
    
    # Save index
    index_path = INDICES_DIR / "radiopaedia_index.faiss"
    metadata = {
        "index_type": index_type,
        "num_vectors": int(index.ntotal),
        "embedding_dim": embeddings.shape[1],
        "embeddings_path": str(embeddings_path),
        **embedding_metadata
    }
    
    builder.save_index(index, index_path, metadata)
    
    # Test index
    builder.test_index(index, embeddings, k=10)
    
    print("\n" + "="*60)
    print("INDEX BUILDING COMPLETE")
    print("="*60)
    print(f"Index saved: {index_path}")
    print(f"Type: {index_type}")
    print(f"Vectors: {index.ntotal}")
    print("="*60)


if __name__ == "__main__":
    main()
