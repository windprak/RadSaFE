#!/usr/bin/env python3
"""
Step 2: Build embeddings for Radiopaedia chunks using BGE-large-en-v1.5
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


class EmbeddingBuilder:
    """Build embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Embed all chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            Numpy array of embeddings (n_chunks x embedding_dim)
        """
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"\nEmbedding {len(texts)} chunks...")
        print(f"Batch size: {self.batch_size}")
        
        # Encode in batches with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings
    
    def process_corpus(self, chunks_path: Path, output_path: Path):
        """
        Load chunks, embed them, and save embeddings
        
        Args:
            chunks_path: Path to chunks JSONL file
            output_path: Path to save embeddings (.npy)
        """
        print(f"Loading chunks from: {chunks_path}")
        
        chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Build embeddings
        embeddings = self.embed_chunks(chunks)
        
        # Save embeddings
        print(f"\nSaving embeddings to: {output_path}")
        np.save(output_path, embeddings)
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                "model": self.model_name,
                "embedding_dim": embeddings.shape[1],
                "num_chunks": embeddings.shape[0],
                "chunks_path": str(chunks_path),
                "normalized": True
            }, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
        print(f"\nEmbedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        
        return embeddings


def main():
    """Main execution"""
    
    # Create output directory
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding builder
    builder = EmbeddingBuilder(
        model_name=EMBEDDING_MODEL,
        batch_size=32
    )
    
    # Process Radiopaedia chunks
    chunks_path = EMBEDDINGS_DIR / "radiopaedia_chunks.jsonl"
    embeddings_path = EMBEDDINGS_DIR / "radiopaedia_embeddings.npy"
    
    if not chunks_path.exists():
        print(f"Error: Chunks file not found: {chunks_path}")
        print("Please run 01_chunk_radiopaedia.py first")
        sys.exit(1)
    
    embeddings = builder.process_corpus(chunks_path, embeddings_path)
    
    print("\n" + "="*60)
    print("EMBEDDING COMPLETE")
    print("="*60)
    print(f"Embeddings saved: {embeddings_path}")
    print(f"Shape: {embeddings.shape}")
    print("="*60)


if __name__ == "__main__":
    main()
