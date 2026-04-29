#!/usr/bin/env python3
"""
Step 4: Retrieve relevant context for each question in all datasets
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import tiktoken

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


class ContextRetriever:
    """Retrieve relevant chunks for questions using FAISS index"""
    
    def __init__(self, 
                 index_path: Path,
                 chunks_path: Path,
                 embeddings_path: Path,
                 model_name: str = EMBEDDING_MODEL):
        """
        Initialize retriever
        
        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks JSONL
            embeddings_path: Path to embeddings .npy
            model_name: Embedding model name
        """
        print("="*60)
        print("INITIALIZING CONTEXT RETRIEVER")
        print("="*60)
        
        # Load FAISS index
        print(f"\nLoading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))
        print(f"Index loaded: {self.index.ntotal} vectors")
        
        # Load chunks
        print(f"\nLoading chunks: {chunks_path}")
        self.chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line.strip()))
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Load embeddings (for verification)
        print(f"\nLoading embeddings: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        # Verify consistency
        assert len(self.chunks) == self.embeddings.shape[0], \
            f"Mismatch: {len(self.chunks)} chunks vs {self.embeddings.shape[0]} embeddings"
        assert self.index.ntotal == len(self.chunks), \
            f"Mismatch: {self.index.ntotal} index vectors vs {len(self.chunks)} chunks"
        
        # Load embedding model
        print(f"\nLoading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
        # Tokenizer for token counting
        self.encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
        
        print("\n" + "="*60)
        print("RETRIEVER READY")
        print("="*60 + "\n")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string
        
        Args:
            query: Query text
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k chunks for a query
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores
        """
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            k
        )
        
        # Gather results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            chunk = self.chunks[idx].copy()
            chunk['retrieval_score'] = float(score)
            chunk['retrieval_rank'] = rank
            results.append(chunk)
        
        return results
    
    def build_context(self, 
                      chunks: List[Dict], 
                      max_chunks: int = None,
                      token_limit: int = None) -> Dict:
        """
        Build context from retrieved chunks
        
        Args:
            chunks: Retrieved chunks (in rank order)
            max_chunks: Maximum number of chunks to use
            token_limit: Maximum tokens (if None, use max_chunks)
            
        Returns:
            Context dictionary with text, metadata
        """
        context_parts = []
        total_tokens = 0
        chunks_used = 0
        
        # Determine limit
        limit = max_chunks if max_chunks is not None else len(chunks)
        
        for i, chunk in enumerate(chunks[:limit]):
            chunk_text = chunk['text']
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            # Check token limit
            if token_limit and (total_tokens + chunk_tokens > token_limit):
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            chunks_used += 1
        
        # Join with separators
        context_text = "\n\n---\n\n".join(context_parts)
        
        return {
            'text': context_text,
            'chunks_used': chunks_used,
            'total_tokens': total_tokens,
            'truncated': chunks_used < len(chunks),
            'chunk_ids': [c.get('article_id', 'unknown') for c in chunks[:chunks_used]]
        }
    
    def process_dataset(self, 
                        dataset_path: Path,
                        output_path: Path,
                        question_field: str = "question_text",
                        max_k: int = 10):
        """
        Process entire dataset and retrieve context for all questions
        
        Args:
            dataset_path: Path to dataset JSONL
            output_path: Path to save retrieval results
            question_field: Field name containing question text
            max_k: Maximum k to retrieve
        """
        print(f"\nProcessing dataset: {dataset_path}")
        print(f"Question field: {question_field}")
        print(f"Max k: {max_k}")
        print("="*60)
        
        results = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Retrieving context"):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract question
                    question = record.get(question_field, "")
                    if not question:
                        print(f"Warning: No question found in record")
                        continue
                    
                    # Retrieve chunks
                    retrieved_chunks = self.retrieve(question, k=max_k)
                    
                    # Build different context conditions
                    contexts = {}
                    
                    # Zero-shot (no context)
                    contexts['zero_shot'] = {
                        'text': '',
                        'chunks_used': 0,
                        'total_tokens': 0,
                        'truncated': False
                    }
                    
                    # Top-1, Top-5, Top-10
                    for k in [1, 5, 10]:
                        if k <= len(retrieved_chunks):
                            contexts[f'top_{k}'] = self.build_context(
                                retrieved_chunks,
                                max_chunks=k
                            )
                    
                    # Extended context (token-limited to 150k)
                    contexts['extended_150k'] = self.build_context(
                        retrieved_chunks,
                        token_limit=EXTENDED_TOKEN_LIMIT
                    )
                    
                    # Add to record
                    result = record.copy()
                    result['retrieved_contexts'] = contexts
                    result['retrieved_chunks'] = retrieved_chunks  # Keep for analysis
                    
                    results.append(result)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing record: {e}")
                    continue
        
        # Save results
        print(f"\nSaving {len(results)} records to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Statistics
        print("\n" + "="*60)
        print("RETRIEVAL COMPLETE")
        print("="*60)
        print(f"Records processed: {len(results):,}")
        
        # Sample statistics
        if results:
            sample = results[0]['retrieved_contexts']
            print("\nSample context sizes:")
            for condition, ctx in sample.items():
                print(f"  {condition:15s}: {ctx['chunks_used']:3d} chunks, {ctx['total_tokens']:6,d} tokens")
        
        print("="*60)
        
        return results


def main():
    """Main execution"""
    
    # Paths
    index_path = INDICES_DIR / "radiopaedia_index.faiss"
    chunks_path = EMBEDDINGS_DIR / "radiopaedia_chunks.jsonl"
    embeddings_path = EMBEDDINGS_DIR / "radiopaedia_embeddings.npy"
    
    # Create output directory
    retrieval_dir = PHASE2_DIR / "retrievals"
    retrieval_dir.mkdir(exist_ok=True)
    
    # Initialize retriever
    retriever = ContextRetriever(
        index_path=index_path,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path
    )
    
    # Process each dataset
    datasets = [
        {
            'name': 'radiology_dr',
            'path': RADIOLOGY_DR_PATH,
            'question_field': 'question_text',
            'output': retrieval_dir / 'radiology_dr_with_context.jsonl'
        },
        {
            'name': 'medqa_test',
            'path': MEDQA_PATH,
            'question_field': 'question_text',
            'output': retrieval_dir / 'medqa_test_with_context.jsonl'
        },
        {
            'name': 'pubmedqa',
            'path': PUBMEDQA_PATH,
            'question_field': 'question_text',
            'output': retrieval_dir / 'pubmedqa_with_context.jsonl'
        },
        {
            'name': 'risk_radiorag',
            'path': RISK_RADIORAG_PATH,
            'question_field': 'question_text',
            'output': retrieval_dir / 'risk_radiorag_with_context.jsonl'
        }
    ]
    
    for dataset in datasets:
        if dataset['path'].exists():
            print(f"\n{'='*60}")
            print(f"PROCESSING: {dataset['name'].upper()}")
            print(f"{'='*60}")
            
            retriever.process_dataset(
                dataset_path=dataset['path'],
                output_path=dataset['output'],
                question_field=dataset['question_field'],
                max_k=600  # Retrieve 600 chunks to fill up to 100k tokens
            )
        else:
            print(f"\nWarning: Dataset not found: {dataset['path']}")
    
    print("\n" + "="*60)
    print("ALL DATASETS PROCESSED")
    print("="*60)


if __name__ == "__main__":
    main()
