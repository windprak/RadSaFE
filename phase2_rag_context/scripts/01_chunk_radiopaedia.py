#!/usr/bin/env python3
"""
Step 1: Chunk Radiopaedia corpus into ~300-token passages with 50-token overlap
Following RadioRAG implementation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import tiktoken
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


class RadiopaediaChunker:
    """Chunk Radiopaedia articles into fixed-size token chunks"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
        
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk text into overlapping segments
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Tokenize
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.chunk_size:
            # Text fits in one chunk
            return [{
                "text": text,
                "tokens": len(tokens),
                "chunk_id": 0,
                "total_chunks": 1,
                **metadata
            }]
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "tokens": len(chunk_tokens),
                "chunk_id": chunk_id,
                "start_token": start,
                "end_token": end,
                **metadata
            })
            
            chunk_id += 1
            start += self.chunk_size - self.overlap
        
        # Add total chunks to all
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        return chunks
    
    def process_radiopaedia(self, input_path: Path, output_path: Path):
        """
        Process Radiopaedia JSONL file and create chunks
        
        Args:
            input_path: Path to radiopedia.jsonl
            output_path: Path to output chunked JSONL
        """
        print(f"Processing Radiopaedia corpus: {input_path}")
        print(f"Chunk size: {self.chunk_size} tokens")
        print(f"Overlap: {self.overlap} tokens")
        print("="*60)
        
        all_chunks = []
        total_articles = 0
        total_tokens = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Chunking articles"):
                try:
                    article = json.loads(line.strip())
                    
                    # Extract article content
                    # Adjust field names based on actual radiopedia.jsonl structure
                    article_id = article.get('id', article.get('slug', f'article_{total_articles}'))
                    title = article.get('title', '')
                    content = article.get('content', article.get('text', article.get('markdown', '')))
                    
                    # Combine title and content
                    full_text = f"{title}\n\n{content}" if title else content
                    
                    if not full_text.strip():
                        continue
                    
                    # Create metadata
                    metadata = {
                        "article_id": article_id,
                        "title": title,
                        "source": "radiopaedia"
                    }
                    
                    # Chunk the article
                    chunks = self.chunk_text(full_text, metadata)
                    all_chunks.extend(chunks)
                    
                    total_articles += 1
                    total_tokens += sum(c['tokens'] for c in chunks)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing article: {e}")
                    continue
        
        # Save chunks
        print(f"\nSaving {len(all_chunks)} chunks to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + '\n')
        
        # Statistics
        avg_chunks_per_article = len(all_chunks) / total_articles if total_articles > 0 else 0
        avg_tokens_per_chunk = total_tokens / len(all_chunks) if all_chunks else 0
        
        print("\n" + "="*60)
        print("CHUNKING COMPLETE")
        print("="*60)
        print(f"Total articles processed: {total_articles:,}")
        print(f"Total chunks created: {len(all_chunks):,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average chunks per article: {avg_chunks_per_article:.2f}")
        print(f"Average tokens per chunk: {avg_tokens_per_chunk:.2f}")
        print("="*60)
        
        return all_chunks


def main():
    """Main execution"""
    
    # Create output directory
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize chunker
    chunker = RadiopaediaChunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    # Process Radiopaedia
    output_path = EMBEDDINGS_DIR / "radiopaedia_chunks.jsonl"
    chunks = chunker.process_radiopaedia(RADIOPAEDIA_PATH, output_path)
    
    # Save metadata
    metadata_path = EMBEDDINGS_DIR / "radiopaedia_chunks_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "source": str(RADIOPAEDIA_PATH),
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "total_chunks": len(chunks),
            "output_path": str(output_path)
        }, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
