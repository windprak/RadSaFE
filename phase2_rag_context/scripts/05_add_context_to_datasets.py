#!/usr/bin/env python3
"""
Step 5: Add context fields to standardized datasets
Creates final datasets with all prompt conditions
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import tiktoken
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


class ContextAdder:
    """Add context fields to datasets with proper prompt formatting"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def format_options(self, options: Dict) -> str:
        """
        Format options for prompt
        
        Args:
            options: Dictionary with option keys and values
            
        Returns:
            Formatted options string
        """
        if not options:
            return ""
        
        lines = []
        for key in sorted(options.keys()):
            lines.append(f"{key}. {options[key]}")
        return "\n".join(lines)
    
    def build_prompt(self, 
                     question: str,
                     options: Dict,
                     context: str = "",
                     template: str = None) -> str:
        """
        Build complete prompt with optional context
        
        Args:
            question: Question text
            options: Answer options
            context: Retrieved context (empty for zero-shot)
            template: Prompt template to use
            
        Returns:
            Complete prompt string
        """
        formatted_options = self.format_options(options)
        
        if context:
            # Use context template
            template = template or CONTEXT_TEMPLATE
            prompt = template.format(
                context=context,
                question=question,
                options=formatted_options
            )
        else:
            # Use zero-shot template
            template = template or ZERO_SHOT_TEMPLATE
            prompt = template.format(
                question=question,
                options=formatted_options
            )
        
        return prompt
    
    def create_padded_control(self, 
                              top_1_context: str,
                              target_tokens: int) -> str:
        """
        Create padded control by repeating top-1 context
        
        Args:
            top_1_context: Top-1 retrieved context
            target_tokens: Target token count (e.g., from top-10)
            
        Returns:
            Padded context matching target token count
        """
        if not top_1_context:
            return ""
        
        current_tokens = self.count_tokens(top_1_context)
        
        if current_tokens >= target_tokens:
            return top_1_context
        
        # Repeat context until we reach target
        repetitions = (target_tokens // current_tokens) + 1
        padded = (top_1_context + "\n\n---\n\n") * repetitions
        
        # Truncate to exact target
        tokens = self.encoding.encode(padded)
        truncated_tokens = tokens[:target_tokens]
        
        return self.encoding.decode(truncated_tokens)
    
    def add_contexts_to_record(self, 
                               record: Dict,
                               question_field: str = "question_text",
                               options_field: str = "options") -> Dict:
        """
        Add all context conditions to a single record
        
        Args:
            record: Dataset record with retrieved_contexts
            question_field: Field containing question
            options_field: Field containing options
            
        Returns:
            Record with added prompt_conditions field
        """
        question = record.get(question_field, "")
        options = record.get(options_field, {})
        retrieved = record.get('retrieved_contexts', {})
        
        prompt_conditions = {}
        
        # Zero-shot
        zero_shot_prompt = self.build_prompt(question, options, context="")
        prompt_conditions['zero_shot'] = {
            'prompt': zero_shot_prompt,
            'tokens': self.count_tokens(zero_shot_prompt),
            'chunks_used': 0,
            'truncated': False,
            'condition_type': 'zero_shot'
        }
        
        # Top-k conditions
        for k in [1, 5, 10]:
            condition_key = f'top_{k}'
            if condition_key in retrieved:
                context_data = retrieved[condition_key]
                context_text = context_data.get('text', '')
                
                prompt = self.build_prompt(question, options, context=context_text)
                
                prompt_conditions[condition_key] = {
                    'prompt': prompt,
                    'tokens': self.count_tokens(prompt),
                    'chunks_used': context_data.get('chunks_used', 0),
                    'context_tokens': context_data.get('total_tokens', 0),
                    'truncated': context_data.get('truncated', False),
                    'condition_type': f'retrieval_top_{k}'
                }
        
        # Padded top-1 control (matches top-10 token count)
        if 'top_1' in retrieved and 'top_10' in retrieved:
            top_1_context = retrieved['top_1'].get('text', '')
            top_10_tokens = retrieved['top_10'].get('total_tokens', 0)
            
            padded_context = self.create_padded_control(top_1_context, top_10_tokens)
            padded_prompt = self.build_prompt(question, options, context=padded_context)
            
            prompt_conditions['padded_top_1'] = {
                'prompt': padded_prompt,
                'tokens': self.count_tokens(padded_prompt),
                'chunks_used': 1,
                'context_tokens': self.count_tokens(padded_context),
                'truncated': False,
                'condition_type': 'control_padded',
                'note': 'Top-1 context repeated to match top-10 token count'
            }
        
        # Extended context (150k token limit)
        if 'extended_150k' in retrieved:
            context_data = retrieved['extended_150k']
            context_text = context_data.get('text', '')
            
            prompt = self.build_prompt(question, options, context=context_text)
            
            prompt_conditions['extended_150k'] = {
                'prompt': prompt,
                'tokens': self.count_tokens(prompt),
                'chunks_used': context_data.get('chunks_used', 0),
                'context_tokens': context_data.get('total_tokens', 0),
                'truncated': context_data.get('truncated', False),
                'condition_type': 'extended_150k',
                'note': 'Can be sliced to model-specific limits using model tokenizer'
            }
        
        # Add to record
        result = record.copy()
        result['prompt_conditions'] = prompt_conditions
        
        return result
    
    def process_dataset(self,
                        input_path: Path,
                        output_path: Path,
                        question_field: str = "question_text",
                        options_field: str = "options"):
        """
        Process entire dataset and add prompt conditions
        
        Args:
            input_path: Path to dataset with retrieved contexts
            output_path: Path to save final dataset
            question_field: Field containing question
            options_field: Field containing options
        """
        print(f"\nProcessing: {input_path}")
        print(f"Output: {output_path}")
        print("="*60)
        
        records = []
        stats = {
            'total': 0,
            'conditions': {},
            'truncations': 0
        }
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Adding contexts"):
                try:
                    record = json.loads(line.strip())
                    
                    # Add contexts
                    processed = self.add_contexts_to_record(
                        record,
                        question_field=question_field,
                        options_field=options_field
                    )
                    
                    records.append(processed)
                    stats['total'] += 1
                    
                    # Track condition stats
                    for cond_name, cond_data in processed['prompt_conditions'].items():
                        if cond_name not in stats['conditions']:
                            stats['conditions'][cond_name] = {
                                'count': 0,
                                'total_tokens': 0,
                                'truncated': 0
                            }
                        
                        stats['conditions'][cond_name]['count'] += 1
                        stats['conditions'][cond_name]['total_tokens'] += cond_data['tokens']
                        if cond_data.get('truncated', False):
                            stats['conditions'][cond_name]['truncated'] += 1
                            stats['truncations'] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing record: {e}")
                    continue
        
        # Save results
        print(f"\nSaving {len(records)} records to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        # Save statistics
        stats_path = output_path.with_suffix('.stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print statistics
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total records: {stats['total']:,}")
        print(f"Total truncations: {stats['truncations']:,}")
        print("\nCondition statistics:")
        
        for cond_name, cond_stats in sorted(stats['conditions'].items()):
            avg_tokens = cond_stats['total_tokens'] / cond_stats['count'] if cond_stats['count'] > 0 else 0
            print(f"\n  {cond_name}:")
            print(f"    Records: {cond_stats['count']:,}")
            print(f"    Avg tokens: {avg_tokens:,.0f}")
            print(f"    Truncated: {cond_stats['truncated']:,}")
        
        print("="*60)
        
        return records, stats


def main():
    """Main execution"""
    
    print("="*60)
    print("ADDING CONTEXT TO STANDARDIZED DATASETS")
    print("="*60)
    
    # Create output directory
    final_dir = PHASE2_DIR / "final_datasets"
    final_dir.mkdir(exist_ok=True)
    
    retrieval_dir = PHASE2_DIR / "retrievals"
    
    # Initialize adder
    adder = ContextAdder()
    
    # Process each dataset
    datasets = [
        {
            'name': 'radiology_dr',
            'input': retrieval_dir / 'radiology_dr_with_context.jsonl',
            'output': final_dir / 'radiology_dr_final.jsonl',
            'question_field': 'question_text',
            'options_field': 'options'
        },
        {
            'name': 'medqa_test',
            'input': retrieval_dir / 'medqa_test_with_context.jsonl',
            'output': final_dir / 'medqa_test_final.jsonl',
            'question_field': 'question_text',
            'options_field': 'options'
        },
        {
            'name': 'pubmedqa',
            'input': retrieval_dir / 'pubmedqa_with_context.jsonl',
            'output': final_dir / 'pubmedqa_final.jsonl',
            'question_field': 'question_text',
            'options_field': 'options'
        },
        {
            'name': 'risk_radiorag',
            'input': retrieval_dir / 'risk_radiorag_with_context.jsonl',
            'output': final_dir / 'risk_radiorag_final.jsonl',
            'question_field': 'question_text',
            'options_field': 'options'
        }
    ]
    
    all_stats = {}
    
    for dataset in datasets:
        if dataset['input'].exists():
            print(f"\n{'='*60}")
            print(f"PROCESSING: {dataset['name'].upper()}")
            print(f"{'='*60}")
            
            records, stats = adder.process_dataset(
                input_path=dataset['input'],
                output_path=dataset['output'],
                question_field=dataset['question_field'],
                options_field=dataset['options_field']
            )
            
            all_stats[dataset['name']] = stats
        else:
            print(f"\nWarning: Input not found: {dataset['input']}")
    
    # Save combined statistics
    combined_stats_path = final_dir / 'combined_statistics.json'
    with open(combined_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL DATASETS PROCESSED")
    print(f"Combined statistics: {combined_stats_path}")
    print("="*60)


if __name__ == "__main__":
    main()
