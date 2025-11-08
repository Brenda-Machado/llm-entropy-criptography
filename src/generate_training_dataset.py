"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

generate_training_dataset.py
"""

import json
import secrets
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple
from vector_store import VectorStore, QualityMetrics
from prompt_engineering import PromptTemplate

class TrainingDatasetGenerator:
    
    def __init__(self, 
                 output_dir: str = "datasets",
                 key_size_bits: int = 256):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.key_size_bits = key_size_bits
        vector_store_path = self.output_dir / "vector_store.jsonl"
        self.vector_store = VectorStore(str(vector_store_path))
        self.prompt_templates = [
            PromptTemplate(key_size_bits=key_size_bits, strategy='zero-shot'),
            PromptTemplate(key_size_bits=key_size_bits, strategy='few-shot'),
            PromptTemplate(key_size_bits=key_size_bits, strategy='cot')
        ]
    
    def generate_high_quality_example(self) -> Dict:
        seed = secrets.token_hex(32)
        key_hex = secrets.token_hex(self.key_size_bits // 8)
        key_bytes = bytes.fromhex(key_hex)
        entropy = QualityMetrics.calculate_shannon_entropy(key_bytes)
        unique_bytes = QualityMetrics.count_unique_bytes(key_bytes)
        quality_score = QualityMetrics.calculate_quality_score(key_bytes, self.key_size_bits)
        repetitions = QualityMetrics.check_repetition_pattern(key_bytes)
        sequences = QualityMetrics.check_sequential_pattern(key_bytes)
        
        return {
            'seed': seed,
            'key': key_hex,
            'metrics': {
                'entropy': float(entropy),
                'unique_bytes': int(unique_bytes),
                'unique_ratio': float(unique_bytes / len(key_bytes)),
                'quality_score': float(quality_score),
                'repetition_violations': int(repetitions),
                'sequence_violations': int(sequences)
            }
        }
    
    def format_training_example(self, 
                                seed: str, 
                                key_hex: str, 
                                metrics: Dict,
                                strategy: str = 'instructional') -> Dict:
        if strategy == 'instructional':
            instruction = (
                f"Generate a high-entropy {self.key_size_bits}-bit cryptographic key "
                f"from the following seed.\n\n"
                f"Seed: {seed}\n\n"
                f"Requirements:\n"
                f"- Output exactly {self.key_size_bits // 4} hexadecimal characters\n"
                f"- Maximize entropy (target: ≥7.9 bits/byte)\n"
                f"- Ensure uniform distribution of all hex digits\n"
                f"- Avoid patterns and repetitions\n\n"
                f"Output the key:"
            )
            
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{key_hex}"
        
        elif strategy == 'conversational':
            text = (
                f"User: I need a secure {self.key_size_bits}-bit cryptographic key. "
                f"Here's my seed: {seed}. Can you generate a high-entropy key?\n\n"
                f"Assistant: I'll generate a cryptographically secure {self.key_size_bits}-bit key "
                f"with maximum entropy. Here's your key:\n\n{key_hex}"
            )
        
        elif strategy == 'technical':
            text = (
                f"<task>cryptographic_key_generation</task>\n"
                f"<key_size>{self.key_size_bits}_bits</key_size>\n"
                f"<input_seed>{seed}</input_seed>\n"
                f"<requirements>\n"
                f"- Shannon entropy: ≥7.9 bits/byte\n"
                f"- Unique bytes: ≥90%\n"
                f"- No repetition patterns\n"
                f"- No sequential patterns\n"
                f"</requirements>\n"
                f"<output_key>{key_hex}</output_key>"
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return {
            'text': text,
            'metadata': {
                'seed': seed,
                'key': key_hex,
                'key_size_bits': self.key_size_bits,
                'format_strategy': strategy,
                **metrics
            }
        }
    
    def generate_dataset(self,
                        num_examples: int,
                        min_quality_score: float = 85.0,
                        format_strategy: str = 'instructional',
                        save_to_vector_store: bool = True) -> Tuple[List[Dict], Dict]:

        print(f"\n{'='*80}")
        print(f"GERANDO DATASET DE TREINAMENTO")
        print(f"{'='*80}")
        print(f"Tamanho da chave: {self.key_size_bits} bits")
        print(f"Número de exemplos: {num_examples}")
        print(f"Quality score mínimo: {min_quality_score}")
        print(f"Estratégia de formato: {format_strategy}")
        print(f"{'='*80}\n")
        
        examples = []
        rejected = 0
        quality_scores = []
        entropies = []
        start_time = time.time()
        attempts = 0
        max_attempts = num_examples * 3  
        
        while len(examples) < num_examples and attempts < max_attempts:
            attempts += 1
            example_data = self.generate_high_quality_example()

            if example_data['metrics']['quality_score'] >= min_quality_score:
                formatted = self.format_training_example(
                    seed=example_data['seed'],
                    key_hex=example_data['key'],
                    metrics=example_data['metrics'],
                    strategy=format_strategy
                )
                
                examples.append(formatted)
                quality_scores.append(example_data['metrics']['quality_score'])
                entropies.append(example_data['metrics']['entropy'])
                
                if save_to_vector_store:
                    self.vector_store.add_example(
                        seed=example_data['seed'],
                        key_hex=example_data['key'],
                        metadata={
                            'format_strategy': format_strategy,
                            'dataset_generation': True
                        }
                    )
            else:
                rejected += 1
            
            if len(examples) % 100 == 0 and len(examples) > 0:
                elapsed = time.time() - start_time
                rate = len(examples) / elapsed
                eta = (num_examples - len(examples)) / rate if rate > 0 else 0
                
                print(f"Progresso: {len(examples)}/{num_examples} "
                      f"({len(examples)/num_examples*100:.1f}%) | "
                      f"Rejeitados: {rejected} | "
                      f"Rate: {rate:.1f} ex/s | "
                      f"ETA: {eta:.0f}s")
        
        if save_to_vector_store:
            self.vector_store.save()
        
        elapsed = time.time() - start_time
        stats = {
            'total_examples': len(examples),
            'rejected_examples': rejected,
            'acceptance_rate': len(examples) / attempts * 100 if attempts > 0 else 0,
            'generation_time': elapsed,
            'rate': len(examples) / elapsed if elapsed > 0 else 0
        }
        
        if quality_scores:
            stats.update({
                'quality_score': {
                    'mean': float(np.mean(quality_scores)),
                    'std': float(np.std(quality_scores)),
                    'min': float(np.min(quality_scores)),
                    'max': float(np.max(quality_scores)),
                    'median': float(np.median(quality_scores))
                },
                'entropy': {
                    'mean': float(np.mean(entropies)),
                    'std': float(np.std(entropies)),
                    'min': float(np.min(entropies)),
                    'max': float(np.max(entropies)),
                    'median': float(np.median(entropies))
                },
                'excellent_count': sum(1 for s in quality_scores if s >= 95),
                'high_quality_count': sum(1 for s in quality_scores if s >= 90),
                'valid_entropy_count': sum(1 for e in entropies if e >= 7.9)
            })
        
        return examples, stats
    
    def save_dataset(self, 
                    examples: List[Dict], 
                    filename: str,
                    split: str = 'train'):
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"\n✓ {split.capitalize()} dataset salvo: {filepath}")
        print(f"  Total de exemplos: {len(examples)}")
    
    def analyze_dataset(self, filepath: str):
        examples = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        if not examples:
            print(f"Dataset vazio: {filepath}")
            return
        
        print(f"\n{'='*80}")
        print(f"ANÁLISE DO DATASET: {filepath}")
        print(f"{'='*80}")
        
        quality_scores = [ex['metadata']['quality_score'] for ex in examples if 'metadata' in ex]
        entropies = [ex['metadata']['entropy'] for ex in examples if 'metadata' in ex]
        
        if quality_scores:
            print(f"\nTotal de exemplos: {len(examples)}")
            print(f"\nQuality Score:")
            print(f"  Média:   {np.mean(quality_scores):.2f}")
            print(f"  Mediana: {np.median(quality_scores):.2f}")
            print(f"  Mín-Máx: {np.min(quality_scores):.2f} - {np.max(quality_scores):.2f}")
            print(f"  StdDev:  {np.std(quality_scores):.2f}")
            
            print(f"\nEntropia Shannon:")
            print(f"  Média:   {np.mean(entropies):.4f} bits/byte")
            print(f"  Mediana: {np.median(entropies):.4f}")
            print(f"  Mín-Máx: {np.min(entropies):.4f} - {np.max(entropies):.4f}")
            
            print(f"\nDistribuição por qualidade:")
            print(f"  Excelente (≥95): {sum(1 for s in quality_scores if s >= 95)} "
                  f"({sum(1 for s in quality_scores if s >= 95)/len(quality_scores)*100:.1f}%)")
            print(f"  Alta (≥90):      {sum(1 for s in quality_scores if s >= 90)} "
                  f"({sum(1 for s in quality_scores if s >= 90)/len(quality_scores)*100:.1f}%)")
            print(f"  Boa (≥85):       {sum(1 for s in quality_scores if s >= 85)} "
                  f"({sum(1 for s in quality_scores if s >= 85)/len(quality_scores)*100:.1f}%)")
            
            print(f"\nEntropia válida (≥7.9): {sum(1 for e in entropies if e >= 7.9)} "
                  f"({sum(1 for e in entropies if e >= 7.9)/len(entropies)*100:.1f}%)")

        print(f"\n{'='*80}")
        print("EXEMPLO DE ENTRADA:")
        print(f"{'='*80}")

        example = examples[0]
        print(example['text'][:500] + "..." if len(example['text']) > 500 else example['text'])
        
        if 'metadata' in example:
            print(f"\nMétrica do exemplo:")
            print(f"  Quality Score: {example['metadata']['quality_score']:.2f}")
            print(f"  Entropia: {example['metadata']['entropy']:.4f}")


def generate_multiple_datasets(key_size_bits: int = 256,
                               sizes: Dict[str, Tuple[int, int]] = None):

    if sizes is None:
        sizes = {
            'tiny': (100, 20),
            'small': (1000, 200),
            'medium': (5000, 1000),
            'large': (10000, 2000),
            'xlarge': (50000, 10000)
        }
    
    generator = TrainingDatasetGenerator(key_size_bits=key_size_bits)
    size_name = 'large'  
    train_size, val_size = sizes[size_name]
    
    print(f"\n{'='*80}")
    print(f"CONFIGURAÇÃO: {size_name.upper()}")
    print(f"{'='*80}")
    print(f"Training examples: {train_size:,}")
    print(f"Validation examples: {val_size:,}")
    print(f"Key size: {key_size_bits} bits")
    
    strategies = ['instructional']  
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"ESTRATÉGIA DE FORMATO: {strategy.upper()}")
        print(f"{'='*80}")
        
        print(f"\n--- Gerando Training Dataset ---")

        train_examples, train_stats = generator.generate_dataset(
            num_examples=train_size,
            min_quality_score=85.0,
            format_strategy=strategy,
            save_to_vector_store=True
        )
        
        train_file = f"training_data_{strategy}_{key_size_bits}bit.jsonl"
        generator.save_dataset(train_examples, train_file, split='train')
        
        print(f"\n--- Gerando Validation Dataset ---")

        val_examples, val_stats = generator.generate_dataset(
            num_examples=val_size,
            min_quality_score=85.0,
            format_strategy=strategy,
            save_to_vector_store=True
        )
        
        val_file = f"validation_data_{strategy}_{key_size_bits}bit.jsonl"
        generator.save_dataset(val_examples, val_file, split='validation')
        generator.analyze_dataset(generator.output_dir / train_file)
        generator.analyze_dataset(generator.output_dir / val_file)
    
    print(f"\n{'='*80}")
    print("ESTATÍSTICAS DO VECTOR STORE")
    print(f"{'='*80}")
    
    vs_stats = generator.vector_store.get_statistics()
    print(f"Total de exemplos armazenados: {vs_stats['total_examples']}")
    print(f"Exemplos de alta qualidade (≥90): {vs_stats['high_quality_count']}")
    print(f"Entropia média: {vs_stats['entropy_stats']['mean']:.4f}")
    print(f"Quality score médio: {vs_stats['quality_stats']['mean']:.2f}")
    
    print(f"\n{'='*80}")
    print("DATASETS GERADOS COM SUCESSO!")
    print(f"{'='*80}")
    print(f"\nArquivos salvos em: {generator.output_dir}/")
    print(f"Vector store: {generator.vector_store.storage_path}")


if __name__ == "__main__":
    print("="*80)
    print("GERADOR DE DATASET DE TREINAMENTO")
    print("="*80)
    
    # Gerar datasets para diferentes tamanhos de chave
    for key_size in [256]:  # Pode adicionar: 128, 192
        print(f"\n{'#'*80}")
        print(f"# TAMANHO DE CHAVE: {key_size} bits")
        print(f"{'#'*80}")
        
        generate_multiple_datasets(key_size_bits=key_size)
    
    print("\n" + "="*80)
    print("GERAÇÃO COMPLETA!")
    print("="*80)