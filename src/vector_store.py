"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

vector_store.py
"""

import json
import numpy as np
import secrets
from pathlib import Path
from typing import List, Dict

class QualityMetrics:

    @staticmethod
    def calculate_shannon_entropy(key_bytes: bytes) -> float:
        if len(key_bytes) == 0:
            return 0.0
        
        counts = np.bincount(np.frombuffer(key_bytes, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(key_bytes)

        return float(-np.sum(probs * np.log2(probs)))
    
    @staticmethod
    def count_unique_bytes(key_bytes: bytes) -> int:
        return len(set(key_bytes))
    
    @staticmethod
    def check_repetition_pattern(key_bytes: bytes, window_size: int = 4) -> int:
        violations = 0

        for i in range(len(key_bytes) - window_size + 1):
            window = key_bytes[i:i+window_size]

            if len(set(window)) == 1:
                violations += 1

        return violations
    
    @staticmethod
    def check_sequential_pattern(key_bytes: bytes) -> int:
        violations = 0
        for i in range(len(key_bytes) - 3):
            if (key_bytes[i+1] == key_bytes[i] + 1 and 
                key_bytes[i+2] == key_bytes[i] + 2 and
                key_bytes[i+3] == key_bytes[i] + 3):
                violations += 1

            elif (key_bytes[i+1] == key_bytes[i] - 1 and 
                  key_bytes[i+2] == key_bytes[i] - 2 and
                  key_bytes[i+3] == key_bytes[i] - 3):
                violations += 1

        return violations
    
    @staticmethod
    def calculate_quality_score(key_bytes: bytes, key_size_bits: int = 256) -> float:
        expected_length = key_size_bits // 8

        if len(key_bytes) != expected_length:
            return 0.0
        
        score = 0.0
        entropy = QualityMetrics.calculate_shannon_entropy(key_bytes)

        if entropy >= 7.9:
            score += 40
        elif entropy >= 7.5:
            score += 30
        elif entropy >= 7.0:
            score += 20
        elif entropy >= 6.5:
            score += 10
        
        unique_ratio = QualityMetrics.count_unique_bytes(key_bytes) / expected_length

        if unique_ratio >= 0.9:
            score += 30
        elif unique_ratio >= 0.75:
            score += 20
        elif unique_ratio >= 0.5:
            score += 10
        
        repetitions = QualityMetrics.check_repetition_pattern(key_bytes)

        if repetitions == 0:
            score += 15
        elif repetitions <= 2:
            score += 10
        elif repetitions <= 5:
            score += 5

        sequences = QualityMetrics.check_sequential_pattern(key_bytes)

        if sequences == 0:
            score += 15
        elif sequences <= 2:
            score += 10
        
        return score


class VectorStore:
    def __init__(self, storage_path: str = "datasets/vector_store.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.examples = []
        self.load()
    
    def add_example(self, seed: str, key_hex: str, metadata: Dict = None):
        key_bytes = bytes.fromhex(key_hex)
        key_size_bits = len(key_bytes) * 8
        entropy = QualityMetrics.calculate_shannon_entropy(key_bytes)
        unique_bytes = QualityMetrics.count_unique_bytes(key_bytes)
        quality_score = QualityMetrics.calculate_quality_score(key_bytes, key_size_bits)
        repetitions = QualityMetrics.check_repetition_pattern(key_bytes)
        sequences = QualityMetrics.check_sequential_pattern(key_bytes)
        
        example = {
            'seed': seed,
            'key': key_hex,
            'key_size_bits': key_size_bits,
            'metrics': {
                'entropy': float(entropy),
                'unique_bytes': int(unique_bytes),
                'unique_ratio': float(unique_bytes / len(key_bytes)),
                'quality_score': float(quality_score),
                'repetition_violations': int(repetitions),
                'sequence_violations': int(sequences)
            }
        }
        
        if metadata:
            example['metadata'] = metadata
        
        self.examples.append(example)
    
    def save(self):
        with open(self.storage_path, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example) + '\n')
    
    def load(self):
        if not self.storage_path.exists():
            return
        
        self.examples = []

        with open(self.storage_path, 'r') as f:
            for line in f:

                if line.strip():
                    self.examples.append(json.loads(line))
    
    def get_top_examples(self, 
                         n: int = 10, 
                         key_size_bits: int = 256,
                         min_quality: float = 85.0,
                         min_entropy: float = 7.9) -> List[Dict]:
        filtered = [
            ex for ex in self.examples

            if (ex['key_size_bits'] == key_size_bits and
                ex['metrics']['quality_score'] >= min_quality and
                ex['metrics']['entropy'] >= min_entropy)
        ]
        
        sorted_examples = sorted(
            filtered,
            key=lambda x: x['metrics']['quality_score'],
            reverse=True
        )
        
        return sorted_examples[:n]
    
    def get_diverse_examples(self,
                            n: int = 10,
                            key_size_bits: int = 256,
                            min_quality: float = 85.0) -> List[Dict]:
        
        top_examples = self.get_top_examples(
            n=n*3, 
            key_size_bits=key_size_bits,
            min_quality=min_quality
        )
        
        if len(top_examples) <= n:
            return top_examples
        
        selected = [top_examples[0]]  
        
        for candidate in top_examples[1:]:
            if len(selected) >= n:
                break
            
            candidate_bytes = bytes.fromhex(candidate['key'])[:8]  
            
            is_diverse = True
            for selected_ex in selected:
                selected_bytes = bytes.fromhex(selected_ex['key'])[:8]
                diff = sum(a != b for a, b in zip(candidate_bytes, selected_bytes))

                if diff < 4: 
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        return selected
    
    def get_statistics(self) -> Dict:
        if not self.examples:
            return {'total_examples': 0}
        
        entropies = [ex['metrics']['entropy'] for ex in self.examples]
        quality_scores = [ex['metrics']['quality_score'] for ex in self.examples]
        by_size = {}

        for ex in self.examples:
            size = ex['key_size_bits']

            if size not in by_size:
                by_size[size] = 0
            by_size[size] += 1
        
        return {
            'total_examples': len(self.examples),
            'by_key_size': by_size,
            'entropy_stats': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
                'median': float(np.median(entropies))
            },
            'quality_stats': {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores)),
                'median': float(np.median(quality_scores))
            },
            'high_quality_count': sum(1 for score in quality_scores if score >= 90),
            'excellent_entropy_count': sum(1 for e in entropies if e >= 7.9)
        }


def generate_high_quality_dataset(vector_store: VectorStore,
                                  num_examples: int = 1000,
                                  key_size_bits: int = 256):
    print(f"Gerando {num_examples} exemplos de alta qualidade...")
    print(f"Tamanho da chave: {key_size_bits} bits ({key_size_bits//4} hex chars)")
    
    high_quality_count = 0
    
    for i in range(num_examples):
        seed = secrets.token_hex(32)
        key_hex = secrets.token_hex(key_size_bits // 8)
        
        vector_store.add_example(seed, key_hex)
        
        if vector_store.examples[-1]['metrics']['quality_score'] >= 90:
            high_quality_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Progresso: {i+1}/{num_examples} "
                  f"(Alta qualidade: {high_quality_count}/{i+1} = "
                  f"{high_quality_count/(i+1)*100:.1f}%)")
    
    vector_store.save()
    
    stats = vector_store.get_statistics()
    
    print(f"\nEstatísticas:")
    print(f"  Total de exemplos: {stats['total_examples']}")
    print(f"  Exemplos de alta qualidade (≥90): {stats['high_quality_count']}")
    print(f"  Entropia média: {stats['entropy_stats']['mean']:.4f}")
    print(f"  Quality score médio: {stats['quality_stats']['mean']:.2f}")
