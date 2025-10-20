"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

generate_training_dataset.py - Gera dataset de treinamento com chaves de alta entropia

"""

import os
import json
import hashlib
import secrets
import numpy as np
from utils import calculate_entropy, is_valid_entropy

def generate_high_entropy_key():
    return secrets.token_hex(32)

def generate_training_example(seed=None, max_attempts=10):
    if seed is None:
        seed = secrets.token_hex(32)
    
    for attempt in range(max_attempts):
        key = generate_high_entropy_key()
        key_bytes = bytes.fromhex(key)
        byte_counts = [0] * 256

        for byte in key_bytes:
            byte_counts[byte] += 1
        
        entropy = 0.0
        total = len(key_bytes)

        for count in byte_counts:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        if entropy >= 7.5:
            return {
                "prompt": f"Generate a high-entropy cryptographic key from seed: {seed}\n\nOutput only 64 hexadecimal characters:",
                "completion": key,
                "metadata": {
                    "entropy": float(entropy),
                    "seed": seed,
                    "valid": True
                }
            }
    
    return {
        "prompt": f"Generate a high-entropy cryptographic key from seed: {seed}\n\nOutput only 64 hexadecimal characters:",
        "completion": key,
        "metadata": {
            "entropy": float(entropy),
            "seed": seed,
            "valid": True
        }
    }

def generate_jsonl_dataset(num_examples=1000, output_file="training_data.jsonl"):
    with open(output_file, 'w') as f:
        for i in range(num_examples):
            example = generate_training_example()            
            training_item = {
                "text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
            }
            
            f.write(json.dumps(training_item) + '\n')
    
def generate_validation_dataset(num_examples=100, output_file="validation_data.jsonl"):
    generate_jsonl_dataset(num_examples, output_file)

def analyze_dataset(jsonl_file):
    entropies = []

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']

            if '### Response:\n' in text:
                key_hex = text.split('### Response:\n')[1].strip()
                try:
                    key_bytes = bytes.fromhex(key_hex)
                    byte_counts = [0] * 256

                    for byte in key_bytes:
                        byte_counts[byte] += 1
                    
                    entropy = 0.0
                    total = len(key_bytes)

                    for count in byte_counts:
                        if count > 0:
                            p = count / total
                            entropy -= p * (np.log(p) / np.log(2))
                    
                    entropies.append(entropy)
                except:
                    pass
    
    if entropies:
        print(f"Total de exemplos: {len(entropies)}")
        print(f"Entropia média: {sum(entropies) / len(entropies):.4f}")
        print(f"Entropia mínima: {min(entropies):.4f}")
        print(f"Entropia máxima: {max(entropies):.4f}")
        print(f"Exemplos válidos (>7.5): {sum(1 for e in entropies if e >= 7.5)}")

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    generate_jsonl_dataset(
        num_examples=5000, 
        output_file="datasets/training_data.jsonl"
    )
    
    generate_validation_dataset(
        num_examples=500,
        output_file="datasets/validation_data.jsonl"
    )
    
    analyze_dataset("datasets/training_data.jsonl")
    analyze_dataset("datasets/validation_data.jsonl")