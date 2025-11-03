"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

generate_training_dataset.py
"""

import os
import json
import secrets
import numpy as np
import time

def calculate_entropy_corrected(key_bytes):
    if len(key_bytes) == 0:
        return 0.0
    
    counts = np.bincount(np.frombuffer(key_bytes, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(key_bytes)

    return float(-np.sum(probs * np.log2(probs)))

def generate_training_examples(num_examples, method='secrets'):

    prompt_templates = [
        "Generate a high-entropy cryptographic key from seed: {seed}\n\nOutput only 64 hexadecimal characters:",
        "Create a cryptographically secure 256-bit key using seed: {seed}\n\nProvide exactly 64 hex characters:",
        "Seed: {seed}\n\nGenerate 64 random hex digits (0-9,a-f) with maximum entropy:",
        "Using seed {seed}, produce a 32-byte hexadecimal key.\n\nOutput format: 64 hex characters",
    ]
    
    examples = []
    start_time = time.time()
    
    for i in range(num_examples):
        seed = secrets.token_hex(32)

        if method == 'mixed':
            source1 = secrets.token_bytes(16)
            source2 = secrets.token_bytes(16)
            combined = bytes(a ^ b for a, b in zip(source1, source2))
            key = combined.hex() + secrets.token_hex(16)
        else:
            key = secrets.token_hex(32)
        
        key_bytes = bytes.fromhex(key)
        entropy = calculate_entropy_corrected(key_bytes)
        unique_bytes = len(set(key_bytes))
        prompt = np.random.choice(prompt_templates).format(seed=seed)
        
        examples.append({
            "text": f"### Instruction:\n{prompt}\n\n### Response:\n{key}",
            "metadata": {
                "entropy": entropy,
                "unique_bytes": unique_bytes,
                "method": method,
                "seed": seed
            }
        })
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_examples - i - 1) / rate
            print(f"Progresso: {i+1}/{num_examples} ({(i+1)/num_examples*100:.1f}%) | "
                  f"Rate: {rate:.1f} ex/s | ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    
    return examples

def save_dataset(examples, output_file):
    with open(output_file, 'w') as f:
        for item in examples:
            f.write(json.dumps(item) + '\n')

def analyze_dataset(jsonl_file):
    entropies = []
    unique_counts = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'metadata' in data:
                entropies.append(data['metadata']['entropy'])
                unique_counts.append(data['metadata']['unique_bytes'])
    
    if entropies:

        print(f"Total de exemplos: {len(entropies)}")
        print(f"\nEntropia Shannon (esperado: 5.0-5.5 para 32 bytes):")
        print(f"  Média:   {np.mean(entropies):.4f} bits/byte")
        print(f"  Mediana: {np.median(entropies):.4f}")
        print(f"  StdDev:  {np.std(entropies):.4f}")
        
        print(f"\nBytes únicos (de 32 possíveis):")
        print(f"  Média:   {np.mean(unique_counts):.2f}")

        print(f"\nDistribuição de entropia:")
        print(f"  ≥5.5: {sum(1 for e in entropies if e >= 5.5)} ({sum(1 for e in entropies if e >= 5.5)/len(entropies)*100:.1f}%)")
        print(f"  ≥5.0: {sum(1 for e in entropies if e >= 5.0)} ({sum(1 for e in entropies if e >= 5.0)/len(entropies)*100:.1f}%)")
        print(f"  ≥4.5: {sum(1 for e in entropies if e >= 4.5)} ({sum(1 for e in entropies if e >= 4.5)/len(entropies)*100:.1f}%)")
        
        # Mostra exemplo
        with open(jsonl_file, 'r') as f:
            first = json.loads(f.readline())
            print(f"\nExemplo de entrada:")
            text = first['text']
            parts = text.split('### Response:\n')
            if len(parts) == 2:
                instruction = parts[0].replace('### Instruction:\n', '').strip()
                response = parts[1].strip()
                print(f"  Prompt: {instruction[:80]}...")
                print(f"  Chave:  {response}")
                print(f"  Entropia: {first['metadata']['entropy']:.4f}")

def test_entropy_calculation():
    entropies = []

    for i in range(10):
        key = secrets.token_hex(32)
        key_bytes = bytes.fromhex(key)
        entropy = calculate_entropy_corrected(key_bytes)
        unique = len(set(key_bytes))
        entropies.append(entropy)
        if i < 3:
            print(f"  Chave {i+1}: entropia={entropy:.4f}, únicos={unique}/32")

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    test_entropy_calculation()

    SIZES = {
        'tiny': (100, 20),          
        'small': (1000, 100),        
        'medium': (5000, 500),       
        'large': (10000, 1000),      
        'xlarge': (50000, 5000),     
    }
    
    size = 'large' 
    train_size, val_size = SIZES[size]
    train_file = f"datasets/training_data.jsonl"
    train_examples = generate_training_examples(train_size, method='mixed')

    save_dataset(train_examples, train_file)
    analyze_dataset(train_file)

    val_file = f"datasets/validation_data.jsonl"
    val_examples = generate_training_examples(val_size, method='mixed')
    save_dataset(val_examples, val_file)
    analyze_dataset(val_file)
    
    print(f"\nArquivos gerados:")
    print(f"  📁 {train_file} ({train_size:,} exemplos)")
    print(f"  📁 {val_file} ({val_size:,} exemplos)")
