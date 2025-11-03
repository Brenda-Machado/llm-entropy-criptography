"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

generate_training_dataset.py
"""

import os
import json
import hashlib
import secrets
import numpy as np
import time

def calculate_entropy(data_bytes):
    if not data_bytes:
        return 0.0
    
    byte_counts = [0] * 256

    for byte in data_bytes:
        byte_counts[byte] += 1

    entropy = 0.0
    total = len(data_bytes)

    for count in byte_counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy

def generate_high_entropy_key_v2(seed, method='mixed'):
    if method == 'secrets':
        return secrets.token_hex(32)
    
    elif method == 'hash':
        result = seed.encode() if isinstance(seed, str) else seed

        for i in range(3):
            salt = secrets.token_bytes(16)
            result = hashlib.sha256(result + salt).digest()

        return result.hex()
    
    elif method == 'mixed':
        source1 = secrets.token_bytes(16)
        source2 = hashlib.sha256((seed + str(time.time())).encode()).digest()[:16]
        source3 = secrets.token_bytes(16)
        combined = bytes(a ^ b ^ c for a, b, c in zip(source1, source2, source3))

        return combined.hex() + secrets.token_hex(16)
    
    elif method == 'prng':
        rng = np.random.RandomState(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))

        return ''.join(format(rng.randint(0, 255), '02x') for _ in range(32))
    
    else:
        return secrets.token_hex(32)

def validate_key_quality(key_hex):
    try:
        key_bytes = bytes.fromhex(key_hex)

    except ValueError:
        return {'valid': False, 'error': 'Invalid hex'}
    
    if len(key_bytes) != 32:
        return {'valid': False, 'error': 'Invalid length'}
    
    entropy = calculate_entropy(key_bytes)
    unique_bytes = len(set(key_bytes))
    has_repetition = False

    for i in range(len(key_bytes) - 3):
        if key_bytes[i] == key_bytes[i+1] == key_bytes[i+2] == key_bytes[i+3]:
            has_repetition = True
            break

    expected = len(key_bytes) / 256
    chi_square = sum((count - expected) ** 2 / expected for count in [key_bytes.count(i) for i in range(256)])
    quality_score = 0

    if entropy >= 7.9:
        quality_score += 40
    elif entropy >= 7.5:
        quality_score += 30
    elif entropy >= 7.0:
        quality_score += 20

    if unique_bytes >= 28:
        quality_score += 30
    elif unique_bytes >= 24:
        quality_score += 20

    if not has_repetition:
        quality_score += 20

    if chi_square < 300:
        quality_score += 10

    return {
        'valid': entropy >= 7.5 and unique_bytes >= 24,
        'entropy': float(entropy),
        'unique_bytes': unique_bytes,
        'has_repetition': has_repetition,
        'chi_square': float(chi_square),
        'quality_score': quality_score
    }

def generate_training_example_v2(seed=None, method='mixed', include_reasoning=False):
    if seed is None:
        seed = secrets.token_hex(32)

    max_attempts = 20

    for attempt in range(max_attempts):
        key = generate_high_entropy_key_v2(seed, method=method)
        quality = validate_key_quality(key)

        if quality['valid'] and quality['quality_score'] >= 80:
            break

    prompt_templates = [
        f"Generate a high-entropy cryptographic key from seed: {seed}\n\nOutput only 64 hexadecimal characters:",
        f"Create a cryptographically secure 256-bit key using seed: {seed}\n\nProvide exactly 64 hex characters:",
        f"Seed: {seed}\n\nGenerate 64 random hex digits (0-9,a-f) with maximum entropy:",
        f"Using seed {seed}, produce a 32-byte hexadecimal key with high Shannon entropy.\n\nOutput format: 64 hex characters only",
    ]

    prompt = np.random.choice(prompt_templates)

    return {
        "prompt": prompt,
        "completion": key,
        "metadata": {
            "entropy": quality['entropy'],
            "unique_bytes": quality['unique_bytes'],
            "quality_score": quality['quality_score'],
            "method": method,
            "seed": seed,
            "valid": quality['valid']
        }
    }

def generate_diverse_dataset(num_examples=1000, output_file="training_data_v2.jsonl"):
    methods = ['secrets', 'hash', 'mixed', 'prng']
    method_weights = [0.4, 0.2, 0.3, 0.1]
    examples_generated = 0
    high_quality_count = 0

    with open(output_file, 'w') as f:

        while examples_generated < num_examples:
            method = np.random.choice(methods, p=method_weights)
            example = generate_training_example_v2(method=method)

            if example['metadata']['quality_score'] >= 70:
                training_item = {
                    "text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}",
                    "metadata": example['metadata']
                }
                f.write(json.dumps(training_item) + '\n')
                examples_generated += 1

                if example['metadata']['quality_score'] >= 90:
                    high_quality_count += 1
    
    print(f"\n Dataset gerado: {output_file}")

def analyze_dataset_v2(jsonl_file):
    entropies = []
    unique_bytes_list = []
    quality_scores = []
    methods_count = {}


    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            if 'metadata' in data:
                meta = data['metadata']
                entropies.append(meta['entropy'])
                unique_bytes_list.append(meta['unique_bytes'])
                quality_scores.append(meta['quality_score'])
                method = meta.get('method', 'unknown')
                methods_count[method] = methods_count.get(method, 0) + 1
    
    for method, count in sorted(methods_count.items()):
            print(f"  {method:10s}: {count:4d} ({count/len(entropies)*100:.1f}%)")

def create_stratified_splits(input_file, train_ratio=0.8):
    examples = []

    with open(input_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    high_quality = [e for e in examples if e.get('metadata', {}).get('quality_score', 0) >= 90]
    mid_quality = [e for e in examples if 80 <= e.get('metadata', {}).get('quality_score', 0) < 90]
    np.random.shuffle(high_quality)
    np.random.shuffle(mid_quality)
    train_high = high_quality[:int(len(high_quality) * train_ratio)]
    val_high = high_quality[int(len(high_quality) * train_ratio):]
    train_mid = mid_quality[:int(len(mid_quality) * train_ratio)]
    val_mid = mid_quality[int(len(mid_quality) * train_ratio):]
    train_data = train_high + train_mid
    val_data = val_high + val_mid
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    with open("datasets/training_data_v2.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open("datasets/validation_data_v2.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    temp_file = "datasets/temp_full_dataset.jsonl"
    generate_diverse_dataset(num_examples=10000, output_file=temp_file)
    analyze_dataset_v2(temp_file)
    create_stratified_splits(temp_file)
    analyze_dataset_v2("datasets/training_data_v2.jsonl")
    analyze_dataset_v2("datasets/validation_data_v2.jsonl")
