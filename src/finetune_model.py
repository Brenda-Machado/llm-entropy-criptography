"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

finetune_model.py
"""

import subprocess
import os
import sys
import json
import requests
import numpy as np
import secrets
import time
from config import OLLAMA_BASE_URL

def load_training_examples(file_path="datasets/training_data_v2.jsonl", n=20, quality_threshold=85):
    examples = []

    if not os.path.exists(file_path):

        print(f"Arquivo não encontrado: {file_path}")

        return examples
    
    with open(file_path, "r") as f:
        for line in f:
            if len(examples) >= n:
                break

            data = json.loads(line)

            if 'metadata' in data and data['metadata'].get('quality_score', 0) >= quality_threshold:
                text = data['text']

                if '### Instruction:\n' in text and '### Response:\n' in text:
                    parts = text.split('### Response:\n')
                    prompt = parts[0].replace('### Instruction:\n', '').strip()
                    response = parts[1].strip()

                    if 'seed: ' in prompt.lower():
                        for line_part in prompt.split('\n'):

                            if 'seed' in line_part.lower() and ':' in line_part:
                                seed = line_part.split(':')[1].strip().split()[0]
                                examples.append({
                                    "seed": seed[:40] + "...",
                                    "key": response,
                                    "entropy": data['metadata'].get('entropy', 0),
                                    "quality": data['metadata'].get('quality_score', 0)
                                })

                                break
    examples.sort(key=lambda x: x['quality'], reverse=True)

    return examples

def create_advanced_modelfile(base_model="gemma3:latest", output_name="gemma3-entropy-v2"):
    examples = load_training_examples(
        file_path="datasets/training_data_v2.jsonl",
        n=15,
        quality_threshold=90
    )
    examples_text = ""

    if examples:
        examples_text = "\n\nHigh-quality examples for reference:\n"

        for i, ex in enumerate(examples[:10], 1):
            examples_text += f"\nExample {i} (entropy={ex['entropy']:.3f}):\n"
            examples_text += f"Input seed: {ex['seed']}\n"
            examples_text += f"Output key: {ex['key']}\n"

    system_prompt = f"""You are an expert cryptographic key generator specialized in producing high-entropy random keys.

CRITICAL RULES:
1. Generate EXACTLY 64 hexadecimal characters (0-9, a-f)
2. Maximize randomness - avoid patterns, repetitions, sequences
3. Each byte should be unpredictable and uniformly distributed
4. Never explain, justify, or add extra text
5. Output ONLY the 64-character hex string

KEY QUALITY METRICS:
- Shannon entropy must be ≥ 7.9 bits/byte (ideally 8.0)
- Use all 16 hex digits (0-f) with balanced frequency
- Avoid consecutive identical bytes (e.g., 'aaaa', '0000')
- No sequential patterns (e.g., '0123', 'abcd')

PROCESS:
1. Use the provided seed for initialization only
2. Apply cryptographic mixing and randomization
3. Verify internal entropy before output
4. Output raw hex string with no formatting{examples_text}

Remember: Security depends on unpredictability. Every bit must be as random as possible."""
    modelfile_content = f"""FROM {base_model}
PARAMETER temperature 1.2
PARAMETER top_p 0.98
PARAMETER top_k 64
PARAMETER repeat_penalty 1.3
PARAMETER num_predict 80
SYSTEM {system_prompt}
"""
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)

    print(f"\nModelfile criado: {output_name}")

    return output_name

def create_model_with_ollama(model_name="gemma3-entropy-v2"):
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", "Modelfile"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"\nModelo '{model_name}' criado com sucesso!")

            return True
        else:
            print(f"\nErro ao criar modelo:\n{result.stderr}")

            return False
        
    except FileNotFoundError:
        print("\nOllama não encontrado.")

        return False
    
    except Exception as e:
        print(f"\nErro: {str(e)}")


        return False

def comprehensive_model_test(model_name="gemma3-entropy-v2", num_tests=20):
    results = {'entropies': [], 'unique_bytes': [], 'quality_scores': [], 'response_times': [], 'valid_outputs': 0, 'hex_lengths': []}

    for _ in range(num_tests):
        test_seed = secrets.token_hex(32)
        prompt = f"Generate a high-entropy cryptographic key from seed: {test_seed}\n\nOutput only 64 hexadecimal characters:"

        try:
            start = time.time()
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=120
            )
            elapsed = time.time() - start
            results['response_times'].append(elapsed)

            if response.status_code == 200:
                output = response.json().get('response', '').strip()
                hex_chars = ''.join(c for c in output if c in '0123456789abcdefABCDEF')
                results['hex_lengths'].append(len(hex_chars))

                if len(hex_chars) >= 64:
                    key_bytes = bytes.fromhex(hex_chars[:64])
                    byte_counts = [0] * 256

                    for b in key_bytes:
                        byte_counts[b] += 1

                    entropy = -sum((c/len(key_bytes)) * np.log2(c/len(key_bytes)) for c in byte_counts if c > 0)
                    unique = len(set(key_bytes))
                    quality = 0

                    if entropy >= 7.9: quality += 40

                    elif entropy >= 7.5: quality += 30
                    
                    if unique >= 28: quality += 30

                    elif unique >= 24: quality += 20

                    if all(key_bytes[i:i+4].count(key_bytes[i]) < 4 for i in range(len(key_bytes)-3)):
                        quality += 20

                    results['entropies'].append(entropy)
                    results['unique_bytes'].append(unique)
                    results['quality_scores'].append(quality)

                    if entropy >= 7.5 and unique >= 24:
                        results['valid_outputs'] += 1
        except Exception:
            continue

    return results

def compare_with_baseline(finetuned_model="gemma3-entropy-v2", base_model="gemma3:latest", num_tests=30):
    base_results = comprehensive_model_test(base_model, num_tests)
    ft_results = comprehensive_model_test(finetuned_model, num_tests)
    metrics = [
        ("Entropia Média", np.mean(base_results['entropies']), np.mean(ft_results['entropies']), 'higher'),
        ("Quality Score", np.mean(base_results['quality_scores']), np.mean(ft_results['quality_scores']), 'higher'),
        ("Taxa de Sucesso (%)", base_results['valid_outputs']/num_tests*100, ft_results['valid_outputs']/num_tests*100, 'higher'),
        ("Tempo Médio (s)", np.mean(base_results['response_times']), np.mean(ft_results['response_times']), 'lower'),
    ]

    for name, base_val, ft_val, better in metrics:
        if better == 'higher':
            improvement = ((ft_val - base_val) / base_val * 100) if base_val > 0 else 0
        else:
            improvement = ((base_val - ft_val) / base_val * 100) if base_val > 0 else 0
        
        print(f"{name:<30} {base_val:<15.4f} {ft_val:<15.4f} {'↑' if improvement>0 else '↓'} {abs(improvement):.1f}%")
    
    return base_results, ft_results

if __name__ == "__main__":
    if not os.path.exists("datasets/training_data_v2.jsonl"):
        print("\nDataset não encontrado")
        sys.exit(1)

    model_name = "gemma3-entropy-v2"
    create_advanced_modelfile("gemma3:latest", model_name)

    if create_model_with_ollama(model_name):
        test_results = comprehensive_model_test(model_name, num_tests=20)
        response = input("Comparar com modelo base? [s/N]: ").lower()
        
        if response == 's':
            compare_with_baseline(model_name, "gemma3:latest", num_tests=30)
    else:
        print("\nCriação do modelo falhou.")
