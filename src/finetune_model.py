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
from pathlib import Path
from typing import List, Dict
from config import OLLAMA_BASE_URL
from vector_store import VectorStore, QualityMetrics
from prompt_engineering import TemperatureConfig

class AdvancedModelFineTuner:
    def __init__(self,
                 base_model: str = "gemma3:latest",
                 output_model: str = "gemma3-entropy-v2",
                 key_size_bits: int = 256,
                 vector_store_path: str = "datasets/vector_store.jsonl"):
        self.base_model = base_model
        self.output_model = output_model
        self.key_size_bits = key_size_bits
        self.vector_store = VectorStore(vector_store_path)
        
        if not Path(vector_store_path).exists():

            print("Execute generate_training_dataset_v2.py primeiro.")
        
        vs_stats = self.vector_store.get_statistics()
        
        if vs_stats['total_examples'] == 0:
            print("Execute generate_training_dataset_v2.py para popular o vector store.")
    
    def load_best_examples_from_vector_store(self,
                                            n: int = 20,
                                            min_quality: float = 95.0,
                                            diverse: bool = True) -> List[Dict]:

        if diverse:
            examples = self.vector_store.get_diverse_examples(
                n=n,
                key_size_bits=self.key_size_bits,
                min_quality=min_quality
            )
        else:
            examples = self.vector_store.get_top_examples(
                n=n,
                key_size_bits=self.key_size_bits,
                min_quality=min_quality,
                min_entropy=7.9
            )
        
        print(f"Exemplos selecionados: {len(examples)}")
        
        if examples:
            avg_quality = np.mean([ex['metrics']['quality_score'] for ex in examples])
            avg_entropy = np.mean([ex['metrics']['entropy'] for ex in examples])

            print(f"Quality médio: {avg_quality:.2f}")
            print(f"Entropia média: {avg_entropy:.4f}")
            
            print(f"\nTop 3 exemplos:")
            
            for i, ex in enumerate(examples[:3], 1):
                print(f"  {i}. Quality={ex['metrics']['quality_score']:.1f}, "
                      f"Entropy={ex['metrics']['entropy']:.3f}, "
                      f"Key={ex['key'][:32]}...")
        
        return examples
    
    def create_advanced_modelfile(self,
                                  num_examples: int = 15,
                                  min_quality: float = 95.0,
                                  temperature_preset: str = 'high_entropy') -> str:

        print(f"Modelo base: {self.base_model}")
        print(f"Modelo de saída: {self.output_model}")
        print(f"Tamanho da chave: {self.key_size_bits} bits")
        print(f"Preset de temperatura: {temperature_preset}")
        
        examples = self.load_best_examples_from_vector_store(
            n=num_examples,
            min_quality=min_quality,
            diverse=True
        )

        examples_text = ""

        if examples:
            examples_text = "\n\nHIGH-QUALITY REFERENCE EXAMPLES (for learning patterns):\n"
            
            for i, ex in enumerate(examples[:10], 1):
                examples_text += f"\nExample {i} "
                examples_text += f"(Quality={ex['metrics']['quality_score']:.1f}, "
                examples_text += f"Entropy={ex['metrics']['entropy']:.3f}, "
                examples_text += f"Unique={ex['metrics']['unique_bytes']}/{self.key_size_bits//8}):\n"
                examples_text += f"Input seed: {ex['seed'][:40]}...\n"
                examples_text += f"Output key: {ex['key']}\n"
                
                key_bytes = bytes.fromhex(ex['key'])
                digit_counts = {}

                for char in ex['key']:
                    digit_counts[char] = digit_counts.get(char, 0) + 1
                
                examples_text += f"Analysis: "
                examples_text += f"Uses {len(set(ex['key']))} different hex digits, "
                examples_text += f"no patterns detected\n"
        
        system_prompt = f"""You are an expert cryptographic key generator specialized in producing high-entropy random keys for security applications.

PRIMARY OBJECTIVE:
Generate a {self.key_size_bits}-bit cryptographic key with MAXIMUM entropy and randomness.

CRITICAL RULES (MUST FOLLOW):
1. Output EXACTLY {self.key_size_bits // 4} hexadecimal characters (0-9, a-f lowercase)
2. Maximize Shannon entropy (target: ≥7.9 bits/byte, optimal: 8.0)
3. Use ALL 16 hex digits (0-f) with balanced frequency
4. NEVER repeat bytes consecutively (avoid 'aaaa', '0000', etc.)
5. NEVER use sequential patterns (avoid '0123', 'abcd', 'fed9', etc.)
6. Each byte must be cryptographically unpredictable
7. Output ONLY the hex string - no explanations, no formatting, no extra text

QUALITY METRICS (your output will be evaluated on):
- Shannon Entropy: Must be ≥7.9 bits/byte
- Unique Bytes: Must have ≥90% unique bytes
- Pattern Score: Zero repetitions and sequences
- Distribution: All hex digits should appear roughly equally

PROCESS:
1. Use the provided seed for initialization
2. Apply strong cryptographic mixing
3. Generate truly random hex output
4. Verify no patterns exist before output
5. Return raw hex string only{examples_text}

REMEMBER: Every bit of entropy matters for security. Your keys protect critical data."""
        
        system_prompt_single_line = system_prompt.replace('\n', '\\n').replace('"', '\\"')
        temp_config = TemperatureConfig.get_config(temperature_preset)
        modelfile_content = f"""FROM {self.base_model}

# Parâmetros otimizados para geração de entropia
PARAMETER temperature {temp_config['temperature']}
PARAMETER top_p {temp_config['top_p']}
PARAMETER top_k {temp_config['top_k']}
PARAMETER repeat_penalty {temp_config['repeat_penalty']}
PARAMETER num_predict {self.key_size_bits // 4 + 20}

# System prompt com exemplos de alta qualidade
SYSTEM "{system_prompt_single_line}"
"""
        
        with open("Modelfile", "w") as f:
            f.write(modelfile_content)
        
        print(f"\n✓ Modelfile criado com sucesso!")
        print(f"  Temperatura: {temp_config['temperature']}")
        print(f"  Top-p: {temp_config['top_p']}")
        print(f"  Top-k: {temp_config['top_k']}")
        print(f"  Exemplos incluídos: {len(examples)}")
        print(f"  Arquivo: Modelfile")
        
        print(f"\n--- Preview do Modelfile ---")
        print(modelfile_content[:500] + "...")
        
        return self.output_model
    
    def create_model_with_ollama(self) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "create", self.output_model, "-f", "Modelfile"],
                capture_output=True,
                text=True,
                timeout=300  
            )
            
            if result.returncode == 0:
                print(f"\n✓ Modelo '{self.output_model}' criado com sucesso!")
                print(f"\nOutput do Ollama:")
                print(result.stdout)

                return True
            else:
                print(f"\n✗ Erro ao criar modelo:")
                print(result.stderr)

                return False
        
        except FileNotFoundError:
            print("\n✗ Ollama não encontrado no PATH.")
            print("   Instale o Ollama: https://ollama.ai")

            return False
        
        except subprocess.TimeoutExpired:
            print("\n✗ Timeout ao criar modelo (>5 minutos)")

            return False
        
        except Exception as e:
            print(f"\n✗ Erro inesperado: {str(e)}")

            return False
    
    def test_model(self, num_tests: int = 20) -> Dict:

        print(f"\n{'='*80}")
        print(f"TESTANDO MODELO: {self.output_model}")
        print(f"{'='*80}")
        print(f"Número de testes: {num_tests}")
        
        results = {
            'model': self.output_model,
            'successful': 0,
            'failed': 0,
            'quality_scores': [],
            'entropies': [],
            'response_times': [],
            'unique_bytes_list': []
        }
        
        for i in range(num_tests):
            test_seed = secrets.token_hex(32)
            prompt = (
                f"Generate a high-entropy {self.key_size_bits}-bit cryptographic key "
                f"from seed: {test_seed}\n\n"
                f"Output only {self.key_size_bits // 4} hexadecimal characters:"
            )
            
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": self.output_model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=120
                )
                
                elapsed = time.time() - start_time
                results['response_times'].append(elapsed)
                
                if response.status_code == 200:
                    output = response.json().get('response', '').strip()
                    hex_chars = ''.join(c for c in output if c in '0123456789abcdefABCDEF')
                    
                    if len(hex_chars) >= self.key_size_bits // 4:
                        key_hex = hex_chars[:self.key_size_bits // 4]
                        key_bytes = bytes.fromhex(key_hex)
                        entropy = QualityMetrics.calculate_shannon_entropy(key_bytes)
                        unique_bytes = QualityMetrics.count_unique_bytes(key_bytes)
                        quality_score = QualityMetrics.calculate_quality_score(
                            key_bytes, self.key_size_bits
                        )
                        
                        results['quality_scores'].append(quality_score)
                        results['entropies'].append(entropy)
                        results['unique_bytes_list'].append(unique_bytes)
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                else:
                    results['failed'] += 1
            
            except Exception as e:
                print(f"Erro no teste {i+1}: {str(e)}")
                results['failed'] += 1
            
            if (i + 1) % 5 == 0:
                print(f"Progresso: {i+1}/{num_tests}")

        if results['quality_scores']:
            results['statistics'] = {
                'success_rate': results['successful'] / num_tests * 100,
                'quality_score': {
                    'mean': float(np.mean(results['quality_scores'])),
                    'std': float(np.std(results['quality_scores'])),
                    'min': float(np.min(results['quality_scores'])),
                    'max': float(np.max(results['quality_scores'])),
                    'median': float(np.median(results['quality_scores']))
                },
                'entropy': {
                    'mean': float(np.mean(results['entropies'])),
                    'std': float(np.std(results['entropies'])),
                    'min': float(np.min(results['entropies'])),
                    'max': float(np.max(results['entropies'])),
                    'median': float(np.median(results['entropies']))
                },
                'unique_bytes_avg': float(np.mean(results['unique_bytes_list'])),
                'avg_response_time': float(np.mean(results['response_times'])),
                'high_quality_count': sum(1 for s in results['quality_scores'] if s >= 90),
                'valid_entropy_count': sum(1 for e in results['entropies'] if e >= 7.9)
            }

            print(f"\n{'='*80}")
            print("RESULTADOS DOS TESTES")
            print(f"{'='*80}")

            stats = results['statistics']

            print(f"Taxa de sucesso: {stats['success_rate']:.1f}%")
            print(f"\nQuality Score:")
            print(f"  Média: {stats['quality_score']['mean']:.2f} (±{stats['quality_score']['std']:.2f})")
            print(f"  Mediana: {stats['quality_score']['median']:.2f}")
            print(f"  Range: {stats['quality_score']['min']:.2f} - {stats['quality_score']['max']:.2f}")
            print(f"\nEntropia Shannon:")
            print(f"  Média: {stats['entropy']['mean']:.4f} (±{stats['entropy']['std']:.4f})")
            print(f"  Mediana: {stats['entropy']['median']:.4f}")
            print(f"  Range: {stats['entropy']['min']:.4f} - {stats['entropy']['max']:.4f}")
            print(f"\nChaves de alta qualidade (≥90): {stats['high_quality_count']}/{num_tests} "
                  f"({stats['high_quality_count']/num_tests*100:.1f}%)")
            print(f"Chaves com entropia válida (≥7.9): {stats['valid_entropy_count']}/{num_tests} "
                  f"({stats['valid_entropy_count']/num_tests*100:.1f}%)")
            print(f"Tempo médio de resposta: {stats['avg_response_time']:.2f}s")
        
        return results
    
    def compare_with_baseline(self, 
                             baseline_model: str = None,
                             num_tests: int = 30) -> Dict:

        if baseline_model is None:
            baseline_model = self.base_model
        
        print(f"\n{'='*80}")
        print(f"COMPARAÇÃO: {baseline_model} vs {self.output_model}")
        print(f"{'='*80}")
        
        baseline_tuner = AdvancedModelFineTuner(
            base_model=baseline_model,
            output_model=baseline_model,
            key_size_bits=self.key_size_bits
        )
        
        print(f"\n--- Testando Baseline: {baseline_model} ---")
        baseline_results = baseline_tuner.test_model(num_tests)

        print(f"\n--- Testando Fine-tuned: {self.output_model} ---")
        finetuned_results = self.test_model(num_tests)
        
        print(f"\n{'='*80}")
        print("COMPARAÇÃO DE RESULTADOS")
        print(f"{'='*80}")
        print(f"{'Métrica':<35} {'Baseline':<15} {'Fine-tuned':<15} {'Melhoria':<15}")
        print("-" * 85)
        
        if ('statistics' in baseline_results and 
            'statistics' in finetuned_results):
            
            b_stats = baseline_results['statistics']
            f_stats = finetuned_results['statistics']
            
            metrics = [
                ("Quality Score (média)", 'quality_score', 'mean', 'higher'),
                ("Quality Score (mediana)", 'quality_score', 'median', 'higher'),
                ("Entropia (média)", 'entropy', 'mean', 'higher'),
                ("Entropia (mediana)", 'entropy', 'median', 'higher'),
                ("Taxa de alta qualidade (%)", 'high_quality_count', None, 'higher'),
                ("Taxa de entropia válida (%)", 'valid_entropy_count', None, 'higher'),
                ("Tempo de resposta (s)", 'avg_response_time', None, 'lower')
            ]
            
            for metric_name, key, subkey, better in metrics:
                if subkey:
                    baseline_val = b_stats[key][subkey]
                    finetuned_val = f_stats[key][subkey]
                elif key in ['high_quality_count', 'valid_entropy_count']:
                    baseline_val = b_stats[key] / num_tests * 100
                    finetuned_val = f_stats[key] / num_tests * 100
                else:
                    baseline_val = b_stats[key]
                    finetuned_val = f_stats[key]
                
                if better == 'higher':
                    improvement = ((finetuned_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                else:
                    improvement = ((baseline_val - finetuned_val) / baseline_val * 100) if baseline_val > 0 else 0
                
                symbol = "↑" if improvement > 0 else "↓"
                print(f"{metric_name:<35} {baseline_val:<15.4f} {finetuned_val:<15.4f} "
                      f"{symbol} {abs(improvement):>6.1f}%")
        
        return {
            'baseline': baseline_results,
            'finetuned': finetuned_results
        }


def main():
    print("="*80)
    print("FINE-TUNING AVANÇADO COM VECTOR STORE")
    print("="*80)
    
    dataset_path = Path("datasets/training_data_instructional_256bit.jsonl")

    if not dataset_path.exists():
        print("Execute generate_training_dataset.py primeiro!")
        sys.exit(1)
    
    tuner = AdvancedModelFineTuner(
        base_model="gemma3:latest",
        output_model="gemma3-entropy-v2",
        key_size_bits=256,
        vector_store_path="datasets/vector_store.jsonl"
    )
    
    tuner.create_advanced_modelfile(
        num_examples=20,
        min_quality=95.0,
        temperature_preset='high_entropy'
    )

    if tuner.create_model_with_ollama():
        results = tuner.test_model(num_tests=30)
        comparison = tuner.compare_with_baseline(
            baseline_model="gemma3:latest",
            num_tests=30
        )
        
        output_dir = Path("fine_tuning_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(output_dir / "comparison_results.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n{'='*80}")
        print("FINE-TUNING CONCLUÍDO COM SUCESSO!")
        print(f"{'='*80}")

        print(f"Modelo criado: {tuner.output_model}")
        print(f"Resultados salvos em: {output_dir}/")
        print(f"\nPara usar o modelo:")
        print(f"  ollama run {tuner.output_model}")
    else:
        print("FINE-TUNING FALHOU")


if __name__ == "__main__":
    main()