"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

llm_client.py - Integração com Ollama (Gemma3 270M)

"""

import requests
import time
import secrets
import numpy as np
from typing import Dict, Tuple
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from prompt_engineering import PromptTemplate, TemperatureConfig
from vector_store import VectorStore, QualityMetrics

class LLMClient:
    def __init__(self,
                 base_url: str = OLLAMA_BASE_URL,
                 model: str = OLLAMA_MODEL,
                 key_size_bits: int = 256,
                 strategy: str = 'few-shot',
                 temperature_preset: str = 'high_entropy',
                 vector_store_path: str = "datasets/vector_store.jsonl",
                 max_retries: int = 3,
                 timeout: int = 300):

        self.base_url = base_url
        self.model = model
        self.key_size_bits = key_size_bits
        self.strategy = strategy
        self.temperature_preset = temperature_preset
        self.max_retries = max_retries
        self.timeout = timeout
        self.prompt_template = PromptTemplate(
            key_size_bits=key_size_bits,
            strategy=strategy
        )
        
        self.vector_store = VectorStore(vector_store_path)
        self.temperature_config = TemperatureConfig.get_ollama_params(temperature_preset)
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0.0,
            'avg_quality_score': 0.0
        }
    
    def generate_key(self, seed: str, store_result: bool = True) -> Dict:
        self.stats['total_requests'] += 1
        start_time = time.time()
        prompt = self.prompt_template.generate_prompt(seed)

        request_params = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "num_predict": self.key_size_bits // 4 + 50,  
            **self.temperature_config
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=request_params,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                output = data.get('response', '').strip()
                hex_chars = self._extract_hex(output)
                result = self._process_output(seed, hex_chars, output, start_time)
                
                if store_result and result['metrics']['quality_score'] >= 85:
                    self.vector_store.add_example(
                        seed=seed,
                        key_hex=result['key_hex'],
                        metadata={
                            'model': self.model,
                            'strategy': self.strategy,
                            'temperature_preset': self.temperature_preset,
                            'generation_time': result['generation_time']
                        }
                    )
                
                self.stats['successful_requests'] += 1
                self.stats['total_time'] += result['generation_time']
                
                return result
                
            except requests.exceptions.ConnectionError as e:
                if attempt == self.max_retries - 1:
                    self.stats['failed_requests'] += 1
                    return self._error_result(
                        f"Falha ao conectar ao Ollama em {self.base_url}. "
                        f"Certifique-se de que está rodando: ollama serve"
                    )
                time.sleep(1) 
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.stats['failed_requests'] += 1
                    return self._error_result(f"Erro ao gerar chave: {str(e)}")
                time.sleep(1)
        
        self.stats['failed_requests'] += 1

        return self._error_result("Número máximo de tentativas excedido")
    
    def _extract_hex(self, output: str) -> str:
        return ''.join(
            c.lower() for c in output 
            if c in '0123456789abcdefABCDEF'
        )
    
    def _process_output(self, seed: str, hex_chars: str, raw_output: str, start_time: float) -> Dict:
        generation_time = time.time() - start_time
        expected_length = self.key_size_bits // 4
        
        if len(hex_chars) < expected_length:
            return {
                'success': False,
                'error': f"Hex insuficiente: {len(hex_chars)} < {expected_length}",
                'key_hex': None,
                'raw_output': raw_output[:200],
                'hex_extracted': hex_chars,
                'generation_time': generation_time
            }
        
        key_hex = hex_chars[:expected_length]
        key_bytes = bytes.fromhex(key_hex)

        entropy = QualityMetrics.calculate_shannon_entropy(key_bytes)
        unique_bytes = QualityMetrics.count_unique_bytes(key_bytes)
        quality_score = QualityMetrics.calculate_quality_score(key_bytes, self.key_size_bits)
        repetitions = QualityMetrics.check_repetition_pattern(key_bytes)
        sequences = QualityMetrics.check_sequential_pattern(key_bytes)
        
        return {
            'success': True,
            'key_hex': key_hex,
            'key_size_bits': self.key_size_bits,
            'metrics': {
                'entropy': float(entropy),
                'unique_bytes': int(unique_bytes),
                'unique_ratio': float(unique_bytes / len(key_bytes)),
                'quality_score': float(quality_score),
                'repetition_violations': int(repetitions),
                'sequence_violations': int(sequences),
                'valid_entropy': entropy >= 7.9
            },
            'generation_info': {
                'model': self.model,
                'strategy': self.strategy,
                'temperature_preset': self.temperature_preset,
                'temperature': self.temperature_config['temperature'],
                'generation_time': generation_time,
                'seed': seed
            },
            'raw_output': raw_output[:200] if len(raw_output) > 200 else raw_output
        }
    
    def _error_result(self, error_message: str) -> Dict:
        return {
            'success': False,
            'error': error_message,
            'key_hex': None,
            'metrics': None,
            'generation_info': {
                'model': self.model,
                'strategy': self.strategy,
                'temperature_preset': self.temperature_preset
            }
        }
    
    def get_statistics(self) -> Dict:
        avg_time = self.stats['total_time'] / max(self.stats['successful_requests'], 1)
        success_rate = self.stats['successful_requests'] / max(self.stats['total_requests'], 1)
        
        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': success_rate * 100,
            'average_generation_time': avg_time,
            'total_time': self.stats['total_time']
        }
    
    def batch_generate(self, num_keys: int, progress_callback=None) -> Tuple[list[Dict], Dict]:
        results = []
        quality_scores = []
        entropies = []
        
        print(f"Gerando {num_keys} chaves usando {self.strategy} com {self.temperature_preset}...")
        
        for i in range(num_keys):
            seed = secrets.token_hex(32)
            result = self.generate_key(seed, store_result=True)
            
            results.append(result)
            
            if result['success']:
                quality_scores.append(result['metrics']['quality_score'])
                entropies.append(result['metrics']['entropy'])
            
            if progress_callback:
                progress_callback(i + 1, num_keys, result)
            
            if (i + 1) % 10 == 0:
                print(f"Progresso: {i+1}/{num_keys}")
        
        successful_results = [r for r in results if r['success']]
        
        aggregate_stats = {
            'total': num_keys,
            'successful': len(successful_results),
            'failed': num_keys - len(successful_results),
            'success_rate': len(successful_results) / num_keys * 100 if num_keys > 0 else 0
        }
        
        if quality_scores:
            aggregate_stats.update({
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
                'high_quality_count': sum(1 for s in quality_scores if s >= 90),
                'valid_entropy_count': sum(1 for e in entropies if e >= 7.9)
            })
        
        return results, aggregate_stats