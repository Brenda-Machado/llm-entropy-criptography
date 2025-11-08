"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

prompt_engineering.py
"""

from typing import List, Dict, Tuple
import secrets
import numpy as np

class PromptTemplate:
    KEY_SIZES = {
        'AES-128': 128,  # 32 caracteres hex
        'AES-192': 192,  # 48 caracteres hex
        'AES-256': 256,  # 64 caracteres hex
        'ChaCha20': 256, # 64 caracteres hex
    }
    
    def __init__(self, key_size_bits: int = 256, strategy: str = 'few-shot'):
        self.key_size_bits = key_size_bits
        self.hex_length = key_size_bits // 4  
        self.strategy = strategy
        
    def _get_instruction(self) -> str:
        return f"""You are a specialized cryptographic key generator. Your task is to generate a {self.key_size_bits}-bit cryptographic key with maximum entropy.

CRITICAL REQUIREMENTS:
1. Output EXACTLY {self.hex_length} hexadecimal characters (0-9, a-f)
2. Maximize Shannon entropy (target: ≥7.9 bits/byte, ideal: 8.0)
3. Ensure uniform distribution across all 16 hex digits
4. Avoid any patterns, sequences, or repetitions
5. Each byte must be cryptographically unpredictable"""

    def _get_context(self, seed: str) -> str:
        return f"""
CONTEXT:
- Input seed (for initialization only): {seed}
- Key standard: {self.key_size_bits}-bit ({self.hex_length} hex chars)
- Application: Cryptographic key generation
- Security requirement: High entropy for cryptographic strength

The seed should be used ONLY as an initialization vector. The output must be cryptographically random and not predictable from the seed alone."""

    def _get_few_shot_examples(self) -> str:
        examples = []

        for i in range(3):
            example_seed = secrets.token_hex(32)[:40]
            example_key = secrets.token_hex(self.key_size_bits // 8)
            key_bytes = bytes.fromhex(example_key)
            entropy = self._calculate_entropy(key_bytes)
            unique_bytes = len(set(key_bytes))
            
            examples.append(f"""
Example {i+1} (Entropy: {entropy:.3f}, Unique bytes: {unique_bytes}/{len(key_bytes)}):
Input seed: {example_seed}...
Output key: {example_key}""")
        
        return "\n".join(examples)

    def _get_chain_of_thought(self) -> str:
        return """
REASONING PROCESS (Chain-of-Thought):
1. **Initialization**: Use the seed to initialize the random state
2. **Mixing**: Apply cryptographic mixing functions to break patterns
3. **Entropy verification**: Internally verify that output has high entropy
   - Check: All 16 hex digits (0-f) should appear
   - Check: No consecutive identical bytes (e.g., 'aaaa')
   - Check: No sequential patterns (e.g., '0123', 'abcd')
4. **Quality assurance**: Ensure Shannon entropy ≥ 7.9 bits/byte
5. **Output**: Return ONLY the {self.hex_length}-character hex string

Think through each step, but output ONLY the final hex key."""

    def _get_output_format(self) -> str:
        return f"""
OUTPUT FORMAT:
- Format: Plain hexadecimal string
- Length: EXACTLY {self.hex_length} characters
- Characters: Only 0-9 and a-f (lowercase)
- No spaces, no separators, no explanations
- No prefix (e.g., no '0x')
- No additional text before or after

Example format: {'a' * self.hex_length}
"""

    def _calculate_entropy(self, key_bytes: bytes) -> float:
        if len(key_bytes) == 0:
            return 0.0
        
        counts = np.bincount(np.frombuffer(key_bytes, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(key_bytes)
        return float(-np.sum(probs * np.log2(probs)))

    def generate_prompt(self, seed: str) -> str:
        if self.strategy == 'zero-shot':
            return self._generate_zero_shot(seed)
        elif self.strategy == 'few-shot':
            return self._generate_few_shot(seed)
        elif self.strategy == 'cot':
            return self._generate_chain_of_thought(seed)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def _generate_zero_shot(self, seed: str) -> str:
        return f"""{self._get_instruction()}

{self._get_context(seed)}

{self._get_output_format()}

Now generate the {self.key_size_bits}-bit key:"""

    def _generate_few_shot(self, seed: str) -> str:
        return f"""{self._get_instruction()}

{self._get_few_shot_examples()}

{self._get_context(seed)}

{self._get_output_format()}

Now generate the {self.key_size_bits}-bit key for the provided seed:"""

    def _generate_chain_of_thought(self, seed: str) -> str:
        return f"""{self._get_instruction()}

{self._get_chain_of_thought()}

{self._get_context(seed)}

{self._get_output_format()}

Now, following the reasoning process above, generate the {self.key_size_bits}-bit key:"""

    def get_context_window_info(self) -> Dict[str, int]:
        sample_prompt = self.generate_prompt(secrets.token_hex(32))
        estimated_tokens = len(sample_prompt) // 4
        
        return {
            'prompt_chars': len(sample_prompt),
            'estimated_tokens': estimated_tokens,
            'response_tokens_needed': self.hex_length + 20,  # Margem de segurança
            'total_tokens': estimated_tokens + self.hex_length + 20
        }


class TemperatureConfig:
    PRESETS = {
        'deterministic': {
            'temperature': 0.3,
            'top_p': 0.7,
            'top_k': 20,
            'repeat_penalty': 1.5,
            'description': 'Baixa aleatoriedade, mais determinístico'
        },
        'balanced': {
            'temperature': 0.9,
            'top_p': 0.9,
            'top_k': 50,
            'repeat_penalty': 1.3,
            'description': 'Equilíbrio entre aleatoriedade e coerência'
        },
        'high_entropy': {
            'temperature': 1.3,
            'top_p': 0.95,
            'top_k': 80,
            'repeat_penalty': 1.2,
            'description': 'Alta aleatoriedade, máxima entropia'
        },
        'extreme_random': {
            'temperature': 1.8,
            'top_p': 0.98,
            'top_k': 100,
            'repeat_penalty': 1.1,
            'description': 'Aleatoriedade extrema (pode gerar ruído)'
        }
    }
    
    @classmethod
    def get_config(cls, preset: str = 'high_entropy') -> Dict:
        if preset not in cls.PRESETS:
            raise ValueError(f"Preset inválido. Opções: {list(cls.PRESETS.keys())}")
        
        return cls.PRESETS[preset].copy()
    
    @classmethod
    def get_ollama_params(cls, preset: str = 'high_entropy') -> Dict:
        config = cls.get_config(preset)
        config.pop('description', None)

        return config


def compare_prompt_strategies(seed: str, key_size: int = 256) -> Dict[str, str]:
    strategies = ['zero-shot', 'few-shot', 'cot']
    prompts = {}
    
    for strategy in strategies:
        template = PromptTemplate(key_size_bits=key_size, strategy=strategy)
        prompts[strategy] = template.generate_prompt(seed)
        info = template.get_context_window_info()
        prompts[f"{strategy}_info"] = info
    
    return prompts
