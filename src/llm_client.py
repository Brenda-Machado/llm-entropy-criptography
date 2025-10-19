"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

llm_client.py - Integração com Ollama (Gemma3 270M)

"""

import requests
import json
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

def generate_llm_sequence(seed, max_length=256):
    prompt = f"""Given this cryptographic seed: {seed}

Generate a high-entropy cryptographic sequence. Output only hexadecimal characters (0-9, a-f).
Sequence:"""
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.95,
                "top_p": 0.95,
                "num_predict": max_length
            },
            timeout=60
        )
        
        response.raise_for_status()
        data = response.json()
        content = data.get('response', '').strip()
        
        if "Sequence:" in content:
            content = content.split("Sequence:")[-1].strip()
        
        return content
    
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Erro: Não foi possível conectar ao Ollama em {OLLAMA_BASE_URL}. "
            "Certifique-se de que o Ollama está rodando com: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Erro ao gerar sequência LLM: {str(e)}")


def generate_entropy_bytes(seed, length=32):
    sequence = generate_llm_sequence(seed)
    hex_chars = ''.join(c for c in sequence if c in '0123456789abcdefABCDEF')

    if len(hex_chars) < length * 2:
        hex_chars = (hex_chars * ((length * 2 // len(hex_chars)) + 1))[:length * 2]
    
    entropy_bytes = bytes.fromhex(hex_chars[:length * 2])
    return entropy_bytes