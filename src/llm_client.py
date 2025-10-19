"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

llm_client.py - Integração com Ollama (Gemma3 270M)

"""

import requests
import json
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

def generate_llm_sequence(seed, max_length=256):
    prompt = f"""Seed: {seed}

Output 64 random hexadecimal characters (0-9, a-f):"""
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.9,
                "top_p": 0.9,
                "num_predict": max_length
            },
            timeout=300
        )
        
        response.raise_for_status()
        data = response.json()
        content = data.get('response', '').strip()
        
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
    hex_chars = ''.join(c.lower() for c in sequence if c in '0123456789abcdefABCDEF')
    
    print(f"[DEBUG] Sequência bruta: {sequence[:100]}")
    print(f"[DEBUG] Hex extraído: {hex_chars[:100]}")
    print(f"[DEBUG] Comprimento hex: {len(hex_chars)}")
    
    if len(hex_chars) < length * 2:
        print(f"[WARNING] Hex insuficiente ({len(hex_chars)} < {length * 2}), expandindo...")
        hex_chars = (hex_chars * ((length * 2 // len(hex_chars)) + 2))[:length * 2]
    
    try:
        entropy_bytes = bytes.fromhex(hex_chars[:length * 2])
        seed_bytes = bytes.fromhex(seed[:length * 2])
        entropy_bytes = bytes(a ^ b for a, b in zip(entropy_bytes, seed_bytes))
        
        return entropy_bytes
    except ValueError as e:
        raise RuntimeError(f"Erro ao converter hex para bytes: {str(e)}, hex: {hex_chars[:length * 2]}")