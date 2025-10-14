"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

llm_client.py

"""

import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_llm_sequence(seed):
    prompt = f"Using this random seed from League of Entropy, generate a secure cryptographic key: {seed}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64
    )
    content = response['choices'][0]['message']['content']
    return content.strip()
