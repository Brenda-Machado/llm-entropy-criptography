"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

llm_client.py - Integração com Hugging Face

"""

from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1


text_generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def generate_llm_sequence(seed, max_length=128):

    prompt = f"""Given this cryptographic seed: {seed}

Generate a high-entropy cryptographic sequence. Output only hexadecimal characters (0-9, a-f).
Sequence:"""
    
    try:
        response = text_generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.9,  
            top_p=0.95,
            do_sample=True
        )
        
        content = response[0]['generated_text']
        if "Sequence:" in content:
            content = content.split("Sequence:")[-1].strip()
        
        return content
    
    except Exception as e:
        raise RuntimeError(f"Erro ao gerar sequência LLM: {str(e)}")


def generate_entropy_bytes(seed, length=32):
    sequence = generate_llm_sequence(seed)
    hex_chars = ''.join(c for c in sequence if c in '0123456789abcdefABCDEF')
    
    if len(hex_chars) < length * 2:
        hex_chars = (hex_chars * ((length * 2 // len(hex_chars)) + 1))[:length * 2]
    
    entropy_bytes = bytes.fromhex(hex_chars[:length * 2])
    return entropy_bytes