"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

finetune_model.py

"""

import subprocess
import os
import sys
import requests
from utils import calculate_entropy
from config import OLLAMA_BASE_URL

def create_modelfile(base_model="gemma3:latest", output_name="gemma3-entropy"):
    modelfile_content = f"""FROM {base_model}
        # Parâmetros otimizados para geração de entropia
        PARAMETER temperature 0.9
        PARAMETER top_p 0.95
        PARAMETER top_k 50

        # System prompt específico para geração de chaves
        SYSTEM You are a cryptographic key generator. Your task is to generate high-entropy hexadecimal sequences for cryptographic keys. Always output exactly 64 hexadecimal characters (0-9, a-f) with maximum randomness and entropy. Never explain, just output the hex string.

        # Adicionar exemplos de treinamento
        ADAPTER ./datasets/training_data.jsonl
        """
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    print(f"✓ Modelfile criado para fine-tuning de {base_model}")
    return output_name

def finetune_with_ollama(model_name="gemma3-entropy"):
    try:
        cmd = ["ollama", "create", model_name, "-f", "Modelfile"]
        
        print(f"\nExecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✓ Modelo '{model_name}' criado com sucesso!")
            print(f"\nPara usar o modelo fine-tuned, atualize config.py:")
            print(f'OLLAMA_MODEL = "{model_name}"')
            return True
        else:
            print(f"\n✗ Erro ao criar modelo:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("\n✗ Erro: Ollama não encontrado. Certifique-se de que está instalado e no PATH.")
        return False
    except Exception as e:
        print(f"\n✗ Erro inesperado: {str(e)}")
        return False

def test_finetuned_model(model_name="gemma3-entropy"):
    test_seed = "a1b2c3d4e5f6789012345678901234567890abcdefabcdef1234567890abcdef"
    prompt = f"Generate a high-entropy cryptographic key from seed: {test_seed}\n\nOutput only 64 hexadecimal characters:"
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            output = data.get('response', '').strip()
            
            print(f"\nPrompt: {prompt[:80]}...")
            print(f"Resposta: {output[:100]}...")
            
            hex_chars = ''.join(c for c in output if c in '0123456789abcdefABCDEF')

            if len(hex_chars) >= 64:
                key_bytes = bytes.fromhex(hex_chars[:64])
                entropy = calculate_entropy(key_bytes)
                print(f"\nEntropia: {entropy:.4f}")
                print(f"Válida: {entropy >= 7.99}")
            else:
                print(f"\n✗ Saída insuficiente: apenas {len(hex_chars)} chars hex")
        else:
            print(f"✗ Erro HTTP: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Erro ao testar: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists("datasets/training_data.jsonl"):
        print("✗ Dataset de treinamento não encontrado.")
        sys.exit(1)
    
    model_name = "gemma3-entropy"
    create_modelfile(
        base_model="gemma3:latest",
        output_name=model_name
    )
    
    success = finetune_with_ollama(model_name)
    
    if success:
        test_finetuned_model(model_name)
    else:
        print("\n✗ Fine-tuning falhou. Verifique os logs acima.")