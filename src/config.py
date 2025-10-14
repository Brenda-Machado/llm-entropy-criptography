"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

config.py

"""

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sua-chave-openai")
DRAND_URL = "https://drand.cloudflare.com/public/latest"
