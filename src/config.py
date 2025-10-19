"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

config.py - Configuração para Ollama

"""

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
DRAND_URL = "https://drand.cloudflare.com/public/latest"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"