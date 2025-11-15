"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

drand_client.py

"""

import requests
from config import DRAND_URL

def get_entropy_seed():
    response = requests.get(DRAND_URL)
    response.raise_for_status()
    data = response.json()
    
    return data['randomness']      
