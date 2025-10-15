"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

utils.py

"""

import hashlib
import numpy as np
from scipy.stats import entropy

def postprocess_key(key_str):
    digest = hashlib.sha256(key_str.encode("utf-8")).digest()
    key = digest[:32]

    return key

def calculate_entropy(key_bytes):
    hist, _ = np.histogram(list(key_bytes), bins=256)
    hx = entropy(hist)

    return hx

def is_valid_entropy(key_bytes, threshold=7.99):
    hx = calculate_entropy(key_bytes)

    return hx >= threshold
