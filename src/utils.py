import hashlib
import numpy as np
from scipy.stats import entropy

def postprocess_key(key_str):
    # Trunca ou hasheia a saída para 256 bits (32 bytes)
    digest = hashlib.sha256(key_str.encode("utf-8")).digest()
    key = digest[:32]
    return key

def calculate_entropy(key_bytes):
    # Calcula entropia Shannon
    hist, _ = np.histogram(list(key_bytes), bins=256)
    hx = entropy(hist)
    return hx

def is_valid_entropy(key_bytes, threshold=7.99):
    # Critério simplificado: entropia mínima por byte
    hx = calculate_entropy(key_bytes)
    return hx >= threshold
