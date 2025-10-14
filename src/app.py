"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

app.py

"""

from flask import Flask, jsonify
from drand_client import get_entropy_seed
from llm_client import generate_llm_sequence
from utils import postprocess_key, is_valid_entropy, calculate_entropy

app = Flask(__name__)

@app.route("/generate_key", methods=["GET"])
def generate_key_endpoint():
    try:
        seed = get_entropy_seed()
        candidate = generate_llm_sequence(seed)
        key = postprocess_key(candidate)
        hx = calculate_entropy(key)
        valid = is_valid_entropy(key)

        return jsonify({
            "key_hex": key.hex(),
            "entropy": hx,
            "valid_entropy": valid,
            "llm_candidate": candidate,
            "drand_seed": seed
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
