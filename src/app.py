"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

app.py

"""

from flask import Flask, jsonify
from drand_client import get_entropy_seed
from llm_client import generate_llm_sequence, generate_entropy_bytes
from utils import postprocess_key, is_valid_entropy, calculate_entropy
import traceback

app = Flask(__name__)

@app.route("/generate_key", methods=["GET"])
def generate_key_endpoint():
    try:
        seed = get_entropy_seed()
        llm_entropy = generate_entropy_bytes(seed, length=32)
        candidate = generate_llm_sequence(seed)
        key = postprocess_key(llm_entropy.hex())
        hx = calculate_entropy(key)
        valid = is_valid_entropy(key)
        
        return jsonify({
            "success": True,
            "key_hex": key.hex(),
            "entropy_shannon": round(float(hx), 4),
            "valid_entropy": bool(valid),
            "llm_candidate": candidate,
            "drand_seed": seed,
            "llm_entropy_hex": llm_entropy.hex()
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)