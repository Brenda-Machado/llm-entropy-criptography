"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

app.py

"""

import traceback
import os
import argparse
from flask_cors import CORS
from flask import Flask, jsonify, request, render_template, send_from_directory
from drand_client import get_entropy_seed
from llm_client import LLMClient
from vector_store import VectorStore
from prompt_engineering import TemperatureConfig
from nist_endpoints import add_nist_endpoints
from json_encoder import configure_json_encoder

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='static')
CORS(app)

configure_json_encoder(app)

DEFAULT_KEY_SIZE = int(os.getenv("KEY_SIZE_BITS", "256"))
DEFAULT_STRATEGY = os.getenv("PROMPT_STRATEGY", "few-shot")
DEFAULT_TEMPERATURE = os.getenv("TEMPERATURE_PRESET", "high_entropy")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")

clients_cache = {}

def get_or_create_client(model: str, 
                         key_size_bits: int,
                         strategy: str,
                         temperature_preset: str) -> LLMClient:

    cache_key = f"{model}_{key_size_bits}_{strategy}_{temperature_preset}"
    
    if cache_key not in clients_cache:
        clients_cache[cache_key] = LLMClient(
            model=model,
            key_size_bits=key_size_bits,
            strategy=strategy,
            temperature_preset=temperature_preset
        )
    
    return clients_cache[cache_key]

add_nist_endpoints(app, clients_cache, get_or_create_client)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/")
def api_docs():
    return jsonify({
        "name": "Cryptographic Key Generation API v2",
        "description": "Sistema avançado de geração de entropia com IA",
        "version": "2.0",
        "endpoints": {
            "/": "Interface web",
            "/api/": "Documentação da API",
            "/generate_key": {
                "methods": ["GET", "POST"],
                "description": "Gera uma chave criptográfica",
                "params": {
                    "key_size": "Tamanho em bits (128, 192, 256)",
                    "strategy": "Estratégia de prompt (zero-shot, few-shot, cot)",
                    "temperature": "Preset de temperatura",
                    "model": "Nome do modelo",
                    "use_drand": "Usar drand como seed",
                    "seed": "Seed customizado"
                }
            },
            "/generate_batch": {
                "methods": ["POST"],
                "description": "Gera múltiplas chaves",
                "params": {
                    "num_keys": "Número de chaves (máx 100)"
                }
            },
            "/vector_store/stats": {
                "methods": ["GET"],
                "description": "Estatísticas do vector store"
            },
            "/vector_store/top_examples": {
                "methods": ["GET"],
                "description": "Top exemplos do vector store"
            },
            "/client/stats": {
                "methods": ["GET"],
                "description": "Estatísticas de uso dos clientes"
            },
            "/temperature/presets": {
                "methods": ["GET"],
                "description": "Lista presets de temperatura"
            },
            "/config": {
                "methods": ["GET"],
                "description": "Configuração atual"
            },
            "/health": {
                "methods": ["GET"],
                "description": "Health check"
            }
        }
    })


@app.route("/generate_key", methods=["GET", "POST", "OPTIONS"])
def generate_key_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        if request.method == "POST":
            params = request.get_json() or {}
        else:
            params = request.args.to_dict()
        
        key_size_bits = int(params.get('key_size', DEFAULT_KEY_SIZE))
        strategy = params.get('strategy', DEFAULT_STRATEGY)
        temperature_preset = params.get('temperature', DEFAULT_TEMPERATURE)
        model = params.get('model', DEFAULT_MODEL)
        
        use_drand_param = params.get('use_drand', True)
        if isinstance(use_drand_param, str):
            use_drand = use_drand_param.lower() == 'true'
        else:
            use_drand = bool(use_drand_param)
        
        custom_seed = params.get('seed')

        if key_size_bits not in [128, 192, 256]:
            return jsonify({
                "success": False,
                "error": "key_size deve ser 128, 192 ou 256"
            }), 400
        
        if strategy not in ['zero-shot', 'few-shot', 'cot']:
            return jsonify({
                "success": False,
                "error": "strategy deve ser zero-shot, few-shot ou cot"
            }), 400
        
        if temperature_preset not in ['deterministic', 'balanced', 'high_entropy', 'extreme_random']:
            return jsonify({
                "success": False,
                "error": "temperature deve ser deterministic, balanced, high_entropy ou extreme_random"
            }), 400

        if use_drand:
            seed = get_entropy_seed()
        elif custom_seed:
            seed = custom_seed
        else:
            return jsonify({
                "success": False,
                "error": "Deve fornecer seed ou usar use_drand=true"
            }), 400
        
        client = get_or_create_client(
            model=model,
            key_size_bits=key_size_bits,
            strategy=strategy,
            temperature_preset=temperature_preset
        )
        result = client.generate_key(seed, store_result=True)
        
        if result['success']:
            response = {
                "success": True,
                "key_hex": result['key_hex'],
                "key_size_bits": result['key_size_bits'],
                "metrics": result['metrics'],
                "generation_info": result['generation_info'],
                "drand_seed": seed if use_drand else None,
                "custom_seed": seed if not use_drand else None
            }
            
            warnings = []

            if result['metrics']['quality_score'] < 75:
                warnings.append("Quality score abaixo do ideal (<75)")
            if result['metrics']['entropy'] < 7.5:
                warnings.append("Entropia abaixo do ideal (<7.5)")
            if result['metrics']['repetition_violations'] > 0:
                warnings.append(f"Padrões de repetição detectados ({result['metrics']['repetition_violations']})")
            
            if warnings:
                response['warnings'] = warnings
            
            return jsonify(response)
        else:
            return jsonify({
                "success": False,
                "error": result['error'],
                "generation_info": result['generation_info']
            }), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/generate_batch", methods=["POST", "OPTIONS"])
def generate_batch_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    try:
        params = request.get_json() or {}
        
        num_keys = int(params.get('num_keys', 10))
        key_size_bits = int(params.get('key_size', DEFAULT_KEY_SIZE))
        strategy = params.get('strategy', DEFAULT_STRATEGY)
        temperature_preset = params.get('temperature', DEFAULT_TEMPERATURE)
        model = params.get('model', DEFAULT_MODEL)
        
        if num_keys > 100:
            return jsonify({
                "success": False,
                "error": "num_keys deve ser ≤ 100"
            }), 400
        
        client = get_or_create_client(
            model=model,
            key_size_bits=key_size_bits,
            strategy=strategy,
            temperature_preset=temperature_preset
        )
        
        results, stats = client.batch_generate(num_keys)
        
        return jsonify({
            "success": True,
            "results": results,
            "statistics": stats,
            "configuration": {
                "model": model,
                "key_size_bits": key_size_bits,
                "strategy": strategy,
                "temperature_preset": temperature_preset
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/vector_store/stats", methods=["GET"])
def vector_store_stats_endpoint():
    try:
        vector_store = VectorStore("datasets/vector_store.jsonl")
        stats = vector_store.get_statistics()
        
        return jsonify({
            "success": True,
            "statistics": stats
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/vector_store/top_examples", methods=["GET"])
def vector_store_top_examples_endpoint():
    try:
        n = int(request.args.get('n', 10))
        key_size_bits = int(request.args.get('key_size', 256))
        min_quality = float(request.args.get('min_quality', 85.0))
        min_entropy = float(request.args.get('min_entropy', 7.9))
        
        vector_store = VectorStore("datasets/vector_store.jsonl")
        examples = vector_store.get_top_examples(
            n=n,
            key_size_bits=key_size_bits,
            min_quality=min_quality,
            min_entropy=min_entropy
        )
        
        return jsonify({
            "success": True,
            "examples": examples,
            "count": len(examples)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/client/stats", methods=["GET"])
def client_stats_endpoint():
    try:
        stats = {}
        
        for cache_key, client in clients_cache.items():
            stats[cache_key] = client.get_statistics()
        
        return jsonify({
            "success": True,
            "clients": stats,
            "total_clients": len(clients_cache)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/temperature/presets", methods=["GET"])
def temperature_presets_endpoint():
    try:
        presets = {}
        
        for preset_name, preset_config in TemperatureConfig.PRESETS.items():
            presets[preset_name] = preset_config.copy()
        
        return jsonify({
            "success": True,
            "presets": presets
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "version": "2.0",
        "features": {
            "vector_store": True,
            "prompt_engineering": True,
            "multiple_strategies": True,
            "temperature_control": True,
            "multiple_key_sizes": True,
            "web_interface": True
        }
    }), 200


@app.route("/config", methods=["GET"])
def config_endpoint():
    return jsonify({
        "success": True,
        "default_config": {
            "key_size_bits": DEFAULT_KEY_SIZE,
            "strategy": DEFAULT_STRATEGY,
            "temperature_preset": DEFAULT_TEMPERATURE,
            "model": DEFAULT_MODEL
        },
        "available_options": {
            "key_sizes": [128, 192, 256],
            "strategies": ["zero-shot", "few-shot", "cot"],
            "temperature_presets": list(TemperatureConfig.PRESETS.keys()),
            "models": ["gemma3:latest", "gemma3-entropy-v2"]
        }
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Flask para geração de chaves criptográficas")
    parser.add_argument("--host", default="0.0.0.0", help="Host (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Porta (padrão: 5000)")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    
    args = parser.parse_args()
    
    print("="*80)
    print("API FLASK - GERAÇÃO DE CHAVES CRIPTOGRÁFICAS")
    print("="*80)
    print(f"Configuração padrão:")
    print(f"  Key size: {DEFAULT_KEY_SIZE} bits")
    print(f"  Strategy: {DEFAULT_STRATEGY}")
    print(f"  Temperature: {DEFAULT_TEMPERATURE}")
    print(f"  Model: {DEFAULT_MODEL}")
    print("="*80)
    print(f"Interface Web: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/api/")
    print("="*80)
    
    app.run(debug=args.debug, host=args.host, port=args.port)