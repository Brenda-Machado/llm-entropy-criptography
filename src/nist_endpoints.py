"""
Endpoints Flask para integração dos testes NIST SP 800-22
"""

from flask import jsonify, request
from nist_tests import NISTTests
from test_nist_integration import NISTValidator
from json_encoder import ensure_json_compatible
import traceback


def add_nist_endpoints(app, clients_cache, get_or_create_client):
    @app.route("/nist/validate_key", methods=["POST", "OPTIONS"])
    def nist_validate_key():
        """
        Valida uma chave existente com NIST SP 800-22
        
        Request body:
        {
            "key_hex": "a3f5b82c..."
        }
        """
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        try:
            params = request.get_json() or {}
            key_hex = params.get('key_hex')
            
            if not key_hex:
                return jsonify({
                    "success": False,
                    "error": "key_hex é obrigatório"
                }), 400
            
            # Valida formato hex
            try:
                bytes.fromhex(key_hex)
            except ValueError:
                return jsonify({
                    "success": False,
                    "error": "key_hex deve ser uma string hexadecimal válida"
                }), 400
            
            # Executa testes NIST
            results = NISTTests.run_all_tests_from_hex(key_hex)
            
            # Converte tipos NumPy para JSON
            results = ensure_json_compatible(results)
            
            return jsonify({
                "success": True,
                "key_hex": key_hex,
                "nist_results": results,
                "formatted_report": NISTTests.format_results(results, detailed=False)
            })
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    
    @app.route("/nist/generate_and_validate", methods=["POST", "OPTIONS"])
    def nist_generate_and_validate():
        """
        Gera uma chave E valida com NIST em uma única chamada
        
        Request body:
        {
            "key_size": 256,
            "strategy": "few-shot",
            "temperature": "high_entropy",
            "model": "gemma3:latest",
            "use_drand": true,
            "seed": "optional_custom_seed"
        }
        """
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        try:
            params = request.get_json() or {}
            
            key_size_bits = int(params.get('key_size', 256))
            strategy = params.get('strategy', 'few-shot')
            temperature_preset = params.get('temperature', 'high_entropy')
            model = params.get('model', 'gemma3:latest')
            
            use_drand_param = params.get('use_drand', True)
            if isinstance(use_drand_param, str):
                use_drand = use_drand_param.lower() == 'true'
            else:
                use_drand = bool(use_drand_param)
            
            custom_seed = params.get('seed')
            
            # Validações
            if key_size_bits not in [128, 192, 256]:
                return jsonify({
                    "success": False,
                    "error": "key_size deve ser 128, 192 ou 256"
                }), 400
            
            # Obtém cliente LLM
            client = get_or_create_client(
                model=model,
                key_size_bits=key_size_bits,
                strategy=strategy,
                temperature_preset=temperature_preset
            )
            
            # Cria validador
            validator = NISTValidator(client)
            
            # Gera e valida
            if use_drand:
                result = validator.generate_and_validate(use_drand=True)
            elif custom_seed:
                result = validator.generate_and_validate(seed=custom_seed, use_drand=False)
            else:
                return jsonify({
                    "success": False,
                    "error": "Deve fornecer seed ou usar use_drand=true"
                }), 400
            
            if result['success']:
                response = {
                    "success": True,
                    "key_hex": result['key_hex'],
                    "seed": result['seed'],
                    "generation_metrics": result['generation_metrics'],
                    "generation_info": result['generation_info'],
                    "nist_validation": result['nist_validation'],
                    "overall_quality": result['overall_quality'],
                    "formatted_report": NISTTests.format_results(
                        result['nist_validation'], 
                        detailed=False
                    )
                }
                
                # Converte tipos NumPy para JSON
                response = ensure_json_compatible(response)
                
                return jsonify(response)
            else:
                return jsonify(result), 500
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    
    @app.route("/nist/batch_validate", methods=["POST", "OPTIONS"])
    def nist_batch_validate():
        """
        Gera e valida múltiplas chaves com NIST
        
        Request body:
        {
            "num_keys": 10,
            "key_size": 256,
            "strategy": "few-shot",
            "temperature": "high_entropy",
            "model": "gemma3:latest"
        }
        """
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        try:
            params = request.get_json() or {}
            
            num_keys = int(params.get('num_keys', 10))
            key_size_bits = int(params.get('key_size', 256))
            strategy = params.get('strategy', 'few-shot')
            temperature_preset = params.get('temperature', 'high_entropy')
            model = params.get('model', 'gemma3:latest')
            
            if num_keys > 50:
                return jsonify({
                    "success": False,
                    "error": "num_keys deve ser ≤ 50 para validação NIST"
                }), 400
            
            # Obtém cliente LLM
            client = get_or_create_client(
                model=model,
                key_size_bits=key_size_bits,
                strategy=strategy,
                temperature_preset=temperature_preset
            )
            
            # Cria validador
            validator = NISTValidator(client)
            
            # Executa batch
            batch_results = validator.batch_validate(num_keys=num_keys)
            
            response = {
                "success": True,
                "statistics": batch_results['statistics'],
                "results": [
                    {
                        'key_hex': r['key_hex'],
                        'nist_pass_rate': r['nist_validation']['summary']['pass_rate'],
                        'overall_quality': r['overall_quality']
                    } for r in batch_results['results'] if r['success']
                ],
                "configuration": {
                    "model": model,
                    "key_size_bits": key_size_bits,
                    "strategy": strategy,
                    "temperature_preset": temperature_preset
                }
            }
            
            # Converte tipos NumPy para JSON
            response = ensure_json_compatible(response)
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    
    @app.route("/nist/test_details", methods=["GET"])
    def nist_test_details():
        """
        Retorna informações sobre os testes NIST disponíveis
        """
        tests_info = [
            {
                "id": 1,
                "name": "Frequency (Monobit) Test",
                "description": "Verifica se o número de 0s e 1s é aproximadamente igual",
                "purpose": "Detecta viés na sequência"
            },
            {
                "id": 2,
                "name": "Block Frequency Test",
                "description": "Verifica a proporção de 1s dentro de blocos",
                "purpose": "Detecta padrões locais"
            },
            {
                "id": 3,
                "name": "Runs Test",
                "description": "Verifica a alternância entre 0s e 1s",
                "purpose": "Detecta oscilação inadequada"
            },
            {
                "id": 4,
                "name": "Longest Run of Ones Test",
                "description": "Verifica a duração da maior sequência de 1s",
                "purpose": "Detecta sequências longas anormais"
            },
            {
                "id": 5,
                "name": "Binary Matrix Rank Test",
                "description": "Verifica independência linear de substrings",
                "purpose": "Detecta dependência linear"
            },
            {
                "id": 6,
                "name": "Discrete Fourier Transform Test",
                "description": "Detecta componentes periódicas",
                "purpose": "Detecta padrões repetitivos"
            },
            {
                "id": 7,
                "name": "Non-overlapping Template Matching Test",
                "description": "Busca por padrões específicos sem sobreposição",
                "purpose": "Detecta templates específicos"
            },
            {
                "id": 8,
                "name": "Overlapping Template Matching Test",
                "description": "Busca por padrões específicos com sobreposição",
                "purpose": "Detecta templates sobrepostos"
            },
            {
                "id": 9,
                "name": "Maurer's Universal Statistical Test",
                "description": "Detecta compressibilidade da sequência",
                "purpose": "Verifica previsibilidade"
            },
            {
                "id": 10,
                "name": "Linear Complexity Test",
                "description": "Mede complexidade usando Berlekamp-Massey",
                "purpose": "Verifica complexidade estrutural"
            },
            {
                "id": 11,
                "name": "Serial Test",
                "description": "Verifica frequência de padrões m-bit",
                "purpose": "Detecta não-aleatoriedade serial"
            },
            {
                "id": 12,
                "name": "Approximate Entropy Test",
                "description": "Compara frequências de padrões sobrepostos",
                "purpose": "Mede imprevisibilidade local"
            },
            {
                "id": 13,
                "name": "Cumulative Sums Test",
                "description": "Verifica somas cumulativas (forward/backward)",
                "purpose": "Detecta viés cumulativo"
            },
            {
                "id": 14,
                "name": "Random Excursions Test",
                "description": "Analisa ciclos em random walk",
                "purpose": "Verifica comportamento de random walk"
            },
            {
                "id": 15,
                "name": "Random Excursions Variant Test",
                "description": "Variante do teste de random excursions",
                "purpose": "Análise complementar de random walk"
            }
        ]
        
        return jsonify({
            "success": True,
            "total_tests": len(tests_info),
            "tests": tests_info,
            "significance_level": NISTTests.ALPHA,
            "interpretation": {
                "pass_threshold": 0.01,
                "description": "P-value >= 0.01 indica que a sequência passou no teste",
                "recommendations": {
                    "excellent": "≥95% dos testes passaram - Adequado para uso criptográfico",
                    "good": "≥85% dos testes passaram - Pode ser adequado para alguns usos",
                    "moderate": "≥70% dos testes passaram - Não recomendado para criptografia",
                    "poor": "<70% dos testes passaram - Inadequado para uso criptográfico"
                }
            }
        })
    
    
    @app.route("/nist/compare_strategies", methods=["POST", "OPTIONS"])
    def nist_compare_strategies():
        """
        Compara diferentes estratégias de prompt usando validação NIST
        
        Request body:
        {
            "strategies": ["zero-shot", "few-shot", "cot"],
            "num_keys_per_strategy": 5,
            "key_size": 256,
            "temperature": "high_entropy",
            "model": "gemma3:latest"
        }
        """
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200
        
        try:
            params = request.get_json() or {}
            
            strategies = params.get('strategies', ['zero-shot', 'few-shot', 'cot'])
            num_keys = int(params.get('num_keys_per_strategy', 5))
            key_size_bits = int(params.get('key_size', 256))
            temperature_preset = params.get('temperature', 'high_entropy')
            model = params.get('model', 'gemma3:latest')
            
            if num_keys > 10:
                return jsonify({
                    "success": False,
                    "error": "num_keys_per_strategy deve ser ≤ 10"
                }), 400
            
            results_by_strategy = {}
            
            for strategy in strategies:
                # Obtém cliente
                client = get_or_create_client(
                    model=model,
                    key_size_bits=key_size_bits,
                    strategy=strategy,
                    temperature_preset=temperature_preset
                )
                
                # Cria validador
                validator = NISTValidator(client)
                
                # Executa batch
                batch_results = validator.batch_validate(num_keys=num_keys)
                
                results_by_strategy[strategy] = {
                    'statistics': batch_results['statistics'],
                    'best_key': max(
                        [r for r in batch_results['results'] if r['success']],
                        key=lambda r: r['overall_quality']['combined_score']
                    )['key_hex'] if batch_results['statistics']['successful_generations'] > 0 else None
                }
            
            # Determina melhor estratégia
            best_strategy = max(
                results_by_strategy.items(),
                key=lambda x: x[1]['statistics'].get('avg_combined_score', 0)
            )
            
            response = {
                "success": True,
                "comparison": results_by_strategy,
                "best_strategy": {
                    "name": best_strategy[0],
                    "avg_combined_score": best_strategy[1]['statistics'].get('avg_combined_score', 0),
                    "avg_nist_pass_rate": best_strategy[1]['statistics'].get('avg_nist_pass_rate', 0)
                },
                "configuration": {
                    "num_keys_per_strategy": num_keys,
                    "key_size_bits": key_size_bits,
                    "temperature_preset": temperature_preset,
                    "model": model
                }
            }
            
            # Converte tipos NumPy para JSON
            response = ensure_json_compatible(response)
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500
    
    
    print("\n✓ Endpoints NIST adicionados:")
    print("  - POST /nist/validate_key")
    print("  - POST /nist/generate_and_validate")
    print("  - POST /nist/batch_validate")
    print("  - GET  /nist/test_details")
    print("  - POST /nist/compare_strategies")