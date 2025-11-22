"""
JSON Encoder customizado para Flask que suporta tipos NumPy
"""

from flask.json.provider import DefaultJSONProvider
import numpy as np
from datetime import datetime, date
from decimal import Decimal


class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        """
        Converte objetos não-serializáveis para formatos JSON-compatíveis
        """
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # NumPy integers
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8, np.uint16,
                           np.uint32, np.uint64)):
            return int(obj)
        
        # NumPy floats
        if isinstance(obj, (np.floating, np.float_, np.float16, np.float32,
                           np.float64)):
            return float(obj)
        
        # NumPy booleans
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        
        # NumPy void/complex (converte para string)
        if isinstance(obj, (np.void, np.complex_, np.complex64, np.complex128)):
            return str(obj)
        
        # Datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        
        # Fallback para o encoder padrão
        return super().default(obj)


def configure_json_encoder(app):
    """
    Configura o encoder JSON customizado na aplicação Flask
    
    Usage:
        from json_encoder import configure_json_encoder
        
        app = Flask(__name__)
        configure_json_encoder(app)
    """
    app.json = NumpyJSONProvider(app)
    print("✓ JSON encoder customizado configurado (suporte a NumPy)")


# Versão alternativa: função helper para conversão manual
def ensure_json_compatible(obj):
    """
    Converte recursivamente objetos com tipos NumPy para tipos Python nativos.
    Use esta função se não quiser modificar o encoder do Flask.
    
    Usage:
        result = ensure_json_compatible(my_data)
        return jsonify(result)
    """
    if obj is None:
        return None
    
    # NumPy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8, np.uint16,
                       np.uint32, np.uint64)):
        return int(obj)
    
    if isinstance(obj, (np.floating, np.float_, np.float16, np.float32,
                       np.float64)):
        return float(obj)
    
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    
    # Python bool (força conversão para garantir)
    if type(obj).__name__ == 'bool':
        return bool(obj)
    
    # Collections
    if isinstance(obj, dict):
        return {key: ensure_json_compatible(value) for key, value in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [ensure_json_compatible(item) for item in obj]
    
    if isinstance(obj, set):
        return [ensure_json_compatible(item) for item in obj]
    
    # Datetime
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    
    # Bytes
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    
    # Outros tipos (strings, int, float, etc.)
    return obj


# Para uso em decorators
from functools import wraps
from flask import jsonify as flask_jsonify


def jsonify_safe(*args, **kwargs):
    """
    Versão segura do jsonify que converte tipos NumPy automaticamente
    
    Usage:
        from json_encoder import jsonify_safe as jsonify
        
        @app.route('/api/data')
        def get_data():
            return jsonify(my_numpy_data)
    """
    if args:
        data = args[0] if len(args) == 1 else args
    else:
        data = kwargs
    
    # Converte dados
    safe_data = ensure_json_compatible(data)
    
    return flask_jsonify(safe_data)


# Testes
if __name__ == "__main__":
    import json
    
    print("="*70)
    print("TESTE DO JSON ENCODER CUSTOMIZADO")
    print("="*70)
    
    # Dados de teste com tipos NumPy
    test_data = {
        'int': np.int64(42),
        'float': np.float64(3.14159),
        'bool': np.bool_(True),
        'array': np.array([1, 2, 3, 4, 5]),
        'nested': {
            'value': np.int32(100),
            'flag': np.bool_(False),
            'numbers': np.array([1.5, 2.5, 3.5])
        },
        'list_with_numpy': [np.int16(1), np.float32(2.5), np.bool_(True)],
        'python_bool': True,
        'python_int': 123,
        'string': 'test'
    }
    
    print("\n1. Testando ensure_json_compatible()...")
    try:
        converted = ensure_json_compatible(test_data)
        json_str = json.dumps(converted, indent=2)
        print("✓ Conversão bem-sucedida!")
        print(f"  Tamanho JSON: {len(json_str)} bytes")
        
        # Verifica tipos
        print("\n2. Verificando tipos convertidos...")
        print(f"  int: {type(converted['int'])} = {converted['int']}")
        print(f"  float: {type(converted['float'])} = {converted['float']}")
        print(f"  bool: {type(converted['bool'])} = {converted['bool']}")
        print(f"  array: {type(converted['array'])} = {converted['array']}")
        print(f"  nested.flag: {type(converted['nested']['flag'])} = {converted['nested']['flag']}")
        
        print("\n✓ Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n✗ ERRO: {e}")
        import traceback
        traceback.print_exc()