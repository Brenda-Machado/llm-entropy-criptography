"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

json_encoder.py
"""

import numpy as np
from datetime import datetime, date
from decimal import Decimal
from flask.json.provider import DefaultJSONProvider
from functools import wraps
from flask import jsonify as flask_jsonify

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                           np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        if hasattr(np, 'void') and isinstance(obj, np.void):
            return str(obj)
        
        if isinstance(obj, (np.complexfloating, np.complex64, np.complex128)):
            return str(obj)
        
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        if isinstance(obj, Decimal):
            return float(obj)
        
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        
        return super().default(obj)


def configure_json_encoder(app):
    app.json = NumpyJSONProvider(app)

def ensure_json_compatible(obj):
    if obj is None:
        return None
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                       np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    if type(obj).__name__ == 'bool':
        return bool(obj)
    
    if isinstance(obj, (np.complexfloating, np.complex64, np.complex128)):
        return str(obj)
    
    if isinstance(obj, dict):
        return {key: ensure_json_compatible(value) for key, value in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [ensure_json_compatible(item) for item in obj]
    
    if isinstance(obj, set):
        return [ensure_json_compatible(item) for item in obj]
    
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    if isinstance(obj, Decimal):
        return float(obj)
    
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    
    return obj

def jsonify_safe(*args, **kwargs):
    if args:
        data = args[0] if len(args) == 1 else args
    else:
        data = kwargs
    
    safe_data = ensure_json_compatible(data)
    
    return flask_jsonify(safe_data)


def get_numpy_version_info():
    version = np.__version__
    major_version = int(version.split('.')[0])
    
    return {
        'version': version,
        'major': major_version,
        'is_numpy2': major_version >= 2,
        'has_float_': hasattr(np, 'float_'),
        'has_int_': hasattr(np, 'int_'),
    }
