# Exemplos Práticos de Uso - Sistema V2

Este documento contém exemplos práticos de uso do sistema aprimorado.

---

## 📚 Índice

1. [Instalação e Configuração](#instalação-e-configuração)
2. [Gerando Chaves Básicas](#gerando-chaves-básicas)
3. [Usando Vector Store](#usando-vector-store)
4. [Diferentes Estratégias de Prompt](#diferentes-estratégias-de-prompt)
5. [Configurando Temperatura](#configurando-temperatura)
6. [Gerando Datasets](#gerando-datasets)
7. [Fine-tuning](#fine-tuning)
8. [Avaliação e Comparação](#avaliação-e-comparação)
9. [Usando a API REST](#usando-a-api-rest)
10. [Casos de Uso Avançados](#casos-de-uso-avançados)

---

## Instalação e Configuração

### Passo 1: Verificar Dependências

```bash
# Instalar dependências
pip install numpy scipy flask requests matplotlib seaborn

# Verificar Ollama
ollama --version

# Se não instalado, instalar de: https://ollama.ai
```

### Passo 2: Baixar Modelo Base

```bash
# Baixar Gemma3
ollama pull gemma3:latest

# Verificar instalação
ollama list
```

### Passo 3: Executar Testes

```bash
# Executar suite de testes
python3 test_system.py

# Deve mostrar: "✓ TODOS OS TESTES PASSARAM!"
```

---

## Gerando Chaves Básicas

### Exemplo 1: Chave AES-256 Simples

```python
from llm_client_improved import LLMClient
import secrets

# Criar cliente
client = LLMClient(
    model="gemma3:latest",
    key_size_bits=256,
    strategy='few-shot',
    temperature_preset='high_entropy'
)

# Gerar chave
seed = secrets.token_hex(32)
result = client.generate_key(seed)

if result['success']:
    print(f"Chave gerada: {result['key_hex']}")
    print(f"Quality Score: {result['metrics']['quality_score']:.2f}/100")
    print(f"Entropia: {result['metrics']['entropy']:.4f} bits/byte")
    print(f"Válida: {'✓' if result['metrics']['valid_entropy'] else '✗'}")
else:
    print(f"Erro: {result['error']}")
```

### Exemplo 2: Chave AES-128

```python
# Chave menor (128 bits)
client_128 = LLMClient(
    key_size_bits=128,  # 32 caracteres hex
    strategy='few-shot',
    temperature_preset='high_entropy'
)

result = client_128.generate_key(secrets.token_hex(32))
print(f"Chave AES-128: {result['key_hex']}")  # 32 chars
```

### Exemplo 3: Lote de Chaves

```python
# Gerar 10 chaves de uma vez
results, stats = client.batch_generate(num_keys=10)

print(f"Geradas: {stats['successful']}/{stats['total']}")
print(f"Quality médio: {stats['quality_score']['mean']:.2f}")
print(f"Alta qualidade: {stats['high_quality_count']}")

# Filtrar apenas chaves de alta qualidade
high_quality = [
    r for r in results 
    if r['success'] and r['metrics']['quality_score'] >= 90
]

print(f"\nChaves de alta qualidade:")
for r in high_quality:
    print(f"  {r['key_hex'][:32]}... (Q={r['metrics']['quality_score']:.1f})")
```

---

## Usando Vector Store

### Exemplo 4: Criar e Popular Vector Store

```python
from vector_store_system import VectorStore, generate_high_quality_dataset

# Criar vector store
vs = VectorStore("datasets/my_vector_store.jsonl")

# Gerar 1000 exemplos de alta qualidade
generate_high_quality_dataset(
    vector_store=vs,
    num_examples=1000,
    key_size_bits=256
)

# Ver estatísticas
stats = vs.get_statistics()
print(f"Total: {stats['total_examples']}")
print(f"Alta qualidade (≥90): {stats['high_quality_count']}")
print(f"Entropia média: {stats['entropy_stats']['mean']:.4f}")
```

### Exemplo 5: Recuperar Melhores Exemplos

```python
# Top 10 exemplos
top = vs.get_top_examples(
    n=10,
    key_size_bits=256,
    min_quality=95.0,
    min_entropy=7.9
)

print("Top 10 exemplos:")
for i, ex in enumerate(top, 1):
    print(f"{i}. Quality={ex['metrics']['quality_score']:.1f}, "
          f"Entropy={ex['metrics']['entropy']:.3f}")
    print(f"   Key: {ex['key']}")
```

### Exemplo 6: Exemplos Diversos

```python
# Buscar exemplos diversos (não similares entre si)
diverse = vs.get_diverse_examples(
    n=20,
    key_size_bits=256,
    min_quality=90.0
)

print(f"Recuperados {len(diverse)} exemplos diversos")
```

---

## Diferentes Estratégias de Prompt

### Exemplo 7: Comparar Estratégias

```python
seed = secrets.token_hex(32)

strategies = ['zero-shot', 'few-shot', 'cot']

for strategy in strategies:
    client = LLMClient(
        key_size_bits=256,
        strategy=strategy,
        temperature_preset='high_entropy'
    )
    
    result = client.generate_key(seed)
    
    print(f"\n{strategy.upper()}:")
    print(f"  Quality: {result['metrics']['quality_score']:.2f}")
    print(f"  Entropy: {result['metrics']['entropy']:.4f}")
    print(f"  Tempo: {result['generation_info']['generation_time']:.2f}s")
```

### Exemplo 8: Ver Prompt Gerado

```python
from prompt_engineering import PromptTemplate

template = PromptTemplate(
    key_size_bits=256,
    strategy='few-shot'
)

seed = secrets.token_hex(32)
prompt = template.generate_prompt(seed)

print("Prompt gerado:")
print(prompt[:500] + "...")

# Ver tamanho
info = template.get_context_window_info()
print(f"\nTamanho: {info['estimated_tokens']} tokens")
```

---

## Configurando Temperatura

### Exemplo 9: Testar Diferentes Temperaturas

```python
from prompt_engineering import TemperatureConfig

presets = ['balanced', 'high_entropy', 'extreme_random']

for preset in presets:
    client = LLMClient(
        key_size_bits=256,
        strategy='few-shot',
        temperature_preset=preset
    )
    
    result = client.generate_key(secrets.token_hex(32))
    
    config = TemperatureConfig.get_config(preset)
    
    print(f"\n{preset.upper()} (temp={config['temperature']}):")
    print(f"  Quality: {result['metrics']['quality_score']:.2f}")
    print(f"  Entropy: {result['metrics']['entropy']:.4f}")
    print(f"  Unique bytes: {result['metrics']['unique_bytes']}/32")
```

### Exemplo 10: Ver Todos os Presets

```python
from prompt_engineering import TemperatureConfig

print("Presets disponíveis:")
for name, config in TemperatureConfig.PRESETS.items():
    print(f"\n{name}:")
    print(f"  {config['description']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Top-p: {config['top_p']}")
    print(f"  Top-k: {config['top_k']}")
```

---

## Gerando Datasets

### Exemplo 11: Dataset Pequeno para Testes

```python
from generate_training_dataset_v2 import TrainingDatasetGenerator

generator = TrainingDatasetGenerator(key_size_bits=256)

# Gerar 100 exemplos
examples, stats = generator.generate_dataset(
    num_examples=100,
    min_quality_score=85.0,
    format_strategy='instructional',
    save_to_vector_store=True
)

# Salvar
generator.save_dataset(examples, "test_dataset.jsonl", split='train')

# Analisar
generator.analyze_dataset("datasets/test_dataset.jsonl")
```

### Exemplo 12: Dataset Grande para Produção

```python
from generate_training_dataset_v2 import generate_multiple_datasets

# Gerar datasets completos (10k train + 2k val)
generate_multiple_datasets(
    key_size_bits=256,
    sizes={'production': (10000, 2000)}
)

# Arquivos gerados:
# - datasets/training_data_instructional_256bit.jsonl
# - datasets/validation_data_instructional_256bit.jsonl
# - datasets/vector_store.jsonl (atualizado)
```

### Exemplo 13: Diferentes Formatos

```python
strategies = ['instructional', 'conversational', 'technical']

for strategy in strategies:
    examples, _ = generator.generate_dataset(
        num_examples=10,
        format_strategy=strategy
    )
    
    print(f"\n{strategy.upper()} format:")
    print(examples[0]['text'][:200] + "...")
```

---

## Fine-tuning

### Exemplo 14: Fine-tuning Básico

```python
from finetune_model_v2 import AdvancedModelFineTuner

tuner = AdvancedModelFineTuner(
    base_model="gemma3:latest",
    output_model="my-crypto-model",
    key_size_bits=256
)

# Criar Modelfile
tuner.create_advanced_modelfile(
    num_examples=20,
    min_quality=95.0,
    temperature_preset='high_entropy'
)

# Criar modelo
if tuner.create_model_with_ollama():
    print("✓ Modelo criado!")
    
    # Testar
    results = tuner.test_model(num_tests=20)
    print(f"Quality médio: {results['statistics']['quality_score']['mean']:.2f}")
```

### Exemplo 15: Fine-tuning Personalizado

```python
# Configuração avançada
tuner = AdvancedModelFineTuner(
    base_model="gemma3:latest",
    output_model="gemma3-crypto-extreme",
    key_size_bits=256
)

# Usar apenas os MELHORES exemplos
tuner.create_advanced_modelfile(
    num_examples=30,           # Mais exemplos
    min_quality=98.0,          # Apenas excelentes
    temperature_preset='extreme_random'  # Máxima aleatoriedade
)

tuner.create_model_with_ollama()
```

### Exemplo 16: Comparar Modelos

```python
# Comparar base vs fine-tuned
comparison = tuner.compare_with_baseline(
    baseline_model="gemma3:latest",
    num_tests=50
)

# Resultados salvos em fine_tuning_results/comparison_results.json
```

---

## Avaliação e Comparação

### Exemplo 17: Avaliação Rápida

```python
from comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()

# Avaliar uma configuração
evaluator.evaluate_configuration(
    config_name="test_config",
    model="gemma3:latest",
    key_size_bits=256,
    strategy='few-shot',
    temperature_preset='high_entropy',
    num_tests=30
)
```

### Exemplo 18: Comparar Estratégias

```python
# Comparar todas as estratégias de prompt
evaluator.compare_strategies(
    model="gemma3:latest",
    key_size_bits=256,
    temperature_preset='high_entropy',
    num_tests=50
)

# Gerar relatório com gráficos
evaluator.generate_comparison_report('strategies')

# Gráfico salvo em: evaluation_results/comparison_strategies.png
```

### Exemplo 19: Comparar Temperaturas

```python
# Comparar presets de temperatura
evaluator.compare_temperatures(
    model="gemma3:latest",
    strategy='few-shot',
    key_size_bits=256,
    num_tests=50
)

evaluator.generate_comparison_report('temperatures')
```

### Exemplo 20: Relatório Completo

```python
# Executar todas as comparações
evaluator.compare_strategies(...)
evaluator.compare_temperatures(...)
evaluator.compare_key_sizes(...)

# Gerar relatório markdown completo
evaluator.generate_full_report()

# Arquivo: evaluation_results/full_report.md
```

---

## Usando a API REST

### Exemplo 21: Iniciar API

```bash
# Iniciar servidor
python3 app_v2.py --host 0.0.0.0 --port 5000

# Com debug
python3 app_v2.py --debug
```

### Exemplo 22: Gerar Chave via API

```bash
# GET simples (configuração padrão)
curl http://localhost:5000/generate_key

# POST com configuração customizada
curl -X POST http://localhost:5000/generate_key \
  -H "Content-Type: application/json" \
  -d '{
    "key_size": 256,
    "strategy": "few-shot",
    "temperature": "high_entropy",
    "model": "gemma3:latest",
    "use_drand": true
  }'
```

### Exemplo 23: Gerar Lote via API

```bash
curl -X POST http://localhost:5000/generate_batch \
  -H "Content-Type: application/json" \
  -d '{
    "num_keys": 10,
    "key_size": 256,
    "strategy": "few-shot",
    "temperature": "high_entropy"
  }'
```

### Exemplo 24: Consultar Vector Store via API

```bash
# Estatísticas
curl http://localhost:5000/vector_store/stats

# Top exemplos
curl "http://localhost:5000/vector_store/top_examples?n=5&min_quality=95"

# Estatísticas dos clientes
curl http://localhost:5000/client/stats
```

### Exemplo 25: Python + Requests

```python
import requests

# Gerar chave
response = requests.post(
    'http://localhost:5000/generate_key',
    json={
        'key_size': 256,
        'strategy': 'few-shot',
        'temperature': 'high_entropy'
    }
)

data = response.json()

if data['success']:
    print(f"Chave: {data['key_hex']}")
    print(f"Quality: {data['metrics']['quality_score']:.2f}")
else:
    print(f"Erro: {data['error']}")
```

---

## Casos de Uso Avançados

### Exemplo 26: Pipeline Completo

```python
# 1. Gerar vector store
from vector_store_system import VectorStore, generate_high_quality_dataset

vs = VectorStore("datasets/production_vs.jsonl")
generate_high_quality_dataset(vs, 5000, 256)

# 2. Gerar datasets
from generate_training_dataset_v2 import generate_multiple_datasets

generate_multiple_datasets(256, {'prod': (10000, 2000)})

# 3. Fine-tuning
from finetune_model_v2 import AdvancedModelFineTuner

tuner = AdvancedModelFineTuner(
    output_model="gemma3-production"
)
tuner.create_advanced_modelfile(num_examples=30, min_quality=95)
tuner.create_model_with_ollama()

# 4. Avaliar
from comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
evaluator.compare_with_baseline(num_tests=100)
evaluator.generate_full_report()
```

### Exemplo 27: Monitoramento de Qualidade

```python
import time

client = LLMClient(
    model="gemma3-production",
    strategy='few-shot',
    temperature_preset='high_entropy'
)

# Monitorar qualidade ao longo do tempo
quality_history = []

for i in range(100):
    result = client.generate_key(secrets.token_hex(32))
    
    if result['success']:
        quality_history.append({
            'timestamp': time.time(),
            'quality': result['metrics']['quality_score'],
            'entropy': result['metrics']['entropy']
        })
    
    time.sleep(1)

# Análise
import numpy as np
qualities = [h['quality'] for h in quality_history]
print(f"Quality médio: {np.mean(qualities):.2f}")
print(f"Desvio padrão: {np.std(qualities):.2f}")
print(f"Abaixo de 90: {sum(1 for q in qualities if q < 90)}")
```

### Exemplo 28: Integração com Sistema Existente

```python
class CryptoKeyManager:
    """Gerenciador de chaves criptográficas"""
    
    def __init__(self):
        self.client = LLMClient(
            model="gemma3-production",
            key_size_bits=256,
            strategy='few-shot',
            temperature_preset='high_entropy'
        )
        
        self.vector_store = VectorStore("datasets/production_vs.jsonl")
    
    def generate_key(self, purpose: str) -> dict:
        """Gera chave para propósito específico"""
        from drand_client import get_entropy_seed
        
        # Usar drand como seed
        seed = get_entropy_seed()
        
        # Gerar chave
        result = self.client.generate_key(seed)
        
        if result['success']:
            # Adicionar ao vector store se alta qualidade
            if result['metrics']['quality_score'] >= 90:
                self.vector_store.add_example(
                    seed=seed,
                    key_hex=result['key_hex'],
                    metadata={'purpose': purpose}
                )
                self.vector_store.save()
            
            return {
                'key': result['key_hex'],
                'quality': result['metrics']['quality_score'],
                'entropy': result['metrics']['entropy'],
                'purpose': purpose
            }
        else:
            raise RuntimeError(f"Failed to generate key: {result['error']}")

# Uso
manager = CryptoKeyManager()
key_info = manager.generate_key("database_encryption")
print(f"Chave gerada: {key_info['key']}")
```

---

## Dicas e Boas Práticas

### 💡 Para Máxima Qualidade

```python
client = LLMClient(
    model="gemma3-entropy-v2",        # Modelo fine-tuned
    strategy='few-shot',               # Usa exemplos de alta qualidade
    temperature_preset='high_entropy', # Máxima entropia
    key_size_bits=256                 # AES-256
)
```

### 💡 Para Experimentação

```python
client = LLMClient(
    strategy='cot',                    # Raciocínio explícito
    temperature_preset='extreme_random', # Teste de limites
    key_size_bits=128                  # Teste com chaves menores
)
```

### 💡 Para Produção

```python
# 1. Fine-tune com dataset grande
# 2. Use few-shot + high_entropy
# 3. Monitore métricas
# 4. Mantenha vector store atualizado
# 5. Execute avaliações periódicas
```

---

## Troubleshooting Rápido

### Problema: Quality score baixo

```python
# Solução: Aumentar temperatura ou usar modelo fine-tuned
client = LLMClient(
    model="gemma3-entropy-v2",
    temperature_preset='high_entropy'
)
```

### Problema: Hex insuficiente

```python
# Solução: O modelo não está gerando output suficiente
# Verifique o prompt ou aumente num_predict
```

### Problema: Ollama não conecta

```bash
# Solução: Iniciar Ollama
ollama serve

# Em outro terminal
python3 app_v2.py
```

---

Para mais exemplos, consulte:
- `test_system.py` - Testes funcionais
- `comprehensive_evaluation.py` - Avaliações avançadas
- `README_IMPROVEMENTS.md` - Documentação técnica