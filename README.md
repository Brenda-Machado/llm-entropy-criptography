# PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

## Objetivo

Prova de Conceito (PoC) para geração e validação de chaves criptográficas, integrando uma LLM (via API) e fonte pública de entropia (League of Entropy/drand). 

## Funcionalidades

- Consome entropia diretamente da API drand.
- Gera chaves usando uma LLM baseada em seed público.
- Aplica pós-processamento e validação estatística (entropia/NIST).
- Expõe serviço REST via Flask.


## Instalação

```bash
git clone https://github.com/Brenda-Machado/llm-entropy-criptography.git
cd llm-entropy-criptography
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

ou

```bash
curl http://localhost:5000/generate_key
```

## Estrutura

- `app.py`: API Flask e orquestração
- `drand_client.py`: integração League of Entropy
- `llm_client.py`: integração LLM
- `utils.py`: processamento/validação
- `requirements.txt`: dependências


## Licença

MIT

***
