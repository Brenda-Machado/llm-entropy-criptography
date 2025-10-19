# Guia de Setup - Ollama + Gemma3 270M

## 1. Instalar Ollama

Acesse [ollama.ai](https://ollama.ai) e baixe a versão para seu SO (Windows, macOS, Linux).

## 2. Puxar o Modelo Gemma3

Abra um terminal/cmd e execute:

```bash
ollama pull gemma3
```

Isso pode levar alguns minutos dependendo de sua conexão.

## 3. Iniciar Ollama Server

Em um terminal, inicie o servidor:

```bash
ollama serve
```

Você verá algo como:
```
2024/01/15 10:30:45 "Listening on 127.0.0.1:11434"
```

**Deixe este terminal aberto** - é o servidor Ollama.

## 4. Instalar Dependências Python

Em outro terminal/cmd, no diretório do projeto:

```bash
pip install -r requirements.txt
```

## 5. Executar a Aplicação

```bash
python app.py
```

Você verá:
```
 * Running on http://127.0.0.1:5000
```

## 6. Testar o Endpoint

Em outro terminal:

```bash
curl http://localhost:5000/generate_key
```

Ou acesse no navegador:
```
http://localhost:5000/generate_key
```

## Resposta Esperada

```json
{
  "success": true,
  "key_hex": "a1b2c3d4...",
  "entropy_shannon": 7.95,
  "valid_entropy": true,
  "llm_candidate": "1a2b3c4d...",
  "drand_seed": "...",
  "llm_entropy_hex": "..."
}
```

## Variáveis de Ambiente (Opcional)

```bash
# Linux/macOS
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=gemma3
export DEBUG=true

# Windows (PowerShell)
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="gemma3"
$env:DEBUG="true"
```

## Troubleshooting

**Erro: "Não foi possível conectar ao Ollama"**
- Certifique-se de que `ollama serve` está rodando em outro terminal
- Verifique se a porta 11434 não está bloqueada

**Erro: "Modelo não encontrado"**
- Execute: `ollama pull gemma3`
- Verifique com: `ollama list`

**Aplicação lenta**
- Primeira execução carrega o modelo em RAM
- Gemma3 270M precisa de ~600MB RAM
- Próximas requisições serão mais rápidas
