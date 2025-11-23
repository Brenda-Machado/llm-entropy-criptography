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

Deixe este terminal aberto, é o servidor Ollama.
