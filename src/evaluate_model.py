"""
PoC : Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

evaluate_model.py
"""

import requests
import json
import secrets
from utils import calculate_entropy, is_valid_entropy
from config import OLLAMA_BASE_URL
import matplotlib.pyplot as plt
import numpy as np

def test_model_entropy(model_name, num_tests=50):
    print(f"\n=== Testando modelo: {model_name} ===")
    print(f"Número de testes: {num_tests}")
    
    results = {
        "model": model_name,
        "entropies": [],
        "valid_count": 0,
        "invalid_count": 0,
        "errors": 0,
        "avg_response_time": []
    }
    
    for i in range(num_tests):
        seed = secrets.token_hex(32)
        prompt = f"Generate a high-entropy cryptographic key from seed: {seed}\n\nOutput only 64 hexadecimal characters:"
        
        try:
            import time
            start_time = time.time()
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.9
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            results["avg_response_time"].append(elapsed)
            
            if response.status_code == 200:
                data = response.json()
                output = data.get('response', '').strip()
                hex_chars = ''.join(c for c in output if c in '0123456789abcdefABCDEF')
                
                if len(hex_chars) >= 64:
                    key_bytes = bytes.fromhex(hex_chars[:64])
                    entropy = calculate_entropy(key_bytes)
                    valid = is_valid_entropy(key_bytes)
                    
                    results["entropies"].append(entropy)
                    if valid:
                        results["valid_count"] += 1
                    else:
                        results["invalid_count"] += 1
                else:
                    results["errors"] += 1
            else:
                results["errors"] += 1
                
        except Exception as e:
            print(f"Erro no teste {i+1}: {str(e)}")
            results["errors"] += 1
        
        if (i + 1) % 10 == 0:
            print(f"Progresso: {i+1}/{num_tests}")
    
    if results["entropies"]:
        results["avg_entropy"] = np.mean(results["entropies"])
        results["std_entropy"] = np.std(results["entropies"])
        results["min_entropy"] = np.min(results["entropies"])
        results["max_entropy"] = np.max(results["entropies"])
        results["median_entropy"] = np.median(results["entropies"])
        results["avg_time"] = np.mean(results["avg_response_time"])
    
    return results

def compare_models(base_model="gemma3:latest", finetuned_model="gemma3-entropy", num_tests=50):
    base_results = test_model_entropy(base_model, num_tests)
    finetuned_results = test_model_entropy(finetuned_model, num_tests)
    
    print(f"\n{'Métrica':<30} {'Base':<20} {'Fine-tuned':<20} {'Melhoria':<15}")
    print("-"*85)
    
    if base_results["entropies"] and finetuned_results["entropies"]:
        metrics = [
            ("Entropia Média", "avg_entropy", "higher"),
            ("Desvio Padrão", "std_entropy", "lower"),
            ("Entropia Mínima", "min_entropy", "higher"),
            ("Entropia Máxima", "max_entropy", "higher"),
            ("Entropia Mediana", "median_entropy", "higher"),
            ("Chaves Válidas (%)", "valid_count", "higher"),
            ("Tempo Médio (s)", "avg_time", "lower")
        ]
        
        for metric_name, metric_key, better in metrics:
            if metric_key == "valid_count":
                base_val = (base_results[metric_key] / num_tests) * 100
                fine_val = (finetuned_results[metric_key] / num_tests) * 100
                improvement = fine_val - base_val
            elif metric_key in base_results and metric_key in finetuned_results:
                base_val = base_results[metric_key]
                fine_val = finetuned_results[metric_key]
                if better == "higher":
                    improvement = ((fine_val - base_val) / base_val) * 100
                else:
                    improvement = ((base_val - fine_val) / base_val) * 100
            else:
                continue
            
            symbol = "↑" if improvement > 0 else "↓"
            print(f"{metric_name:<30} {base_val:<20.4f} {fine_val:<20.4f} {symbol} {abs(improvement):.2f}%")
    
    results = {
        "base_model": base_results,
        "finetuned_model": finetuned_results
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dumps(results, f, indent=2)
    
    plot_comparison(base_results, finetuned_results)
    
    return results

def plot_comparison(base_results, finetuned_results):
    if not base_results["entropies"] or not finetuned_results["entropies"]:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(base_results["entropies"], bins=20, alpha=0.5, label="Base Model", color="blue")
    axes[0].hist(finetuned_results["entropies"], bins=20, alpha=0.5, label="Fine-tuned", color="green")
    axes[0].axvline(7.99, color='red', linestyle='--', label="Threshold (7.99)")
    axes[0].set_xlabel("Entropia Shannon")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Distribuição de Entropia")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    data_to_plot = [base_results["entropies"], finetuned_results["entropies"]]
    axes[1].boxplot(data_to_plot, labels=["Base", "Fine-tuned"])
    axes[1].axhline(7.99, color='red', linestyle='--', label="Threshold")
    axes[1].set_ylabel("Entropia Shannon")
    axes[1].set_title("Comparação de Entropia")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("entropy_comparison.png", dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico salvo em: entropy_comparison.png")

if __name__ == "__main__":
    compare_models(
        base_model="gemma3:latest",
        finetuned_model="gemma3-entropy",
        num_tests=100
    )