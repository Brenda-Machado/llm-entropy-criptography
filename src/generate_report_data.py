#!/usr/bin/env python3
"""
Script para gerar dados quantitativos para o relatório
Compara modelo base, fine-tuned e referência secrets
"""

import sys
import os
import argparse
sys.path.insert(0, 'src')
import numpy as np
import secrets as py_secrets
import json
import matplotlib.pyplot as plt
from typing import Dict, List
from llm_client import LLMClient
from nist_tests import NISTTests
from test_nist_integration import NISTValidator
from scipy import stats
from collections import Counter
from vector_store import QualityMetrics

class ReportDataGenerator:
    def __init__(self):
        self.results = {
            'base_model': {'keys': [], 'metrics': []},
            'few_shot': {'keys': [], 'metrics': []},
            'zero_shot': {'keys': [], 'metrics': []},
            'cot': {'keys': [], 'metrics': []},
            'secrets_ref': {'keys': [], 'metrics': []}
        }
    
    def generate_baseline_data(self, num_samples: int = 100):
        client = LLMClient(
            model='gemma3:latest', 
            key_size_bits=256,
            strategy='few-shot',
            temperature_preset='high_entropy'
        )
        
        for i in range(num_samples):
            seed = py_secrets.token_hex(32)
            result = client.generate_key(seed, store_result=False)
            
            if result['success']:
                self.results['base_model']['keys'].append(result['key_hex'])
                self.results['base_model']['metrics'].append(result['metrics'])
            
            if (i + 1) % 10 == 0:
                print(f"  Progresso: {i+1}/{num_samples}")
        
        self._print_summary('base_model')
    
    def generate_finetuned_data(self, num_samples: int = 100):
        strategies = {
            'few_shot': 'few-shot',
            'zero_shot': 'zero-shot',
            'cot': 'cot'
        }
        
        for key, strategy in strategies.items():
            client = LLMClient(
                model='gemma3-entropy-v2',  
                key_size_bits=256,
                strategy=strategy,
                temperature_preset='high_entropy'
            )
            
            for i in range(num_samples):
                seed = py_secrets.token_hex(32)
                result = client.generate_key(seed, store_result=False)
                
                if result['success']:
                    self.results[key]['keys'].append(result['key_hex'])
                    self.results[key]['metrics'].append(result['metrics'])
                
                if (i + 1) % 10 == 0:
                    print(f"  Progresso: {i+1}/{num_samples}")
            
            self._print_summary(key)
    
    def generate_reference_data(self, num_samples: int = 100):
        for i in range(num_samples):
            key_bytes = py_secrets.token_bytes(32) 
            key_hex = key_bytes.hex()
            
            entropy = QualityMetrics.calculate_shannon_entropy(key_bytes)
            unique_bytes = QualityMetrics.count_unique_bytes(key_bytes)
            quality_score = QualityMetrics.calculate_quality_score(key_bytes, 256)
            
            self.results['secrets_ref']['keys'].append(key_hex)
            self.results['secrets_ref']['metrics'].append({
                'entropy': float(entropy),
                'unique_bytes': int(unique_bytes),
                'quality_score': float(quality_score),
                'valid_entropy': entropy >= 7.9
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Progresso: {i+1}/{num_samples}")
        
        self._print_summary('secrets_ref')
    
    def _print_summary(self, config_name: str):
        metrics = self.results[config_name]['metrics']
        
        if not metrics:
            print(f"  Nenhuma métrica para {config_name}")
            return
        
        entropies = [m['entropy'] for m in metrics]
        quality_scores = [m['quality_score'] for m in metrics]
        high_quality = sum(1 for q in quality_scores if q >= 90)
        valid_entropy = sum(1 for e in entropies if e >= 7.9)
        
        print(f"\n  Resumo ({config_name}):")
        print(f"    Amostras: {len(metrics)}")
        print(f"    Entropia: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
        print(f"    Quality Score: {np.mean(quality_scores):.1f} ± {np.std(quality_scores):.1f}")
        print(f"    Alta Qualidade: {high_quality}/{len(metrics)} ({high_quality/len(metrics)*100:.1f}%)")
        print(f"    Entropia Válida: {valid_entropy}/{len(metrics)} ({valid_entropy/len(metrics)*100:.1f}%)")
    
    def generate_comparison_table(self) -> Dict:
        comparison = {}
        
        for config_name in ['base_model', 'few_shot', 'secrets_ref']:
            metrics = self.results[config_name]['metrics']
            
            if not metrics:
                continue
            
            entropies = [m['entropy'] for m in metrics]
            quality_scores = [m['quality_score'] for m in metrics]
            high_quality_pct = sum(1 for q in quality_scores if q >= 90) / len(metrics) * 100
            valid_entropy_pct = sum(1 for e in entropies if e >= 7.9) / len(metrics) * 100
            
            comparison[config_name] = {
                'entropy_mean': np.mean(entropies),
                'entropy_std': np.std(entropies),
                'quality_mean': np.mean(quality_scores),
                'quality_std': np.std(quality_scores),
                'high_quality_pct': high_quality_pct,
                'valid_entropy_pct': valid_entropy_pct
            }
        
        if 'base_model' in comparison and 'few_shot' in comparison:
            base = comparison['base_model']
            fs = comparison['few_shot']
            ref = comparison['secrets_ref']
            
            comparison['improvements'] = {
                'entropy_abs': fs['entropy_mean'] - base['entropy_mean'],
                'entropy_rel': (fs['entropy_mean'] - base['entropy_mean']) / base['entropy_mean'] * 100,
                'quality_abs': fs['quality_mean'] - base['quality_mean'],
                'quality_rel': (fs['quality_mean'] - base['quality_mean']) / base['quality_mean'] * 100,
                'high_quality_pp': fs['high_quality_pct'] - base['high_quality_pct'],
                'high_quality_rel': (fs['high_quality_pct'] - base['high_quality_pct']) / base['high_quality_pct'] * 100,
                'valid_entropy_pp': fs['valid_entropy_pct'] - base['valid_entropy_pct'],
                'valid_entropy_rel': (fs['valid_entropy_pct'] - base['valid_entropy_pct']) / base['valid_entropy_pct'] * 100
            }
            
            comparison['gap_to_reference'] = {
                'entropy': fs['entropy_mean'] - ref['entropy_mean'],
                'quality': fs['quality_mean'] - ref['quality_mean'],
                'high_quality_pp': fs['high_quality_pct'] - ref['high_quality_pct'],
                'valid_entropy_pp': fs['valid_entropy_pct'] - ref['valid_entropy_pct']
            }
        
        self._print_comparison_table(comparison)
        
        return comparison
    
    def _print_comparison_table(self, comparison: Dict):
        print("\n" + "="*80)
        print("Comparação entre modelo base, fine-tuned e referência")
        print("="*80)
        print()
        
        base = comparison.get('base_model', {})
        fs = comparison.get('few_shot', {})
        ref = comparison.get('secrets_ref', {})
        imp = comparison.get('improvements', {})
        gap = comparison.get('gap_to_reference', {})
        
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\caption{Comparação entre modelo base, fine-tuned e referência}")
        print("\\label{tab:comparison-baseline}")
        print("\\begin{tabular}{@{}lcccc@{}}")
        print("\\toprule")
        print("\\textbf{Configuração} & \\textbf{Entropia} & \\textbf{Quality} & \\textbf{Alta Q.} & \\textbf{Valid E.} \\\\")
        print(" & \\textbf{(bits/byte)} & \\textbf{Score} & \\textbf{(\\%)} & \\textbf{(\\%)} \\\\ \\midrule")
        
        if base:
            print(f"Gemma3 base & {base['entropy_mean']:.3f} $\\pm$ {base['entropy_std']:.3f} & "
                  f"{base['quality_mean']:.1f} & {base['high_quality_pct']:.0f}\\% & {base['valid_entropy_pct']:.0f}\\% \\\\")
        
        if fs:
            print(f"Few-shot (fine-tuned) & {fs['entropy_mean']:.3f} $\\pm$ {fs['entropy_std']:.3f} & "
                  f"{fs['quality_mean']:.1f} & {fs['high_quality_pct']:.0f}\\% & {fs['valid_entropy_pct']:.0f}\\% \\\\")
        
        if ref:
            print(f"\\texttt{{secrets}} (ref.) & {ref['entropy_mean']:.3f} $\\pm$ {ref['entropy_std']:.3f} & "
                  f"{ref['quality_mean']:.1f} & {ref['high_quality_pct']:.0f}\\% & {ref['valid_entropy_pct']:.0f}\\% \\\\")
        
        print("\\midrule")
        
        if imp:
            print(f"Melhoria absoluta & +{imp['entropy_abs']:.3f} & +{imp['quality_abs']:.1f} & "
                  f"+{imp['high_quality_pp']:.0f}pp & +{imp['valid_entropy_pp']:.0f}pp \\\\")
            print(f"Melhoria relativa & +{imp['entropy_rel']:.1f}\\% & +{imp['quality_rel']:.1f}\\% & "
                  f"+{imp['high_quality_rel']:.0f}\\% & +{imp['valid_entropy_rel']:.0f}\\% \\\\")
        
        if gap:
            print(f"Gap para referência & {gap['entropy']:.3f} & {gap['quality']:.1f} & "
                  f"{gap['high_quality_pp']:.0f}pp & {gap['valid_entropy_pp']:.0f}pp \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def analyze_distribution(self):
        for config_name in ['few_shot', 'secrets_ref', 'base_model']:
            if config_name not in self.results or not self.results[config_name]['keys']:
                continue
            
            print(f"\n{config_name.upper()}:")
            
            all_bytes = []

            for key_hex in self.results[config_name]['keys']:
                key_bytes = bytes.fromhex(key_hex)
                all_bytes.extend(key_bytes)
            
            observed_freq = Counter(all_bytes)
            n_bytes = len(all_bytes)
            expected_freq = n_bytes / 256
            observed = [observed_freq.get(i, 0) for i in range(256)]
            expected = [expected_freq] * 256
            
            chi2, p_value = stats.chisquare(observed, expected)
            
            print(f"  Total de bytes: {n_bytes}")
            print(f"  Chi-quadrado: χ² = {chi2:.1f}")
            print(f"  Graus de liberdade: {255}")
            print(f"  P-value: p = {p_value:.3f}")
            print(f"  Resultado: {'UNIFORME (não rejeita H0)' if p_value > 0.05 else 'NÃO UNIFORME (rejeita H0)'}")
    
    def analyze_autocorrelation(self, max_lag: int = 10):
        for config_name in ['few_shot', 'secrets_ref']:
            if config_name not in self.results or not self.results[config_name]['keys']:
                continue
            
            print(f"\n{config_name.upper()}:")

            all_bytes = []

            for key_hex in self.results[config_name]['keys']:
                key_bytes = bytes.fromhex(key_hex)
                all_bytes.extend(key_bytes)

            series = np.array(all_bytes, dtype=float)
            mean = np.mean(series)
            var = np.var(series)
            autocorr = []
            n = len(series)
            confidence_interval = 2 / np.sqrt(n)
            
            for lag in range(1, max_lag + 1):
                c = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
                r = c / var
                autocorr.append(r)
            
            print(f"  Lags testados: 1-{max_lag}")
            print(f"  Intervalo de confiança 95%: ±{confidence_interval:.4f}")
            print(f"  Autocorrelações:")
            
            all_within_bounds = True

            for lag, r in enumerate(autocorr, 1):
                within = "✓" if abs(r) < confidence_interval else "✗"

                print(f"    Lag {lag}: {r:+.4f} {within}")

                if abs(r) >= confidence_interval:
                    all_within_bounds = False
            
            print(f"\n  Resultado: {'SEM estrutura temporal significativa' if all_within_bounds else 'Estrutura temporal detectada'}")
    
    def run_nist_tests(self, num_samples: int = 10):
        nist_results = {}
        
        for config_name in ['few_shot', 'base_model', 'secrets_ref']:
            if config_name not in self.results or not self.results[config_name]['keys']:
                continue
            
            print(f"\n{config_name.upper()}:")
            
            keys_sample = self.results[config_name]['keys'][:num_samples]
            pass_rates = []
            
            for i, key_hex in enumerate(keys_sample):
                print(f"  Testando amostra {i+1}/{len(keys_sample)}...", end=' ')
                
                results = NISTTests.run_all_tests_from_hex(key_hex)
                pass_rate = results['summary']['pass_rate']
                pass_rates.append(pass_rate)
                
                print(f"Pass rate: {pass_rate:.1f}%")
            
            nist_results[config_name] = {
                'pass_rates': pass_rates,
                'mean': np.mean(pass_rates),
                'std': np.std(pass_rates),
                'min': np.min(pass_rates),
                'max': np.max(pass_rates)
            }
            
            print(f"\n  Resumo NIST:")
            print(f"    Taxa média de aprovação: {nist_results[config_name]['mean']:.1f}% ± {nist_results[config_name]['std']:.1f}%")
            print(f"    Min/Max: {nist_results[config_name]['min']:.1f}% / {nist_results[config_name]['max']:.1f}%")
        
        return nist_results
    
    def plot_entropy_distribution(self, save_path: str = 'entropy_distribution.pdf'):
        plt.figure(figsize=(12, 6))
        
        colors = {
            'base_model': '#ef4444',
            'zero_shot': '#f59e0b',
            'cot': '#10b981',
            'few_shot': '#3b82f6',
            'secrets_ref': '#6366f1'
        }
        
        labels = {
            'base_model': 'Gemma3 Base',
            'zero_shot': 'Zero-shot',
            'cot': 'Chain-of-Thought',
            'few_shot': 'Few-shot (fine-tuned)',
            'secrets_ref': 'secrets (referência)'
        }
        
        for config_name, color in colors.items():
            if config_name not in self.results or not self.results[config_name]['metrics']:
                continue
            
            entropies = [m['entropy'] for m in self.results[config_name]['metrics']]
            plt.hist(entropies, bins=30, alpha=0.6, label=labels[config_name], 
                    color=color, edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Entropia Shannon (bits/byte)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição de Entropia: Comparação entre Configurações', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Gráfico salvo em: {save_path}")
        plt.close()
    
    def save_results(self, filename: str = 'report_data.json'):
        output = {}

        for config, data in self.results.items():
            output[config] = {
                'num_samples': len(data['keys']),
                'keys': data['keys'][:10], 
                'metrics_summary': {
                    'entropy_mean': float(np.mean([m['entropy'] for m in data['metrics']])) if data['metrics'] else 0,
                    'entropy_std': float(np.std([m['entropy'] for m in data['metrics']])) if data['metrics'] else 0,
                    'quality_mean': float(np.mean([m['quality_score'] for m in data['metrics']])) if data['metrics'] else 0,
                    'quality_std': float(np.std([m['quality_score'] for m in data['metrics']])) if data['metrics'] else 0
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Gera dados para o relatório")
    parser.add_argument('--samples', type=int, default=100, help='Número de amostras por configuração')
    parser.add_argument('--nist-samples', type=int, default=10, help='Número de amostras para testes NIST')
    parser.add_argument('--only-gemma', action='store_true', help='Gera apenas dados do Gemma (recomendado)')
    parser.add_argument('--skip-baseline', action='store_true', help='Pula geração do modelo base')
    parser.add_argument('--skip-finetuned', action='store_true', help='Pula geração do modelo fine-tuned')
    parser.add_argument('--skip-reference', action='store_true', help='Pula geração da referência')
    
    args = parser.parse_args()
    
    generator = ReportDataGenerator()
    
    if args.only_gemma:
        generator.generate_finetuned_data(args.samples)
        generator.generate_reference_data(min(args.samples, 100)) 
    else:
        if not args.skip_finetuned:
            generator.generate_finetuned_data(args.samples)
        
        if not args.skip_baseline:
            generator.generate_baseline_data(args.samples)

        if not args.skip_reference:
            generator.generate_reference_data(min(args.samples, 100))
    
    generator.generate_comparison_table()
    generator.analyze_distribution()
    generator.analyze_autocorrelation()
    generator.run_nist_tests(args.nist_samples)
    generator.plot_entropy_distribution()
    generator.save_results()
    
    print("\n" + "="*80)
    print("INTERPRETAÇÃO DOS RESULTADOS:")
    print("="*80)
    print("\nDADOS PRINCIPAIS (Gemma3):")
    print("   - few_shot: modelo fine-tuned com estratégia few-shot")
    print("   - zero_shot: modelo fine-tuned com estratégia zero-shot")
    print("   - cot: modelo fine-tuned com estratégia chain-of-thought")
    print("\nBASELINE DE COMPARAÇÃO:")
    print("   - secrets: gerador criptográfico estabelecido (Python CSPRNG)")
    print("   - base_model: Gemma3 sem fine-tuning (baseline interno)")
    print("\nMÉTRICAS DE AVALIAÇÃO:")
    print("   - Entropia Shannon: deve estar próxima de 8.0 bits/byte")
    print("   - Quality Score: deve ser ≥ 90")
    print("   - Testes NIST: taxa de aprovação deve ser ≥ 95%")


if __name__ == "__main__":
    main()