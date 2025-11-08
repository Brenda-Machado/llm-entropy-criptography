"""
PoC: Avaliação do Uso de Inteligência Artificial na Geração de Entropia para Chaves Criptográficas

Author: Brenda Silva Machado

comprehensive_evaluation.py 
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from llm_client_improved import LLMClient

class ComprehensiveEvaluator:
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def evaluate_configuration(self,
                              config_name: str,
                              model: str,
                              key_size_bits: int,
                              strategy: str,
                              temperature_preset: str,
                              num_tests: int = 50) -> Dict:

        print(f"Avaliando: {config_name}")
        print(f"  Modelo: {model}")
        print(f"  Tamanho da chave: {key_size_bits} bits")
        print(f"  Estratégia: {strategy}")
        print(f"  Temperatura: {temperature_preset}")
        print(f"  Número de testes: {num_tests}")
        
        client = LLMClient(
            model=model,
            key_size_bits=key_size_bits,
            strategy=strategy,
            temperature_preset=temperature_preset
        )
        
        results, stats = client.batch_generate(num_tests)
        result_data = {
            'config_name': config_name,
            'config': {
                'model': model,
                'key_size_bits': key_size_bits,
                'strategy': strategy,
                'temperature_preset': temperature_preset,
                'num_tests': num_tests
            },
            'statistics': stats,
            'individual_results': results
        }
        
        self.results[config_name] = result_data
        output_file = self.output_dir / f"{config_name}.json"
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        

        print(f"  Taxa de sucesso: {stats['success_rate']:.2f}%")

        if 'quality_score' in stats:
            print(f"  Quality score médio: {stats['quality_score']['mean']:.2f}")
            print(f"  Entropia média: {stats['entropy']['mean']:.4f}")
            print(f"  Chaves de alta qualidade (≥90): {stats['high_quality_count']}/{num_tests}")
            print(f"  Chaves com entropia válida (≥7.9): {stats['valid_entropy_count']}/{num_tests}")
        
        return result_data
    
    def compare_strategies(self,
                          model: str,
                          key_size_bits: int = 256,
                          temperature_preset: str = 'high_entropy',
                          num_tests: int = 50):
        strategies = ['zero-shot', 'few-shot', 'cot']
        
        for strategy in strategies:
            config_name = f"{model}_{key_size_bits}bit_{strategy}_{temperature_preset}"
            self.evaluate_configuration(
                config_name=config_name,
                model=model,
                key_size_bits=key_size_bits,
                strategy=strategy,
                temperature_preset=temperature_preset,
                num_tests=num_tests
            )
    
    def compare_temperatures(self,
                            model: str,
                            strategy: str = 'few-shot',
                            key_size_bits: int = 256,
                            num_tests: int = 50):
        presets = ['balanced', 'high_entropy', 'extreme_random']
        
        for preset in presets:
            config_name = f"{model}_{key_size_bits}bit_{strategy}_{preset}"
            self.evaluate_configuration(
                config_name=config_name,
                model=model,
                key_size_bits=key_size_bits,
                strategy=strategy,
                temperature_preset=preset,
                num_tests=num_tests
            )
    
    def compare_key_sizes(self,
                         model: str,
                         strategy: str = 'few-shot',
                         temperature_preset: str = 'high_entropy',
                         num_tests: int = 50):
        key_sizes = [128, 192, 256]
        
        for key_size in key_sizes:
            config_name = f"{model}_{key_size}bit_{strategy}_{temperature_preset}"
            self.evaluate_configuration(
                config_name=config_name,
                model=model,
                key_size_bits=key_size,
                strategy=strategy,
                temperature_preset=temperature_preset,
                num_tests=num_tests
            )
    
    def generate_comparison_report(self, comparison_type: str = 'strategies'):
        if not self.results:
            print("Nenhum resultado para comparar. Execute avaliações primeiro.")

            return

        comparison_data = []
        
        for config_name, data in self.results.items():
            if 'statistics' not in data or 'quality_score' not in data['statistics']:
                continue
            
            stats = data['statistics']
            config = data['config']
            
            comparison_data.append({
                'config_name': config_name,
                'strategy': config['strategy'],
                'temperature': config['temperature_preset'],
                'key_size': config['key_size_bits'],
                'success_rate': stats['success_rate'],
                'quality_mean': stats['quality_score']['mean'],
                'quality_std': stats['quality_score']['std'],
                'entropy_mean': stats['entropy']['mean'],
                'entropy_std': stats['entropy']['std'],
                'high_quality_pct': stats['high_quality_count'] / stats['total'] * 100,
                'valid_entropy_pct': stats['valid_entropy_count'] / stats['total'] * 100
            })
        
        if not comparison_data:
            print("Dados insuficientes para comparação.")
            return
        
        print(f"{'Configuração':<50} {'Quality':<12} {'Entropy':<12} {'Alta Q%':<10} {'Valid E%':<10}")
        print("-" * 100)
        
        for item in comparison_data:
            print(f"{item['config_name']:<50} "
                  f"{item['quality_mean']:>6.2f}±{item['quality_std']:<4.2f} "
                  f"{item['entropy_mean']:>6.3f}±{item['entropy_std']:<4.3f} "
                  f"{item['high_quality_pct']:>7.1f}% "
                  f"{item['valid_entropy_pct']:>8.1f}%")
        
        self._plot_comparison(comparison_data, comparison_type)
    
    def _plot_comparison(self, comparison_data: List[Dict], comparison_type: str):
        if len(comparison_data) < 2:
            print("Dados insuficientes para gráficos (mínimo 2 configurações).")
            return
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comparação de Configurações - {comparison_type.upper()}', 
                     fontsize=16, fontweight='bold')
        
        config_names = [d['config_name'] for d in comparison_data]
        quality_means = [d['quality_mean'] for d in comparison_data]
        quality_stds = [d['quality_std'] for d in comparison_data]
        entropy_means = [d['entropy_mean'] for d in comparison_data]
        entropy_stds = [d['entropy_std'] for d in comparison_data]
        high_quality_pcts = [d['high_quality_pct'] for d in comparison_data]
        valid_entropy_pcts = [d['valid_entropy_pct'] for d in comparison_data]
        
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(config_names)), quality_means, 
                        yerr=quality_stds, capsize=5, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Configuração')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Quality Score Médio (com desvio padrão)')
        ax1.set_xticks(range(len(config_names)))
        ax1.set_xticklabels([name.split('_')[2] for name in config_names], rotation=45)
        ax1.axhline(y=90, color='r', linestyle='--', label='Threshold (90)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(config_names)), entropy_means,
                        yerr=entropy_stds, capsize=5, alpha=0.7, color='forestgreen')
        ax2.set_xlabel('Configuração')
        ax2.set_ylabel('Entropia Shannon (bits/byte)')
        ax2.set_title('Entropia Média (com desvio padrão)')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels([name.split('_')[2] for name in config_names], rotation=45)
        ax2.axhline(y=7.9, color='r', linestyle='--', label='Threshold (7.9)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(config_names)), high_quality_pcts,
                        alpha=0.7, color='coral')
        ax3.set_xlabel('Configuração')
        ax3.set_ylabel('Porcentagem (%)')
        ax3.set_title('Chaves de Alta Qualidade (≥90)')
        ax3.set_xticks(range(len(config_names)))
        ax3.set_xticklabels([name.split('_')[2] for name in config_names], rotation=45)
        ax3.set_ylim(0, 100)
        ax3.grid(alpha=0.3)

        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(config_names)), valid_entropy_pcts,
                        alpha=0.7, color='mediumpurple')
        ax4.set_xlabel('Configuração')
        ax4.set_ylabel('Porcentagem (%)')
        ax4.set_title('Chaves com Entropia Válida (≥7.9)')
        ax4.set_xticks(range(len(config_names)))
        ax4.set_xticklabels([name.split('_')[2] for name in config_names], rotation=45)
        ax4.set_ylim(0, 100)
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()

        output_path = self.output_dir / f'comparison_{comparison_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfico salvo em: {output_path}")
        
        plt.close()
    
    def generate_full_report(self):
        report_path = self.output_dir / "full_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Relatório Completo de Avaliação\n\n")
            f.write("## Configurações Avaliadas\n\n")
            
            for config_name, data in self.results.items():
                f.write(f"### {config_name}\n\n")
                f.write(f"**Configuração:**\n")
                for key, value in data['config'].items():
                    f.write(f"- {key}: {value}\n")
                
                if 'statistics' in data and 'quality_score' in data['statistics']:
                    stats = data['statistics']
                    f.write(f"\n**Estatísticas:**\n")
                    f.write(f"- Taxa de sucesso: {stats['success_rate']:.2f}%\n")
                    f.write(f"- Quality score médio: {stats['quality_score']['mean']:.2f} (±{stats['quality_score']['std']:.2f})\n")
                    f.write(f"- Entropia média: {stats['entropy']['mean']:.4f} (±{stats['entropy']['std']:.4f})\n")
                    f.write(f"- Chaves de alta qualidade: {stats['high_quality_count']}/{stats['total']} ({stats['high_quality_count']/stats['total']*100:.1f}%)\n")
                    f.write(f"- Chaves com entropia válida: {stats['valid_entropy_count']}/{stats['total']} ({stats['valid_entropy_count']/stats['total']*100:.1f}%)\n")
                
                f.write("\n---\n\n")


if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    
    print("\n1. Comparando estratégias de prompt...")

    evaluator.compare_strategies(
        model="gemma3:latest",
        key_size_bits=256,
        temperature_preset='high_entropy',
        num_tests=30 
    )
    
    evaluator.generate_comparison_report('strategies')
    
    print("\n2. Comparando configurações de temperatura...")

    evaluator.compare_temperatures(
        model="gemma3:latest",
        strategy='few-shot',
        key_size_bits=256,
        num_tests=30
    )
    
    evaluator.generate_comparison_report('temperatures')
    evaluator.generate_full_report()
    