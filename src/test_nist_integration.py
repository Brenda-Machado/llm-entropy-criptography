"""
Integração dos testes NIST SP 800-22 com o LLM Client
Script para testar chaves geradas pela IA
"""

import json
import time
from typing import Dict, List, Any
from llm_client import LLMClient
from nist_tests import NISTTests
from drand_client import get_entropy_seed
import numpy as np


def ensure_json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj


class NISTValidator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.validation_history = []
    
    def validate_key(self, key_hex: str, seed: str = None) -> Dict:
        """
        Valida uma chave usando todos os testes NIST
        """
        print(f"\n{'='*80}")
        print(f"Validando chave com NIST SP 800-22...")
        print(f"{'='*80}")
        print(f"Chave: {key_hex[:32]}...")
        
        # Executa testes NIST
        nist_results = NISTTests.run_all_tests_from_hex(key_hex)
        
        # Adiciona ao histórico
        validation_record = {
            'timestamp': time.time(),
            'key_hex': key_hex,
            'seed': seed,
            'nist_results': nist_results
        }
        self.validation_history.append(validation_record)
        
        return nist_results
    
    def generate_and_validate(self, seed: str = None, use_drand: bool = True) -> Dict:
        """
        Gera uma chave e valida com NIST
        """
        # Gera seed se necessário
        if use_drand and not seed:
            seed = get_entropy_seed()
        
        # Gera chave
        print(f"\nGerando chave com LLM...")
        result = self.llm_client.generate_key(seed, store_result=True)
        
        if not result['success']:
            return {
                'success': False,
                'error': result['error'],
                'generation_result': result
            }
        
        key_hex = result['key_hex']
        
        # Valida com NIST
        nist_results = self.validate_key(key_hex, seed)
        
        # Combina resultados
        combined_results = {
            'success': True,
            'key_hex': key_hex,
            'seed': seed,
            'generation_metrics': result['metrics'],
            'generation_info': result['generation_info'],
            'nist_validation': nist_results,
            'overall_quality': self._assess_overall_quality(result, nist_results)
        }
        
        return combined_results
    
    def _assess_overall_quality(self, gen_result: Dict, nist_results: Dict) -> Dict:
        """
        Avalia qualidade geral combinando métricas de geração e NIST
        """
        gen_score = gen_result['metrics']['quality_score']
        nist_pass_rate = nist_results['summary']['pass_rate']
        
        # Score combinado
        combined_score = (gen_score * 0.3 + nist_pass_rate * 0.7)
        
        # Classificação
        if combined_score >= 95 and nist_pass_rate >= 95:
            grade = 'EXCELLENT'
            recommendation = 'Altamente recomendado para uso criptográfico'
        elif combined_score >= 85 and nist_pass_rate >= 85:
            grade = 'GOOD'
            recommendation = 'Adequado para uso criptográfico'
        elif combined_score >= 70 and nist_pass_rate >= 70:
            grade = 'MODERATE'
            recommendation = 'Uso criptográfico não recomendado'
        else:
            grade = 'POOR'
            recommendation = 'NÃO adequado para uso criptográfico'
        
        return {
            'combined_score': combined_score,
            'generation_score': gen_score,
            'nist_pass_rate': nist_pass_rate,
            'grade': grade,
            'recommendation': recommendation,
            'cryptographic_grade': nist_pass_rate >= 95
        }
    
    def batch_validate(self, num_keys: int = 10, progress_callback=None) -> Dict:
        """
        Gera e valida múltiplas chaves
        """
        results = []
        stats = {
            'total': num_keys,
            'successful_generations': 0,
            'nist_excellent': 0,  # >= 95%
            'nist_good': 0,       # >= 85%
            'nist_moderate': 0,   # >= 70%
            'nist_poor': 0,       # < 70%
            'cryptographic_grade': 0,
            'generation_scores': [],
            'nist_pass_rates': [],
            'combined_scores': []
        }
        
        print(f"\n{'='*80}")
        print(f"Batch Validation: Gerando e validando {num_keys} chaves")
        print(f"{'='*80}\n")
        
        for i in range(num_keys):
            print(f"\n[{i+1}/{num_keys}] Gerando chave...")
            
            result = self.generate_and_validate()
            results.append(result)
            
            if result['success']:
                stats['successful_generations'] += 1
                
                # Coleta métricas
                quality = result['overall_quality']
                stats['generation_scores'].append(quality['generation_score'])
                stats['nist_pass_rates'].append(quality['nist_pass_rate'])
                stats['combined_scores'].append(quality['combined_score'])
                
                # Classifica
                if quality['nist_pass_rate'] >= 95:
                    stats['nist_excellent'] += 1
                elif quality['nist_pass_rate'] >= 85:
                    stats['nist_good'] += 1
                elif quality['nist_pass_rate'] >= 70:
                    stats['nist_moderate'] += 1
                else:
                    stats['nist_poor'] += 1
                
                if quality['cryptographic_grade']:
                    stats['cryptographic_grade'] += 1
            
            if progress_callback:
                progress_callback(i + 1, num_keys, result)
            
            # Pequeno delay para não sobrecarregar
            time.sleep(0.5)
        
        # Calcula estatísticas agregadas
        if stats['generation_scores']:
            stats['avg_generation_score'] = np.mean(stats['generation_scores'])
            stats['avg_nist_pass_rate'] = np.mean(stats['nist_pass_rates'])
            stats['avg_combined_score'] = np.mean(stats['combined_scores'])
            stats['std_generation_score'] = np.std(stats['generation_scores'])
            stats['std_nist_pass_rate'] = np.std(stats['nist_pass_rates'])
            stats['cryptographic_grade_rate'] = (stats['cryptographic_grade'] / stats['successful_generations']) * 100
        
        return {
            'results': results,
            'statistics': stats
        }
    
    def save_results(self, filepath: str = "nist_validation_results.json"):
        """
        Salva histórico de validações
        """
        with open(filepath, 'w') as f:
            json.dump(self.validation_history, f, indent=2)
        print(f"\nResultados salvos em: {filepath}")
    
    def generate_report(self, batch_results: Dict, filepath: str = "nist_report.txt"):
        """
        Gera relatório detalhado das validações
        """
        stats = batch_results['statistics']
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE VALIDAÇÃO NIST SP 800-22")
        report.append("=" * 80)
        report.append("")
        
        # Resumo geral
        report.append("RESUMO GERAL")
        report.append("-" * 80)
        report.append(f"Total de chaves testadas: {stats['total']}")
        report.append(f"Gerações bem-sucedidas: {stats['successful_generations']}")
        report.append(f"Taxa de sucesso na geração: {(stats['successful_generations']/stats['total']*100):.2f}%")
        report.append("")
        
        # Classificação NIST
        report.append("CLASSIFICAÇÃO NIST")
        report.append("-" * 80)
        report.append(f"Excelente (≥95%): {stats['nist_excellent']} chaves")
        report.append(f"Bom (≥85%):       {stats['nist_good']} chaves")
        report.append(f"Moderado (≥70%):  {stats['nist_moderate']} chaves")
        report.append(f"Ruim (<70%):      {stats['nist_poor']} chaves")
        report.append("")
        
        # Grau criptográfico
        report.append("ADEQUAÇÃO CRIPTOGRÁFICA")
        report.append("-" * 80)
        report.append(f"Chaves com grau criptográfico: {stats['cryptographic_grade']}")
        if stats['successful_generations'] > 0:
            report.append(f"Taxa de adequação: {stats['cryptographic_grade_rate']:.2f}%")
        report.append("")
        
        # Estatísticas
        if 'avg_generation_score' in stats:
            report.append("ESTATÍSTICAS")
            report.append("-" * 80)
            report.append(f"Score de Geração:")
            report.append(f"  Média: {stats['avg_generation_score']:.2f}")
            report.append(f"  Desvio padrão: {stats['std_generation_score']:.2f}")
            report.append("")
            report.append(f"Taxa de Aprovação NIST:")
            report.append(f"  Média: {stats['avg_nist_pass_rate']:.2f}%")
            report.append(f"  Desvio padrão: {stats['std_nist_pass_rate']:.2f}%")
            report.append("")
            report.append(f"Score Combinado:")
            report.append(f"  Média: {stats['avg_combined_score']:.2f}")
            report.append("")
        
        # Análise individual dos testes mais críticos
        report.append("ANÁLISE DE TESTES CRÍTICOS")
        report.append("-" * 80)
        
        critical_tests = [
            'Frequency (Monobit)',
            'Block Frequency',
            'Runs',
            'Binary Matrix Rank',
            'Discrete Fourier Transform'
        ]
        
        test_stats = {test: {'passed': 0, 'failed': 0} for test in critical_tests}
        
        for result in batch_results['results']:
            if result['success']:
                for test in result['nist_validation']['tests']:
                    test_name = test.get('test_name')
                    if test_name in critical_tests:
                        if test.get('passed', False):
                            test_stats[test_name]['passed'] += 1
                        else:
                            test_stats[test_name]['failed'] += 1
        
        for test_name, counts in test_stats.items():
            total = counts['passed'] + counts['failed']
            if total > 0:
                pass_rate = (counts['passed'] / total) * 100
                report.append(f"{test_name}:")
                report.append(f"  Aprovações: {counts['passed']}/{total} ({pass_rate:.1f}%)")
        
        report.append("")
        report.append("=" * 80)
        
        # Salva relatório
        report_text = "\n".join(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nRelatório salvo em: {filepath}")
        return report_text


def test_single_key():
    """
    Teste de uma única chave
    """
    print("\n" + "="*80)
    print("TESTE INDIVIDUAL")
    print("="*80)
    
    # Inicializa cliente LLM
    client = LLMClient(
        key_size_bits=256,
        strategy='few-shot',
        temperature_preset='high_entropy'
    )
    
    # Cria validador
    validator = NISTValidator(client)
    
    # Gera e valida
    result = validator.generate_and_validate()
    
    if result['success']:
        # Exibe resultados
        print("\n" + NISTTests.format_results(result['nist_validation'], detailed=True))
        
        print("\n" + "="*80)
        print("AVALIAÇÃO GERAL")
        print("="*80)
        quality = result['overall_quality']
        print(f"Score Combinado: {quality['combined_score']:.2f}")
        print(f"Score de Geração: {quality['generation_score']:.2f}")
        print(f"Taxa de Aprovação NIST: {quality['nist_pass_rate']:.2f}%")
        print(f"Classificação: {quality['grade']}")
        print(f"Recomendação: {quality['recommendation']}")
        print(f"Grau Criptográfico: {'✓ SIM' if quality['cryptographic_grade'] else '✗ NÃO'}")
    else:
        print(f"\nERRO: {result['error']}")


def test_batch():
    """
    Teste em lote
    """
    print("\n" + "="*80)
    print("TESTE EM LOTE")
    print("="*80)
    
    # Inicializa cliente LLM
    client = LLMClient(
        key_size_bits=256,
        strategy='few-shot',
        temperature_preset='high_entropy'
    )
    
    # Cria validador
    validator = NISTValidator(client)
    
    # Executa batch
    batch_results = validator.batch_validate(num_keys=10)
    
    # Gera e exibe relatório
    report = validator.generate_report(batch_results)
    print("\n" + report)
    
    # Salva resultados
    validator.save_results()


def compare_strategies():
    """
    Compara diferentes estratégias de prompt
    """
    print("\n" + "="*80)
    print("COMPARAÇÃO DE ESTRATÉGIAS")
    print("="*80)
    
    strategies = ['zero-shot', 'few-shot', 'cot']
    results_by_strategy = {}
    
    for strategy in strategies:
        print(f"\n\nTestando estratégia: {strategy}")
        print("-" * 80)
        
        client = LLMClient(
            key_size_bits=256,
            strategy=strategy,
            temperature_preset='high_entropy'
        )
        
        validator = NISTValidator(client)
        batch_results = validator.batch_validate(num_keys=5)
        
        results_by_strategy[strategy] = batch_results['statistics']
    
    # Compara resultados
    print("\n\n" + "="*80)
    print("COMPARAÇÃO FINAL")
    print("="*80)
    
    for strategy, stats in results_by_strategy.items():
        print(f"\n{strategy.upper()}:")
        if 'avg_combined_score' in stats:
            print(f"  Score Combinado: {stats['avg_combined_score']:.2f}")
            print(f"  Taxa NIST: {stats['avg_nist_pass_rate']:.2f}%")
            print(f"  Grau Criptográfico: {stats['cryptographic_grade']}/{stats['successful_generations']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validação NIST para chaves geradas por IA")
    parser.add_argument('--mode', choices=['single', 'batch', 'compare'], 
                       default='single', help='Modo de teste')
    parser.add_argument('--num-keys', type=int, default=10, 
                       help='Número de chaves para teste em lote')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_single_key()
    elif args.mode == 'batch':
        test_batch()
    elif args.mode == 'compare':
        compare_strategies()