"""
NIST SP 800-22 Statistical Test Suite
Bateria completa de testes de aleatoriedade para validação criptográfica

Referência: NIST Special Publication 800-22rev1a
"""

import numpy as np
from scipy import special as spc
from scipy.fft import fft
from typing import Dict, List, Tuple
import math


class NISTTests:
    """
    Implementação completa dos 15 testes NIST SP 800-22
    Nível de significância (alpha) = 0.01
    P-value >= 0.01 indica que a sequência passou no teste
    """
    
    ALPHA = 0.01  # Nível de significância
    
    @staticmethod
    def hex_to_bits(hex_string: str) -> np.ndarray:
        """Converte string hexadecimal para array de bits"""
        byte_array = bytes.fromhex(hex_string)
        bits = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
        return bits
    
    @staticmethod
    def bits_to_pm1(bits: np.ndarray) -> np.ndarray:
        """Converte bits (0,1) para (+1,-1)"""
        return 2 * bits - 1
    
    # ==================== TESTE 1: FREQUENCY (MONOBIT) ====================
    @staticmethod
    def frequency_test(bits: np.ndarray) -> Dict:
        """
        Teste 1: Frequency (Monobit) Test
        Verifica se o número de 0s e 1s é aproximadamente igual
        """
        n = len(bits)
        s = np.sum(NISTTests.bits_to_pm1(bits))
        s_obs = abs(s) / np.sqrt(n)
        p_value = spc.erfc(s_obs / np.sqrt(2))
        
        return {
            'test_name': 'Frequency (Monobit)',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(s_obs)
        }
    
    # ==================== TESTE 2: BLOCK FREQUENCY ====================
    @staticmethod
    def block_frequency_test(bits: np.ndarray, block_size: int = 128) -> Dict:
        """
        Teste 2: Frequency Test within a Block
        Verifica a proporção de 1s dentro de blocos
        """
        n = len(bits)
        N = n // block_size
        
        if N < 1:
            return {
                'test_name': 'Block Frequency',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        proportions = []
        for i in range(N):
            block = bits[i * block_size:(i + 1) * block_size]
            pi = np.sum(block) / block_size
            proportions.append(pi)
        
        chi_squared = 4 * block_size * np.sum((np.array(proportions) - 0.5) ** 2)
        p_value = spc.gammaincc(N / 2, chi_squared / 2)
        
        return {
            'test_name': 'Block Frequency',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'block_size': block_size,
            'num_blocks': N
        }
    
    # ==================== TESTE 3: RUNS ====================
    @staticmethod
    def runs_test(bits: np.ndarray) -> Dict:
        """
        Teste 3: Runs Test
        Verifica a alternância entre 0s e 1s
        """
        n = len(bits)
        pi = np.sum(bits) / n
        
        # Pré-requisito
        if abs(pi - 0.5) >= 2 / np.sqrt(n):
            return {
                'test_name': 'Runs',
                'p_value': 0.0,
                'passed': False,
                'error': 'Pre-test failed (pi far from 0.5)'
            }
        
        # Conta runs (sequências)
        v_obs = 1 + np.sum(bits[:-1] != bits[1:])
        
        numerator = abs(v_obs - 2 * n * pi * (1 - pi))
        denominator = 2 * np.sqrt(2 * n) * pi * (1 - pi)
        
        p_value = spc.erfc(numerator / denominator)
        
        return {
            'test_name': 'Runs',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': int(v_obs),
            'pi': float(pi)
        }
    
    # ==================== TESTE 4: LONGEST RUN OF ONES ====================
    @staticmethod
    def longest_run_test(bits: np.ndarray) -> Dict:
        """
        Teste 4: Test for the Longest Run of Ones in a Block
        """
        n = len(bits)
        
        if n < 128:
            return {
                'test_name': 'Longest Run of Ones',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data (n < 128)'
            }
        
        # Parâmetros baseados no tamanho
        if n < 6272:
            K, M, N = 3, 8, 16
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
            v_values = [1, 2, 3, 4]
        elif n < 750000:
            K, M, N = 5, 128, 49
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
            v_values = [4, 5, 6, 7, 8, 9]
        else:
            K, M, N = 6, 10000, 75
            pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
            v_values = [10, 11, 12, 13, 14, 15, 16]
        
        # Trunca para o tamanho adequado
        bits = bits[:N * M]
        
        # Conta o longest run em cada bloco
        v = np.zeros(K + 1)
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            run_lengths = []
            current_run = 0
            
            for bit in block:
                if bit == 1:
                    current_run += 1
                else:
                    if current_run > 0:
                        run_lengths.append(current_run)
                    current_run = 0
            if current_run > 0:
                run_lengths.append(current_run)
            
            longest = max(run_lengths) if run_lengths else 0
            
            # Classifica o longest run
            if longest <= v_values[0]:
                v[0] += 1
            elif longest >= v_values[-1]:
                v[K] += 1
            else:
                for j in range(len(v_values) - 1):
                    if v_values[j] < longest <= v_values[j + 1]:
                        v[j + 1] += 1
                        break
        
        # Calcula chi-squared
        chi_squared = np.sum((v - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p_value = spc.gammaincc(K / 2, chi_squared / 2)
        
        return {
            'test_name': 'Longest Run of Ones',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'M': M,
            'N': N
        }
    
    # ==================== TESTE 5: BINARY MATRIX RANK ====================
    @staticmethod
    def matrix_rank_test(bits: np.ndarray) -> Dict:
        """
        Teste 5: Binary Matrix Rank Test
        Verifica a independência linear de substrings
        """
        M = Q = 32  # Dimensão das matrizes
        n = len(bits)
        N = n // (M * Q)
        
        if N == 0:
            return {
                'test_name': 'Binary Matrix Rank',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        def binary_rank(matrix):
            """Calcula o rank de uma matriz binária"""
            m = matrix.copy()
            rows, cols = m.shape
            rank = 0
            
            for col in range(min(rows, cols)):
                # Encontra pivot
                pivot_row = None
                for row in range(rank, rows):
                    if m[row, col] == 1:
                        pivot_row = row
                        break
                
                if pivot_row is None:
                    continue
                
                # Troca linhas
                if pivot_row != rank:
                    m[[rank, pivot_row]] = m[[pivot_row, rank]]
                
                # Eliminação
                for row in range(rows):
                    if row != rank and m[row, col] == 1:
                        m[row] = (m[row] + m[rank]) % 2
                
                rank += 1
            
            return rank
        
        # Conta ranks
        F_M = 0  # Full rank
        F_M1 = 0  # Rank M-1
        
        for k in range(N):
            block = bits[k * M * Q:(k + 1) * M * Q]
            matrix = block.reshape(M, Q)
            rank = binary_rank(matrix)
            
            if rank == M:
                F_M += 1
            elif rank == M - 1:
                F_M1 += 1
        
        # Chi-squared
        chi_squared = ((F_M - 0.2888 * N) ** 2 / (0.2888 * N) +
                       (F_M1 - 0.5776 * N) ** 2 / (0.5776 * N) +
                       ((N - F_M - F_M1) - 0.1336 * N) ** 2 / (0.1336 * N))
        
        p_value = np.exp(-chi_squared / 2)
        
        return {
            'test_name': 'Binary Matrix Rank',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'full_rank': int(F_M),
            'rank_minus_1': int(F_M1)
        }
    
    # ==================== TESTE 6: DISCRETE FOURIER TRANSFORM ====================
    @staticmethod
    def dft_test(bits: np.ndarray) -> Dict:
        """
        Teste 6: Discrete Fourier Transform (Spectral) Test
        Detecta componentes periódicas
        """
        n = len(bits)
        X = NISTTests.bits_to_pm1(bits)
        
        # FFT
        S = np.abs(fft(X))[:n // 2]
        
        # Threshold
        T = np.sqrt(np.log(1 / 0.05) * n)
        
        # Conta picos acima do threshold
        N0 = 0.95 * n / 2
        N1 = np.sum(S < T)
        
        # Estatística
        d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
        p_value = spc.erfc(abs(d) / np.sqrt(2))
        
        return {
            'test_name': 'Discrete Fourier Transform',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(d),
            'peaks_below_threshold': int(N1)
        }
    
    # ==================== TESTE 7: NON-OVERLAPPING TEMPLATE MATCHING ====================
    @staticmethod
    def non_overlapping_template_test(bits: np.ndarray, template: np.ndarray = None) -> Dict:
        """
        Teste 7: Non-overlapping Template Matching Test
        """
        if template is None:
            template = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])  # Template padrão
        
        m = len(template)
        n = len(bits)
        M = 968  # Tamanho do bloco
        N = n // M
        
        if N == 0:
            return {
                'test_name': 'Non-overlapping Template Matching',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        # Calcula mu e sigma²
        mu = (M - m + 1) / (2 ** m)
        sigma_squared = M * ((1 / (2 ** m)) - ((2 * m - 1) / (2 ** (2 * m))))
        
        # Conta matches em cada bloco
        W = []
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            matches = 0
            j = 0
            
            while j < M - m + 1:
                if np.array_equal(block[j:j + m], template):
                    matches += 1
                    j += m  # Non-overlapping
                else:
                    j += 1
            
            W.append(matches)
        
        # Chi-squared
        chi_squared = np.sum((np.array(W) - mu) ** 2) / sigma_squared
        p_value = spc.gammaincc(N / 2, chi_squared / 2)
        
        return {
            'test_name': 'Non-overlapping Template Matching',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'template_size': m,
            'num_blocks': N
        }
    
    # ==================== TESTE 8: OVERLAPPING TEMPLATE MATCHING ====================
    @staticmethod
    def overlapping_template_test(bits: np.ndarray, template: np.ndarray = None) -> Dict:
        """
        Teste 8: Overlapping Template Matching Test
        """
        if template is None:
            template = np.ones(9, dtype=int)  # 9 ones
        
        m = len(template)
        n = len(bits)
        M = 1032  # Tamanho do bloco
        N = n // M
        
        if N == 0:
            return {
                'test_name': 'Overlapping Template Matching',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        # Parâmetros
        K = 5
        lambda_val = (M - m + 1) / (2 ** m)
        eta = lambda_val / 2.0
        
        # Probabilidades teóricas
        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865]
        
        # Conta matches em cada bloco
        v = np.zeros(K + 1)
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            matches = 0
            
            for j in range(M - m + 1):
                if np.array_equal(block[j:j + m], template):
                    matches += 1
            
            if matches <= K:
                v[matches] += 1
            else:
                v[K] += 1
        
        # Chi-squared
        chi_squared = np.sum((v - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p_value = spc.gammaincc(K / 2, chi_squared / 2)
        
        return {
            'test_name': 'Overlapping Template Matching',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'template_size': m
        }
    
    # ==================== TESTE 9: UNIVERSAL STATISTICAL ====================
    @staticmethod
    def universal_test(bits: np.ndarray) -> Dict:
        """
        Teste 9: Maurer's "Universal Statistical" Test
        """
        n = len(bits)
        
        # Parâmetros baseados em n
        if n >= 387840:
            L, Q = 7, 1280
        elif n >= 904960:
            L, Q = 8, 1536
        elif n >= 2068480:
            L, Q = 9, 1792
        elif n >= 4654080:
            L, Q = 10, 2048
        elif n >= 10342400:
            L, Q = 11, 2304
        elif n >= 22753280:
            L, Q = 12, 2560
        elif n >= 49643520:
            L, Q = 13, 2816
        elif n >= 107560960:
            L, Q = 14, 3072
        elif n >= 231669760:
            L, Q = 15, 3328
        else:
            L, Q = 6, 640
        
        K = n // L - Q
        
        if K <= 0 or Q <= 0:
            return {
                'test_name': 'Universal Statistical',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        # Valores esperados (tabela NIST)
        expected_values = {
            6: (5.2177052, 2.954),
            7: (6.1962507, 3.125),
            8: (7.1836656, 3.238),
            9: (8.1764248, 3.311),
            10: (9.1723243, 3.356),
            11: (10.170032, 3.384),
            12: (11.168765, 3.401),
            13: (12.168070, 3.410),
            14: (13.167693, 3.416),
            15: (14.167488, 3.419),
            16: (15.167379, 3.421)
        }
        
        expected_value, variance = expected_values.get(L, (0, 0))
        
        # Tabela de inicialização
        T = {}
        for i in range(Q):
            block = tuple(bits[i * L:(i + 1) * L])
            T[block] = i + 1
        
        # Cálculo do teste
        sum_log = 0.0
        for i in range(Q, Q + K):
            block = tuple(bits[i * L:(i + 1) * L])
            if block in T:
                sum_log += np.log2(i + 1 - T[block])
            T[block] = i + 1
        
        fn = sum_log / K
        
        # Estatística
        c = 0.7 - 0.8 / L + (4 + 32 / L) * (K ** (-3 / L)) / 15
        sigma = c * np.sqrt(variance / K)
        
        p_value = spc.erfc(abs((fn - expected_value) / (np.sqrt(2) * sigma)))
        
        return {
            'test_name': 'Universal Statistical',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(fn),
            'L': L,
            'Q': Q,
            'K': K
        }
    
    # ==================== TESTE 10: LINEAR COMPLEXITY ====================
    @staticmethod
    def linear_complexity_test(bits: np.ndarray, M: int = 500) -> Dict:
        """
        Teste 10: Linear Complexity Test
        Usa algoritmo de Berlekamp-Massey
        """
        n = len(bits)
        N = n // M
        
        if N < 1:
            return {
                'test_name': 'Linear Complexity',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        def berlekamp_massey(block):
            """Algoritmo de Berlekamp-Massey para calcular complexidade linear"""
            n = len(block)
            c = np.zeros(n, dtype=int)
            b = np.zeros(n, dtype=int)
            c[0] = b[0] = 1
            l, m, d = 0, -1, 1
            
            for i in range(n):
                d = block[i]
                for j in range(1, l + 1):
                    d ^= c[j] & block[i - j]
                
                if d == 1:
                    t = c.copy()
                    p = np.zeros(n, dtype=int)
                    for j in range(i - m, n):
                        p[j] = b[j - (i - m)]
                    c = (c + p) % 2
                    
                    if l <= i / 2:
                        l = i + 1 - l
                        m = i
                        b = t
            
            return l
        
        # Calcula complexidade para cada bloco
        complexities = []
        for i in range(N):
            block = bits[i * M:(i + 1) * M]
            L = berlekamp_massey(block)
            complexities.append(L)
        
        # Calcula T_i (desvio normalizado)
        mu = M / 2.0 + (9.0 + (-1) ** (M + 1)) / 36.0 - 1.0 / (2 ** M) * (M / 3.0 + 2.0 / 9.0)
        
        T = []
        for L in complexities:
            T_i = (-1) ** M * (L - mu) + 2.0 / 9.0
            T.append(T_i)
        
        # Conta frequências
        v = [0, 0, 0, 0, 0, 0, 0]  # [-inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, inf]
        
        for t in T:
            if t <= -2.5:
                v[0] += 1
            elif t <= -1.5:
                v[1] += 1
            elif t <= -0.5:
                v[2] += 1
            elif t <= 0.5:
                v[3] += 1
            elif t <= 1.5:
                v[4] += 1
            elif t <= 2.5:
                v[5] += 1
            else:
                v[6] += 1
        
        # Probabilidades teóricas
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        
        # Chi-squared
        chi_squared = np.sum((np.array(v) - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p_value = spc.gammaincc(3, chi_squared / 2)
        
        return {
            'test_name': 'Linear Complexity',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'M': M,
            'N': N,
            'mean_complexity': float(np.mean(complexities))
        }
    
    # ==================== TESTE 11: SERIAL ====================
    @staticmethod
    def serial_test(bits: np.ndarray, m: int = 16) -> Dict:
        """
        Teste 11: Serial Test
        """
        n = len(bits)
        
        if n < m:
            return {
                'test_name': 'Serial',
                'p_value': 0.0,
                'p_value2': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        def psi_squared(m_val, bits_val):
            """Calcula ψ²_m"""
            counts = {}
            for i in range(len(bits_val)):
                pattern = tuple(bits_val[i:i + m_val])
                if len(pattern) == m_val:
                    counts[pattern] = counts.get(pattern, 0) + 1
            
            psi_sq = 0
            for count in counts.values():
                psi_sq += count ** 2
            
            psi_sq = (2 ** m_val / len(bits_val)) * psi_sq - len(bits_val)
            return psi_sq
        
        # Cria sequência circular
        circular_bits = np.concatenate([bits, bits[:m - 1]])
        
        # Calcula ψ² para m, m-1, m-2
        psi_m = psi_squared(m, circular_bits)
        psi_m1 = psi_squared(m - 1, circular_bits)
        psi_m2 = psi_squared(m - 2, circular_bits)
        
        # Estatísticas
        delta1 = psi_m - psi_m1
        delta2 = psi_m - 2 * psi_m1 + psi_m2
        
        p_value1 = spc.gammaincc(2 ** (m - 2), delta1 / 2)
        p_value2 = spc.gammaincc(2 ** (m - 3), delta2 / 2)
        
        return {
            'test_name': 'Serial',
            'p_value': float(p_value1),
            'p_value2': float(p_value2),
            'passed': p_value1 >= NISTTests.ALPHA and p_value2 >= NISTTests.ALPHA,
            'statistic': float(delta1),
            'statistic2': float(delta2),
            'm': m
        }
    
    # ==================== TESTE 12: APPROXIMATE ENTROPY ====================
    @staticmethod
    def approximate_entropy_test(bits: np.ndarray, m: int = 10) -> Dict:
        """
        Teste 12: Approximate Entropy Test
        """
        n = len(bits)
        
        if n < m:
            return {
                'test_name': 'Approximate Entropy',
                'p_value': 0.0,
                'passed': False,
                'error': 'Insufficient data'
            }
        
        def phi(m_val):
            """Calcula Φ(m)"""
            counts = {}
            for i in range(n):
                pattern = tuple(bits[i:i + m_val])
                if len(pattern) == m_val:
                    counts[pattern] = counts.get(pattern, 0) + 1
            
            phi_val = 0.0
            for count in counts.values():
                if count > 0:
                    phi_val += count * np.log(count / n)
            
            return phi_val / n
        
        phi_m = phi(m)
        phi_m1 = phi(m + 1)
        
        appen = phi_m - phi_m1
        chi_squared = 2 * n * (np.log(2) - appen)
        p_value = spc.gammaincc(2 ** (m - 1), chi_squared / 2)
        
        return {
            'test_name': 'Approximate Entropy',
            'p_value': float(p_value),
            'passed': p_value >= NISTTests.ALPHA,
            'statistic': float(chi_squared),
            'appen': float(appen),
            'm': m
        }
    
    # ==================== TESTE 13: CUMULATIVE SUMS ====================
    @staticmethod
    def cumulative_sums_test(bits: np.ndarray) -> Dict:
        """
        Teste 13: Cumulative Sums (Cusum) Test
        Forward e Backward
        """
        n = len(bits)
        X = NISTTests.bits_to_pm1(bits)
        
        # Forward
        S_forward = np.cumsum(X)
        z_forward = np.max(np.abs(S_forward))
        
        # Backward
        S_backward = np.cumsum(X[::-1])
        z_backward = np.max(np.abs(S_backward))
        
        def compute_p_value(z):
            """Calcula p-value para cumsum"""
            sum_val = 0.0
            start = int((-n / z + 1) / 4)
            end = int((n / z - 1) / 4)
            
            for k in range(start, end + 1):
                term1 = spc.ndtr((4 * k + 1) * z / np.sqrt(n))
                term2 = spc.ndtr((4 * k - 1) * z / np.sqrt(n))
                sum_val += term1 - term2
            
            start2 = int((-n / z - 3) / 4)
            end2 = int((n / z - 1) / 4)
            
            for k in range(start2, end2 + 1):
                term1 = spc.ndtr((4 * k + 3) * z / np.sqrt(n))
                term2 = spc.ndtr((4 * k + 1) * z / np.sqrt(n))
                sum_val += term1 - term2
            
            return 1.0 - sum_val
        
        p_value_forward = compute_p_value(z_forward)
        p_value_backward = compute_p_value(z_backward)
        
        return {
            'test_name': 'Cumulative Sums',
            'p_value_forward': float(p_value_forward),
            'p_value_backward': float(p_value_backward),
            'passed': p_value_forward >= NISTTests.ALPHA and p_value_backward >= NISTTests.ALPHA,
            'z_forward': float(z_forward),
            'z_backward': float(z_backward)
        }
    
    # ==================== TESTE 14: RANDOM EXCURSIONS ====================
    @staticmethod
    def random_excursions_test(bits: np.ndarray) -> Dict:
        """
        Teste 14: Random Excursions Test
        """
        n = len(bits)
        X = NISTTests.bits_to_pm1(bits)
        
        # Calcula cumulative sum
        S = np.zeros(n + 1)
        S[1:] = np.cumsum(X)
        
        # Conta ciclos (retornos ao zero)
        cycles = []
        cycle_start = 0
        
        for i in range(1, n + 1):
            if S[i] == 0 and i > cycle_start:
                cycles.append(S[cycle_start:i + 1])
                cycle_start = i
        
        J = len(cycles)
        
        if J < 500:
            return {
                'test_name': 'Random Excursions',
                'p_value': 0.0,
                'passed': False,
                'error': f'Insufficient cycles: {J} < 500'
            }
        
        # Estados para teste
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        
        results = {}
        all_passed = True
        
        for x in states:
            # Conta visitas ao estado x em cada ciclo
            v = np.zeros(6)  # [0, 1, 2, 3, 4, 5+]
            
            for cycle in cycles:
                count = np.sum(cycle == x)
                if count >= 5:
                    v[5] += 1
                else:
                    v[count] += 1
            
            # Probabilidades teóricas
            pi = NISTTests._compute_pi(x)
            
            # Chi-squared
            chi_squared = np.sum((v - J * np.array(pi)) ** 2 / (J * np.array(pi)))
            p_value = spc.gammaincc(2.5, chi_squared / 2)
            
            results[f'state_{x}'] = {
                'p_value': float(p_value),
                'passed': p_value >= NISTTests.ALPHA,
                'chi_squared': float(chi_squared)
            }
            
            if p_value < NISTTests.ALPHA:
                all_passed = False
        
        return {
            'test_name': 'Random Excursions',
            'cycles': J,
            'states': results,
            'passed': all_passed
        }
    
    @staticmethod
    def _compute_pi(x: int) -> np.ndarray:
        """Calcula probabilidades teóricas para Random Excursions"""
        pi = np.zeros(6)
        
        # Fórmulas da especificação NIST
        if x == -4 or x == 4:
            pi[0] = 0.5
            pi[1] = 0.25
            pi[2] = 0.125
            pi[3] = 0.0625
            pi[4] = 0.0312
            pi[5] = 0.0312
        elif x == -3 or x == 3:
            pi[0] = 0.75
            pi[1] = 0.0625
            pi[2] = 0.0469
            pi[3] = 0.0352
            pi[4] = 0.0264
            pi[5] = 0.0791
        elif x == -2 or x == 2:
            pi[0] = 0.8333
            pi[1] = 0.0278
            pi[2] = 0.0231
            pi[3] = 0.0193
            pi[4] = 0.0161
            pi[5] = 0.0804
        elif x == -1 or x == 1:
            pi[0] = 0.5
            pi[1] = 0.25
            pi[2] = 0.125
            pi[3] = 0.0625
            pi[4] = 0.0312
            pi[5] = 0.0312
        
        return pi
    
    # ==================== TESTE 15: RANDOM EXCURSIONS VARIANT ====================
    @staticmethod
    def random_excursions_variant_test(bits: np.ndarray) -> Dict:
        """
        Teste 15: Random Excursions Variant Test
        """
        n = len(bits)
        X = NISTTests.bits_to_pm1(bits)
        
        # Calcula cumulative sum
        S = np.zeros(n + 1)
        S[1:] = np.cumsum(X)
        
        # Conta ciclos
        J = np.sum(S == 0) - 1
        
        if J < 500:
            return {
                'test_name': 'Random Excursions Variant',
                'p_value': 0.0,
                'passed': False,
                'error': f'Insufficient cycles: {J} < 500'
            }
        
        # Estados para teste
        states = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        results = {}
        all_passed = True
        
        for x in states:
            # Conta ocorrências do estado x
            count = np.sum(S == x)
            
            # P-value
            p_value = spc.erfc(abs(count - J) / np.sqrt(2 * J * (4 * abs(x) - 2)))
            
            results[f'state_{x}'] = {
                'p_value': float(p_value),
                'passed': p_value >= NISTTests.ALPHA,
                'count': int(count)
            }
            
            if p_value < NISTTests.ALPHA:
                all_passed = False
        
        return {
            'test_name': 'Random Excursions Variant',
            'cycles': int(J),
            'states': results,
            'passed': all_passed
        }
    
    # ==================== EXECUTAR TODOS OS TESTES ====================
    @staticmethod
    def run_all_tests(bits: np.ndarray) -> Dict:
        """
        Executa todos os 15 testes NIST SP 800-22
        """
        results = {
            'total_bits': len(bits),
            'tests': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0
            }
        }
        
        # Lista de testes
        tests = [
            ('frequency_test', {}),
            ('block_frequency_test', {}),
            ('runs_test', {}),
            ('longest_run_test', {}),
            ('matrix_rank_test', {}),
            ('dft_test', {}),
            ('non_overlapping_template_test', {}),
            ('overlapping_template_test', {}),
            ('universal_test', {}),
            ('linear_complexity_test', {}),
            ('serial_test', {}),
            ('approximate_entropy_test', {}),
            ('cumulative_sums_test', {}),
            ('random_excursions_test', {}),
            ('random_excursions_variant_test', {})
        ]
        
        for test_name, kwargs in tests:
            try:
                test_func = getattr(NISTTests, test_name)
                result = test_func(bits, **kwargs)
                results['tests'].append(result)
                
                # Contabiliza resultados
                if 'passed' in result:
                    results['summary']['total_tests'] += 1
                    if result['passed']:
                        results['summary']['passed'] += 1
                    else:
                        results['summary']['failed'] += 1
            
            except Exception as e:
                results['tests'].append({
                    'test_name': test_name,
                    'error': str(e),
                    'passed': False
                })
                results['summary']['total_tests'] += 1
                results['summary']['failed'] += 1
        
        # Calcula taxa de sucesso
        if results['summary']['total_tests'] > 0:
            results['summary']['pass_rate'] = (
                results['summary']['passed'] / results['summary']['total_tests'] * 100
            )
        
        return results
    
    @staticmethod
    def run_all_tests_from_hex(hex_string: str) -> Dict:
        """
        Executa todos os testes a partir de uma string hexadecimal
        """
        bits = NISTTests.hex_to_bits(hex_string)
        return NISTTests.run_all_tests(bits)
    
    @staticmethod
    def format_results(results: Dict, detailed: bool = True) -> str:
        """
        Formata os resultados dos testes para exibição
        """
        output = []
        output.append("=" * 80)
        output.append("NIST SP 800-22 STATISTICAL TEST SUITE RESULTS")
        output.append("=" * 80)
        output.append(f"Total bits tested: {results['total_bits']}")
        output.append(f"Significance level (α): {NISTTests.ALPHA}")
        output.append("")
        
        # Resultados individuais
        for i, test in enumerate(results['tests'], 1):
            output.append(f"{i}. {test.get('test_name', 'Unknown Test')}")
            
            if 'error' in test:
                output.append(f"   ERROR: {test['error']}")
            else:
                # P-value principal
                if 'p_value' in test:
                    p_val = test['p_value']
                    status = "PASS ✓" if test.get('passed', False) else "FAIL ✗"
                    output.append(f"   P-value: {p_val:.6f} - {status}")
                
                # P-value secundário (para alguns testes)
                if 'p_value2' in test:
                    p_val2 = test['p_value2']
                    output.append(f"   P-value2: {p_val2:.6f}")
                
                if 'p_value_forward' in test:
                    output.append(f"   P-value (forward): {test['p_value_forward']:.6f}")
                    output.append(f"   P-value (backward): {test['p_value_backward']:.6f}")
                
                # Detalhes adicionais
                if detailed:
                    if 'statistic' in test:
                        output.append(f"   Statistic: {test['statistic']:.6f}")
                    
                    if 'cycles' in test:
                        output.append(f"   Cycles: {test['cycles']}")
                    
                    if 'states' in test and isinstance(test['states'], dict):
                        passed_states = sum(1 for s in test['states'].values() if s.get('passed', False))
                        total_states = len(test['states'])
                        output.append(f"   States passed: {passed_states}/{total_states}")
            
            output.append("")
        
        # Sumário
        output.append("=" * 80)
        output.append("SUMMARY")
        output.append("=" * 80)
        output.append(f"Total tests: {results['summary']['total_tests']}")
        output.append(f"Passed: {results['summary']['passed']}")
        output.append(f"Failed: {results['summary']['failed']}")
        output.append(f"Pass rate: {results['summary']['pass_rate']:.2f}%")
        output.append("")
        
        # Conclusão
        if results['summary']['pass_rate'] >= 95:
            output.append("CONCLUSION: The sequence demonstrates EXCELLENT randomness quality.")
            output.append("✓ Suitable for cryptographic applications.")
        elif results['summary']['pass_rate'] >= 85:
            output.append("CONCLUSION: The sequence demonstrates GOOD randomness quality.")
            output.append("⚠ May be suitable for some cryptographic applications.")
        elif results['summary']['pass_rate'] >= 70:
            output.append("CONCLUSION: The sequence demonstrates MODERATE randomness quality.")
            output.append("⚠ NOT recommended for cryptographic applications.")
        else:
            output.append("CONCLUSION: The sequence demonstrates POOR randomness quality.")
            output.append("✗ NOT suitable for cryptographic applications.")
        
        output.append("=" * 80)
        
        return "\n".join(output)
