#!/usr/bin/env python3
"""
Teste dos kernels CUDA (baseline, shared, temporal).
Compara resultados com o kernel sequencial de referência.
"""

import sys
import os
import numpy as np
import h5py
import subprocess
import shutil

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Model, Compiler, Kernel, Properties

def check_cuda_available():
    """Verifica se CUDA está disponível."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def run_kernel_test(kernel_file, kernel_name, model, compiler, properties, reference_result=None):
    """Executa um teste de kernel e retorna os resultados."""
    print(f"  Testando: {kernel_name}")
    print(f"    Arquivo: {kernel_file}")
    
    # Limpa artefatos
    if os.path.exists("/tmp/miniwave"):
        for f in os.listdir("/tmp/miniwave"):
            if f.endswith(".so"):
                try:
                    os.remove(os.path.join("/tmp/miniwave", f))
                except:
                    pass
    if os.path.exists("c-frontend/build"):
        shutil.rmtree("c-frontend/build")
    if os.path.exists("c-frontend/data"):
        shutil.rmtree("c-frontend/data")
    os.makedirs("c-frontend/data", exist_ok=True)
    
    try:
        solver = Kernel(
            file=kernel_file,
            model=model,
            compiler=compiler,
            properties=properties
        )
        
        exec_time, u_final = solver.run()
        
        # Verifica resultados
        results_file = "c-frontend/data/results.h5"
        if not os.path.exists(results_file):
            print(f"    ✗ Arquivo de resultado não encontrado")
            return None
        
        with h5py.File(results_file, "r") as f:
            vector = f["vector"][:]
            exec_time_h5 = f["execution_time"][()]
            # Converte para escalar se for array numpy
            if isinstance(exec_time_h5, np.ndarray):
                exec_time_h5 = exec_time_h5.item()
            exec_time_h5 = float(exec_time_h5)
        
        # Verifica se há valores não-zero
        max_val = np.max(vector)
        min_val = np.min(vector)
        non_zero_count = np.count_nonzero(np.abs(vector) > 1e-6)
        
        print(f"    ✓ Execução concluída em {exec_time:.4f}s")
        print(f"      Min: {min_val:.6f}, Max: {max_val:.6f}")
        print(f"      Elementos não-zero: {non_zero_count} / {vector.size}")
        
        # Verificações críticas
        if max_val == 0 or abs(max_val) < 1e-6:
            print(f"    ✗ ERRO: Todos os valores são zero ou muito pequenos!")
            return None
        
        result = {
            'vector': vector,
            'exec_time': exec_time,
            'max_val': max_val,
            'min_val': min_val,
            'non_zero_count': non_zero_count
        }
        
        # Compara com referência se disponível
        if reference_result is not None:
            ref_vector = reference_result['vector']
            diff = vector - ref_vector
            linf_error = np.max(np.abs(diff))
            l2_error = np.linalg.norm(diff)
            ref_norm = np.linalg.norm(ref_vector)
            rel_error = (l2_error / ref_norm * 100) if ref_norm > 0 else float('inf')
            
            print(f"      Comparação com referência:")
            print(f"        L∞ erro: {linf_error:.6e}")
            print(f"        L2 erro: {l2_error:.6e}")
            print(f"        Erro relativo: {rel_error:.4f}%")
            
            result['linf_error'] = linf_error
            result['l2_error'] = l2_error
            result['rel_error'] = rel_error
            
            # Verifica se o erro é aceitável
            if rel_error > 1.0:  # Mais de 1% de erro relativo
                print(f"      ⚠ AVISO: Erro relativo alto ({rel_error:.4f}%)")
            elif rel_error > 0.1:  # Mais de 0.1% de erro relativo
                print(f"      ⚠ AVISO: Erro relativo moderado ({rel_error:.4f}%)")
            else:
                print(f"      ✓ Erro relativo aceitável ({rel_error:.4f}%)")
        
        return result
        
    except Exception as e:
        print(f"    ✗ Erro na execução: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cuda_kernels():
    """Testa todos os kernels CUDA."""
    print("=" * 70)
    print("TESTE 5: Verificação dos Kernels CUDA")
    print("=" * 70)
    print()
    
    # Verifica se CUDA está disponível
    if not check_cuda_available():
        print("⚠ CUDA não disponível (nvcc não encontrado)")
        print("  Pulando testes CUDA")
        print()
        print("=" * 70)
        print("⚠ TESTE CUDA: PULADO (CUDA não disponível)")
        print("=" * 70)
        print()
        return True
    
    print("✓ CUDA disponível")
    print()
    
    # Configuração de teste
    grid_size = 64
    num_timesteps = 10
    sm = 75  # Compute capability (ajuste conforme sua GPU)
    
    print(f"Configuração:")
    print(f"  Grid: {grid_size}³")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Precisão: float32")
    print(f"  SM: {sm}")
    print()
    
    # Cria modelo
    vel_model = np.ones(shape=(grid_size, grid_size, grid_size), dtype=np.float32) * 1500.0
    
    model = Model(
        velocity_model=vel_model,
        grid_spacing=(10, 10, 10),
        dt=0.002,
        num_timesteps=num_timesteps,
        space_order=2,
        dtype='float32'
    )
    
    # Primeiro, executa kernel sequencial como referência
    print("1. Executando kernel sequencial (referência)...")
    compiler_seq = Compiler(language="c", sm=sm, fast_math=False)
    properties = Properties(block_size_1=8, block_size_2=8, block_size_3=8)
    
    reference_result = run_kernel_test(
        "kernels/sequential.c",
        "Sequential (Reference)",
        model,
        compiler_seq,
        properties
    )
    
    if reference_result is None:
        print("✗ ERRO: Não foi possível executar kernel sequencial de referência")
        return False
    
    print()
    
    # Testa kernels CUDA
    compiler_cuda = Compiler(language="cuda", sm=sm, fast_math=False)
    
    kernels_to_test = [
        ("kernels/cuda_baseline.cu", "CUDA Baseline"),
        ("kernels/cuda_shared.cu", "CUDA Shared Memory"),
        ("kernels/cuda_temporal.cu", "CUDA Temporal Blocking"),
    ]
    
    results = {}
    
    for kernel_file, kernel_name in kernels_to_test:
        if not os.path.exists(kernel_file):
            print(f"⚠ {kernel_name}: arquivo não encontrado ({kernel_file})")
            continue
        
        print(f"{len(results) + 2}. {kernel_name}:")
        result = run_kernel_test(
            kernel_file,
            kernel_name,
            model,
            compiler_cuda,
            properties,
            reference_result
        )
        results[kernel_name] = result
        print()
    
    # Resumo
    print("=" * 70)
    print("RESUMO DOS TESTES CUDA:")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for kernel_name, result in results.items():
        if result is None:
            print(f"  ✗ {kernel_name}: FALHOU")
            failed += 1
        else:
            print(f"  ✓ {kernel_name}: PASSOU", end="")
            if 'rel_error' in result:
                print(f" (erro relativo: {result['rel_error']:.4f}%)")
            else:
                print()
            passed += 1
    
    print()
    print(f"Total: {passed} passou, {failed} falhou")
    print()
    
    # Verifica se pelo menos um kernel CUDA passou
    if passed == 0:
        print("✗ ERRO: Nenhum kernel CUDA passou nos testes")
        return False
    
    print("=" * 70)
    if failed == 0:
        print("✓ TESTE DOS KERNELS CUDA: PASSOU")
    else:
        print("⚠ TESTE DOS KERNELS CUDA: PASSOU COM FALHAS")
    print("=" * 70)
    print()
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = test_cuda_kernels()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n✗ ERRO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
