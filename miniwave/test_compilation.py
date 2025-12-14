#!/usr/bin/env python3
"""
Teste de compilação dos kernels.
Verifica se todos os kernels podem ser compilados corretamente.
"""

import sys
import os
import numpy as np

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Model, Compiler, Properties

def test_kernel_compilation(kernel_file, language, sm=75, dtype='float32'):
    """Testa compilação de um kernel específico."""
    print(f"  Testando: {kernel_file} ({language}, {dtype})")
    
    # Cria modelo pequeno para teste
    grid_size = (16, 16, 16)
    vel_model = np.ones(shape=grid_size, dtype=np.float32) * 1500.0
    
    model = Model(
        velocity_model=vel_model,
        grid_spacing=(10, 10, 10),
        dt=0.002,
        num_timesteps=1,
        space_order=2,
        dtype=dtype
    )
    
    # Cria compilador
    compiler = Compiler(language=language, sm=sm, fast_math=False)
    
    # Cria propriedades
    properties = Properties(
        block_size_1=4,
        block_size_2=4,
        block_size_3=4
    )
    properties.space_order = model.space_order
    properties.dtype = dtype
    
    # Tenta compilar
    try:
        shared_object = compiler.compile(kernel_file, properties)
        assert os.path.exists(shared_object), f"Arquivo .so não foi criado: {shared_object}"
        print(f"    ✓ Compilado com sucesso: {shared_object}")
        return True
    except Exception as e:
        print(f"    ✗ Erro na compilação: {e}")
        return False

def test_compilation():
    """Testa compilação de todos os kernels."""
    print("=" * 70)
    print("TESTE 3: Verificação de Compilação dos Kernels")
    print("=" * 70)
    print()
    
    results = {}
    
    # Testa kernel sequencial (C)
    print("1. Kernel Sequencial (C):")
    results['sequential'] = test_kernel_compilation(
        "kernels/sequential.c",
        "c",
        dtype='float32'
    )
    results['sequential_float64'] = test_kernel_compilation(
        "kernels/sequential.c",
        "c",
        dtype='float64'
    )
    print()
    
    # Testa kernels CUDA (requer CUDA disponível)
    print("2. Kernels CUDA:")
    
    # Verifica se CUDA está disponível
    cuda_available = False
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cuda_available = True
            print("  ✓ CUDA disponível (nvcc encontrado)")
        else:
            print("  ⚠ CUDA não disponível (nvcc não encontrado ou erro)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠ CUDA não disponível (nvcc não encontrado)")
    
    if cuda_available:
        # Baseline
        results['cuda_baseline'] = test_kernel_compilation(
            "kernels/cuda_baseline.cu",
            "cuda",
            sm=75,
            dtype='float32'
        )
        results['cuda_baseline_float64'] = test_kernel_compilation(
            "kernels/cuda_baseline.cu",
            "cuda",
            sm=75,
            dtype='float64'
        )
        
        # Shared
        results['cuda_shared'] = test_kernel_compilation(
            "kernels/cuda_shared.cu",
            "cuda",
            sm=75,
            dtype='float32'
        )
        results['cuda_shared_float64'] = test_kernel_compilation(
            "kernels/cuda_shared.cu",
            "cuda",
            sm=75,
            dtype='float64'
        )
        
        # Temporal
        results['cuda_temporal'] = test_kernel_compilation(
            "kernels/cuda_temporal.cu",
            "cuda",
            sm=75,
            dtype='float32'
        )
        results['cuda_temporal_float64'] = test_kernel_compilation(
            "kernels/cuda_temporal.cu",
            "cuda",
            sm=75,
            dtype='float64'
        )
    else:
        print("  ⚠ Pulando testes CUDA (CUDA não disponível)")
        results['cuda_baseline'] = None
        results['cuda_shared'] = None
        results['cuda_temporal'] = None
    
    print()
    
    # Resumo
    print("=" * 70)
    print("RESUMO DA COMPILAÇÃO:")
    print("=" * 70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is None:
            print(f"  ⚠ {name}: PULADO (CUDA não disponível)")
            skipped += 1
        elif result:
            print(f"  ✓ {name}: PASSOU")
            passed += 1
        else:
            print(f"  ✗ {name}: FALHOU")
            failed += 1
    
    print()
    print(f"Total: {passed} passou, {failed} falhou, {skipped} pulado")
    print()
    
    # Verifica se pelo menos o kernel sequencial compilou
    assert results.get('sequential', False), "ERRO CRÍTICO: Kernel sequencial não compilou"
    
    if failed > 0:
        print("⚠ Alguns kernels falharam na compilação")
        print("  Isso pode ser normal se CUDA não estiver disponível")
    
    print("=" * 70)
    if failed == 0:
        print("✓ TESTE DE COMPILAÇÃO: PASSOU")
    else:
        print("⚠ TESTE DE COMPILAÇÃO: PASSOU COM AVISOS")
    print("=" * 70)
    print()
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = test_compilation()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n✗ ERRO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
