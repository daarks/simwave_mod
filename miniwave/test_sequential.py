#!/usr/bin/env python3
"""
Teste do kernel sequencial como referência.
Verifica se o kernel sequencial gera resultados válidos e não zerados.
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

def test_sequential_kernel():
    """Testa o kernel sequencial como referência."""
    print("=" * 70)
    print("TESTE 4: Verificação do Kernel Sequencial (Referência)")
    print("=" * 70)
    print()
    
    # Configuração de teste pequeno
    grid_size = 64
    num_timesteps = 10
    
    print(f"Configuração:")
    print(f"  Grid: {grid_size}³")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Precisão: float32")
    print()
    
    # Limpa artefatos anteriores
    print("Limpando artefatos anteriores...")
    if os.path.exists("/tmp/miniwave"):
        for f in os.listdir("/tmp/miniwave"):
            if f.endswith(".so"):
                os.remove(os.path.join("/tmp/miniwave", f))
    if os.path.exists("c-frontend/build"):
        shutil.rmtree("c-frontend/build")
    if os.path.exists("c-frontend/data"):
        shutil.rmtree("c-frontend/data")
    os.makedirs("c-frontend/data", exist_ok=True)
    print("✓ Limpeza concluída")
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
    
    # Obtém arrays iniciais
    prev_u_init, next_u_init = model.u_arrays
    
    # Verifica inicialização
    max_init = np.max(prev_u_init)
    print(f"Valor máximo inicial: {max_init:.6f}")
    assert max_init > 0, "Erro: dados iniciais estão zerados"
    print("✓ Dados iniciais contêm valores não-zero")
    print()
    
    # Cria compilador e kernel
    compiler = Compiler(language="c", sm=75, fast_math=False)
    properties = Properties(block_size_1=8, block_size_2=8, block_size_3=8)
    
    solver = Kernel(
        file="kernels/sequential.c",
        model=model,
        compiler=compiler,
        properties=properties
    )
    
    # Executa simulação
    print("Executando simulação sequencial...")
    try:
        exec_time, u_final = solver.run()
        print(f"✓ Simulação concluída em {exec_time:.4f} segundos")
        print()
    except Exception as e:
        print(f"✗ Erro na execução: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verifica resultados
    print("Verificando resultados...")
    
    # Verifica se o arquivo de resultado foi criado
    results_file = "c-frontend/data/results.h5"
    assert os.path.exists(results_file), f"Arquivo de resultado não encontrado: {results_file}"
    print(f"✓ Arquivo de resultado criado: {results_file}")
    
    # Lê e verifica o arquivo HDF5
    with h5py.File(results_file, "r") as f:
        # Verifica datasets
        assert "vector" in f, "Dataset 'vector' não encontrado"
        assert "execution_time" in f, "Dataset 'execution_time' não encontrado"
        print("✓ Datasets necessários presentes")
        
        # Verifica vector
        vector = f["vector"][:]
        assert vector.shape == (grid_size, grid_size, grid_size), \
            f"Shape incorreto: esperado {(grid_size, grid_size, grid_size)}, obtido {vector.shape}"
        print(f"✓ Shape correto: {vector.shape}")
        
        # Verifica se há valores não-zero
        max_val = np.max(vector)
        min_val = np.min(vector)
        non_zero_count = np.count_nonzero(np.abs(vector) > 1e-6)
        total_elements = vector.size
        
        print(f"  Min: {min_val:.6f}")
        print(f"  Max: {max_val:.6f}")
        print(f"  Elementos não-zero (|u| > 1e-6): {non_zero_count} / {total_elements} ({100*non_zero_count/total_elements:.2f}%)")
        print()
        
        # Verificações críticas
        assert max_val != 0, "ERRO CRÍTICO: Todos os valores são zero!"
        assert abs(max_val) > 1e-6, f"ERRO: Valor máximo muito pequeno: {max_val}"
        assert non_zero_count > 0, "ERRO: Nenhum valor não-zero encontrado"
        
        # Verifica se a onda se propagou (deve haver mais valores não-zero após simulação)
        init_nonzero = np.count_nonzero(np.abs(prev_u_init) > 1e-6)
        print(f"  Elementos não-zero inicial: {init_nonzero}")
        print(f"  Elementos não-zero final: {non_zero_count}")
        
        # A onda deve se propagar (mais valores não-zero após simulação)
        # Mas para timesteps pequenos, pode não haver muita propagação
        if num_timesteps >= 5:
            # Para timesteps suficientes, esperamos alguma propagação
            propagation_ratio = non_zero_count / max(init_nonzero, 1)
            print(f"  Razão de propagação: {propagation_ratio:.2f}x")
            # Não falha se não houver propagação, apenas avisa
            if propagation_ratio < 1.1:
                print("  ⚠ Aviso: Pouca propagação detectada (pode ser normal para timesteps pequenos)")
        
        # Verifica execution_time
        exec_time_h5 = f["execution_time"][()]
        # Converte para escalar se for array numpy
        if isinstance(exec_time_h5, np.ndarray):
            exec_time_h5 = exec_time_h5.item()
        exec_time_h5 = float(exec_time_h5)
        assert exec_time_h5 > 0, f"Tempo de execução inválido: {exec_time_h5}"
        print(f"✓ Tempo de execução no HDF5: {exec_time_h5:.6f} segundos")
        print()
        
        # Verifica valores no centro (onde a fonte estava)
        center_z, center_y, center_x = grid_size // 2, grid_size // 2, grid_size // 2
        center_init = prev_u_init[center_z, center_y, center_x]
        center_final = vector[center_z, center_y, center_x]
        
        print(f"Valor no centro:")
        print(f"  Inicial: {center_init:.6f}")
        print(f"  Final: {center_final:.6f}")
        print(f"  Mudança: {center_final - center_init:+.6f}")
        print()
        
        # O valor no centro deve ter mudado após a simulação
        assert abs(center_final - center_init) > 1e-6, \
            "ERRO: Valor no centro não mudou após simulação"
        print("✓ Valor no centro mudou após simulação (simulação funcionou)")
    
    # Verifica se u_final corresponde ao vector do HDF5
    assert np.allclose(u_final, vector, rtol=1e-5), \
        "Valores retornados não correspondem ao arquivo HDF5"
    print("✓ Valores retornados correspondem ao arquivo HDF5")
    print()
    
    print("=" * 70)
    print("✓ TESTE DO KERNEL SEQUENCIAL: PASSOU")
    print("=" * 70)
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_sequential_kernel()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n✗ ERRO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
