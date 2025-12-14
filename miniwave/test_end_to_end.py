#!/usr/bin/env python3
"""
Teste end-to-end completo do simulador miniwave.
Valida todo o fluxo desde a inicialização até a geração dos resultados.
"""

import sys
import os
import numpy as np
import h5py
import shutil

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Model, Compiler, Kernel, Properties

def test_end_to_end():
    """Teste end-to-end completo."""
    print("=" * 70)
    print("TESTE 6: Teste End-to-End Completo")
    print("=" * 70)
    print()
    
    # Configurações de teste
    test_configs = [
        {
            'name': 'Pequeno (32³, 5 timesteps)',
            'grid_size': 32,
            'num_timesteps': 5,
            'dtype': 'float32'
        },
        {
            'name': 'Médio (64³, 10 timesteps)',
            'grid_size': 64,
            'num_timesteps': 10,
            'dtype': 'float32'
        },
    ]
    
    all_passed = True
    
    for config in test_configs:
        print(f"Configuração: {config['name']}")
        print(f"  Grid: {config['grid_size']}³")
        print(f"  Timesteps: {config['num_timesteps']}")
        print(f"  Precisão: {config['dtype']}")
        print()
        
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
        
        # Cria modelo
        grid_size = config['grid_size']
        vel_model = np.ones(shape=(grid_size, grid_size, grid_size), 
                           dtype=np.float32) * 1500.0
        
        model = Model(
            velocity_model=vel_model,
            grid_spacing=(10, 10, 10),
            dt=0.002,
            num_timesteps=config['num_timesteps'],
            space_order=2,
            dtype=config['dtype']
        )
        
        # Verifica inicialização
        prev_u_init, next_u_init = model.u_arrays
        max_init = np.max(prev_u_init)
        assert max_init > 0, "Dados iniciais zerados"
        print("  ✓ Inicialização: OK")
        
        # Cria compilador e kernel
        compiler = Compiler(language="c", sm=75, fast_math=False)
        properties = Properties(block_size_1=4, block_size_2=4, block_size_3=4)
        
        solver = Kernel(
            file="kernels/sequential.c",
            model=model,
            compiler=compiler,
            properties=properties
        )
        
        # Executa simulação
        try:
            exec_time, u_final = solver.run()
            print(f"  ✓ Execução: OK ({exec_time:.4f}s)")
        except Exception as e:
            print(f"  ✗ Execução falhou: {e}")
            all_passed = False
            continue
        
        # Verifica arquivo HDF5 de entrada
        input_file = "c-frontend/data/miniwave_data.h5"
        if os.path.exists(input_file):
            with h5py.File(input_file, "r") as f:
                assert "prev_u" in f, "prev_u não encontrado no HDF5 de entrada"
                assert "next_u" in f, "next_u não encontrado no HDF5 de entrada"
                assert "vel_model" in f, "vel_model não encontrado no HDF5 de entrada"
            print("  ✓ HDF5 de entrada: OK")
        else:
            print(f"  ✗ HDF5 de entrada não encontrado: {input_file}")
            all_passed = False
            continue
        
        # Verifica arquivo HDF5 de saída
        output_file = "c-frontend/data/results.h5"
        if not os.path.exists(output_file):
            print(f"  ✗ HDF5 de saída não encontrado: {output_file}")
            all_passed = False
            continue
        
        with h5py.File(output_file, "r") as f:
            # Verifica datasets
            assert "vector" in f, "Dataset 'vector' não encontrado"
            assert "execution_time" in f, "Dataset 'execution_time' não encontrado"
            print("  ✓ HDF5 de saída: datasets presentes")
            
            # Verifica vector
            vector = f["vector"][:]
            assert vector.shape == (grid_size, grid_size, grid_size), \
                f"Shape incorreto: {vector.shape}"
            print(f"  ✓ HDF5 de saída: shape correto {vector.shape}")
            
            # Verifica valores não-zero
            max_val = np.max(vector)
            min_val = np.min(vector)
            non_zero = np.count_nonzero(np.abs(vector) > 1e-6)
            
            assert max_val != 0, "Todos os valores são zero!"
            assert abs(max_val) > 1e-6, f"Valor máximo muito pequeno: {max_val}"
            assert non_zero > 0, "Nenhum valor não-zero encontrado"
            
            print(f"  ✓ HDF5 de saída: valores válidos (min={min_val:.6f}, max={max_val:.6f}, non-zero={non_zero})")
            
            # Verifica execution_time
            exec_time_h5 = f["execution_time"][()]
            # Converte para escalar se for array numpy
            if isinstance(exec_time_h5, np.ndarray):
                exec_time_h5 = exec_time_h5.item()
            exec_time_h5 = float(exec_time_h5)
            assert exec_time_h5 > 0, f"Tempo de execução inválido: {exec_time_h5}"
            print(f"  ✓ HDF5 de saída: execution_time válido ({exec_time_h5:.6f}s)")
            
            # Verifica correspondência com valores retornados
            assert np.allclose(u_final, vector, rtol=1e-5), \
                "Valores retornados não correspondem ao HDF5"
            print("  ✓ Valores retornados correspondem ao HDF5")
        
        # Verifica propagação da onda
        center_z, center_y, center_x = grid_size // 2, grid_size // 2, grid_size // 2
        center_init = prev_u_init[center_z, center_y, center_x]
        center_final = u_final[center_z, center_y, center_x]
        
        assert abs(center_final - center_init) > 1e-6, \
            "Valor no centro não mudou (simulação não funcionou)"
        print(f"  ✓ Propagação: valor no centro mudou ({center_init:.6f} → {center_final:.6f})")
        
        print()
        print(f"  ✓ Configuração '{config['name']}': PASSOU")
        print()
    
    print("=" * 70)
    if all_passed:
        print("✓ TESTE END-TO-END: PASSOU")
    else:
        print("✗ TESTE END-TO-END: FALHOU")
    print("=" * 70)
    print()
    
    return all_passed

def test_multiple_precisions():
    """Testa com diferentes precisões."""
    print("=" * 70)
    print("TESTE 6b: Verificação de Precisões (float32 vs float64)")
    print("=" * 70)
    print()
    
    grid_size = 32
    num_timesteps = 5
    
    results = {}
    
    for dtype in ['float32', 'float64']:
        print(f"Testando com {dtype}...")
        
        # Limpa artefatos
        if os.path.exists("c-frontend/build"):
            shutil.rmtree("c-frontend/build")
        if os.path.exists("c-frontend/data"):
            shutil.rmtree("c-frontend/data")
        os.makedirs("c-frontend/data", exist_ok=True)
        
        # Cria modelo
        if dtype == 'float32':
            vel_model = np.ones(shape=(grid_size, grid_size, grid_size), 
                               dtype=np.float32) * 1500.0
        else:
            vel_model = np.ones(shape=(grid_size, grid_size, grid_size), 
                               dtype=np.float64) * 1500.0
        
        model = Model(
            velocity_model=vel_model,
            grid_spacing=(10, 10, 10),
            dt=0.002,
            num_timesteps=num_timesteps,
            space_order=2,
            dtype=dtype
        )
        
        compiler = Compiler(language="c", sm=75, fast_math=False)
        properties = Properties(block_size_1=4, block_size_2=4, block_size_3=4)
        
        solver = Kernel(
            file="kernels/sequential.c",
            model=model,
            compiler=compiler,
            properties=properties
        )
        
        try:
            exec_time, u_final = solver.run()
            
            # Verifica resultado
            output_file = "c-frontend/data/results.h5"
            with h5py.File(output_file, "r") as f:
                vector = f["vector"][:]
                max_val = np.max(vector)
                assert max_val > 0, f"Valores zerados com {dtype}"
            
            results[dtype] = {
                'vector': vector,
                'max_val': max_val,
                'dtype': vector.dtype
            }
            
            print(f"  ✓ {dtype}: OK (max={max_val:.6f}, dtype={vector.dtype})")
            
        except Exception as e:
            print(f"  ✗ {dtype}: FALHOU - {e}")
            results[dtype] = None
    
    # Compara resultados se ambos passaram
    if results.get('float32') and results.get('float64'):
        vec32 = results['float32']['vector']
        vec64 = results['float64']['vector']
        
        # Converte para mesmo tipo para comparação
        vec32_float64 = vec32.astype(np.float64)
        diff = np.abs(vec32_float64 - vec64)
        max_diff = np.max(diff)
        
        print()
        print(f"Diferença máxima entre float32 e float64: {max_diff:.6e}")
        
        if max_diff > 1e-3:
            print("  ⚠ AVISO: Diferença significativa entre precisões")
        else:
            print("  ✓ Diferença entre precisões é aceitável")
    
    print()
    print("=" * 70)
    if all(r is not None for r in results.values()):
        print("✓ TESTE DE PRECISÕES: PASSOU")
    else:
        print("⚠ TESTE DE PRECISÕES: PASSOU COM AVISOS")
    print("=" * 70)
    print()
    
    return all(r is not None for r in results.values())

if __name__ == "__main__":
    try:
        success1 = test_end_to_end()
        success2 = test_multiple_precisions()
        sys.exit(0 if (success1 and success2) else 1)
    except AssertionError as e:
        print(f"\n✗ ERRO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
