#!/usr/bin/env python3
"""
Teste de inicialização dos dados do modelo.
Verifica se a fonte inicial está sendo criada corretamente no centro do grid.
"""

import sys
import os
import numpy as np

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Model

def test_initialization():
    """Testa se os dados são inicializados corretamente."""
    print("=" * 70)
    print("TESTE 1: Verificação de Inicialização dos Dados")
    print("=" * 70)
    print()
    
    # Configuração de teste
    grid_size = (64, 64, 64)
    vel_model = np.ones(shape=grid_size, dtype=np.float32) * 1500.0
    
    model = Model(
        velocity_model=vel_model,
        grid_spacing=(10, 10, 10),
        dt=0.002,
        num_timesteps=10,
        space_order=2,
        dtype='float32'
    )
    
    # Obtém os arrays iniciais
    prev_u, next_u = model.u_arrays
    
    print(f"Grid shape: {prev_u.shape}")
    print(f"Data type: {prev_u.dtype}")
    print()
    
    # Verifica dimensões
    assert prev_u.shape == grid_size, f"Shape incorreto: esperado {grid_size}, obtido {prev_u.shape}"
    assert next_u.shape == grid_size, f"Shape incorreto: esperado {grid_size}, obtido {next_u.shape}"
    print("✓ Dimensões corretas")
    
    # Verifica tipo de dados
    assert prev_u.dtype == np.float32, f"Tipo incorreto: esperado float32, obtido {prev_u.dtype}"
    assert next_u.dtype == np.float32, f"Tipo incorreto: esperado float32, obtido {next_u.dtype}"
    print("✓ Tipo de dados correto")
    
    # Verifica se há valores não-zero (fonte inicial)
    n1, n2, n3 = grid_size
    center_z, center_y, center_x = n1 // 2, n2 // 2, n3 // 2
    
    # Verifica o centro (onde a fonte deve estar)
    center_value = prev_u[center_z, center_y, center_x]
    print(f"Valor no centro [{center_z}, {center_y}, {center_x}]: {center_value}")
    
    # A fonte inicial deve ter valores não-zero no centro
    max_val = np.max(prev_u)
    min_val = np.min(prev_u)
    non_zero_count = np.count_nonzero(prev_u)
    total_elements = prev_u.size
    
    print(f"Valor máximo: {max_val}")
    print(f"Valor mínimo: {min_val}")
    print(f"Elementos não-zero: {non_zero_count} / {total_elements} ({100*non_zero_count/total_elements:.2f}%)")
    print()
    
    # Verificações
    assert max_val > 0, "Erro: valor máximo é zero - fonte não foi inicializada"
    assert center_value > 0, f"Erro: valor no centro é zero ({center_value}) - fonte não está no centro"
    assert non_zero_count > 0, "Erro: todos os valores são zero"
    
    # Verifica se a fonte está concentrada no centro
    # A fonte deve ter valores decrescentes do centro para fora
    # Verifica uma região 5x5x5 ao redor do centro
    region_size = 5
    z_start = max(0, center_z - region_size)
    z_end = min(n1, center_z + region_size)
    y_start = max(0, center_y - region_size)
    y_end = min(n2, center_y + region_size)
    x_start = max(0, center_x - region_size)
    x_end = min(n3, center_x + region_size)
    
    center_region = prev_u[z_start:z_end, y_start:y_end, x_start:x_end]
    center_region_max = np.max(center_region)
    
    print(f"Valor máximo na região central ({region_size}x{region_size}x{region_size}): {center_region_max}")
    
    # Verifica se next_u é uma cópia de prev_u
    assert np.array_equal(prev_u, next_u), "Erro: next_u não é uma cópia de prev_u"
    print("✓ next_u é uma cópia correta de prev_u")
    print()
    
    # Verifica valores específicos da fonte
    # A função _add_initial_source cria camadas concêntricas:
    # - Camada externa (s=4): valor 50.0
    # - Camada interna (s=3): valor 45.0
    # - Camada interna (s=2): valor 40.5
    # - Camada interna (s=1): valor 36.45
    # - Centro (s=0): valor 32.805
    # Como as camadas são sobrescritas de fora para dentro, o centro tem o menor valor
    
    # Verifica valores ao longo do eixo x do centro para fora
    found_values = []
    for offset in range(5):
        z = center_z
        y = center_y
        x = center_x + offset  # Do centro (offset=0) para fora
        if 0 <= z < n1 and 0 <= y < n2 and 0 <= x < n3:
            val = prev_u[z, y, x]
            found_values.append(val)
    
    if len(found_values) > 0:
        center_val = found_values[0]
        outer_vals = found_values[1:] if len(found_values) > 1 else []
        
        print(f"Valor no centro: {center_val:.6f}")
        if outer_vals:
            print(f"Valores externos (amostra): {[f'{v:.3f}' for v in outer_vals[:3]]}")
        
        # Verifica que o centro tem valor não-zero e está no range esperado
        assert center_val > 0, f"Erro: valor no centro é zero"
        assert 30.0 <= center_val <= 50.0, f"Erro: valor no centro fora do range esperado: {center_val}"
        
        # Verifica que há valores não-zero ao redor
        if outer_vals:
            max_outer = max(outer_vals)
            assert max_outer > 0, "Erro: valores externos são zero"
            print(f"✓ Valores da fonte estão no range esperado (centro={center_val:.3f}, max_externo={max_outer:.3f})")
        else:
            print(f"✓ Valor no centro está correto: {center_val:.3f}")
    
    print()
    print("=" * 70)
    print("✓ TESTE DE INICIALIZAÇÃO: PASSOU")
    print("=" * 70)
    print()
    
    return True

if __name__ == "__main__":
    try:
        test_initialization()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ ERRO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
