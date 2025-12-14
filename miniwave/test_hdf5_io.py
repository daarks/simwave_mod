#!/usr/bin/env python3
"""
Teste de integridade dos arquivos HDF5.
Verifica se os dados estão sendo escritos e lidos corretamente.
"""

import sys
import os
import numpy as np
import h5py

# Adiciona o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Model, DatasetWriter

def test_hdf5_write_read():
    """Testa escrita e leitura de HDF5."""
    print("=" * 70)
    print("TESTE 2: Verificação de Integridade HDF5")
    print("=" * 70)
    print()
    
    # Configuração de teste
    grid_size = (32, 32, 32)
    vel_model = np.ones(shape=grid_size, dtype=np.float32) * 1500.0
    
    model = Model(
        velocity_model=vel_model,
        grid_spacing=(10, 10, 10),
        dt=0.002,
        num_timesteps=10,
        space_order=2,
        dtype='float32'
    )
    
    prev_u, next_u = model.u_arrays
    
    # Prepara dados para escrita
    data = {
        "prev_u": {"dataset_data": prev_u, "dataset_attributes": {}},
        "next_u": {"dataset_data": next_u, "dataset_attributes": {}},
        "vel_model": {
            "dataset_data": model.velocity_model,
            "dataset_attributes": {},
        },
        "coefficient": {
            "dataset_data": model.stencil_coefficients,
            "dataset_attributes": {},
        },
        "d1": {"dataset_data": 10.0, "dataset_attributes": {}},
        "d2": {"dataset_data": 10.0, "dataset_attributes": {}},
        "d3": {"dataset_data": 10.0, "dataset_attributes": {}},
        "dt": {"dataset_data": 0.002, "dataset_attributes": {}},
        "n1": {"dataset_data": grid_size[0], "dataset_attributes": {}},
        "n2": {"dataset_data": grid_size[1], "dataset_attributes": {}},
        "n3": {"dataset_data": grid_size[2], "dataset_attributes": {}},
        "iterations": {"dataset_data": 10, "dataset_attributes": {}},
        "stencil_radius": {"dataset_data": 1, "dataset_attributes": {}},
        "block_size_1": {"dataset_data": 8, "dataset_attributes": {}},
        "block_size_2": {"dataset_data": 8, "dataset_attributes": {}},
        "block_size_3": {"dataset_data": 8, "dataset_attributes": {}},
    }
    
    # Cria diretório se não existir
    test_file = "test_data.h5"
    os.makedirs(os.path.dirname(test_file) if os.path.dirname(test_file) else ".", exist_ok=True)
    
    # Escreve arquivo HDF5
    print(f"Escrevendo arquivo HDF5: {test_file}")
    DatasetWriter.write_dataset(data, test_file)
    print("✓ Arquivo escrito com sucesso")
    print()
    
    # Lê e verifica o arquivo
    print("Lendo e verificando arquivo HDF5...")
    with h5py.File(test_file, "r") as f:
        # Verifica se todos os datasets existem
        required_datasets = ["prev_u", "next_u", "vel_model", "coefficient", "scalar_data"]
        for ds_name in required_datasets:
            assert ds_name in f, f"Dataset '{ds_name}' não encontrado no arquivo"
        print("✓ Todos os datasets necessários estão presentes")
        
        # Verifica prev_u
        prev_u_read = f["prev_u"][:]
        assert prev_u_read.shape == prev_u.shape, \
            f"Shape de prev_u incorreto: esperado {prev_u.shape}, obtido {prev_u_read.shape}"
        assert prev_u_read.dtype == prev_u.dtype, \
            f"Tipo de prev_u incorreto: esperado {prev_u.dtype}, obtido {prev_u_read.dtype}"
        assert np.allclose(prev_u_read, prev_u, rtol=1e-5), \
            "Valores de prev_u não correspondem"
        print("✓ prev_u: shape, tipo e valores corretos")
        
        # Verifica next_u
        next_u_read = f["next_u"][:]
        assert next_u_read.shape == next_u.shape, \
            f"Shape de next_u incorreto: esperado {next_u.shape}, obtido {next_u_read.shape}"
        assert next_u_read.dtype == next_u.dtype, \
            f"Tipo de next_u incorreto: esperado {next_u.dtype}, obtido {next_u_read.dtype}"
        assert np.allclose(next_u_read, next_u, rtol=1e-5), \
            "Valores de next_u não correspondem"
        print("✓ next_u: shape, tipo e valores corretos")
        
        # Verifica vel_model
        vel_model_read = f["vel_model"][:]
        assert vel_model_read.shape == vel_model.shape, \
            f"Shape de vel_model incorreto: esperado {vel_model.shape}, obtido {vel_model_read.shape}"
        assert np.allclose(vel_model_read, vel_model, rtol=1e-5), \
            "Valores de vel_model não correspondem"
        print("✓ vel_model: shape e valores corretos")
        
        # Verifica coefficient
        coeff_read = f["coefficient"][:]
        coeff_expected = model.stencil_coefficients
        assert coeff_read.shape == coeff_expected.shape, \
            f"Shape de coefficient incorreto: esperado {coeff_expected.shape}, obtido {coeff_read.shape}"
        assert np.allclose(coeff_read, coeff_expected, rtol=1e-5), \
            "Valores de coefficient não correspondem"
        print("✓ coefficient: shape e valores corretos")
        
        # Verifica atributos escalares
        scalar_data = f["scalar_data"]
        attrs_to_check = ["d1", "d2", "d3", "dt", "n1", "n2", "n3", 
                         "iterations", "stencil_radius", "block_size_1", 
                         "block_size_2", "block_size_3"]
        
        for attr_name in attrs_to_check:
            assert attr_name in scalar_data.attrs, \
                f"Atributo '{attr_name}' não encontrado em scalar_data"
            attr_value = float(scalar_data.attrs[attr_name])
            expected_value = data[attr_name]["dataset_data"]
            assert abs(attr_value - expected_value) < 1e-5, \
                f"Valor do atributo '{attr_name}' incorreto: esperado {expected_value}, obtido {attr_value}"
        print("✓ Todos os atributos escalares estão corretos")
        
        # Verifica se há valores não-zero
        prev_u_max = np.max(prev_u_read)
        prev_u_min = np.min(prev_u_read)
        prev_u_nonzero = np.count_nonzero(prev_u_read)
        
        print()
        print(f"Estatísticas de prev_u no HDF5:")
        print(f"  Min: {prev_u_min:.6f}")
        print(f"  Max: {prev_u_max:.6f}")
        print(f"  Elementos não-zero: {prev_u_nonzero} / {prev_u_read.size} ({100*prev_u_nonzero/prev_u_read.size:.2f}%)")
        
        assert prev_u_max > 0, "Erro: todos os valores de prev_u são zero no HDF5"
        assert prev_u_nonzero > 0, "Erro: nenhum valor não-zero encontrado em prev_u"
        print("✓ prev_u contém valores não-zero")
    
    # Limpa arquivo de teste
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"✓ Arquivo de teste removido: {test_file}")
    
    print()
    print("=" * 70)
    print("✓ TESTE DE INTEGRIDADE HDF5: PASSOU")
    print("=" * 70)
    print()
    
    return True

def test_hdf5_float64():
    """Testa escrita e leitura com float64."""
    print("=" * 70)
    print("TESTE 2b: Verificação HDF5 com float64")
    print("=" * 70)
    print()
    
    grid_size = (16, 16, 16)
    vel_model = np.ones(shape=grid_size, dtype=np.float64) * 1500.0
    
    model = Model(
        velocity_model=vel_model,
        grid_spacing=(10, 10, 10),
        dt=0.002,
        num_timesteps=5,
        space_order=2,
        dtype='float64'
    )
    
    prev_u, next_u = model.u_arrays
    
    data = {
        "prev_u": {"dataset_data": prev_u, "dataset_attributes": {}},
        "next_u": {"dataset_data": next_u, "dataset_attributes": {}},
        "vel_model": {"dataset_data": model.velocity_model, "dataset_attributes": {}},
        "coefficient": {"dataset_data": model.stencil_coefficients, "dataset_attributes": {}},
    }
    
    test_file = "test_data_float64.h5"
    DatasetWriter.write_dataset(data, test_file)
    
    with h5py.File(test_file, "r") as f:
        prev_u_read = f["prev_u"][:]
        assert prev_u_read.dtype == np.float64, \
            f"Tipo incorreto: esperado float64, obtido {prev_u_read.dtype}"
        assert np.allclose(prev_u_read, prev_u, rtol=1e-10), \
            "Valores não correspondem para float64"
        print("✓ float64: tipo e valores corretos")
    
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print()
    print("=" * 70)
    print("✓ TESTE HDF5 FLOAT64: PASSOU")
    print("=" * 70)
    print()
    
    return True

if __name__ == "__main__":
    try:
        test_hdf5_write_read()
        test_hdf5_float64()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ ERRO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
