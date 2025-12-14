#!/usr/bin/env bash
# Script para executar todos os testes do miniwave em sequência

set -e  # Para na primeira falha

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Verifica se está no .venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠ AVISO: Não está no ambiente virtual (.venv)${NC}"
    echo -e "${YELLOW}  Execute: source .venv/bin/activate${NC}"
    echo ""
    read -p "Continuar mesmo assim? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ambiente virtual ativo: $VIRTUAL_ENV${NC}"
    echo ""
fi

# Banner
echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "          MINIWAVE - SUITE COMPLETA DE TESTES"
echo "======================================================================"
echo -e "${NC}"
echo ""

# Contadores
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Função para executar um teste
run_test() {
    local test_name="$1"
    local test_file="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${BLUE}${BOLD}[$TOTAL_TESTS] Executando: $test_name${NC}"
    echo -e "${BLUE}  Arquivo: $test_file${NC}"
    echo ""
    
    if [ ! -f "$test_file" ]; then
        echo -e "${RED}✗ ERRO: Arquivo não encontrado: $test_file${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    # Executa o teste
    if python3 "$test_file" 2>&1; then
        echo ""
        echo -e "${GREEN}✓ TESTE PASSOU: $test_name${NC}"
        echo ""
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo ""
        echo -e "${RED}✗ TESTE FALHOU: $test_name${NC}"
        echo ""
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Lista de testes em ordem
TESTS=(
    "Verificação de Inicialização|test_initialization.py"
    "Verificação de HDF5 I/O|test_hdf5_io.py"
    "Verificação de Compilação|test_compilation.py"
    "Teste do Kernel Sequencial|test_sequential.py"
    "Teste dos Kernels CUDA|test_cuda_kernels.py"
    "Teste End-to-End|test_end_to_end.py"
)

# Executa todos os testes
for test_info in "${TESTS[@]}"; do
    IFS='|' read -r test_name test_file <<< "$test_info"
    run_test "$test_name" "$test_file"
    
    # Pausa entre testes (opcional)
    sleep 1
done

# Resumo final
echo ""
echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "                    RESUMO FINAL DOS TESTES"
echo "======================================================================"
echo -e "${NC}"
echo ""

echo -e "Total de testes: ${BOLD}$TOTAL_TESTS${NC}"
echo -e "${GREEN}Testes que passaram: $PASSED_TESTS${NC}"
echo -e "${RED}Testes que falharam: $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ TODOS OS TESTES PASSARAM!${NC}"
    echo ""
    echo -e "${GREEN}O simulador miniwave está funcionando corretamente.${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}${BOLD}✗ ALGUNS TESTES FALHARAM${NC}"
    echo ""
    echo -e "${YELLOW}Revise os erros acima e corrija os problemas antes de continuar.${NC}"
    echo ""
    exit 1
fi
