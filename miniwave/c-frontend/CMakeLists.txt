cmake_minimum_required(VERSION 3.18)
project(miniwave)

# Habilita CUDA se KERNEL_TYPE for cuda ou mpi_cuda
if(KERNEL_TYPE STREQUAL "cuda" OR KERNEL_TYPE STREQUAL "mpi_cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
endif()

# Gera selected_kernel.h a partir do template
configure_file(${CMAKE_SOURCE_DIR}/selected_kernel.h.in
               ${CMAKE_BINARY_DIR}/selected_kernel.h)
include_directories(${CMAKE_BINARY_DIR})

# Usa o kernel passado como variável ou fallback para sequential
if(NOT DEFINED KERNEL_SOURCE)
    set(KERNEL_SOURCE "../kernels/sequential.c")
    message(STATUS "KERNEL_SOURCE not set, using default: ${KERNEL_SOURCE}")
else()
    message(STATUS "Using kernel: ${KERNEL_SOURCE}")
    # Ajusta o caminho relativo para ser a partir do diretório c-frontend
    set(KERNEL_SOURCE "../${KERNEL_SOURCE}")
endif()

# Compila o frontend com o kernel especificado
add_executable(miniwave ${KERNEL_SOURCE} miniwave.c)

# Includes para os kernels
target_include_directories(miniwave PUBLIC ${CMAKE_SOURCE_DIR}/../kernels)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C)
target_include_directories(miniwave PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(miniwave PRIVATE ${HDF5_LIBRARIES} m)

# Link CUDA runtime se necessário
if(KERNEL_TYPE STREQUAL "cuda" OR KERNEL_TYPE STREQUAL "mpi_cuda")
    target_link_libraries(miniwave PRIVATE cudart)
    set_target_properties(miniwave PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE}
    )
endif()

# MPI se necessário
if(KERNEL_TYPE STREQUAL "mpi" OR KERNEL_TYPE STREQUAL "mpi_cuda" OR KERNEL_TYPE STREQUAL "ompc")
    find_package(MPI REQUIRED)
    target_link_libraries(miniwave PRIVATE MPI::MPI_C)
    target_compile_definitions(miniwave PRIVATE CONTAINS_MPI)
endif()
