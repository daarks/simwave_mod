#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_kernel.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

__global__ void wave_propagation_kernel(
    f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
    f_type d1Squared, f_type d2Squared, f_type d3Squared, f_type dtSquared,
    int n1, int n2, int n3, int stencil_radius)
{
    int i = blockIdx.z * blockDim.z + threadIdx.z + stencil_radius;
    int j = blockIdx.y * blockDim.y + threadIdx.y + stencil_radius;
    int k = blockIdx.x * blockDim.x + threadIdx.x + stencil_radius;

    if (i < n1 - stencil_radius && j < n2 - stencil_radius && k < n3 - stencil_radius) {
        int current = (i * n2 + j) * n3 + k;

        f_type value = coefficient[0] * (prev_u[current]/d1Squared + 
                                         prev_u[current]/d2Squared + 
                                         prev_u[current]/d3Squared);

        for (int ir = 1; ir <= stencil_radius; ir++) {
            value += coefficient[ir] * (
                ((prev_u[current + ir] + prev_u[current - ir]) / d3Squared) +
                ((prev_u[current + (ir * n3)] + prev_u[current - (ir * n3)]) / d2Squared) +
                ((prev_u[current + (ir * n2 * n3)] + prev_u[current - (ir * n2 * n3)]) / d1Squared));
        }

        value *= dtSquared * vel_model[current] * vel_model[current];
        next_u[current] = 2.0 * prev_u[current] - next_u[current] + value;
    }
}

int forward(f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
            f_type d1, f_type d2, f_type d3, f_type dt, int n1, int n2, int n3,
            int iterations, int stencil_radius,
            int block_size_1, int block_size_2, int block_size_3)
{
    f_type d1Squared = d1 * d1;
    f_type d2Squared = d2 * d2;
    f_type d3Squared = d3 * d3;
    f_type dtSquared = dt * dt;

    size_t grid_size = n1 * n2 * n3 * sizeof(f_type);
    size_t coeff_size = (stencil_radius + 1) * sizeof(f_type);

    // Allocate device memory
    f_type *d_prev_u, *d_next_u, *d_vel_model, *d_coefficient;
    CUDA_CHECK(cudaMalloc(&d_prev_u, grid_size));
    CUDA_CHECK(cudaMalloc(&d_next_u, grid_size));
    CUDA_CHECK(cudaMalloc(&d_vel_model, grid_size));
    CUDA_CHECK(cudaMalloc(&d_coefficient, coeff_size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_prev_u, prev_u, grid_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_u, next_u, grid_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_model, vel_model, grid_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coefficient, coefficient, coeff_size, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 blockSize(block_size_3, block_size_2, block_size_1);
    dim3 gridSize(
        (n3 - 2 * stencil_radius + blockSize.x - 1) / blockSize.x,
        (n2 - 2 * stencil_radius + blockSize.y - 1) / blockSize.y,
        (n1 - 2 * stencil_radius + blockSize.z - 1) / blockSize.z
    );

    // Time iteration loop
    for (int t = 0; t < iterations; t++) {
        wave_propagation_kernel<<<gridSize, blockSize>>>(
            d_prev_u, d_next_u, d_vel_model, d_coefficient,
            d1Squared, d2Squared, d3Squared, dtSquared,
            n1, n2, n3, stencil_radius
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers for next iteration
        f_type *temp = d_next_u;
        d_next_u = d_prev_u;
        d_prev_u = temp;
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(prev_u, d_prev_u, grid_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(next_u, d_next_u, grid_size, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_prev_u);
    cudaFree(d_next_u);
    cudaFree(d_vel_model);
    cudaFree(d_coefficient);

    return 0;
}
