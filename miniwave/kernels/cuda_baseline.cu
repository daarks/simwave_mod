#include "cuda_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err!=cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); return -1; } } while(0)

__global__
void wave_kernel_baseline(
    const f_type *prev_u, f_type *next_u, const f_type *vel_model, const f_type *coefficient,
    f_type d1Squared, f_type d2Squared, f_type d3Squared, f_type dtSquared,
    int n1, int n2, int n3, int stencil_radius)
{
    // blockIdx.z -> i (n1), blockIdx.y -> j (n2), blockIdx.x -> k (n3)
    int i = blockIdx.z * blockDim.z + threadIdx.z + stencil_radius;
    int j = blockIdx.y * blockDim.y + threadIdx.y + stencil_radius;
    int k = blockIdx.x * blockDim.x + threadIdx.x + stencil_radius;

    if (i >= n1 - stencil_radius || j >= n2 - stencil_radius || k >= n3 - stencil_radius)
        return;

    int current = (i * n2 + j) * n3 + k;

    f_type value = coefficient[0] * (prev_u[current]/d1Squared +
                                     prev_u[current]/d2Squared +
                                     prev_u[current]/d3Squared);

    for (int ir = 1; ir <= stencil_radius; ++ir) {
        value += coefficient[ir] * (
            (prev_u[current + ir] + prev_u[current - ir]) / d3Squared +
            (prev_u[current + ir * n3] + prev_u[current - ir * n3]) / d2Squared +
            (prev_u[current + ir * n2 * n3] + prev_u[current - ir * n2 * n3]) / d1Squared
        );
    }

    value *= dtSquared * vel_model[current] * vel_model[current];
    next_u[current] = 2.0 * prev_u[current] - next_u[current] + value;
}

extern "C"
int forward(f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
            f_type d1, f_type d2, f_type d3, f_type dt, int n1, int n2, int n3,
            int iterations, int stencil_radius,
            int block_size_1, int block_size_2, int block_size_3)
{
    size_t grid_elems = (size_t)n1 * n2 * n3;
    size_t grid_bytes = grid_elems * sizeof(f_type);
    size_t coeff_bytes = (stencil_radius + 1) * sizeof(f_type);

    // device pointers
    f_type *d_prev, *d_next, *d_vel, *d_coeff;
    CUDA_CHECK(cudaMalloc(&d_prev, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_next, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_coeff, coeff_bytes));

    CUDA_CHECK(cudaMemcpy(d_prev, prev_u, grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, next_u, grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, vel_model, grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeff, coefficient, coeff_bytes, cudaMemcpyHostToDevice));

    // launch config (note ordering: x->k, y->j, z->i)
    dim3 block(block_size_3, block_size_2, block_size_1);
    dim3 grid( (n3 - 2*stencil_radius + block.x - 1)/block.x,
               (n2 - 2*stencil_radius + block.y - 1)/block.y,
               (n1 - 2*stencil_radius + block.z - 1)/block.z );

    f_type d1S = d1*d1, d2S = d2*d2, d3S = d3*d3, dtS = dt*dt;

    for (int t=0; t<iterations; ++t) {
        wave_kernel_baseline<<<grid, block>>>(
            d_prev, d_next, d_vel, d_coeff, d1S, d2S, d3S, dtS, n1, n2, n3, stencil_radius
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // swap
        f_type *tmp = d_prev; d_prev = d_next; d_next = tmp;
    }

    CUDA_CHECK(cudaMemcpy(prev_u, d_prev, grid_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(next_u, d_next, grid_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_prev); cudaFree(d_next); cudaFree(d_vel); cudaFree(d_coeff);
    return 0;
}
