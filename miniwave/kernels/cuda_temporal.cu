#include "cuda_kernel.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            return -1;                                                        \
        }                                                                     \
    } while (0)

__device__ __forceinline__
int idx3d(int i, int j, int k, int n2, int n3) {
    return (i * n2 + j) * n3 + k;
}

// Kernel para 1 passo de tempo (fallback sem grid-wide sync)
__global__
void wave_kernel_step(
    const f_type *d_prev,
    f_type *d_next,
    const f_type *d_vel,
    const f_type *d_coeff,
    f_type d1S, f_type d2S, f_type d3S, f_type dtS,
    int n1, int n2, int n3,
    int stencil_radius)
{
    const int r = stencil_radius;

    int i = blockIdx.z * blockDim.z + threadIdx.z + r;
    int j = blockIdx.y * blockDim.y + threadIdx.y + r;
    int k = blockIdx.x * blockDim.x + threadIdx.x + r;

    if (i < r || i >= n1 - r ||
        j < r || j >= n2 - r ||
        k < r || k >= n3 - r) {
        return;
    }

    int g = idx3d(i, j, k, n2, n3);
    f_type vel = d_vel[g];
    f_type vel2dt2 = dtS * vel * vel;

    f_type center = d_prev[g];

    // Calcula Laplaciano (usa u^t = d_prev)
    f_type value = d_coeff[0] * (center / d1S + center / d2S + center / d3S);

    for (int ir = 1; ir <= r; ++ir) {
        f_type xp = d_prev[idx3d(i,     j,     k + ir, n2, n3)];
        f_type xn = d_prev[idx3d(i,     j,     k - ir, n2, n3)];
        f_type yp = d_prev[idx3d(i,     j + ir, k,     n2, n3)];
        f_type yn = d_prev[idx3d(i,     j - ir, k,     n2, n3)];
        f_type zp = d_prev[idx3d(i + ir, j,     k,     n2, n3)];
        f_type zn = d_prev[idx3d(i - ir, j,     k,     n2, n3)];

        value += d_coeff[ir] * (
            (xp + xn) / d3S +
            (yp + yn) / d2S +
            (zp + zn) / d1S
        );
    }

    value *= vel2dt2;
    // u_tm1 = d_next (entrada contém u^{t-1})
    f_type u_new = 2.0 * center - d_next[g] + value;
    d_next[g] = u_new;
}

// Kernel com grid-wide sync (cooperative groups) para temporal blocking
__global__
void wave_kernel_temporal(
    f_type *d_prev,
    f_type *d_next,
    const f_type *d_vel,
    const f_type *d_coeff,
    f_type d1S, f_type d2S, f_type d3S, f_type dtS,
    int n1, int n2, int n3,
    int iterations,
    int stencil_radius)
{
    cg::grid_group grid = cg::this_grid();
    const int r = stencil_radius;

    int i = blockIdx.z * blockDim.z + threadIdx.z + r;
    int j = blockIdx.y * blockDim.y + threadIdx.y + r;
    int k = blockIdx.x * blockDim.x + threadIdx.x + r;

    if (i < r || i >= n1 - r ||
        j < r || j >= n2 - r ||
        k < r || k >= n3 - r) {
        return;
    }

    int g = idx3d(i, j, k, n2, n3);
    f_type vel = d_vel[g];
    f_type vel2dt2 = dtS * vel * vel;

    for (int t = 0; t < iterations; ++t) {
        f_type *u_t   = (t % 2 == 0) ? d_prev : d_next;
        f_type *u_tm1 = (t % 2 == 0) ? d_next : d_prev;

        f_type center = u_t[g];

        f_type value = d_coeff[0] * (center / d1S + center / d2S + center / d3S);

        for (int ir = 1; ir <= r; ++ir) {
            f_type xp = u_t[idx3d(i,     j,     k + ir, n2, n3)];
            f_type xn = u_t[idx3d(i,     j,     k - ir, n2, n3)];
            f_type yp = u_t[idx3d(i,     j + ir, k,     n2, n3)];
            f_type yn = u_t[idx3d(i,     j - ir, k,     n2, n3)];
            f_type zp = u_t[idx3d(i + ir, j,     k,     n2, n3)];
            f_type zn = u_t[idx3d(i - ir, j,     k,     n2, n3)];

            value += d_coeff[ir] * (
                (xp + xn) / d3S +
                (yp + yn) / d2S +
                (zp + zn) / d1S
            );
        }

        value *= vel2dt2;
        f_type u_new = 2.0 * center - u_tm1[g] + value;

        if (t % 2 == 0) {
            d_next[g] = u_new;
        } else {
            d_prev[g] = u_new;
        }

        // sincronização global entre todos os blocos
        grid.sync();
    }
}

extern "C"
int forward(f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
            f_type d1, f_type d2, f_type d3, f_type dt, int n1, int n2, int n3,
            int iterations, int stencil_radius,
            int block_size_1, int block_size_2, int block_size_3)
{
    // Verifica suporte a cooperative launch
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    bool coop_supported = prop.cooperativeLaunch;

    const int r = stencil_radius;

    size_t grid_elems = (size_t)n1 * n2 * n3;
    size_t grid_bytes = grid_elems * sizeof(f_type);
    size_t coeff_bytes = (stencil_radius + 1) * sizeof(f_type);

    f_type *d_prev, *d_next, *d_vel, *d_coeff;

    CUDA_CHECK(cudaMalloc(&d_prev,  grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_next,  grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel,   grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_coeff, coeff_bytes));

    CUDA_CHECK(cudaMemcpy(d_prev,  prev_u,      grid_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next,  next_u,      grid_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel,   vel_model,   grid_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeff, coefficient, coeff_bytes, cudaMemcpyHostToDevice));

    dim3 block(block_size_3, block_size_2, block_size_1);
    dim3 grid(
        (n3 - 2 * r + block.x - 1) / block.x,
        (n2 - 2 * r + block.y - 1) / block.y,
        (n1 - 2 * r + block.z - 1) / block.z
    );

    // Estima se cabe em cooperative launch
    long long total_blocks = (long long)grid.x * grid.y * grid.z;
    long long max_coop_blocks = (long long)prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
    bool can_coop = coop_supported && (total_blocks <= max_coop_blocks);

    f_type d1S = d1 * d1;
    f_type d2S = d2 * d2;
    f_type d3S = d3 * d3;
    f_type dtS = dt * dt;

    f_type *d_final;
    
    if (can_coop) {
        void *args[] = { &d_prev, &d_next, &d_vel, &d_coeff,
                         &d1S, &d2S, &d3S, &dtS,
                         &n1, &n2, &n3, &iterations, &stencil_radius };
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)wave_kernel_temporal, grid, block, args));
        CUDA_CHECK(cudaDeviceSynchronize());
        // No kernel cooperativo: se iterations é par, último write foi em d_prev
        // se ímpar, último write foi em d_next
        d_final = (iterations % 2 == 0) ? d_prev : d_next;
    } else {
        // Fallback: um passo por kernel (correto, porém mais launches)
        for (int t = 0; t < iterations; ++t) {
            wave_kernel_step<<<grid, block>>>(
                d_prev, d_next, d_vel, d_coeff,
                d1S, d2S, d3S, dtS,
                n1, n2, n3, r
            );
            CUDA_CHECK(cudaGetLastError());
            // swap após cada iteração
            f_type *tmp = d_prev; d_prev = d_next; d_next = tmp;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        // No fallback: após N swaps, se N é par, resultado está em d_prev
        // se N é ímpar, resultado está em d_next
        d_final = (iterations % 2 == 0) ? d_prev : d_next;
    }

    CUDA_CHECK(cudaMemcpy(prev_u, d_final, grid_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(next_u, d_final, grid_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_prev);
    cudaFree(d_next);
    cudaFree(d_vel);
    cudaFree(d_coeff);

    return 0;
}
