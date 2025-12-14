#include "cuda_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

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
int sIndex_dev(int lz, int ly, int lx, int sx, int sy) {
    return (lz * sy + ly) * sx + lx;
}

__global__
void wave_kernel_shared(
    const f_type *d_prev,
    f_type       *d_next,
    const f_type *d_vel,
    const f_type *d_coeff,
    f_type d1S, f_type d2S, f_type d3S, f_type dtS,
    int n1, int n2, int n3,
    int stencil_radius)
{
    extern __shared__ f_type s_prev[];

    const int r = stencil_radius;

    const int bx = blockDim.x;
    const int by = blockDim.y;
    const int bz = blockDim.z;

    const int sx = bx + 2 * r;
    const int sy = by + 2 * r;
    const int sz = bz + 2 * r;

    const int threads_per_block = bx * by * bz;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = (tz * by + ty) * bx + tx;

    // Carrega tile completo para shared memory
    const int total_shared_elems = sx * sy * sz;

    for (int idx = tid; idx < total_shared_elems; idx += threads_per_block) {
        int tmp = idx;
        int lx  = tmp % sx;
        tmp    /= sx;
        int ly  = tmp % sy;
        int lz  = tmp / sy;

        // Coordenadas globais (pode estar no halo)
        // lz vai de 0 a sz-1 = 0 a (bz+2r-1), mapeia para blockIdx.z*bz até blockIdx.z*bz+bz+2r-1
        int i = blockIdx.z * bz + lz;
        int j = blockIdx.y * by + ly;
        int k = blockIdx.x * bx + lx;

        f_type val = 0.0;

        // Carrega só se estiver dentro do domínio
        if (i >= 0 && i < n1 && j >= 0 && j < n2 && k >= 0 && k < n3) {
            int gidx = (i * n2 + j) * n3 + k;
            val = d_prev[gidx];
        }

        s_prev[sIndex_dev(lz, ly, lx, sx, sy)] = val;
    }

    __syncthreads();

    // Coordenadas globais da thread (região interior)
    int gi = blockIdx.z * bz + tz + r;
    int gj = blockIdx.y * by + ty + r;
    int gk = blockIdx.x * bx + tx + r;

    // CRITICAL FIX: Boundary check completo (superior E inferior)
    if (gi < r || gi >= n1 - r ||
        gj < r || gj >= n2 - r ||
        gk < r || gk >= n3 - r) {
        return;
    }

    // Coordenadas locais em shared (deslocadas por r)
    int lz = tz + r;
    int ly = ty + r;
    int lx = tx + r;

    int gidx = (gi * n2 + gj) * n3 + gk;

    // Ponto central
    f_type center = s_prev[sIndex_dev(lz, ly, lx, sx, sy)];

    // Laplaciano
    f_type value = d_coeff[0] * (center / d1S + center / d2S + center / d3S);

    for (int ir = 1; ir <= r; ++ir) {
        f_type xp = s_prev[sIndex_dev(lz,     ly,     lx + ir, sx, sy)];
        f_type xn = s_prev[sIndex_dev(lz,     ly,     lx - ir, sx, sy)];
        f_type yp = s_prev[sIndex_dev(lz,     ly + ir, lx,     sx, sy)];
        f_type yn = s_prev[sIndex_dev(lz,     ly - ir, lx,     sx, sy)];
        f_type zp = s_prev[sIndex_dev(lz + ir, ly,     lx,     sx, sy)];
        f_type zn = s_prev[sIndex_dev(lz - ir, ly,     lx,     sx, sy)];

        value += d_coeff[ir] * (
            (xp + xn) / d3S +
            (yp + yn) / d2S +
            (zp + zn) / d1S
        );
    }

    f_type vel = d_vel[gidx];
    value *= dtS * vel * vel;

    d_next[gidx] = 2.0 * center - d_next[gidx] + value;
}

extern "C"
int forward(f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
            f_type d1, f_type d2, f_type d3, f_type dt, int n1, int n2, int n3,
            int iterations, int stencil_radius,
            int block_size_1, int block_size_2, int block_size_3)
{
    const int r = stencil_radius;

    size_t grid_elems  = (size_t)n1 * n2 * n3;
    size_t grid_bytes  = grid_elems * sizeof(f_type);
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

    int sx = block.x + 2 * r;
    int sy = block.y + 2 * r;
    int sz = block.z + 2 * r;

    size_t shared_bytes = (size_t)sx * sy * sz * sizeof(f_type);

    f_type d1S = d1 * d1;
    f_type d2S = d2 * d2;
    f_type d3S = d3 * d3;
    f_type dtS = dt * dt;

    for (int t = 0; t < iterations; ++t) {
        wave_kernel_shared<<<grid, block, shared_bytes>>>(
            d_prev, d_next, d_vel, d_coeff,
            d1S, d2S, d3S, dtS,
            n1, n2, n3, stencil_radius
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap buffers
        f_type *tmp = d_prev;
        d_prev = d_next;
        d_next = tmp;
    }

    CUDA_CHECK(cudaMemcpy(prev_u, d_prev, grid_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(next_u, d_next, grid_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_prev);
    cudaFree(d_next);
    cudaFree(d_vel);
    cudaFree(d_coeff);

    return 0;
}
