#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>

__device__ inline void update_best(float& best_val, int& best_idx, float val, int idx) {
    if (val > best_val || (val == best_val && idx < best_idx)) {
        best_val = val;
        best_idx = idx;
    }
}

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = -FLT_MAX;
    int index = INT_MAX;

    if (idx < N) {
        val = input[idx];
        index = idx;
    }

    s_vals[tid] = val;
    s_idxs[tid] = index;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v1 = s_vals[tid];
            int i1 = s_idxs[tid];
            float v2 = s_vals[tid + s];
            int i2 = s_idxs[tid + s];

            if (v2 > v1 || (v2 == v1 && i2 < i1)) {
                s_vals[tid] = v2;
                s_idxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;

    float best_val = -FLT_MAX;
    int best_idx = INT_MAX;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        int idx = block_idxs[i];
        update_best(best_val, best_idx, v, idx);
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v1 = s_vals[tid];
            int i1 = s_idxs[tid];
            float v2 = s_vals[tid + s];
            int i2 = s_idxs[tid + s];

            if (v2 > v1 || (v2 == v1 && i2 < i1)) {
                s_vals[tid] = v2;
                s_idxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s_idxs[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}