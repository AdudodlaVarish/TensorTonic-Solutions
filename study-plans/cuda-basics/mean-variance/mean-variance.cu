#include <cuda_runtime.h>

__global__ void reduce_sum_kernel(const float* input, float* block_sums, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    if (idx < N) {
        sum = input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void finalize_mean_kernel(const float* block_sums, float* mean_out, int num_blocks, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean_out[0] = sdata[0] / (float)N;
    }
}

__global__ void reduce_var_kernel(const float* input, const float* mean_out, float* block_vars, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float mean = mean_out[0];
    float sum_sq_diff = 0.0f;
    if (idx < N) {
        float diff = input[idx] - mean;
        sum_sq_diff = diff * diff;
    }
    sdata[tid] = sum_sq_diff;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vars[blockIdx.x] = sdata[0];
    }
}

__global__ void finalize_var_kernel(const float* block_vars, float* var_out, int num_blocks, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_vars[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        var_out[0] = sdata[0] / (float)N;
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_sums = nullptr;
    float* block_vars = nullptr;
    cudaMalloc(&block_sums, blocks * sizeof(float));
    cudaMalloc(&block_vars, blocks * sizeof(float));

    reduce_sum_kernel<<<blocks, threads>>>(input, block_sums, N);
    finalize_mean_kernel<<<1, threads>>>(block_sums, mean_out, blocks, N);

    reduce_var_kernel<<<blocks, threads>>>(input, mean_out, block_vars, N);
    finalize_var_kernel<<<1, threads>>>(block_vars, var_out, blocks, N);

    cudaDeviceSynchronize();

    cudaFree(block_sums);
    cudaFree(block_vars);
}