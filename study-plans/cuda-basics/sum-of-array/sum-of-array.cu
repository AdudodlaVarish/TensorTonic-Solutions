#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    
    float val = 0.0f;
    if (idx < N) {
        val = input[idx];
    }
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float);
    
    // Initialize the result buffer to 0 before accumulating
    cudaMemset(result, 0, sizeof(float));
    
    sum_kernel<<<blocks, threads, sharedMemSize>>>(input, result, N);
    cudaDeviceSynchronize();
}