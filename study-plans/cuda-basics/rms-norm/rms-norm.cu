#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    extern __shared__ float sdata[];

    float sum_sq = 0.0f;
    for (int j = tid; j < N; j += blockSize) {
        float val = row_in[j];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sdata[0] / (float)N;
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    for (int j = tid; j < N; j += blockSize) {
        row_out[j] = row_in[j] * inv_rms * gamma[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    int blocks = M;
    size_t shared_mem_bytes = threads * sizeof(float);

    rms_norm_kernel<<<blocks, threads, shared_mem_bytes>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}