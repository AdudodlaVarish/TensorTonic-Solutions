#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float sigmoid_ = 1 + exp(-input[i]);
        float sigmoid = 1 / sigmoid_;
        output[i] = sigmoid;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}