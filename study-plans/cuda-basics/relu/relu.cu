#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
      output[i] = fmaxf(input[i], 0);
}
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}