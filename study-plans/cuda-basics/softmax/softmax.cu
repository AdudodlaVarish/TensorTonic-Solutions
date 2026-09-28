#include <cuda_runtime.h>
#include <float.h>

__global__ void reduce_max(const float* in, float* out, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float v = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) v = fmaxf(v, in[i]);
    s[tid] = v;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) s[tid] = fmaxf(s[tid], s[tid + st]);
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}

__global__ void exp_and_sum(const float* in, float* out, const float* maxv, float* sumv, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float m = *maxv;
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float e = expf(in[i] - m);
        out[i] = e;
        local += e;
    }
    s[tid] = local;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0) *sumv = s[0];
}

__global__ void normalize(float* out, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] /= *sumv;
}

extern "C" void solve(const float* input, float* output, int N) {
    float *d_max, *d_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    reduce_max<<<1, 256>>>(input, d_max, N);
    exp_and_sum<<<1, 256>>>(input, output, d_max, d_sum, N);
    int threads = 256, blocks = (N + threads - 1) / threads;
    normalize<<<blocks, threads>>>(output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_max);
    cudaFree(d_sum);
}
