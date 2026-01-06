// very naive reduction
// we first implement the version that operates on one block
__global__ void naiveReductionSum(float *input, float *output) {
    // choose every second element so that on the first iteration
    // of the loop we add every second element, on second every 4th etc.
    const int idx = threadIdx.x * 2;

    //  first iteration 1 0 1 0 1 0 1 0 
    // second iteration 1 0 0 0 1 0 0 0 
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (idx % stride == 0) {
            input[idx] += input[idx + stride];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *output = input[0];
    }
}

// in terms of idea this kernel is completely the same to the kernel above
// it just has better memory access pattern
__global__ void reductionSumMemPattern(float* input, float *output) {
    const int idx = threadIdx.x;

    //  first iteration 1 1 1 1 0 0 0 0
    // second iteration 1 1 0 0 0 0 0 0
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (idx < stride == 0) {
            input[idx] += inpu[idx + stride];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *output = input[0];
    }
}

// here the idea is the same as in the last kernel, we just use shared memory
// to store intermediate results to reduce overall global memory traffic
// also this allows us to not modify the input array which is good practice
__global__ void reductionSumMemPatternSharedMem(const float *input, float* output) {
    __shared__ float input_s[blockDim.x];

    const int idx = threadIdx.x;
    input_s[idx] = input[idx] + input[idx + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        // we are synching in the beginning of the loop to avoid racing to shared memory
        __syncthreads();
        if (idx < stride) {
            input_s[idx] = input_s[idx] + input_s[idx + stride];
        }
    }

    if (idx == 0) {
        *output = input_s[0];
    }
}

// creating a global recution tree that scales to an arbitrary number of threadblocks
// but is very insefficient in terms of hardware utilization on the final stages
__global__ void reductionSumSharedMemBlocks(const float* input, float* output) {
    __shared__ float input_s[blockDim.x];

    unsigned int GLOBAL_STRIDE = 2 * blockDim.x * blockIdx.x;
    const int idx = GLOBAL_STRIDE + threadIdx.x;
    input_s[idx] = input[idx] + input[idx + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (idx < GLOBAL_STRIDE + stride) {
            input_s[idx] = input_s[idx] + input_s[idx + stride];
        }
    }

    if (idx % GLOBAL_STRIDE == 0) {
        atomicAdd(output, input_s[0]);
    }
}

