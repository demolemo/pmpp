#include <iostream>


// general case tiled matmul with tiles loaded in the shared memory
// A /in N x M, B /in M x K, C /in N x K
// for simplicity we assume that N, M, K are all divisible by TILE_SIZE
__global__ tiledMatmul(
        float* A, float *B, float *C,
        int N, int, M, int K,
        int TILE_SIZE
        )
