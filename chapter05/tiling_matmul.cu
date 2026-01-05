#include <iostream>

// TILE_SIZE should be known on compilation to work correctly with shared memory
#define TILE_SIZE 16

// square case tiled matmul 
// A /in N x N, B /in N x N, C /in N x N
// for the simple case we assume that N % TILE_SIZE == 0

// algorithm:
// take a tile from the first matrix
// take a tile from the second matrix
// do the naive matmul
// index corresponds to what?
//
// in the first implementation each thread is responsible for calculating one tile completely

// each thread calculates one output element without tiling
// the problem with this approch is too low utilization of the GPU. we are doing too little ops and too many reads
__global__ void naiveMatmul(
    const float* A, const float* B, float* C, int N
    ){
    const int row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    float P_val = 0;
    for (int i = 0; i < N; ++i) {
        P_val += A[row_idx * N + i] * B[i * N + col_idx];
    }
    C[row_idx * N + col_idx] = P_val;
}

// each thread calculates the whole tile in the output matrix
// this approach is a little bit better than the last because we decrease the number of memory reads using coalescing
__global__ void naiveTiledMatmul(
    const float* A, const float *B, float *C,
    int N
    ) {
    // index of the tile that we calculate
    const int row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    for (int elem_row = 0; elem_row < TILE_SIZE; ++elem_row) {
        for (int elem_col = 0; elem_col < TILE_SIZE; ++elem_col) {

            float P_val = 0;
            for (int elem_idx = 0; elem_idx < N; ++elem_idx) {
                // get relative index of the element to the start of the grid
                P_val += A[N * (row_idx * TILE_SIZE + elem_row) + elem_idx] * B[N * elem_idx + TILE_SIZE * col_idx + elem_col];
            }
            C[N * (row_idx * TILE_SIZE + elem_row) + col_idx * TILE_SIZE + elem_col] = P_val;
        }
    }
}

// errors while working on this kernel
// not drawing a picture from the start
// incorrectly naming column and row because thought about them in the order of occurance
// used flat indexing inside of the inner loop
// went through indexing in shared matrices by vibe, should have deconstructed them and worked with them by hand
__global__ void sharedMemTiledMatmul (
        const float *A, const float *B, float *C, int N
        ) {
    // find the coordinates of the thread in the grid
    const int bx = blockIdx.x; const int tx = threadIdx.x;
    const int by = blockIdx.y; const int ty = threadIdx.y;

    // inititialize shared memory for loading tiles from matrices A and B
    __shared__ float Ta[TILE_SIZE][TILE_SIZE];
    __shared__ float Tb[TILE_SIZE][TILE_SIZE];

    // find out the index of output element that is being calculated relative to the start of the matrix
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // initialize the element to hold the results of calculations
    float P_val = 0;
    for (int tile_idx = 0; tile_idx < N / TILE_SIZE; ++tile_idx) {
        // first axis calculations:
        // by * TILE_SIZE * N - how many rows we need to skip to get to the start of the tile
        // ty * N - on which row we are currently
        // second axis calcualtions:
        // tile_idx * TILE_SIZE - how far we are from the beginning of the row in absolute terms
        // tx - the index of the concrete element in this row
        Ta[ty][tx] = A[(by * TILE_SIZE * N + ty * N) + (tile_idx * TILE_SIZE + tx)];

        // first axis calculations:
        // tile_idx * TILE_SIZE * N - how many rows we need to skip to get to the start of the tile
        // ty * N - on which row we are currently
        // second axis calculations:
        // bx * TILE_SIZE - how far tile starts from the beginning of the row in absolute terms
        // tx - the index of the concrete element in this row
        Tb[ty][tx] = B[(tile_idx * TILE_SIZE * N + ty * N) + (bx * TILE_SIZE + tx)];
        __syncthreads();

        // for the inner loop we abuse the fact that Ta and Tb are 2D arrays
        for (int i = 0; i < TILE_SIZE; i++) {
            P_val += Ta[ty][i] * Tb[i][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = P_val;
}

// for this kernel we assume that N, M, K are all divisible by TILE_SIZE, that is
// N % TILE_SIZE == 0, M % TILE_SIZE == 0, K % TILE_SIZE == 0
// it seems right after writing the kernel that there is no need in N
__global__ void sharedMemTiledMatmulArbDivSizes(
        const float *A, const float *B, float *C,
        int N, int M, int K
        ) {
    const int bx = blockIdx.x; const int tx = threadIdx.x;
    const int by = blockIdx.y; const int ty = threadIdx.y;

    __shared__ float Ta[TILE_SIZE][TILE_SIZE];
    __shared__ float Tb[TILE_SIZE][TILE_SIZE];

    const int col = bx * TILE_SIZE + tx;
    const int row = by * TILE_SIZE + ty;

    float P_val = 0;
    for (int tile_idx = 0; tile_idx < M / TILE_SIZE; ++tile_idx) {
        Ta[ty][tx] = A[(by * TILE_SIZE * M + ty * M) + (tile_idx * TILE_SIZE + tx)];
        Tb[ty][tx] = B[(tile_idx * TILE_SIZE * K + ty * K) + col];
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            P_val += Ta[ty][i] * Tb[i][tx];
        }
        __syncthreads();
    }
    C[row * K + col] = P_val;
}


// for this kernel we make no assumptions N, M, K - they could be arbitrary
__global__ void sharedMemTiledMatmulArbSizes(
        const float *A, const float *B, float *C,
        int N, int M, int K
        ) {
    const int bx = blockIdx.x; const int tx = threadIdx.x;
    const int by = blockIdx.y; const int ty = threadIdx.y;

    const int col = bx * TILE_SIZE + tx;
    const int row = by * TILE_SIZE + ty;

    __shared__ float Ta[TILE_SIZE][TILE_SIZE];
    __shared__ float Tb[TILE_SIZE][TILE_SIZE];

    float P_val = 0;
    for (int tile_idx = 0; tile_idx < (M + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        if (by * TILE_SIZE + ty < N && tile_idx * TILE_SIZE + tx < M) {
            Ta[ty][tx] = A[(by * TILE_SIZE * M + ty * M) + (tile_idx * TILE_SIZE + tx)];
        } else {
            Ta[ty][tx] = 0;
        }
        if ((tile_idx * TILE_SIZE + ty) < M && col < K) {
            Tb[ty][tx] = B[(tile_idx * TILE_SIZE * K + ty * K) + col];
        } else {
            Tb[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            P_val += Ta[ty][i] * Tb[i][tx];
        }
        __syncthreads();
    }

    if (row < N && col < K) {
        C[row * K + col] = P_val;
    }
}
