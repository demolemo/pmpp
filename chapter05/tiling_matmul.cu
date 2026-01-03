#include <iostream>


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
    float* A, float* B, float* C, int N
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
    float* A, float *B, float *C,
    int N, int TILE_SIZE
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


