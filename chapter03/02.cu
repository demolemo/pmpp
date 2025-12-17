#include <iostream>

// A: n x 1
// B: n x m
// C: m x 1
__global__ void matrixVecMul(
        float *A, float *B, float *C, 
        int n, int m
    ) {
    // one thread corresponds to one output element in the final vector
    int idx = blockDim.x * blockSize.x + threadIdx.x;

    // check that idx is not out of bounds
    if (idx < n) {
        // accumulate the value according to the formula
        float P = 0;
        for (int i = 0; i < m; ++i) {
            P += B[i * n + idx] * C[idx];
        }
        A[idx] = P;
    }
}

int main() {
    // call this kernel with appropriate params
    return 0;
}
