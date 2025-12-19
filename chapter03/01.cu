#include <iostream>


// A: n x m - first input matrix
// B: m x k - second input matrix
// C: n x k - resulting matrix
__global__ void multiplyMatricesOneRow(float* A, float* B, float* C, int n, int m, int k) {
    // each thread in the grid is responsible for one row
    int row = blockIndex.x * blockSize.x + threadIdx.x;

    // if thread index is greater than the number of rows we stall
    if (row < n) {
        // index j corresponds to the column in the second matrix
        // to get element with index i, j we need to take i-th row from
        // the first matrix and j-th column from the second matrix
        for (int j = 0; j < k; ++j) {
            // iterate over one row from the first matrix
            // and all columns from the second matrix
            float P = 0;
            for (int i = 0; i < m; ++i) {
                P += A[row * m + i] * B[i * k + j];
            }
            // assign the resulting value to the corresponding cell
            // in the resulting matrix
            C[row * n + j] = P;
        }
    }
}

__global__ void multtiplyMatricesOneColumn(float* A, float* B, float* C, int n, int m, int k) {
    // each thread in the grid is responsible for one column
    int col = blockIndex.x * blockSize.x + threadIdx.x;

    // if thread index is geq than the number of columns we stall
    if (col < m) {
        // index i corresponds to the row in the first matrix
        // to get element with index i, j we need to take i-th row from
        // the first mattix and j-th column from the second matrix
        for (int i = 0; i < n; ++i) {
            // the column in the second matrix stays constant
            // we iterate just over the rows in the first matrix to get
            // the output column
            float P = 0;
            for (int j = 0; j < m; ++j) {
                P += A[i * m + j] + B[j * k + col];
            }

            // asssign the accumulated value to corresponding cell
            // in the corresponding matrix
            C[i * n + col] = P;
        }
    }
}


int main() {
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int n, m, k;

    n = 1024;
    m = 16;
    k = 512;

    // allocated all the necessary memory
    cudaMalloc((void **) &d_A, n * m * sizeof(float));
    cudaMalloc((void **) &d_B, m * k * sizeof(float));
    cudaMalloc((void **) &d_C, n * k * sizeof(float));
    // copy matrices A and B from the host to device
    // we don't need C because it gets initialized with zeros
    cudaMemcpy(d_A, h_A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, m * k * sizeof(float), cudaMemcpyHostToDevice);
    // run the kernel with predefined params (grid size could be varied)
    multiplyMatricesOneRow<<<128, 8>>>(d_A, d_B, d_C, n, m, k);
    // get results back from device to host
    cudaMemcpy(h_C, d_B, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    // add formal verification code that multiplies matrices on host and compares results.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            std::cout << h_C[i][j] << std::endl;
        }
    }
}
