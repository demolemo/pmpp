## Question 1
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

- **a.** Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.
- **b.** Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.
- **c.** Analyze the pros and cons of each of the two kernel designs.

## Question 2
A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, A[i] = ∑ʲ B[i][j] + C[j]. For simplicity we will handle only square matrices whose elements are single-precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

## Question 3
Consider the following CUDA kernel and the corresponding host function that calls it:
```cuda
01      __global__ void foo_kernel(float* a, float* b, unsigned int M,
        unsigned int N) {
02          unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
03          unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
04          if(row < M && col < N) {
05              b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06          }
07      }
08      void foo(float* a_d, float* b_d) {
09          unsigned int M = 150;
10          unsigned int N = 300;
11          dim3 bd(16, 32);
12          dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
13          foo_kernel <<< gd, bd >>>(a_d, b_d, M, N);
14      }
```

- **a.** What is the number of threads per block?
- **b.** What is the number of threads in the grid?
- **c.** What is the number of blocks in the grid?
- **d.** What is the number of threads that execute the code on line 05?

## Question 4
Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:

- **a.** If the matrix is stored in row-major order.
- **b.** If the matrix is stored in column-major order.

## Question 5
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.
