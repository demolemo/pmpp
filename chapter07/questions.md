# CUDA Exercises - Chapter 7

## Exercise 1

Calculate the P[0] value in Fig. 7.3.

## Exercise 2

Consider performing a 1D convolution on array N = {4, 1, 3, 2, 3} with filter F = {2, 1, 4}. What is the resulting output array?

## Exercise 3

What do you think the following 1D convolution filters are doing?

**a.** [0 1 0]

**b.** [0 0 1]

**c.** [1 0 0]

**d.** [−1/2 0 1/2]

**e.** [1/3 1/3 1/3]

## Exercise 4

Consider performing a 1D convolution on an array of size N with a filter of size M:

**a.** How many ghost cells are there in total?

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?

## Exercise 5

Consider performing a 2D convolution on a square matrix of size N × N with a square filter of size M × M:

**a.** How many ghost cells are there in total?

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?

## Exercise 6

Consider performing a 2D convolution on a rectangular matrix of size N₁ × N₂ with a rectangular mask of size M₁ × M₂:

**a.** How many ghost cells are there in total?

**b.** How many multiplications are performed if ghost cells are treated as multiplications (by 0)?

**c.** How many multiplications are performed if ghost cells are not treated as multiplications?

## Exercise 7

Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size N × N with a filter of size M × M using an output tile of size T × T.

**a.** How many thread blocks are needed?

**b.** How many threads are needed per block?

**c.** How much shared memory is needed per block?

**d.** Repeat the same questions if you were using the kernel in Fig. 7.15.

## Exercise 8

Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.

## Exercise 9

Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.

## Exercise 10

Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.
