## Question 1
We can use coalesced memory accesses to reduce global memory traffic. This is done if we have two matrices in row-major order and we do vanilla addition.

## Question 2
Drawing goes here

## Question 3
One thread could start loading elements for the next tile before the current one is processed. This leads to incorrect results for calculations. A complete mess.

## Question 4
To reduce overall global memory traffic. Many CUDA optimizations are trying to make a problem compute bound instead of being memory bound.

## Question 5
We reduce memory traffic by 32 times. (for each element we needed to load it n times now we need to load it n / 32 times).

## Question 6
512 * 1000 versions of the variable, one for each thread

## Question 7
1000 versions of the variable, one for each block

## Question 8
1. N times (to get final matrix we must take each row and multiply it with each column)
2. N / T (now we need to multiply each row tile with each column tile which leads to reducdtion of memory traffic by the factor of T)

## Question 9
I'm too lazy for this

## Question 10
a. LMAO, for None
b. We need an intermediate variable to store one of the values. Also, there is no point in using so many threads, we can do the transpose operation using $N^2/2$ threads. Assymptotically it's the same but it's CUDA god damn it.

## Question 11
1. 128 * 8 = 1024 (for each thread we have a different version of variable `i`)
2. 128 * 8 = 1024 (for each thread we have a different version of variable `x[]`)
3. 128 (one version of variable `y` for each thread block)
4. 128 (one version of array `b_s[]` for each thread block)
5. 1 * 4 + 32 * 4 = 33 * 4 = 132 (assuming float takes 4 bytes)
6. what a hell is this thing

## Question 12
- 2048 threads/SM
- 32 blocks/SM
- 65536 registers/SM
- 96 KB shared memory/SM

1. 32 * 64 = 2048 (so this configuration consumes all possible threads), 2048 * 27 ~ 55k, so we underutilize the registers and finally we underutilize shared memory heavy.
2. of course the fucking threads are the limiting factor in this configuration, we cannot consume so much threads. only 1/4 of all possible threads will be in work in each moment in time (sic!). although in modern GPUs i think the real limit is 1024 threads per block.
