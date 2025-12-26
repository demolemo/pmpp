## Question 1
a. Each block contains of 128 threads and each warp contains of 32 threads -> each block contains of 4 warps
b. Blocks in the grid = 8, warps per block = 4 -> warps in the grid = 32
c. 
- For each block 3 warps execute this statement -> 8 * 3 = 24 warps are active
- 2 out of 3 active warps in the block are divergent -> 8 * 2 = 16 warps
- 100% (not sure on the math but this warp is not divergent)
- 25% (8 out of 32 threads are doing the work)
- 75% (24 out of 32 threads are doing the work)
d.
- all warps are active -> 32 warps
- all warps are divergent -> 32 warps (each warp contains odd and even indices)
- 50% (half of threads execute one instruction and other half execute other instruction). still need to check if i understand this concept correctly
e.
- 2 iterations (each thread does at least two iterations)
- 3 iterations (the number of actual iterations here depends on `i`)

## Question 2
We need at least 4 blocks of 512 threads => 2048 total threads in the grid

## Question 3
From all of the warps that we have in the grid only one warp would diverge (if we process elements like a smart person would - in a sequential manner). Warp that has 16 threads doing the work and 16 threads stalling.

## Question 4
(this is the answer that i came up with wo reading the chapter) we assume that the percentage of the total occupancy is waiting time / total time.
waiting time for all threads is the time of the longest thread - time of the thread, while total time is time of the longest thread multiplied by the number of threads => ((3 - 2) + (3 - 2.3) + (3 - 3) + (3 - 2.8) + (3 - 2.4) + (3 - 1.9)) / (3 * 6)

## Question 5
Bad idea. There might be the code further that depends on synchronization, for example some kind of accumulation of results. This code would break if we leave out `__syncthreads()` function.

## Question 6
Configuration C - 512 * 3 will take all of the threads available

## Question 7
a - possible, 50%
b - possible, 50%
c - possible, 50%
d - possible, 100%
e - possible, 100%

## Question 8
a - 32 blocks, 128 threads, 30 registers per thread - too much threads, some threads will stall
b - 32 blocks, 32 threads per block and 29 registers per thread - this configuration is fully possible, although we leave too many resources on the table
c - 32 blocks, 256 threads per block and 34 registers per thread - too many threads per block, some threads will stall

## Question 9
I will tell that he is telling a bunch of bullshit because his thread blocks are too large for arch that fits only 512 threads per block.
