## Question 1


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
a - 32 blocks, 128 threads, 30 registers per thread - 
