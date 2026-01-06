# CUDA Exercises - Chapter 10

## Exercise 1
For the simple reduction kernel in Fig. 10.6, if the number of elements is 1024 and the warp size is 32, how many warps in the block will have divergence during the fifth iteration?

## Exercise 2
For the improved reduction kernel in Fig. 10.9, if the number of elements is 1024 and the warp size is 32, how many warps will have divergence during the fifth iteration?

## Exercise 3
Modify the kernel in Fig. 10.9 to use the access pattern illustrated below.

*[Diagram shows a parallel reduction pattern where elements are reduced by pairing adjacent elements from left to right, with arrows indicating the reduction flow across 5 iterations, progressively halving the active elements until a single result remains in the rightmost position]*

## Exercise 4
Modify the kernel in Fig. 10.15 to perform a max reduction instead of a sum reduction.

## Exercise 5
Modify the kernel in Fig. 10.15 to work for an arbitrary length input that is not necessarily a multiple of `COARSE_FACTOR*2*blockDim.x`. Add an extra parameter `N` to the kernel that represents the length of the input.

## Exercise 6
Assume that parallel reduction is to be applied on the following input array:

| 6 | 2 | 7 | 4 | 5 | 8 | 3 | 1 |
|---|---|---|---|---|---|---|---|

Show how the contents of the array change after each iteration if:

**a.** The unoptimized kernel in Fig. 10.6 is used.

Initial array: | 6 | 2 | 7 | 4 | 5 | 8 | 3 | 1 |

**b.** The kernel optimized for coalescing and divergence in Fig. 10.9 is used.

Initial array: | 6 | 2 | 7 | 4 | 5 | 8 | 3 | 1 |
