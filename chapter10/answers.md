## Exercise 1
during the fifth iteration `stride` is equal to 16. due to naive access pattern, all of the threads in the grid will diverge.

## Exercise 2
No threads have divergence, some of them are doing the full work and some of them doing none of the work.

## Exercise 3
Make a reverse kernel so that results are saved in the second half not the first half.

## Exercise 4
First write the kernel with coarsening and then sub the operation to max instead of sum.

## Exercise 5
Update the kernel in question and make a note about it.

## Exercise 6
**a.** Initial array: `[6, 2, 7, 4, 5, 8, 3, 1]`
after 1 iteration: `[8, 2, 11, 4, 13, 8, 4, 1]`
after 2 iteration: `[19, 2, 11, 4, 17, 8, 4, 1]`
after 3 iteration: `[36, 2, 11, 4, 17, 8, 4, 1]`

**b.** Initial array: `[6, 2, 7, 4, 5, 8, 3, 1]`
after 1 iteration: `[11, 10, 10, 5, 5, 8, 3, 1]`
after 2 iteration: `[21, 10, 15, 5, 5, 8, 3, 1]`
after 3 iteration: `[36, 10, 15, 5, 5, 8, 3, 1]`

