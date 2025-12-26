## Question 1
We can use coalesced memory accesses to reduce global memory traffic. This is done if we have two matrices in row-major order and we do vanilla addition.

## Question 2


## Question 3


## Question 4
To reduce register pressure? This will leave more space for local variable of each thread (automatic variables of course).

## Question 5
We reduce memory traffic by 32 times. (for each element we needed to load it n times now we need to load it n / 32 times).

## Question 6
512 * 1000 versions of the variable, one for each thread

## Question 7
1000 versions of the variable, one for each block

## Question 8
a. N times (to get final matrix we must take each row and multiply it with each column)
b. N / T (now we need to multiply each row tile with each column tile which leads to reducdtion of memory traffic by the factor of T)

## Question 9

