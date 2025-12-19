## Question 3
a. 16 x 32 = 512
b. `blocksInGrid` = $((150 - 1) // 16 + 1) * ((300 - 1) // 32 + 1)$ = $100$. Threads multiplied by blocks = 100 * 512 = 51_200
c. `blocksInGrid` = 100
d. 150 * 300 (we know that this code will be executed for each pair of M & N).

## Question 4
A: 400 x 500 - matrix

Specify the array index of the matrix element at row 20 and column 10;
a) in row-major order
b) in column-major order

a) In row major order we first traverse rows -> 20 * 400 + 10 = 8010 - the index of the corresponding element in the matrix stored in the row-major order.
b) In column-major order we first traverse columns (sic) -> 10 * 500 + 20 = 5020 - the index of corresponding element in the matrix stored in the column-major order.

## Question 5
w (X) = 400, h (Y) = 500, d (Z) = 300 - source dimensions of the tensors
x = 10, y = 20, z = 5 - final position of the element we are looking for

This better works out on paper. Essentially row major layout means that the first index that changes is x, then y, then z. 
So the final formula for the

We need to take z index, multiply it by w and h (because those are the first dimensions that are changing), then we need to take y and multiply it by w because it's the secound dimension that is changing and then we need to add x to the mix. 

