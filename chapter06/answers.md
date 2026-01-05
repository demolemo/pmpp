## Question 1
This will be implemented later because im lazy fuck

## Question 2
If block size is greater than 32 we are doing uncoalesced access to the global memory

## Question 3
1. this is coalesced access (consecutive threads access consecutive elements)
2. this is coalesced access (consecutive threads access consecutive elements)
3. those are coalesced because in the one iteration of the loop consecutive threads access consecutive elements in the array
4. those are non coalesced because `i` is multiplied by 4 which leads consecutive threads access non consecutive elements in one iteration of the loop
5. coalesced, consecutive threads access consecutive elements in one iteration of the loop
6. coalesced, consecutive threads access consecutive elements
7. coalesced, constant does shifting but it's the same for all threads
8. non coalesced, consecutive threads access elements that are 4 slots apart
9. non coalesced, consecutive threads access elements that are 8 slots apart

## Question 4
1. Look at the kernel and actually answer it, fool
2. Look at the kernel and actually answer this question, fool
3. Look at the kernel and actually answer this question, fool

