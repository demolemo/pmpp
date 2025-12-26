cuda memory model and all kinds of memories - (memories by david guetta and kid cudi)

this fucking performance graph once again - add a picture with explanation

write tiling kernel myself to better grasp all the beauty of it. i think i understand but the understanding is a tricky thing. 

write for the tile of any size

compare the performance of the matmul through the naive approach via tiled approach. try to estimate gflops first and then watch perf to get the idea. 

true and false deps in parallel algorithms
read-after-write
write-after-read

write a kernel that matmuls two matrices that are not the size of the tile

extend the logic to cover the general case matrices

dynamic allocation of shared memory size?
