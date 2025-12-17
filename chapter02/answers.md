1. C - assuming that grid is one dimensional (other assumptions would be wrong)
2. C - we want to assign each thread to the first element in the pair
3. 
4. B - obviously we need 8 blocks each of which consists of 1024 threads.
5. D - first argument in `cudaMalloc` is pointer to the memory location while the second one is the size of allocated memory.
6. D - refer to the last answer and keep in mind that we want to dereference twice because we have an array of elements? (I'm not sure here myself).
7. C - `cudaMemcpy` has the following signature `cudaMemcpy(dst, src, size, direcction)`
8. C - `cudaError_t` looks similar to the C++ syntax.
9. 
    a) 128
    b) 200_064
    c) 1563
    d) 200_064 (all of the threads)
    e) 200_000 (only selected threads)
10. Don't be a dummy and try to use different functions for device part of the code as well as for the host part of the code. Usually, they behave very differently and there's no point in optimizing for both cases at the same time.

