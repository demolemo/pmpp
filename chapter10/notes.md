- [x] write simple reduction kernel with naive access pattern
- [x] write reduction kernel with improved access pattern
- [ ] write version of the kernel that scales on multiple threadblocks

we can avoid modifying `input` array in the reduction pattern by using shared memory


overall outline of the chapter:
- introduce the reduction algorithm
- make it better in terms of memory accesses and control divergence
- make it use shared memory to reduce global memory traffic
- write a version that scales to several threadblocks but is very inefficient in terms of hardware utilization (this will lead to poor perf on machines that receive a large amount of data all at once but don't have the resources to start all of the threadblocks)
- make algo better in terms of hardware utilization by using the coarsening of the threads (more iterations on max work)
