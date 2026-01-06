Naive reduction sum - stride 1 and stride grows by 2 each time until we reach the finishing step. The main con here is that we have bad memory access pattern. Elements are far from each other and we cannot use memory coalescing. 

Patched reduction sum - stride is max first and then we reduce it to one in the end. This leads to elements being near each other which in turn leads to less memory traffic. Also there is less control flow divergence of course.

More patching - Introduce shared memory in the mix to avoid writing back to the global memory. Even less memory traffic here.

Atomic read to scale this shit on any number of thread blocks (finish reading this)

