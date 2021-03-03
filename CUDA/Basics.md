### CUDA Programming Model

The best applications use both GPU and CPU. CPUs can be 10x faster for **sequential**, while GPUs can be 10x faster for **parallel.**

### Host Code (CPU)

1. Allocate space on the GPU : `cudaMalloc`

2. Copy CPU data to the GPU memory : `cudaMemcpy`

3. Launch kernel function(s) on the GPU 

4. Copy results from the GPU to the CPU : `cudaMemcpy`

5. Free GPU memory : `cudaFree`

### Kernal Code
Note: Write kernel code as if it is going to be run on 1 thread. We will use IDs to identify which piece of data is being processed by this thread.

### Example Code
