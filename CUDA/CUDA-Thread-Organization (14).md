### Memory
- Remember that.. the CPU and GPU have different memory spaces! You'll have to move data around if you want to use them in both
- Functions that we learned... `cudaMalloc`, `cudaFree`, and `cudaMemcpy`
- Pointers hold memory addresses in GPU or CPU. Couldn't differentiate between the two just by looking at their values
  - Dereferencing CPU pointer in kernal code -> likely to crash (and GPU pointer in host code)

For revision...
  - `cudaMalloc`: Allocates memory space on the device/GPU
  - `cudaFree`: used to free up memory in the GPU
  - `cudaMemcpy`: Used to copy memory from the GPU to CPU and vice versa 

In the GPU: 
  - 1000s of threads per app
  
### Remember a typical GPU program...
  1. Allocate space on GPU
  2. Copy CPU values to GPU
  3. Launch kernel functions on GPU (define launch-configuration before this)
  4. Copy GPU values to CPU
  5. Free memory on the device (GPU)

##### Kernel code (GPU)
  - Write kernel function as if it would be run on a single thread and use ID's to identify!
  - Parallelism is expressed in the host code!

### Thread Organization 

##### The Basics...

On the software side...
  - *Threads* are grouped into *blocks* (all threads in a block execute the same kernel program - SPMD!)
  - *Blocks* are grouped into *Grids*
  - IDs.. each thread has a unique ID within a block, each block with a unique ID within a grid

On the hardware side...
  - Each block runs on one *SM*.. SM might run more than one block
  - Each threads runs on an SP (within an SM)
    - An SP can only run one thread at a time, although may run successive threads

Threads in blocks can be organized in *1D, 2D or 3D array of threads*

How do we deal with this? (1D, 2D, 3D array of threads)
  - We have built-in variables! : threadIdx.x, threadIdx.y, threadIdx.z

Why do we do this type of organization?
  - This simplifies memory addressing when processing multidimensional data
  - 1D threads good for processing vectors
  - 2D threads are suitable for 2D arrays
  - 3D threads for 3D arrays / environments

#### Blocks in a Grid
  - Kernel code may initiate one or more blocks, each with a certain number of threads
  - `kernel1<<<gridSize,blockSize>>>();`
  - All blocks for a given kernel belong to a grid
  - All blocks in a grid must finish before the next kernel function can run! This is a synchronization point
  - **Each block runs on one SM**
  - Built-in variables for blocks: blockIdx.x, blockIdx.y, blockIdx.z. **Unique within a grid**

#### Dimension Variables
  - Holds the number of elements over this dimension
  - Unique for each grid and are set are launch time
  - Built-in dimension variables: blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z

### Thread Life Cycle in HW
1. Grid is launched `kernelFoo<<<gridSize, blockSize>>>();`
2. Blocks are distributed to SM.. could have more than one block per SM
3. Each SM launches the threads in its block (one thread per core (SP))
4. As blocks complete, resources are freed up

### Kernel Launch Configuration

Example:
`vectorAdd<<<1, N>>>(...);`
- Tells GPU to launch N threads on 1 block

***`kernelFunction<<<gridSize(# of blocks), blockSize(# of threads)>>>();`***

We can....
- Run as many blocks at once (all belong to same grid)
- Each block can have a maximum of 1024 threads on newer GPUs, and 512 on older ones
- ...not specify which block runs before another

How should we choose the kernel launch configuration?
  - Well, we should choose the breakdown that makes the most sense for our problem
  - For a vector, we can choose a 1D setup
  - x-Dimension is used by default for 1D items. We can define a higher dimensionality with **dim3**
  - ***Remember each block is assigned to one SM.. to fully use the GPU, blocks should be >= # of SMs***
  
  Say you need N threads (should be equal to number of data elements)..
  
  How do you determine the launch configuration?
  
  ##### Steps
    1. Choose number of threds per block (nthreads)
    2. Compute the number of blocks as follows.... nblocks = (N-1)/nthreads+1

Remember, again, to fully use the GPU : 
  - threads per block should be >= # of SPs per SM
  - blocks should be >= # of SMs

Remember we use `if (i<n)` to discard extra threads




