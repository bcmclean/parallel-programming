Problem Example:
#### Processing a 100 x 100 image. 
What should we not use?
  - Many 1D blocks
  - Many 2D blocks
  - One big 2D block

You can't use a big 2D block for a 100x100 image because that would use 10,000 threads! The maxiumum is 1024.

Normally for something like 100 x 70, we would use a 2D block.

#### What if we were using a 32x32 image?
- One 2D block would be the least preferred..
  - The single block will be assigned to one SM and the other SMs will be doing nothing 

### Configuring Kernel with Higher Dimensional Grids/Blocks

`kernelFunction<<<gridSize, blockSize>>>();`

##### gridSize:

Dimension and size of the grid in terms of blocks.. could one:
  - dim3(gx, gy, gz) -> 3D grid
  - dim3(gx, gy) -> 2D grid
  - dim3(gx) (or an integer) -> 1D grid

##### blockSize:

Dimension and size of each block in threads.. could be:
 - dim3(bx, by, bz) -> 3D block
 - dim3(bx, by) -> 2D block
 - dim3(bx) -> 1D block


Thread (0, 0, 0) in block (0, 0, 0) says: Hello!

Thread (1, 0, 0) in block (0, 0, 0) says: Hello!

Thread (0, 0, 0) in block (1, 0, 0) says: Hello!

Thread (1, 0, 0) in block (1, 0, 0) says: Hello!

Possible ways to define grid/block size:
1. `hello<<<dim3(2,1,1),dim3(2,1,1)>>>();` (same as `hello<<<2,2>>>`)
2. `dem3 gridSize(2,1,1), blockSize(2,1,1); // hello<<<gridSize, blockSize>>>();`

Can we run printf in kernel?
  - Yes you can do this - but it's not common. Mostly just used for debugging
  - Need to use cudaDeviceSynchronize() -> force printf() in device to flush

Clicker question.. How many stars will be printed?

b0 : [0 1]
     [0 1]

b1 : [0 1]
     [0 1]

b2 : [0 1]
     [0 1]
     
number of blocks:
1 2 3 4 5
2
3
4
5
6
7
8
9
10

- 1 2

blocks.. 100?
1 2 3 4
2
3
4
100 blocks, 1600 threads

### Cuda Limits

- Maximum number of threads per block 1024
- How many blocks? As many as you want (almost)
- Within a block:
  - x, y dimension: 1024
  - z-dimension: 64

### Threads Cooperation

Threads in the ***same block can cooperate***
  - Synchronize their execution
  - Communicate via shared memory
  - Thread/block index is used to assign work and address shared data

Threads in ***different blocks cannot cooperate***
  - Blocks can execute in any order relative to other blocks
  - There is no native way to synchronize all threads in all blocks..
    - To synchronize threads in all blocks, terminate your kernel at the synchronization point, then launch a new kernel which would continue with your job


We should know that:
  - All threads in all blocks run the same kernel
  - Threads within the same block cooperate via shared memory, atomic operations and barrier synchronization
  - Threads in different blocks CANNOT cooperate

The data of matrix d_M is stored in memory as a one dimensional array that follows row-major convention








