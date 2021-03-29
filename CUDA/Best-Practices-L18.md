## CUDA Best Practices

### Measuring Performance

####
NVIDIA suggests APOD Framework

#### APOD Framework
-> Assess, Parallelize, Optimize, Deploy
- Locate the parts that are responsible for most execution time. Decide if they can be parallelized.
- Determine expected speedup (Amdahl's and Gustafson's Laws
- Parallelize code
- Optimize your implementation
  - Memory optimization
  - Instruction optimization
  - Control flow
- Deploy the parallelized components and measure the performance. Compare to your orgiginal expectations.

#### Measure the Performance: Timing
- Can use CPU or GPU timers
CPU Timer
````
double t = clock();
kernel<<<...,...>>>(..);
cudaDeviceSynchronize(); // block host until kernel finishes
t = 1000 * (clock()-t)
````
- Bandwidth (BW) -> Rate at which data can be transferred from/to the GPU global memory
- Theoretical Bandwidth:
  - Can be calculated based on GPU specifications
- Effective Bandwidth
  - Calculated for a specific program
  - `BWEffective = (Rb + Wb) / time`
  - How? Time your kernel. Computer the effective bandwidth based on the amount of data the kernel reads and writes per unit of time.
  - Example... for 1024x1024 float matrix copy, Rb = Wb = 1024^2 x 4 bytes
    - BWEffective = 2 * 2024^2 * 4 / time
    
### Memory Optimization (the important part!)

- Most important area for performance
- The goal is to maximize the use of hardware by maximizing bandwidth
- Two aspects:
  1. Transferring data between host and device
  2. Using different memories of the device

- Bottleneck on PCI-e bus (between CPU and GPU). Bandwidth is only 5gb per second which isn't great.
- Another bottleneck between GPU and the global memory

#### Guidelines
(1) Transferring data between host and device
Guideline 1: **Minimize data transfer between host and device**
- Even if it means running some kernels on the device that don't show performance gains when compared with running them on the host CPU
- Create and destroy intermediate data structures, which are solely used by the device - on the device only
- Batch small transfers into one large transfer (to avoid the overhead associated with each transfer)

(2) Using different memories of the device...
a. (Guideline 2) Use fast memory and avoid slow memory
b. (Guideline 3) Copy frequently accessed data to faster memory to reduce global memory traffic (basically, if it will be used frequently)
c. (Guideline 4) Access memory fast by using **coalesced global memory access**.. also **reduce misaligned** memory access
- One of the MOST IMPORTANT!! performance considerations
- Accessing the global memory is faster when adjacent threads access contiguous memory locations at the same time!
- Global memory accesses by threads of a warp can be performed in as few as one transaction if guideline is followed...

#### Okay, let's look at coalesced global memory access

- Each memory transaction gives access to a chunk of memory at once, even if you are reading and writing to a single memory location.
- If threads in a warp are accessing contiguous locations at the same time, this can be done in one memory transaction
- The concurrent accesses of the threads of a warp with coalesce into a number of transactions equal to the number of cache lines necessary to service all of the threads of the warp

#### Memory Access Patterns
1. Simple (sequential, aligned)
- i-th thread accesses i-th word in a cache line
- Not all threads need to participate
2. Sequential but Misaligned
- Memory allocated using cudaMalloc is aligned to at least 256 bytes.
- So, choosing sensible thread block sizes, like multiple warp size, facilitates memory accesses by warps that are aligned to cache lines
- Guideline 5!!! Choose sensible thread block sizes, such as multiples of the warp size, to avoid misaligned memory accesses
3. Strided
- A "stride" is a # of locations between the beginnings of successive array elements being accessed
- A stride of 2 results in a 50% of load/store efficiency.. since half the elements in the memory transaction aren't used and represent wasted bandwidth
- As the stride increases, the effective bandwidth decreases until the point where 32 lines of cache are needed for the 32 threads in a warp

So to summarize coalesced global memory access....
- Simple access pattern gives the best performance
- Misaligned memory access should be avoided
- Non-unit-stride global memory accesses should be avoided (or at least minimize the stride) whenever possible
- Random global memory access is the woooorssssst


### Optimization
Practice.. Here we have each element read by two threads each.
- Copy C into shared memory! 

````
__shared__ float sh[blockDim.x];
int ix = threadIdx.x;
sh[ix] = c[i];
__syncthreads();

if (i<n && ix>0)
   temp = sh[ix] + sh[ix-1];
   
if (i>0 && ix=0)
temp = sh[ix] + c[i-1];
__syncthreads();

````

Guideline 6: Optimize your code at the instruction level
  - Use CUDA fast math library 
    - eg use **__sin()** instead of sin()
    - Use expf2() instead of expf()
  - Use shift operations to avoid expensive division and modulo calculations
    - multiplying by 2 is same as i<<1; if n is a power of 2, i/n is equivalent to i>>log2(n). i%n equivalent to i&(n-1)
   - Try to avoid automatic conversion of doubles to floats
    - These require more clock cycles!

### Control Flow

- All threads share a program counter.. they run the same instructions

Thread divergence
- All the 32 threads run one statement. Some of them will enter into the if statement. Some will enter and others will do nothing. 
- All threads in warp must run the same instruction
- We want to avoid thread divergence
- Make if statement true for everyone

Guideline 7: Avoid of minimize warp divergence
  - Threads in a warp are doing different things
  - At least one thread takes longer to finish


Could design it so all threads in a warp take the same path..
````
if (threadIdx.x/WARP_SIZE == 2)
{
}
else (...)

````

- Sometimes warp divergence is unavoidable.. but if possible:
  - If the control flow depends on the thread ID minimize the divergence by using warp id to assignm task
  - Assign same task to all threads
- Try to equally distribute the workload to all threads in a warp
- Consider using host to carry out part of the work that causes a load imbalance on the GPU
- Modify the algorithm

### Reduction
PUT IT ON YOUR CHEAT SHEET!!

#### Okay, but what is it?
- Reduction is one of the common algorithms suitable for parallel programs
- Other common algorithms are.. histogram, sort, scan

- Reduction aims to reduce all elements to a single value
-   - max, min, sum, etc

How?

- Serial: run a loop over every element
- Parallel: 
  - partition array into segments
  - each segment is processed by a thread block to find a partial result
  - combine results from different blocks

We'll use in-plane reduction using shared memory
- the original vector is in device global memory
- shared memory used to hold a partial result (sum/min)
- each iteraction brings partial result vector closer to the first result
- final solution will be in element 0

Assume we are finding the sum..
1. Find the partial sum within each store and store. We need shared memory so that block threads can communicate
2. Copy the partial sum from each block into the global array. Global memory so blocks can communicate
3. Find the sum of all partial results stored in the global array.. you can do it on the host

Copy local results to another global array (size = # of blocks)

Copy results to host and reduce on the host



