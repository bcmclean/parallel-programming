## Memory and Performance

### Okay so, let's look at what kind of memories we have here..

### Registers 
- Registers are **private** to each thread
- They are partitioned among threads in a block
- More threads = less registers per thread!

### Shared memory
- These are (obviously) **shared** by threads in the same block
- They are divided among blocks
- So... more blocks = less shared memory per block!

### Global memory
- **Shared.. by ALL threads** in a grid!

### What about performance?
Well..

- Reading and writing to a **Register** takes **1 cycle**
- Reading and writing to **Shared Memory** takes **5 cycles**
- Reading and writing to **Global Memory** takes **500 cycles** (allocated by host using cudaMalloc)
- Reading **Constant Memory** takes **5 cycles with caching**

### Constant Memory
- On DRAM (Dynamic random access memory) but has dedicated on-chip cache
- Initialized in host code
  - Host can read and write
  - But the Kernel can read-only!

Okay.. So what kind of data is stored in constant memory?
  - Varibles **declared as __constant__**
  - **__global__** function parameters are passed to device via constant memory

### Paralell Memory Sharing

#### Registers / Local Memory
- These are private per thread
- Auto variables
- Register spill (when we run out of registers)

#### Shared memory
- Shared by threads in the same block
- Inter-thread communication

#### Global Memory
- Shared by all threads
- Inter-grid communication

### Okay, but where is local memory and where is it used?
It doesn't refer to a physical location. The local memory is on the global memory. It's local because each thread has its own private area. It is also cached (L1). It's used when we run out of registers (register spilling). When we declare arrays inside the kernel, some are stored in registers if they are small and the compiler can resolve indexing.

### Next Question: What are L1 and L2 caches?
- Well, a cache is **non-programmable**
What is their purpose?
- They help threads access the same memory segment without going to DRAM. 
What's the difference?
- L2 is coherent and L1 is not. This means that if 2 SMs are working on the same global memory location, it's not guaranteed that one SM will see the changes made on the other SM.

### Which memory is the faster?
This is how they rank:

- Register, Shared Memory, Local Memory, Global Memory

#### Why would shared memory be faster than local memory?
If what you're looking for isn't in the cache, you have to grab it from the global memory!

## Part 2: Let's improve the memory!

#### CGMA
- Compute to Global Memory Access
- It is the number of floating point calculations performanced for each access to the global memory
- Each access of a float is 4 bytes

If CGMA = 1.. this means each FLOP requires reading one float from global memory. GPU can only perform n/4 FLOPS per sec.

A low CGMA = low performance.

**The aim here would be to increase the CGMA (reduce the number of general memory accesses with respect to the # of floating point operations**

<img src="https://github.com/bcmclean/parallel-programming/blob/main/CUDA/poor-performance-CGMA.png" height="500" width="700">

### What's wrong with the performance of the code above?
Well, it has a CGMA of 1. This means that for each floating point operation, there is 1 memory transaction.

So 1 Floating Point Operation / 1 Memory Access. Remember, a low CGMA means low performance! The goal, again, is to reduce the number of general memory accesses with respect ot the # of floating point operations. 

### What can we do about it?
In the code, every element in M or N is being read twice (when tile_width = 2). Each item is accessed tile_width times.

How many times will each element in M be read by all the threads for computing p? Width times!

### We can use something called ✨ tiling ✨ to reduce global memory traffic
Basically, we can partition data from the global memory into tiles so each tile fits into the shared memory!

This invovles the following steps...
1. Load subset from global memory to shared memory using multiple threads to exploit memory-level parallelism
2. Perform computation on shared memory
3. Copy results from shared memory to global memory

### Why do we have to use shared memory? Why not registers? 
Okay, let's look back at the different types of memory that we have.

#### Registers
- There are a limited number of registers per SM. So.. they are divided amongst threads in a block, remember? This means the more threads we have, the less registers per thread. If we have a lot of threads, it's going to be difficult for some of them to share one register. There simply aren't enough registers to solve this problem!
- For this particular problem, we also have to remember that elements are being accessed multiple times. The fact that the memory is going to be private to each thread is not great! 

### Looking back at the steps again...

1. The block loads data from global memory to shared memory
2. Synchronize threads
3. Threads work on data from shared memory in parallel
4. Block writes data back from shared to global memory-

---

1. Define shared memory matrices (that have the same dimension as the tile width!)
````
__shared__ float Ns[TILE_WIDTH][TILE_WIDTH]
__shared__ float Ms[TILE_WIDTH][TILE_WIDTH]
````
2. Before you do any computation, you want to read the global memory into the shared memory
````
Ms[ty][tx] = M[y * Width + tx];
Ns[ty][tx] = N[ty * Width + x];
````
3. Since step 2 needs to be finished before we move onto the computation, we need to synchronize the threads!

`__syncthreads();`

Note this only works on threads within the same block..

4. Do computation in shared memory
````
value += Ms[ty][k] * Ns[k][tx]
````

#### What if WIDTH > TILE_WIDTH?
- Break up inner product loop of each thread into phases
- At the beginning of each phase, load M and N elements that everyone needs during a phase P into shared memory
- Everyone access the M and N elements from the shared memory during the phase


### L1 and L2
- When reading from global memory L1 and L2 are used
- When writing to global memory only L2 is used
