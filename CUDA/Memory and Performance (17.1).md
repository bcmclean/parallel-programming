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

## Part 2

#### CGMA
- Compute to Global Memory Access
- It is the number of floating point calculations performanced for each access to the global memory
- Each access of a float is 4 bytes

If CGMA = 1.. this means each FLOP requires reading one float from global memory. GPU can only perform n/4 FLOPS per sec.

A low CGMA = low performance.

**The aim here would be to increase the CGMA (reduce the number of general memory accesses with respect to the # of floating point operations**

