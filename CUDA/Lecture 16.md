- GPU is responsible for assigning thread blocks to SMs
  - A block must be assigned to exactly one SM
  - An SM can run more than one thread block

Note: **SM = streaming multiprocessor**

- Threads in the same block may cooperate, but not all threads in the SM can cooperate.. Some will belong to different blocks!
- Using SMs allows for **scalable** architecture

### Warps
1. Blocks are assigned to SMs
2. Each SM splits threads in its blocks into *warps*
  - Warps are scheduling units of the SM
  - Thread IDs within a warp are consecutive and increasing. Warp 0 starts with Thread ID 0
  - Size of the warp is implementation specific. Usually # of threads in a warp = # of SPs in an SM

- Can't rely on ordering with warps, need to synchronize if there are any dependencies

- All threads in a single warp will execute in parallel, and a warp in an SM runs in parallel with warps in other SMs

#### Question: If 3 blocks are assigned to an SM and each block has 256 threads, how many warps are there in an SM? Warp = 32 threads.
- 24 ((256 x 3)/32)
- Each block is divided into 256/32 = 8 warps
- There are 8 * 3 = 24 warps

### Thread Life Cycle
1. The Grid is launched
2. Blocks are assigned to SMs in an arbitrary order 
  - Each block is assigned to one SM
  - Each SM is assigned zero or more blocks
3. Each block is divided into warps whose execution is interleaved
4. Warps are executed by the SM (each SP executes one thread)
  - Threads in a warp run in parallel
  - All threads in a warp execute the same instruction when selected

### Zero-overhead and Latency Tolerance
- Latency hiding:
  - While a warp is waiting for a result from a long-latency operation the SM will pick another warp that's ready to execute. This avoids idle time and makes full use of the hardware despite long latency operations

Note: Latency is the delay between an action and the response to that action

- Zero-overhead thread scheduling
  - Having zero idle time is referred to as zero-overhead thread scheduling in processor designs

- Switching of the warps requires almost no time. Switching allows it to hide the latency. 

### GPU Limits
- Limits on blocks and threads it can simultaneously tracked
- G80: Each SM can track up to 8 blocks / 768 threads at a time
- G200: Each SM can process up to 8 blocks / 1024 threads at a time
- If we assign more than the max amount it will just be scheduled for a later execution
