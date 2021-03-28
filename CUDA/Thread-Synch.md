### Thread Sync

Whenever you share data between threads within shared or global memory,
that means threads are using the same data that's written or read by other threads.
So, you need to synchronize threads so that if a thread is reading a value that's supposed to be written by another thread it will wait until that happens.
If we don't synchronize it, we'll end up with a **race condition**.

### To summarize what we learned previously about memory access (performance):
- Shared Memory = 😄
- Global Memory = 😞

Whenever threads are communicating with one another.. We need to make sure they are on the same page..
We need to **SYNCHRONIZE**!!!

### Synchronization to avoid data race
How do we synchronize? Let's use... **barriers** 🚧 🚧 🚧 

Another technique is memory fences but we're not going to talk about that.

### Barriers
We have two types!

1. **Explicit** barriers **within a block**
  - Like __synchthreads()__ which synchronizes all threads within a block
  - All threads within a block must reach and execute synchtreads() before execution can resume

2. **Implicit** barriers at the **End of each kernel**
  - A kernel must complete before the next kernel can start
  - You'll want to terminate your kernel and start another one

If you use synchthreads twice in your kernel they will be treated as two different barriers. This means that all threads must execute the first function/barrier and all will need to execute the second function/barrier too.

<img src="https://github.com/bcmclean/parallel-programming/blob/main/CUDA/thread-synch-question.png" width="350" height="150">

For the image above, do we have to break the statement into two?

The answer is no. Why is that? It's because we are only reading other elements and copying them into our own. Other elements are not affected by our reading and writing. But if we were writing to an element other than our own, that's when we would need to synchronize them.

<img src="https://github.com/bcmclean/parallel-programming/blob/main/CUDA/thread-synch-question-2.png" width="650" height="250">

For the above image, do these two programs perform the same?

No! The location of syncthreads() matters. If it's outside of the if statement, all threads will run through it. But if it's inside the if statement, since only odd elements will reach it, we will be faced with a deadlock because not all threads will be able to read it and synchronize.

### Atomic Sections

#### Why do we want to avoid critical sections?
It will serialize your threads and lead to low performance!

- Serializes thread accesses to shared data
- Read-modify-write atomic operation on one word in global or shared memory
- Arithmetic: atomicAdd(), atomicSub(), ....
- Bitwise: atomicAnd(), atomicOr(), atomicXor()

Other limitations...
- There's no specific order
- Only certain operations are supported
- Only int is supported for most operations
