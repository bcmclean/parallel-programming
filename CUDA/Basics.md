### CUDA Programming Model

The best applications use both GPU and CPU. CPUs can be 10x faster for **sequential**, while GPUs can be 10x faster for **parallel.**

### Important notes:
- Device -> runs on GPU (coprocessor to the CPU.. in this case a GPU)
- Host -> runs on CPU 
- CPUs can be 10x faster for sequential code
- GPUs can be 10x faster for parallel code
- With CPUs, **latency** matters
- With GPUs, **throughput** wins
- A device has its own DRAM (device memory)
- Kernel Code: Data-parallel portions of an application which run on the device
- C Program = host code (runs on CPU) + device code (runs on GPU)
- Serial or modestly parallel parts in host C code is compiled by host standard compiler
- Highly parallel parts in device SPMD kernel C code is compiled by NVIDIA compiler
- A CUDA kernel is executed by an array of threads
- SPMD -> single program multiple data

Flash cards: https://quizlet.com/575297696/l13-cuda-flash-cards/?new

---

### `_global_`

- Called from CPU, executed on (kernel function) GPU/device. Must return void

### `_device_`

- Called from GPU and executed by the GPU. Can't be called from CPU
- Called from other global/device functions

### `_host_`

- Called and executed by CPU
- Device & host can be used together

---

### CUDA Functions

CUDA functions return an error code if something goes wrong

#### `cudaMalloc(void **d_ptr, size_t n)`

- Allocates n bytes of linear memory on the device and returns in `*d_ptr` a pointer to that allocated memory
- d_ptr: address of pointer to allocated device memory
- n: size of requested memory in bytes

#### `cudaMemcpy(void *dst, void *src, size_t n, dir)`

- Copies data between the host (CPU) and device (GPU)
- dst/src -> pointers to dest/source memory segments
- n -> number of bytes to copy
- dir -> type of transfer which could be
  - cudaMemcpyDeviceToHost
  - cudaMemcpyHostToDevice
  - cudaMemcpyDeviceToDevice
- Starts copying after previous cuda calls are completed
- CPU thread is blocked until after copy is complete

#### `cudaFree(void *d_ptr)`

- Frees memory on device pointed at by d_ptr

#### `cudaMemset(void* d_ptr, int value, size_t n)

- Fills first n bytes of memory area pointed to by d_ptr with a constant value value

---

### How do you parallelize code?

### Host Code (CPU)

1. Allocate space on the GPU : `cudaMalloc`

2. Copy CPU data to the GPU memory : `cudaMemcpy`

3. Launch kernel function(s) on the GPU 

4. Copy results from the GPU to the CPU : `cudaMemcpy`

5. Free GPU memory : `cudaFree`

### Kernal Code
Note: Write kernel code as if it is going to be run on 1 thread. We will use IDs to identify which piece of data is being processed by this thread.

### Example..

Parallelize the following code:

<img src="https://github.com/bcmclean/parallelprogramming/blob/main/CUDA/Screen%20Shot%202021-03-03%20at%203.34.24%20pm.png" width="500" height="400">

#### Step 1: Parallelizing the device code
  
   1.1: Add _global_ 
   
   1.2: Divide data amongst threads
   
   1.3: Don't run on invalid data range

```
void vectorAdd(int* a, int* b, int* c, int n) {
  int i;
  for (i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}
```
Changed to:

```
// global is callable from the host/CPU, executed by the device/GPU
// write kernel code like it will be run by 1 thread
// threadIdx.x is a read only variable
// if(i<n) in case we have more threads than the number of elements

_global_ vectorAdd(int* a, int* b, int* c, int n) {

  int i = threadIdx.x 
  if(i<n) 
    c[i] = a[i] + b[i];
}
```
#### Step 2: Parallelize the host code

  2.1: Allocate memory

  2.2: Copy CPU to GPU

  2.3: Call GPU functions

  2.4: Copy GPU to CPU

  2.5: Free GPU

```
int main() {
int *a, *b, *c;
a = malloc(N * sizeof(int));
b = malloc(N * sizeof(int));
c = malloc(N * sizeof(int));

vectorAdd(a, b, c, N);

free(a); free(b); free(c);
```

Changed to:

```
int main() {
int *a, *b, *c;
int *d_a, *d_b, *d_c;

a = malloc(N * sizeof(int));
b = malloc(N * sizeof(int));
c = malloc(N * sizeof(int));

// allocated memory
cudaMalloc(&d_a, N * sizeof(int));
cudaMalloc(&d_b, N * sizeof(int));
cudaMalloc(&d_c, N * sizeof(int));

// copy CPU to GPU. Don't need to copy C since there are no numbers - it will be calculated by the GPU
cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

vectorAdd<<<1, N>>>(d_a, d_b, d_c, N);

cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
```

---

### Points of confusion

**One:** Passing in `sizeof(int)` or `N * sizeof(int)`

  - These functions take in the number of bytes
  - `sizeof()` just returns the amount of memory that is allocated to that data type 
  - `sizeof(int)` will return 4 since an int uses 4 bytes
  - N would be the size of the array
  - So to get the number of bytes -> N * sizeof(int)

!!!

**Two:** Passing in '&c' in one example and 'c' in another for cudaMemcpy **still trying to figure this one out**
