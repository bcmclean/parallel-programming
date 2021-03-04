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
cudaMalloc(&d_a, N* sizeof(int));
cudaMalloc(&d_b, N* sizeof(int));
cudaMalloc(&d_c, N* sizeof(int));

// copy CPU to GPU. Don't need to copy C since there are no numbers - it will be calculated by the GPU
cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

vectorAdd<<<1, N>>>(d_a, d_b, d_c, N);

cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
```
