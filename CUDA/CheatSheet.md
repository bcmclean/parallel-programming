| GPU  | CPU |
| ------------- | ------------- |
| Device  | Host  |
| 10x faster for parallel  | 10x faster for sequential  |
| Throughput  | Latency  |

int num_bytes = n * sizeof(data);

#### Host Code (CPU)
2. Allocate space on the GPU : cudaMalloc(&d_a, num_bytes)
3. Copy CPU data to the GPU memory : cudaMemcpy(d_a, a, num_bytes, cudaMemcpyHostToDevice );
4. Launch kernel function(s) on the GPU
5. Copy results from the GPU to the CPU : cudaMemcpy
6. Free GPU memory : cudaFree(d_a)

- Each block runs on one SM, Each thread runs on an SP. 
- SP can only run one thread at a time
- Block will be on one SM
- SM can run many blocks
