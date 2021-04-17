### Shared Memory
OpenMP -> Open Mult-Processing

`#include <omp.h>`

#### Two Ways to Divide a Task..
1. Task Parallelism
- Partition various tasks carried out solving the problem among the cores
- Questions 1-5, questions 6-10, etc
2. Data Parallelism
- Partition the data used in solving the problem amongst the cores
- Each core carries out similar operations on its part of the data
- 1/3 exams, 1/3 exams, 1/3 exams

`#pragma omp parallel`
- Compiler directive, telling program to run in parallel
- Anything under this will be run by more than one thread

Runtime library routines...
  - omp_get_thread_num()

Environment variables
  - set OMP_NUM_THREADS = 3

#### OpenMP implements parallelism using **threads** only

Remember for process vs. threads that
- Threads exist in a process!
- They are both **units of execution**
- Processes don't share memory, and threads have access to the same shared memory

### OpenMP uses Fork-Join model
- Synchronization : Everyone must wait until everyone is done before continuing on
- Fork at parallel, then join again after
- Collection of threads executing parallel block -> team.
- Master thread -> ID = 0

### Pragmas
- Special preprocessor instructions that are added to a system to allow behaviours that aren't part of the basic C specification
- `#pragma omp directive [clause [clause]..]`
- Directive: specifies the required directive 
- Clause: information to modify the directive..
  - `#pragma omp parallel num_threads(10)1
  - Continuation on new line use \ in pragma

### Distributing Tasks
- Split the work using the thread ID
`if (omp_get_thread_num() == 2)`

### Distribute the Data
````
// Private to only that thread
int my_a = id * 3;
int my_b = id * 3 + 3;
for (int i = my_a; i < my_b; i++)
````

0*3
1*3

0+3 = 3
0 -> 3

3+3 = 6
3 -> 6

