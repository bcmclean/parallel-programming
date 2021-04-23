## OpenMP
#### Fork-Join Model
- Team of threads execute the parallel block. The original is the master, and the rest are slave threads.

### Pragmas
`#pragma omp parallel` -> Most basic

`#pragma omp parallel num_threads(thread_count)` -> specify number of threads to run
  - Most systems can run hundreds or thousands of threads 

`#pragma omp master` 
  - Executed once by the master thread
  - Different from single directive because it does NOT have implied barrier on exit

`#pragma omp single`
  - Executed once by *any* thead in the team
  - There is an implicit barrier at the end of the single region

### Functions
`omp_get_thread_num()`

`omp_get_num_threads()`

### Area Under a Curve Problem
- Convert to parallel without any type of data race
- Watch out for loop carried dependencies
- Choose between manually dividing threads/parallel for and reduction vs global_result only

The main themes are:

1. With a global result 
    - Data race possible, ensure that you are using a critical section if this is the case
    - Could save function into global_result variable, parallelize for loop only within function 
and return global_result

2. With reduction
    - Global result is private to each thread, on exit they are all added together
    - reduction (<op> : <variable list>).. can use max, min, sum
    
3. Manually divide the threads
    - Iterations must be divisible by number of threads
4. Parallel for / omp for
    - Loop counter of "for" statement (immediately after for directive) is private
    - Loops must be iteration-independent

### Manually divide & Use Reduction

````
// in main
double global_result = 0;
#pragma omp parallel num_threads(thread_count) reduction(+:global_result)
  global_result += Local_trap(...);
printf("%f", global_result)

// trap function
double Local_trap(double a, double b, int n) 
{
double x, my_approx, my_a, my_b;
int thread_count = omp_get_num_threads();
my_n = n / thread_count;
// my_a = id * number of tasks per thread, starting point
my_a = a + my_rank * my_n * h;
// my_b = id * number of tasks per thread + number of tasks per thread, ending point
my_b = my_a + my_n * h;

double h (b-a) / n;
my_approx = (f(my_a) + f(my_b)) / 2.0;
for (i = 1; i <= my_n-1; i++)
  {
  my_approx += f(my_a + i * h);
  }
return h * my_approx;
}

````

### Without Reduction & Without Parallel For

````
#pragma omp parallel num_threads(thread_count)
{
double my_result = Local_trap(a, b, n);
#pragma omp critical
  global_result += my_result;
}
````

### With Parallel For & Reduction
````
// in main
double global_result = Trap(a, b, n, thread_count)

double Trap(double a, double b, int n, int thread_count)
  double h = (b-a)/n;
  double approx = (f(a)+f(b))/2.0;
  int i;
#pragma omp parallel for num_threads(thread_count) reduction(+:approx)
  for (i = 1; i <= n-1; i++)
    approx += f(a+i*h);
  return h * approx;
  
````

### Loop Carried Dependencies
-> Loop carried dependency is what happens when calculations in one iteration depend on the data written by other iteractions...

You must (do one of the following):

  1. Rewrite the algorithm
  2. Don't use parallel for
  3. Order your iterations (poor efficiency)

If you have something like: `a[i] = b[i]`
  - There is no loop carried dependency here.. a and b are completely independent

If you have something like: `a[i] = a[i-1]`
  - Now your program depends on results written in other interactions.. this is a loop carried dependency

### Variable Scopes

1. Shared Variables
  - Exists in one memory location, all threads can access
  - Variables declared BEFORE a parallel block are shared by default

2. Private Variables
  - Accessed by single thread, each thread has its own copy
  - Variables declared WITHIN parallel block are private

Explicitely define the scope:
1. shared(x)
  - x will refer to same memory block for all threads
2. private(y)
  - y will refer to diff memory block for each thread.. each copy of y is uninitialized
3. firstprivate(z)
  - same as private, but each copy of z is initialized with the value that the original z has when the construct is encountered
 
 Variables that can't be changed...
 - Variables declared inside the parallel region are private
 - A loop variable in a parallel loop is private
 - Const variables are shared

`pragma omp parallel num_threads(thread_count) default(none) private(x) shared(y)`

- Default forces the programmer to specify the scope of each variable in a block

### Atomic
- Load-modify-store (updating single memory location) -> has restrictions

### Barriers
- Used for threads synchronization
- All threads must reach the barrier before any of them can proceed

Two types of barriers:
  - Implicit: automatically added
  - Explicit: programmers add them

`#pragma omp barrier`

### Nowait 
- If nowait is used, after 'single' for example, there is no synchronization
- Using barriers is expensive because many threads will be idle, so you can reduce synchronization by using 'nowait'
- Used to cancel implicit barriers at the end of pragma construct
- `#pragma omp single nowait`
- Of course, we only use this if there is no data dependency

### Sections

### Schedule

### Pragma Omp Parallel For
For.. has an implied barrier on exit (unless nowait is used!). No implied barrier on entry.
  - They use existing threads (don't create new ones)

`#pragma omp for` -> Does not create new threads. It MUST be placed WITHIN a parallel region
`#pragma omp parallel for` -> Creates new threads. Parallel block that ONLY includes a for loop...

````
#pragma omp for
for (... i.... ) // parallelized
  for (...j....) // not parallel, each thread will execute all iterations
  // j is not private.. unless its declared right after outer for loop or is explicitely defined as private
  
````
!!!!! go back to slide 20

#### Static scheduling
- Number of iterations is divided equally amongst threads
- Extra iterations would be assigned to first few threads


### Matrix Multiplication

