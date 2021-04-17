### Basics (not related to OpenMP)

#### Processes VS Threads
- Threads exist within a process
- Every process has at least one thread
- A process is multithreaded when it contains more than one thread of execution
- Both threads and processes are **units of execution** (tasks)
- Both are items that can be scheduled for execution
- Processes will by defailt **not share memory**
- Threads of the same process will by default have access to the **same shared memory**
- Data can be shared or private. Private data is only available to the thread that owns it.

#### Scheduler
- Computer application that uses a scheduling policy to decide which processes should run next
- Uses a selection function to make this decision
- It will use several factors like:
  - Resources the processes require (and if they are available)
  - Time the processes have been waiting or if it has been executing 
  - The processes priority
- Scheduling policy tries to optimize with these factors:
  - Responsiveness
  - Turnaround (time it takes for processes to finish)
  - Resource utilization
  - Fairness (dividing the CPU time fairly) (opposite to starvation which is where a process is not givent the chance to run)

#### Scheduling Policies
Non-Pre-Emptive (each tasks runs to completion before another can run)
- **First-in-First Out**: Tasks are placed in a queue as they arrive
- **Shortest Job First**: Process that requires the least execution time is picked first

Pre-Emptive
- **Round-Robin**: Each task is assigned to run for a fixed amount of time before it is required to give way to the next task and move back to the queue. 
- **Earliest Deadline First**: Process with the closes deadline is picked next
- **Shortest-Remaining-Time-First**: Process with the shortest remaining time is picked next

#### Concurrent System
#### What is it?
- A system that can run several activities at the same time
- The benefit is that it's efficient at using resources and faster, but the cost is that is its complexity of hardware and software
#### Two types of concurrent systems
1. Parallel systems
  - More than one processor that can carry out several activities simultaneously
2. Pseudo-parallel systems
  - Share processor time between a number of activities. Two activities running on a single processor but only one is actually running at any one time.

#### Multi-tasking
1. Pre-emptive Multitasking
  - The OS decides when a task should give way to another to allow sharing of resources 
2. Cooperative Multitasking
  - Process is coded such that it decides when to allow other processes to run from time to time

### Condition Synchronization
- Mechanism to make sure process doesn't proceed unless a condition is met

### What is a critical section?
- It is the part of a concurrent program that shouldn't be executed by more than one process at any time
  - Usually when updating a shared resource..
  - Critical sections should be executed in serial

### What is mutual exclusion?
- A mechanism to ensure that no two concurrent processes access a critical section at the same time

### Deadlocks
- Concurrent programs must satisfy two properties: Safety (doesn't enter a bad state) and liveness (must perform progress)
- Speaking of liveness.....
- **Deadlock**
  - When a process is waiting for a resource that will never be available. This resource may be held by another process that waits for the first process to finish first. (Think about the dining philosopher's problem)
- Livelock -> Active but not making any progress.

### Four conditions for a deadlock....
1. Mutual exclusion (the program involves a shared resource thats protected by mutual exclusion and only one process can have that resource)
2. Hold while waiting (A process can hold a resource while its waiting for other resources)
3. No pre-emption (OS doesnt force process to let go, must let go by itself)
4. Circular wait (P1 waiting for resource held by P2, P2 waiting for waiting for resource  held by 1)
