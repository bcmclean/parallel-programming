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

