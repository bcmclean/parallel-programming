### Introduction to MPI

#### Different MPI Components
- MPI_Init -> Initialize MPI computation
- MPI_Finalize -> Terminate MPI computation
- MPI_Comm_size -> Get number of processes
- MPI_Comm_rank -> Get current process ID
- MPI_Send, MPI_Recv -> Blocking send/receive
- MPI_Isend, MPI_Irecv

- With MPI, most functions will return an int error code. We can use MPI_SUCCESS
(no error) to check for errors

#### MPI_Init (&argc, &argv)
- This tells MPI to do all the required setup
  - This includes.. allocating storage for message buffers, deciding which processes get which rank
  - Also includes defining a **communicator** that consists of all the processes created when the program is started (MPI_COMM_WORLD)

**What are the arguments?**
- &argc -> pointer to number of arguments in argc in main()
- &argv -> pointer argument vector argv in main()
- When the program doesn't use these arguments just pass NULL for both of them

#### MPI_Finalize
- MPI cleans everything up for the program

````
MPI_Comm_size(
    MPI_COMM_WORLD, // in: the communicator
    &comm_sz // # of processes in the communicator
)

MPI_Comm_rank(
    MPI_COMM_WORLD, // in: the communicator
    &my_rank // out: get rahnk or process making this call
````

#### MPI_COMM_WORLD
- Default communicator that groups all the processes when the program started

# Compile and Execution
To compile
- `mpicc -o mpi_program mpi_program.c`

To execute
- `mpiexec -n 1 ./mpi_program` (run with one process)
- `mpiexec -n 4 ./mpi_program` (run with 4 processes)

#### Point-to-point has two processes:
- MPI_Send and MPI_Recv
- To send a message we need to specify parameters such as...
- Which process is the senter, the data sent (location, type, size), who is the receiver, data received (location, type, size)

### Wildcard Arguments
- Receiver can get a message without specifying the following..
  - sender of the message (use MPI_ANY_SOURCE wildcard constant for the source)
  - message -> MPI_ANY_TAG
  - amount of data in the message (can be obtained later)

