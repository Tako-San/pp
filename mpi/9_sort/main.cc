/* IMPORTANT: Compile with -lm:
   mpicc mpi_mergesort.c -lm -o mpi_mergesort */

#include <algorithm>
#include <cstddef>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#include <mpi.h>

// Arrays size <= SMALL switches to insertion sort
#define SMALL 32

void merge(int a[], int size, int temp[]);
void insertionSort(int a[], int size);
void mergeSortSerial(int a[], int size, int temp[]);
void mergeSortParallel(std::vector<int> &a, std::vector<int> &temp, int size,
                       int level, int rank, int maxRank, int tag,
                       MPI::Comm &comm);
int topmostLevel(int rank);
void runRootMPI(std::vector<int> &a, std::vector<int> &temp, int maxRank,
                int tag, MPI::Comm &comm);
void runHelperMPI(int rank, int maxRank, int tag, MPI::Comm &comm);
int main(int ac, char **av);

int main(int ac, char **av) {
  // All processes
  MPI::Init(ac, av);

  // Check processes and their ranks
  // number of processes == communicator size
  auto commsize = MPI::COMM_WORLD.Get_size();
  auto myRank = MPI::COMM_WORLD.Get_rank();

  auto maxRank = commsize - 1;
  int tag = 123;

  // Only root process sets test data
  if (myRank != 0) {
    runHelperMPI(myRank, maxRank, tag, MPI::COMM_WORLD);
    MPI::Finalize();
    return 0;
  }

  // Set test data
  std::cout << "-MPI Recursive Mergesort-" << std::endl;

  // Check arguments
  if (ac != 2) {
    std::cerr << "Usage: " << av[0] << " array-size" << std::endl;
    MPI::COMM_WORLD.Abort(1);
  }

  // Get argument
  auto size = atoi(av[1]); // Array size

  std::cout << "Array size = " << size << std::endl;
  std::cout << "Processes = " << commsize << std::endl;

  std::vector<int> a{};
  a.resize(size);
  auto tmp = a;

  // Random array initialization
  srand(314159);
  for (int i = 0; i < size; i++)
    a[i] = rand() % size;

  // Sort with root process
  auto start = MPI::Wtime();
  runRootMPI(a, tmp, maxRank, tag, MPI::COMM_WORLD);
  auto end = MPI::Wtime();

  std::cout << "Elapsed = " << (end - start) << std::endl;

  // Result check
  if (!std::is_sorted(a.begin(), a.end())) {
    MPI::COMM_WORLD.Abort(1);
  }

  MPI::Finalize();
  return 0;
}

// Root process code
void runRootMPI(std::vector<int> &a, std::vector<int> &temp, int maxRank,
                int tag, MPI::Comm &comm) {
  auto rank = comm.Get_rank();
  if (rank != 0) {
    std::cerr << "Error: run_root_mpi called from process " << rank
              << "; must be called from process 0 only" << std::endl;
    MPI::COMM_WORLD.Abort(1);
  }

  mergeSortParallel(a, temp, a.size(), 0, rank, maxRank, tag, comm);
  /* level=0; rank=root_rank=0; */
  return;
}

// Helper process code
void runHelperMPI(int rank, int maxRank, int tag, MPI::Comm &comm) {
  auto level = topmostLevel(rank);
  // probe for a message and determine its size and sender
  MPI::Status status{};
  comm.Probe(MPI::ANY_SOURCE, tag, status);
  auto size = status.Get_count(MPI::INT);
  auto parentRank = status.Get_source();

  std::vector<int> a{};
  a.resize(size);
  std::vector<int> tmp = a;

  comm.Recv(a.data(), size, MPI::INT, parentRank, tag);
  mergeSortParallel(a, tmp, size, level, rank, maxRank, tag, comm);
  // Send sorted array to parent process
  comm.Send(a.data(), size, MPI::INT, parentRank, tag);
  return;
}

// Given a process rank, calculate the top level of the process tree in which
// the process participates Root assumed to always have rank 0 and to
// participate at level 0 of the process tree
int topmostLevel(int rank) {
  int level = 0;
  while (pow(2, level) <= rank)
    level++;
  return level;
}

// MPI merge sort
void mergeSortParallel(std::vector<int> &a, std::vector<int> &temp, int size,
                       int level, int rank, int maxRank, int tag,
                       MPI::Comm &comm) {
  auto helperRank = rank + pow(2, level);
  if (helperRank > maxRank) { // no more processes available
    mergeSortSerial(a.data(), size, temp.data());
    return;
  }

  // printf("Process %d has helper %d\n", rank, helperRank);

  // Send second half, asynchronous
  auto request = comm.Isend(a.data() + size / 2, size - size / 2, MPI::INT,
                            helperRank, tag);
  // Sort first half
  mergeSortParallel(a, temp, size / 2, level + 1, rank, maxRank, tag, comm);
  // Free the async request (matching receive will complete the transfer).
  request.Free();

  // Receive second half sorted
  comm.Recv(a.data() + size / 2, size - size / 2, MPI::INT, helperRank, tag);

  // Merge the two sorted sub-arrays through temp
  merge(a.data(), size, temp.data());
}

void mergeSortSerial(int a[], int size, int temp[]) {
  // Switch to insertion sort for small arrays
  if (size <= SMALL) {
    insertionSort(a, size);
    return;
  }
  mergeSortSerial(a, size / 2, temp);
  mergeSortSerial(a + size / 2, size - size / 2, temp);
  // Merge the two sorted subarrays into a temp array
  merge(a, size, temp);
}

void merge(int a[], int size, int temp[]) {
  int i1 = 0;
  int i2 = size / 2;
  int tempi = 0;
  while (i1 < size / 2 && i2 < size) {
    if (a[i1] < a[i2]) {
      temp[tempi] = a[i1];
      i1++;
    } else {
      temp[tempi] = a[i2];
      i2++;
    }
    tempi++;
  }
  while (i1 < size / 2) {
    temp[tempi] = a[i1];
    i1++;
    tempi++;
  }
  while (i2 < size) {
    temp[tempi] = a[i2];
    i2++;
    tempi++;
  }
  // Copy sorted temp array into main array, a
  memcpy(a, temp, size * sizeof(int));
}

void insertionSort(int a[], int size) {
  int i;
  for (i = 0; i < size; i++) {
    int j, v = a[i];
    for (j = i - 1; j >= 0; j--) {
      if (a[j] <= v)
        break;
      a[j + 1] = a[j];
    }
    a[j + 1] = v;
  }
}
