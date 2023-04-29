#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <mpi.h>

using Vec = std::vector<int>;
using VIter = Vec::iterator;

constexpr int SMALL = 32;

int topmostLevel(int rank);
void insertionSort(VIter a, int size);

void runHelperMPI(int rank, int maxRank, int tag, MPI::Comm &comm);
void runRootMPI(Vec &a, Vec &tmp, int maxRank, int tag, MPI::Comm &comm);

void merge(VIter a, VIter tmp, int size);
void mergeSortSerial(VIter a, VIter tmp, int size);
void mergeSortParallel(Vec &a, Vec &tmp, int size, int level, int rank,
                       int maxRank, int tag, MPI::Comm &comm);

int main(int ac, char **av) {
  MPI::Init(ac, av);

  auto commsize = MPI::COMM_WORLD.Get_size();
  auto rank = MPI::COMM_WORLD.Get_rank();

  auto maxRank = commsize - 1;
  int tag = 123;

  /* Only root process sets test data */
  if (rank != 0) {
    runHelperMPI(rank, maxRank, tag, MPI::COMM_WORLD);
    MPI::Finalize();
    return 0;
  }

  if (ac != 2) {
    std::cerr << "Usage: " << av[0] << " array-size" << std::endl;
    MPI::COMM_WORLD.Abort(1);
  }

  auto arrSize = atoi(av[1]); /* Array size */

  std::cout << "Array size = " << arrSize << std::endl;
  std::cout << "Processes = " << commsize << std::endl;

  Vec a{};
  a.resize(arrSize);
  auto tmp{a};

  /* Random array initialization */
  srand(314159);
  std::generate(a.begin(), a.end(), [arrSize] { return rand() % arrSize; });

  /* Sort with root process */
  auto start = MPI::Wtime();
  runRootMPI(a, tmp, maxRank, tag, MPI::COMM_WORLD);
  auto end = MPI::Wtime();

  std::cout << "Elapsed = " << (end - start) << std::endl;

  /* Result check */
  if (!std::is_sorted(a.begin(), a.end())) {
    MPI::COMM_WORLD.Abort(1);
  }

  MPI::Finalize();
  return 0;
}

/* Root process code */
void runRootMPI(Vec &a, Vec &tmp, int maxRank, int tag, MPI::Comm &comm) {
  auto rank = comm.Get_rank();
  if (rank != 0) {
    std::cerr << "Error: run_root_mpi called from process " << rank
              << "; must be called from process 0 only" << std::endl;
    MPI::COMM_WORLD.Abort(1);
  }

  mergeSortParallel(a, tmp, a.size(), 0, rank, maxRank, tag, comm);
  /* level=0; rank=root_rank=0; */
  return;
}

/* Helper process code */
void runHelperMPI(int rank, int maxRank, int tag, MPI::Comm &comm) {
  auto level = topmostLevel(rank);
  /* probe for a message and determine its size and sender */
  MPI::Status status{};
  comm.Probe(MPI::ANY_SOURCE, tag, status);
  auto size = status.Get_count(MPI::INT);
  auto parentRank = status.Get_source();

  Vec a{};
  a.resize(size);
  Vec tmp = a;

  comm.Recv(a.data(), size, MPI::INT, parentRank, tag);
  mergeSortParallel(a, tmp, size, level, rank, maxRank, tag, comm);
  /* Send sorted array to parent process */
  comm.Send(a.data(), size, MPI::INT, parentRank, tag);
  return;
}

/* Given a process rank, calculate the top level of the process tree in which */
/* the process participates Root assumed to always have rank 0 and to */
/* participate at level 0 of the process tree */
int topmostLevel(int rank) {
  int level = 0;
  while (pow(2, level) <= rank)
    ++level;
  return level;
}

/* MPI merge sort */
void mergeSortParallel(Vec &a, Vec &tmp, int size, int level, int rank,
                       int maxRank, int tag, MPI::Comm &comm) {
  auto helperRank = rank + static_cast<int>(pow(2, level));
  if (helperRank > maxRank) { /* no more processes available */
    mergeSortSerial(a.begin(), tmp.begin(), size);
    return;
  }

  /* Send second half, asynchronous */
  auto request = comm.Isend(a.data() + size / 2, size - size / 2, MPI::INT,
                            helperRank, tag);
  /* Sort first half */
  mergeSortParallel(a, tmp, size / 2, level + 1, rank, maxRank, tag, comm);
  /* Free the async request (matching receive will complete the transfer). */
  request.Free();

  /* Receive second half sorted */
  comm.Recv(a.data() + size / 2, size - size / 2, MPI::INT, helperRank, tag);

  /* Merge the two sorted sub-arrays through tmp */
  merge(a.begin(), tmp.begin(), size);
}

void mergeSortSerial(VIter a, VIter tmp, int size) {
  /* Switch to insertion sort for small arrays */
  if (size <= SMALL) {
    insertionSort(a, size);
    return;
  }
  mergeSortSerial(a, tmp, size / 2);
  mergeSortSerial(a + size / 2, tmp, size - size / 2);
  /* Merge the two sorted subarrays into a tmp array */
  merge(a, tmp, size);
}

void merge(VIter a, VIter tmp, int size) {
  int i1 = 0;
  int i2 = size / 2;
  int tmpi = 0;
  while (i1 < size / 2 && i2 < size) {
    if (a[i1] < a[i2]) {
      tmp[tmpi] = a[i1];
      ++i1;
    } else {
      tmp[tmpi] = a[i2];
      ++i2;
    }
    ++tmpi;
  }
  while (i1 < size / 2) {
    tmp[tmpi] = a[i1];
    ++i1;
    ++tmpi;
  }
  while (i2 < size) {
    tmp[tmpi] = a[i2];
    ++i2;
    ++tmpi;
  }
  /* Copy sorted tmp array into main array, a */
  std::copy_n(tmp, size, a);
}

void insertionSort(VIter a, int size) {
  for (int i = 0; i < size; ++i) {
    int j;
    auto v = a[i];
    for (j = i - 1; j >= 0; j--) {
      if (a[j] <= v)
        break;
      a[j + 1] = a[j];
    }
    a[j + 1] = v;
  }
}
