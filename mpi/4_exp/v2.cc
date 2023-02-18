#include <concepts>
#include <cstddef>
#include <iostream>
#include <limits>

#include <mpi.h>

template <std::integral T> T ceil(T dividend, T divider);
double worker(int N, int curRank, int ncpus);
void manager(int N, int curRank, int ncpus);

int main(int ac, char **av) {
  if (ac < 2) {
    std::cout << "USAGE: " << av[0] << " N" << std::endl;
    return 0;
  }

  auto N = atoi(av[1]);

  MPI::Init(ac, av);
  auto curRank = MPI::COMM_WORLD.Get_rank();
  auto ncpus = MPI::COMM_WORLD.Get_size();

  if (0 == curRank) {
    manager(N, curRank, ncpus);
  } else {
    auto partialSum = worker(N, curRank, ncpus);
    MPI::COMM_WORLD.Send(&partialSum, 1, MPI::DOUBLE, 0, 0);
  }

  MPI::Finalize();
  return 0;
}

template <std::integral T> T ceil(T dividend, T divider) {
  return (dividend + divider - 1) / divider;
}

double worker(int N, int curRank, int ncpus) {
  /* number of terms for each thread */
  auto termN = ceil(N, ncpus);

  double partialSum = 0;
  double factTail = (curRank != 0) ? (curRank * termN) : 1;

  auto start = curRank * termN + 1;
  auto next = start + termN;
  auto end = (next > N) ? (N + 1) : next;
  for (auto n = start; n < end; ++n) {
    partialSum += 1 / factTail;
    factTail *= n;
  }

  return partialSum;
}

void manager(int N, int curRank, int ncpus) {
  /* number of terms for each thread */
  auto n = ceil(N, ncpus);

  auto result = worker(N, curRank, ncpus);

  decltype(result) fact = 1;
  for (auto j = 2; j < n; ++j)
    fact *= j;
  /* here fact equals to (n - 1)! */

  for (int i = 1; i < ncpus; ++i) {
    decltype(result) tmp{};
    MPI::COMM_WORLD.Recv(&tmp, 1, MPI::DOUBLE, i, 0);
    result += tmp / fact;

    for (auto j = i * n, end = (i + 1) * n; j < end; ++j)
      fact *= j;
    /* now fact equals to (i * n - 1)! */
  }

  std::cout << "N = " << N << std::endl;
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cout << "e = " << result << std::endl;
}
