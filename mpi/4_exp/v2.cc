#include <cstddef>
#include <iostream>
#include <limits>

#include <mpi.h>

double worker(int N, int curRank, int ncpus) {
  double fact = 1;
  for (auto j = 2; j <= curRank; ++j)
    fact *= j;

  double partialSum = 0;
  for (auto i = curRank + 1; i <= N; i += ncpus) {
    auto term = 1 / fact;
    for (auto j = i; j < i + ncpus; ++j)
      fact *= j;

    partialSum += term;
  }

  return partialSum;
}

void manager(int N, int curRank, int ncpus) {
  double result = worker(N, curRank, ncpus);

  for (int i = 1; i < ncpus; ++i) {
    double tmp{};
    MPI::COMM_WORLD.Recv(&tmp, 1, MPI::DOUBLE, i, 0);

    result += tmp;
  }

  std::cout << "N = " << N << std::endl;
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cout << "e = " << result << std::endl;
}

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
