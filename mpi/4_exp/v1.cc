#include <cstddef>
#include <iostream>
#include <limits>

#include <mpi.h>

int main(int ac, char **av) {
  if (ac < 2) {
    std::cout << "USAGE: " << av[0] << " N" << std::endl;
    return 0;
  }

  double result = 0;
  auto N = atoi(av[1]);

  MPI::Init(ac, av);
  auto curRank = MPI::COMM_WORLD.Get_rank();
  auto ncpus = MPI::COMM_WORLD.Get_size();

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

  MPI::COMM_WORLD.Reduce(&partialSum, &result, 1, MPI::DOUBLE, MPI::SUM, 0);

  if (0 == curRank) {
    std::cout << "N = " << N << std::endl;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "e = " << result << std::endl;
  }

  MPI::Finalize();
  return 0;
}
