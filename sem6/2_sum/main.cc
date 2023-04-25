#include <iostream>

#include <mpi.h>

int main(int ac, char **av) {
  if (ac < 2) {
    std::cout << "USAGE: " << av[0] << " N" << std::endl;
    return 0;
  }

  double result = 0;
  auto N = atoi(av[1]);

  MPI::Init(ac, av);

  double partialSum = 0;
  auto ncpus = MPI::COMM_WORLD.Get_size();
  for (auto n = MPI::COMM_WORLD.Get_rank() + 1; n <= N; n += ncpus) {
    partialSum += 1.0 / n;
  }

  MPI::COMM_WORLD.Reduce(&partialSum, &result, 1, MPI::DOUBLE, MPI::SUM, 0);

  if (0 == MPI::COMM_WORLD.Get_rank()) {
    std::cout << "N = " << N << std::endl;
    std::cout << "sum = " << result << std::endl;
  }

  MPI::Finalize();
  return 0;
}
