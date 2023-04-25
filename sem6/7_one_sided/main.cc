#include <iostream>

#include <mpi.h>

using T = double;

int main(int ac, char **av) {
  if (ac < 2) {
    std::cout << "USAGE: " << av[0] << " N" << std::endl;
    return 0;
  }

  auto N = atoi(av[1]);
  MPI::Init(ac, av);

  auto commSize = MPI::COMM_WORLD.Get_size();
  auto rank = MPI::COMM_WORLD.Get_rank();

  double sum = 0;
  for (auto n = rank + 1; n <= N; n += commSize) {
    sum += 1.0 / n;
  }

  auto win = MPI::Win::Create(&sum, sizeof(T), sizeof(T), MPI::INFO_NULL,
                              MPI::COMM_WORLD);

  win.Fence(0);
  if (rank != 0)
    win.Accumulate(&sum, 1, MPI::DOUBLE, 0, 0, 1, MPI::DOUBLE, MPI::SUM);
  win.Fence(0);

  if (0 == rank) {
    std::cout << "N = " << N << std::endl;
    std::cout << "sum = " << sum << std::endl;
  }

  win.Free();
  MPI::Finalize();
  return 0;
}
