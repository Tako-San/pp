#include <iostream>

#include <mpi.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);

  if (ac != 2) {
    std::cout << "Usage: " << av[0] << " ITERNUM" << std::endl;
    MPI::Finalize();
    return 0;
  }

  auto size = MPI::COMM_WORLD.Get_size();
  if (size < 2) {
    std::cout << "Need more than 1 thread to test send/recv time" << std::endl;
    MPI::Finalize();
    return 0;
  }

  auto rank = MPI::COMM_WORLD.Get_rank();

  double x = 0;
  if (rank == 0) {
    auto startTime = MPI::Wtime();
    for (std::size_t i = 0; i < 100; ++i)
      MPI::COMM_WORLD.Recv(&x, 1, MPI::DOUBLE, 1, 0);
    auto finishTime = MPI::Wtime();

    std::cout << "Transfer time: " << (finishTime - startTime) * 1e6 << " us" << std::endl;
  } else if (rank == 1)
    for (std::size_t i = 0; i < 100; ++i)
      MPI::COMM_WORLD.Send(&x, 1, MPI::DOUBLE, 0, 0);

  MPI::Finalize();
  return 0;
}
