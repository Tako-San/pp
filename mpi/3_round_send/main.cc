#include <iostream>

#include <mpi.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);

  unsigned msg = 0;
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto size = MPI::COMM_WORLD.Get_size();
  if (0 == rank) {
    MPI::COMM_WORLD.Send(&msg, 1, MPI::UNSIGNED, (rank + 1) % size, 0);
    MPI::COMM_WORLD.Recv(&msg, 1, MPI::UNSIGNED, size - 1, 0);
    std::cout << "T" << rank << ": " << msg << std::endl;
  } else {
    MPI::COMM_WORLD.Recv(&msg, 1, MPI::UNSIGNED, rank - 1, 0);
    std::cout << "T" << rank << ": " << msg++ << std::endl;
    MPI::COMM_WORLD.Send(&msg, 1, MPI::UNSIGNED, (rank + 1) % size, 0);
  }

  MPI::Finalize();
  return 0;
}
