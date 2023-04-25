#include <iostream>

#include <mpi.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);
  std::cout << "Hello World!" << std::endl
            << "size = " << MPI::COMM_WORLD.Get_size() << ", "
            << "rank = " << MPI::COMM_WORLD.Get_rank() << std::endl;
  MPI::Finalize();
  return 0;
}
