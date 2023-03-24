#include <mpi.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);

  MPI::COMM_SELF.Spawn("6_server", MPI::ARGV_NULL, 1, MPI::INFO_NULL, 0);
  MPI::COMM_SELF.Spawn("6_client", MPI::ARGV_NULL, 1, MPI::INFO_NULL, 0);

  MPI::Finalize();
  return 0;
}
