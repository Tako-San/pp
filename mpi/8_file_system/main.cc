#include <mpi.h>
#include <stdio.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commSize = MPI::COMM_WORLD.Get_size();

  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, "test.out", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  char c = '0' + commSize - rank;
  MPI_File_write_at(fh, rank, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  MPI::Finalize();
  return 0;
}