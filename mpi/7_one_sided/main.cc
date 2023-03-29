#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Hey, Buddy, you choose the wrong door, the parallel club has two "
           "args of command line\n");
    return 0;
  }

  int num = atoi(argv[1]);

  int commSize = 0, commRank = 0;
  MPI_Win win;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  double sum = 0;
  for (int i = num / commSize * commRank + 1,
           end = (commRank == commSize - 1)
                     ? num + 1
                     : num / commSize * (commRank + 1) + 1;
       i < end; ++i) {
    sum += 1.0 / i;
  }

  MPI_Win_create(&sum, 1 * sizeof(double), sizeof(double), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &win);

  MPI_Win_fence(0, win);

  if (commRank != 0)
    MPI_Accumulate(&sum, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, win);

  MPI_Win_fence(0, win);

  if (commRank == 0)
    printf("result = %f\n", sum);

  MPI_Win_fence(0, win);

  MPI_Win_free(&win);
  // MPI_Free_mem(result);
  MPI_Finalize();

  return 0;
}