#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <mpi/mpi.h>

/* Function to sum reciprocals */
double calcSum(int64_t start, int64_t end) {
  double partSum = 0.0;

  for (int64_t i = start; i < end; i++)
    partSum += 1.0 / static_cast<double>(i);

  return partSum;
}

int main(int ac, char **av) {
  MPI::Init(ac, av);

  auto commsize = MPI::COMM_WORLD.Get_size();
  auto rank = MPI::COMM_WORLD.Get_rank();

  /* Handle wrong input */
  if (ac < 2) {
    if (!rank)
      std::cout << "Usage: " << av[0] << " [N]" << std::endl;

    MPI::Finalize();
    return 0;
  }

  /* Split COMM_WORLD into two parts: one for proc #0 and second for the rest */
  auto myComm = MPI::COMM_WORLD.Split(!!rank, rank);
  if (MPI::Comm_Null() == myComm) {
    std::cout << "Proc " << rank << ": failed to split communicator"
              << std::endl;
    MPI::Finalize();
    return 0;
  }

  /* Now obtain new rank withing new communicators */
  auto myCommSize = myComm.Get_size();
  auto myCommRank = myComm.Get_rank();

  /* Just print all information */
  std::cout << "COMM_WORLD" << std::endl;
  std::cout << "  rank = " << rank << std::endl;
  std::cout << "  commsize = " << commsize << std::endl;

  std::cout << "myComm" << std::endl;
  std::cout << "  rank = " << myCommRank << std::endl;
  std::cout << "  commsize = " << myCommSize << std::endl << std::endl;

  /* Calculate sum for proc from 1 to N inside myCommcommucator */
  if (rank == 0) {
    MPI::Finalize();
    return 0;
  }

  const auto N = std::atoll(av[1]);
  const auto diff = N / myCommSize;
  auto start = 1 + diff * myCommRank;
  auto end = 1 + diff * (myCommRank + 1);

  if (N % myCommSize) {
    if (myCommRank < N % myCommSize) {
      start += myCommRank;
      end += myCommRank + 1;
    } else {
      start += (N % myCommSize);
      end += (N % myCommSize);
    }
  }

  /* Calculate sum for each process */
  auto partSum = calcSum(start, end);

  /* Collecting all into 0-th process */
  double totalSum{};
  myComm.Reduce(&partSum, &totalSum, 1, MPI::DOUBLE, MPI::SUM, 0);
  if (myCommRank == 0) {
    std::cout << "***************" << std::endl;
    std::cout << "SUM = " << totalSum << std::endl;
    std::cout << "***************" << std::endl;
  }

  MPI::Finalize();
  return 0;
}
