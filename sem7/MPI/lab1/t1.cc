#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <mpi/mpi.h>

constexpr std::size_t ISIZE = 5000;
constexpr std::size_t JSIZE = 5000;

constexpr auto ISIZE_USED = ISIZE - 8;
constexpr auto JSIZE_USED = JSIZE - 3;

using ArrTy = std::vector<std::vector<double>>;
using Dumper = std::function<void(const ArrTy &, std::ostream &)>;

void printArr(const ArrTy &arr, std::ostream &ost) {
  for (auto &&row : arr) {
    for (auto elem : row)
      ost << elem << " ";
    ost << std::endl;
  }
}

using ProcFunc = std::function<void(ArrTy &)>;

void ethalon(ArrTy &arr) {
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  for (std::size_t i = rank; i < ISIZE; i += commsize)
    for (std::size_t j = 0; j < JSIZE; ++j)
      arr[i][j] = std::sin(4 * arr[i][j]);

  for (std::size_t i = rank; i < ISIZE; i += commsize) {
    if (rank != 0) {
      MPI::COMM_WORLD.Send(arr[i].data(), ISIZE, MPI::DOUBLE, 0,
                           static_cast<int>(i));
    } else {
      for (std::size_t id = 1; static_cast<int>(id) < commsize; ++id)
        if (i + id < ISIZE)
          MPI::COMM_WORLD.Recv(arr[i + id].data(), ISIZE, MPI::DOUBLE, id,
                               static_cast<int>(i + id));
    }
  }
}

void processArr(ArrTy &arr) {
  // Original cycle
  // for (std::size_t i = 8; i < ISIZE; i++)
  //   for (std::size_t j = 0; j < JSIZE - 3; j++)
  //     arr[i][j] = std::sin(4 * arr[i - 8][j - 3]);
  // Normalized version
  for (std::size_t i = 0; i < ISIZE_USED; i++)
    for (std::size_t j = 0; j < JSIZE_USED; j++) {
      arr[i + 8][j] = std::sin(4 * arr[i][j + 3]);
      // if (i == ISIZE_USED - 1) {
      //   std::cout << "arr[" << (i + 8) << "][" << j << "] = " << arr[i + 8][j] << std::endl;
      //   std::cout << "arr[" << i << "][" << (j + 3) << "] = " << arr[i][j + 3] << std::endl;
      //   std::cout << std::endl;
      // }
    }
}
// Check bernstein condition:
// F(k1, k2) = (k1 + 8, k2)
// G(l1, l2) = (l1, l2 + 3)
// Have a system
// k1 + 8 = l1
// k2 = l2 + 3
// => l = (k1 + 8, k2 - 3)
// D = l - k = (8, -3) => d = (>, <)
// > ==> i true-dependency
// < ==> j false-dependency

void processArrPar(ArrTy &arrT) {
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  for (std::size_t i = 0; i < ISIZE_USED; i++) {
    MPI::COMM_WORLD.Barrier();
    for (std::size_t j = rank; j < JSIZE_USED; j += commsize) {
      arrT[j][i + 8] = std::sin(4 * arrT[j + 3][i]);
      // if (i == ISIZE_USED - 1) {
      //   std::cout << "arrT[" << j << "][" << (i + 8) << "] = " << arrT[j][i + 8] << std::endl;
      //   std::cout << "arrT[" << (j + 3) << "][" << i << "] = " << arrT[j + 3][i] << std::endl;
      //   std::cout << std::endl;
      // }
    }
  }

  for (std::size_t i = rank; i < ISIZE; i += commsize) {
    if (rank != 0) {
      MPI::COMM_WORLD.Send(arrT[i].data(), ISIZE, MPI::DOUBLE, 0,
                           static_cast<int>(i));
    } else {
      for (std::size_t id = 1; static_cast<int>(id) < commsize; ++id)
        if (i + id < ISIZE)
          MPI::COMM_WORLD.Recv(arrT[i + id].data(), ISIZE, MPI::DOUBLE, id,
                               static_cast<int>(i + id));
    }
  }
}

void measureDump(ProcFunc f, ArrTy &arr, std::string_view filename,
                 Dumper dumper) {
  auto rank = MPI::COMM_WORLD.Get_rank();
  MPI::COMM_WORLD.Barrier();
  auto tic = MPI::Wtime();
  f(arr);
  MPI::COMM_WORLD.Barrier();
  auto toc = MPI::Wtime();

  if (rank != 0) {
    arr.clear();
    return;
  }

  std::cout << "Elapsed time " << (toc - tic) * 1000 << " ms" << std::endl;

  std::ofstream of(filename.data());
  if (!of) {
    std::cerr << "Cannot open file for results" << std::endl;
    return;
  }
  dumper(arr, of);
}

void initArr(ArrTy &a) {
  a.resize(ISIZE);
  // Fill array with data
  for (std::size_t i = 0; i < a.size(); i++) {
    auto &ai = a[i];
    ai.resize(JSIZE);
    for (std::size_t j = 0; j < ai.size(); j++)
      ai[j] = 10 * i + j;
  }
}

void initTArr(ArrTy &a) {
  a.resize(JSIZE);
  // Fill array with data
  for (std::size_t j = 0; j < a.size(); j++) {
    auto &aj = a[j];
    aj.resize(ISIZE);
    for (std::size_t i = 0; i < aj.size(); i++)
      aj[i] = 10 * i + j;
  }
}

void printTArr(const ArrTy &a, std::ostream &ost) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < a[i].size(); ++j)
      ost << a[j][i] << " ";
    ost << std::endl;
  }
}

void doSeq() {
  ArrTy a{};
  initArr(a);

  std::cout << "Sequential:" << std::endl;
  measureDump(processArr, a, "seq.txt", printArr);
}

void doPar() {
  ArrTy aT{};
  initTArr(aT);

  if (MPI::COMM_WORLD.Get_rank() == 0)
    std::cout << "Parallel:" << std::endl;
  measureDump(processArrPar, aT, "par.txt", printTArr);
}

void doEth() {
  ArrTy a{};
  initArr(a);

  if (MPI::COMM_WORLD.Get_rank() == 0)
    std::cout << "Ethalon:" << std::endl;
  measureDump(ethalon, a, "eth.txt", printArr);
}

int main(int argc, char *argv[]) {
  MPI::Init(argc, argv);
  auto commsz = MPI::COMM_WORLD.Get_size();

  if (commsz == 1)
    doSeq();
  // else if (commsz == 3)
    doPar(), doEth();
  // else if (MPI::COMM_WORLD.Get_rank() == 0)
  //   std::cerr << "Commsize is not right " << commsz << " (required 3)"
  //             << std::endl;

  MPI::Finalize();
}
