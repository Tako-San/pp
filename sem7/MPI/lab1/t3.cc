#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <mpi/mpi.h>

constexpr std::size_t ISIZE = 5000;
constexpr std::size_t JSIZE = 5000;

using ArrTy = std::vector<std::vector<double>>;
using Dumper = std::function<void(const ArrTy &, std::ostream &)>;

void printArr(const ArrTy &arr, std::ostream &ost) {
  for (auto &&row : arr) {
    for (auto elem : row)
      ost << elem << " ";
    ost << std::endl;
  }
}

using ProcFunc = std::function<void(ArrTy &, ArrTy &)>;

void ethalon(ArrTy &a, ArrTy &b) {
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  for (std::size_t i = rank; i < ISIZE; i += commsize)
    for (std::size_t j = 0; j < JSIZE; ++j) {
      a[i][j] = std::sin(4 * a[i][j]);
      b[i][j] = a[i][j] * 3;
    }

  for (std::size_t i = rank; i < ISIZE; i += commsize) {
    if (rank != 0) {
      MPI::COMM_WORLD.Send(a[i].data(), ISIZE, MPI::DOUBLE, 0,
                           static_cast<int>(i));
      MPI::COMM_WORLD.Send(b[i].data(), ISIZE, MPI::DOUBLE, 0,
                           static_cast<int>(i));
    } else {
      for (std::size_t id = 1; static_cast<int>(id) < commsize; ++id)
        if (i + id < ISIZE) {
          MPI::COMM_WORLD.Recv(a[i + id].data(), ISIZE, MPI::DOUBLE, id,
                               static_cast<int>(i + id));
          MPI::COMM_WORLD.Recv(b[i + id].data(), ISIZE, MPI::DOUBLE, id,
                               static_cast<int>(i + id));
        }
    }
  }
}

void processArr(ArrTy &a, ArrTy &b) {
  // Original cycles
  //
  // for (std::size_t i = 0; i < ISIZE; i++)
  //   for (std::size_t j = 0; j < JSIZE; j++)
  //     arr[i][j] = std::sin(0.001 * arr[i][j]);
  //
  // for (std::size_t i = 0; i < ISIZE - 3; ++i)
  //   for (std::size_t j = 5; j < JSIZE; ++j)
  //     b[i][j] = a[i + 3][j - 5] * 3;

  // Normalized version
  std::size_t i = 0;
  for (; i < 3; i++) {
    for (std::size_t j = 0; j < JSIZE; j++) {
      a[i][j] = std::sin(0.001 * a[i][j]);
    }
  }

  for (; i < ISIZE; i++) {
    std::size_t j = 0;
    for (; j < JSIZE - 5; j++) {
      a[i][j] = std::sin(0.001 * a[i][j]);
      b[i - 3][j + 5] = a[i][j] * 3;
    }

    for (; j < JSIZE; j++)
      a[i][j] = std::sin(0.001 * a[i][j]);
  }
}

void processArrPar(ArrTy &aT, ArrTy &bT) {
  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  std::size_t i = 0;
  for (; i < 3; i++) {
    for (std::size_t j = rank; j < JSIZE; j += commsize) {
      aT[j][i] = std::sin(0.001 * aT[j][i]);
    }
  }

  for (; i < ISIZE; i++) {
    MPI::COMM_WORLD.Barrier();
    std::size_t j = rank;
    for (; j < JSIZE - 5; j += commsize) {
      aT[j][i] = std::sin(0.001 * aT[j][i]);
      bT[j + 5][i - 3] = aT[j][i] * 3;
    }

    for (; j < JSIZE; j += commsize)
      aT[j][i] = std::sin(0.001 * aT[j][i]);
  }

  for (std::size_t j = rank; j < JSIZE; j += commsize) {
    if (rank != 0) {
      MPI::COMM_WORLD.Send(aT[j].data(), JSIZE, MPI::DOUBLE, 0,
                           static_cast<int>(j));
      if (j + 5 < JSIZE)
        MPI::COMM_WORLD.Send(bT[j + 5].data(), JSIZE, MPI::DOUBLE, 0,
                             static_cast<int>(j + 5));
    } else {
      for (std::size_t id = 1; static_cast<int>(id) < commsize; ++id)
        if (j + id < JSIZE) {
          MPI::COMM_WORLD.Recv(aT[j + id].data(), JSIZE, MPI::DOUBLE, id,
                               static_cast<int>(j + id));
          if (j + id + 5 < JSIZE)
            MPI::COMM_WORLD.Recv(bT[j + id + 5].data(), JSIZE, MPI::DOUBLE, id,
                                 static_cast<int>(j + id + 5));
        }
    }
  }
}

void measureDump(ProcFunc f, ArrTy &a, ArrTy &b, std::string_view filename,
                 Dumper dumper) {
  auto rank = MPI::COMM_WORLD.Get_rank();
  MPI::COMM_WORLD.Barrier();
  auto tic = MPI::Wtime();
  f(a, b);
  MPI::COMM_WORLD.Barrier();
  auto toc = MPI::Wtime();

  if (rank != 0) {
    a.clear();
    return;
  }

  std::cout << "Elapsed time " << (toc - tic) * 1000 << " ms" << std::endl;

  std::ofstream of(filename.data());
  if (!of) {
    std::cerr << "Cannot open file for results" << std::endl;
    return;
  }
  dumper(a, of);
  dumper(b, of);
}

void initArr(ArrTy &a, ArrTy &b) {
  a.resize(ISIZE);
  b.resize(ISIZE);
  // Fill array with data
  for (std::size_t i = 0; i < a.size(); i++) {
    auto &ai = a[i];
    ai.resize(JSIZE);
    b[i].resize(JSIZE);
    for (std::size_t j = 0; j < ai.size(); j++)
      ai[j] = 10 * i + j;
  }
}

void initTArr(ArrTy &a, ArrTy &b) {
  a.resize(JSIZE);
  b.resize(JSIZE);
  // Fill array with data
  for (std::size_t j = 0; j < a.size(); j++) {
    auto &aj = a[j];
    aj.resize(ISIZE);
    b[j].resize(ISIZE);
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
  ArrTy b{};
  initArr(a, b);

  std::cout << "Sequential:" << std::endl;
  measureDump(processArr, a, b, "seq.txt", printArr);
}

void doPar() {
  ArrTy aT{};
  ArrTy bT{};
  initTArr(aT, bT);

  if (MPI::COMM_WORLD.Get_rank() == 0)
    std::cout << "Parallel:" << std::endl;
  measureDump(processArrPar, aT, bT, "par.txt", printTArr);
}

void doEth() {
  ArrTy a{};
  ArrTy b{};
  initArr(a, b);

  if (MPI::COMM_WORLD.Get_rank() == 0)
    std::cout << "Ethalon:" << std::endl;
  measureDump(ethalon, a, b, "eth.txt", printArr);
}

int main(int argc, char *argv[]) {
  MPI::Init(argc, argv);
  auto commsz = MPI::COMM_WORLD.Get_size();

  if (commsz == 1)
    doSeq();
  doPar();
  doEth();

  MPI::Finalize();
}
