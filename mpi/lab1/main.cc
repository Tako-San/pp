#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

using ldbl = long double;

constexpr ldbl a = 1;
constexpr ldbl h = 1e-3;
constexpr ldbl tau = 1e-3;
constexpr ldbl T = 1;
constexpr ldbl X = 1;
constexpr int K = T / tau + 1;
constexpr int M = X / h + 1;
constexpr ldbl PI = 3.14159265358979323846;

ldbl f(ldbl x, ldbl t) { return t + x; }

ldbl phi(ldbl x) { return std::cos(PI * x); }

ldbl psi(ldbl t) { return std::exp(-t); }

void printRes(std::ostream &ost, const std::vector<ldbl> &res) {
  ost << "tau: " << tau << std::endl;
  ost << "h: " << h << std::endl;
  ost << "T: " << T << std::endl;
  ost << "X: " << X << std::endl;
  ost << "M: " << M << std::endl;
  ost << "K: " << K << std::endl;

  for (std::size_t i = 0, sz = K * M; i < sz; ++i)
    ost << res[i] << std::endl;
}

int main(int argc, char *argv[]) {
  MPI::Init(argc, argv);

  std::vector<std::vector<ldbl>> U{};
  U.resize(K);
  for (auto &row : U)
    row.resize(M);

  // Fill initial values
  for (std::size_t k = 0; k < K; ++k)
    U[k][0] = psi(k * tau);

  for (std::size_t m = 0; m < M; ++m)
    U[0][m] = phi(m * h);

  auto rank = MPI::COMM_WORLD.Get_rank();
  auto commsize = MPI::COMM_WORLD.Get_size();

  MPI::COMM_WORLD.Barrier();
  auto tic = MPI::Wtime();

  for (auto k = rank; k < K; k += commsize)
    for (int m = 1; m < M; ++m) {
      if (k != 0) {
        MPI::COMM_WORLD.Recv(&(U[k - 1][m]), 1, MPI::LONG_DOUBLE,
                             rank ? rank - 1 : commsize - 1, m);

        auto fVal = f((m - 0.5) * h, (k - 0.5) * tau);
        U[k][m] =
            U[k - 1][m] + U[k - 1][m - 1] - U[k][m - 1] -
            a * tau / h *
                (-U[k][m - 1] + U[k - 1][m] - U[k - 1][m - 1]) +
            2 * tau * fVal;
        U[k][m] /= 1 + a * tau / h;
      }

      MPI::COMM_WORLD.Send(&(U[k][m]), 1, MPI::LONG_DOUBLE,
                           (k + 1) % commsize, m);
    }

  std::vector<ldbl> res{};
  // Fill K in a way that commsize becomes it's factor
  auto filledK = K + (commsize - K % commsize) % commsize;
  if (rank == 0)
    res.resize(filledK * M);

  // Dummy vector for free processors
  std::vector<ldbl> empty(M, 0);

  for (auto k = rank; k < filledK; k += commsize) {
    if (commsize > 1) {
      auto *src = k < K ? U[k].data() : empty.data();

      MPI::COMM_WORLD.Gather(src, M, MPI::LONG_DOUBLE, res.data() + k * M,
                             M, MPI::LONG_DOUBLE, 0);
    } else if (k < K)
      std::copy(U[k].begin(), U[k].end(), res.begin() + k * M);
    else
      break;
  }

  MPI::COMM_WORLD.Barrier();
  auto toc = MPI::Wtime();

  if (rank == 0) {
    std::cout << "Elapsed time " << toc - tic << " s." << std::endl;
    std::string name = argc > 1 ? argv[1] : "res.txt";
    std::ofstream f(name);
    printRes(f, res);
  }

  MPI::Finalize();
  return 0;
}
