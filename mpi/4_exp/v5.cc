#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmpxx.h>
#include <mpi.h>

static int calculateMaxN(int N);
static void mpz_set_ull(mpz_class &num, int_fast64_t ull);

int main(int ac, char **av) {
  MPI::Init(ac, av);

  auto commsize = MPI::COMM_WORLD.Get_size();
  auto rank = MPI::COMM_WORLD.Get_rank();

  if (ac < 2) {
    if (rank == 0)
      std::cout << "Usage: " << av[0] << " [N]" << std::endl;

    MPI::Finalize();
    return 0;
  }

  const auto N = std::atoi(av[1]);

  // Calculate to which term we must calculate for given accuracy and send to
  // all processes this information Still valid if we have only 1 process
  int maxFact{};
  if (rank == commsize - 1)
    maxFact = calculateMaxN(N);

  MPI::COMM_WORLD.Bcast(&maxFact, 1, MPI::INT, commsize - 1);

  // Distributing starts and ends of summing among processes
  const auto diff = maxFact / commsize;
  auto start = 1 + diff * rank;
  auto end = 1 + diff * (rank + 1);

  if (maxFact % commsize) {
    if (rank < maxFact % commsize) {
      start += rank;
      end += rank + 1;
    } else {
      start += (maxFact % commsize);
      end += (maxFact % commsize);
    }
  }
  mpz_class a_mpz{static_cast<unsigned>(end - 1)};
  mpz_class S_mpz{static_cast<unsigned>(end)};

  // Main algorithm
  mpz_class locCurrFact = 1_mpz;
  mpz_class locSum = 0_mpz;

  for (int i = end - 2; i > start; i--) {
    if (i % (1u << 13) == 0) {
      locSum += locCurrFact * S_mpz;
      locCurrFact *= a_mpz;

      mpz_set_ull(a_mpz, i); // mpzzz_ull
      S_mpz = a_mpz;
    } else {
      a_mpz *= i;
      S_mpz += a_mpz;
    }
  }

  locSum += locCurrFact * S_mpz;
  locCurrFact *= a_mpz;

  // For all we calculate locCurrFact = start * (start + 1) * ... * (end - 2) *
  // (end - 1)
  locCurrFact *= start;

  if (rank != 0) {
    // For all except first processes we receive largest factorial of PREVIOUS
    // process
    MPI::Status status{};

    // Use MPI_Probe and MPI_Get_count to obtain length of passed number
    MPI::COMM_WORLD.Probe(rank - 1, 0, status);
    auto recvLength = status.Get_count(MPI::CHAR);

    std::string factStrRecv{};
    factStrRecv.resize(recvLength);
    // auto *factStrRecv = (char *)calloc(recvLength, sizeof(char));

    MPI::COMM_WORLD.Recv(factStrRecv.data(), recvLength, MPI::CHAR, rank - 1,
                         0);

    // By these two lines RankFactFromStr is the largest factorial of PREVIOUS
    // process
    mpz_class rankFactFromStr{factStrRecv.data(), 32};

    // If THIS process is not last, multiply largest factorial of PREVIOUS
    // process by [start * (start + 1) * ... * (end - 2) * (end - 1)] of THIS
    // process to obtain largest factorial of THIS process
    locCurrFact *= rankFactFromStr;
  }

  // Send NEXT process largest factorial of THIS process
  // Still valid if we have only 1 process
  if (rank < commsize - 1) {
    auto factStrSend = locCurrFact.get_str(32);
    MPI::COMM_WORLD.Isend(factStrSend.data(), factStrSend.size() + 1, MPI::CHAR,
                          rank + 1, 0);
  }

  // By now all processes have value of their maximum factorial stored in
  // LocCurrFact Convert all integer sums to floating point ones and perform
  // division by largest factorial to get true sum of THIS process Precision
  // chosen to be 64 + [ln(10)/ln(2) * N] bits
  mpf_set_default_prec(64 + ceil(3.33 * N));

  mpf_class locSumFloat{locSum};
  mpf_class rankMaxFactFloat{locCurrFact};

  locSumFloat = locSumFloat / rankMaxFactFloat;

  if (rank != 0) {
    auto *tmp = locSumFloat.get_mpf_t();
    MPI::COMM_WORLD.Send(&tmp->_mp_prec, 1, MPI::INT, 0, 0);
    MPI::COMM_WORLD.Send(&tmp->_mp_size, 1, MPI::INT, 0, 0);
    MPI::COMM_WORLD.Send(&tmp->_mp_exp, 1, MPI::LONG, 0, 0);

    int tmpLimbsSize = sizeof(tmp->_mp_d[0]) * tmp->_mp_size;

    MPI::COMM_WORLD.Send(&tmpLimbsSize, 1, MPI::INT, 0, 0);
    MPI::COMM_WORLD.Send(reinterpret_cast<char *>(tmp->_mp_d), tmpLimbsSize,
                         MPI::CHAR, 0, 0);

  } else {
    locSumFloat += 1;

    mpf_class sum_i{};
    for (int i = 1; i < commsize; i++) {
      auto *tmp = sum_i.get_mpf_t();
      MPI::COMM_WORLD.Recv(&tmp->_mp_prec, 1, MPI::INT, i, 0);
      MPI::COMM_WORLD.Recv(&tmp->_mp_size, 1, MPI::INT, i, 0);
      MPI::COMM_WORLD.Recv(&tmp->_mp_exp, 1, MPI::LONG, i, 0);

      int tmpSize{};
      MPI::COMM_WORLD.Recv(&tmpSize, 1, MPI::INT, i, 0);

      auto *tmpLimbs =
          reinterpret_cast<mp_limb_t *>(calloc(tmpSize, sizeof(char)));

      MPI::COMM_WORLD.Recv(reinterpret_cast<char *>(tmpLimbs), tmpSize,
                           MPI::CHAR, i, 0);

      free(tmp->_mp_d);
      tmp->_mp_d = tmpLimbs;

      locSumFloat += sum_i;
    }

    auto *formatStr =
        reinterpret_cast<char *>(calloc(14 + strlen(av[1]), sizeof(char)));

    std::snprintf(formatStr, 13 + strlen(av[1]), "%%.%dFf\b \b\n", N + 1);
    gmp_printf(formatStr, locSumFloat.get_mpf_t());

    free(formatStr);
  }

  MPI::Finalize();
  return 0;
}

int calculateMaxN(int N) {
  auto x_curr = 3.0;
  auto x_prev = x_curr;

  // x_{n + 1} = x_n - f(x_n)/f'(x_n),
  // where f(x) = x*ln(x) - x - N*ln(10)
  do {
    x_prev = x_curr;
    x_curr = (x_curr + N * std::log(10)) / std::log(x_curr);
  } while (std::fabs(x_curr - x_prev) > 1.0);

  return static_cast<int>(std::ceil(x_curr));
}

void mpz_set_ull(mpz_class &num, int_fast64_t ull) {
  auto n = num.get_mpz_t();
  mpz_set_ui(n, static_cast<unsigned>(ull >> 32));
  mpz_mul_2exp(n, n, 32);
  mpz_add_ui(n, n, static_cast<unsigned>(ull));
}
