#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmp.h>
#include <mpi.h>

static inline int calculateMaxN(int N) {
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

static inline void mpz_set_ull(mpz_t n, int_fast64_t ull) {
  mpz_set_ui(n, static_cast<unsigned>(ull >> 32));
  mpz_mul_2exp(n, n, 32);
  mpz_add_ui(n, n, static_cast<unsigned>(ull));
}

int main(int argc, char *argv[]) {
  MPI::Init(argc, argv);

  auto commsize = MPI::COMM_WORLD.Get_size();
  auto rank = MPI::COMM_WORLD.Get_rank();

  if (argc < 2) {
    if (rank == 0)
      std::cout << "Usage: " << argv[0] << " [N]" << std::endl;

    MPI::Finalize();
    return 0;
  }

  const auto N = std::atoi(argv[1]);

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
  mpz_t a_mpz{};
  mpz_init_set_ui(a_mpz, end - 1);

  mpz_t S_mpz{};
  mpz_init_set_ui(S_mpz, end);

  // Main algorithm
  mpz_t locCurrFact{};
  mpz_init_set_ui(locCurrFact, 1);

  mpz_t locSum{};
  mpz_init_set_ui(locSum, 0);

  for (int i = end - 2; i > start; i--) {
    if (i % (1u << 13) == 0) {
      mpz_addmul(locSum, locCurrFact, S_mpz);
      mpz_mul(locCurrFact, locCurrFact, a_mpz);

      mpz_set_ull(a_mpz, i);
      mpz_set(S_mpz, a_mpz);
    } else {
      mpz_mul_ui(a_mpz, a_mpz, i);
      mpz_add(S_mpz, S_mpz, a_mpz);
    }
  }

  mpz_addmul(locSum, locCurrFact, S_mpz);
  mpz_mul(locCurrFact, locCurrFact, a_mpz);
  mpz_clears(a_mpz, S_mpz, NULL);

  // For all we calculate locCurrFact = start * (start + 1) * ... * (end - 2) *
  // (end - 1)
  mpz_mul_ui(locCurrFact, locCurrFact, start);

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
    mpz_t rankFactFromStr{};
    mpz_init_set_str(rankFactFromStr, factStrRecv.data(), 32);

    // If THIS process is not last, multiply largest factorial of PREVIOUS
    // process by [start * (start + 1) * ... * (end - 2) * (end - 1)] of THIS
    // process to obtain largest factorial of THIS process
    mpz_mul(locCurrFact, locCurrFact, rankFactFromStr);

    mpz_clear(rankFactFromStr);
  }

  // Send NEXT process largest factorial of THIS process
  // Still valid if we have only 1 process
  if (rank < commsize - 1) {
    auto *factStrSend = mpz_get_str(NULL, 32, locCurrFact);
    MPI::COMM_WORLD.Isend(factStrSend, strlen(factStrSend) + 1, MPI::CHAR,
                          rank + 1, 0);
  }

  // By now all processes have value of their maximum factorial stored in
  // LocCurrFact Convert all integer sums to floating point ones and perform
  // division by largest factorial to get true sum of THIS process Precision
  // chosen to be 64 + [ln(10)/ln(2) * N] bits
  mpf_set_default_prec(64 + ceil(3.33 * N));

  mpf_t locSumFloat;
  mpf_init(locSumFloat);
  mpf_set_z(locSumFloat, locSum);

  mpf_t rankMaxFactFloat;
  mpf_init(rankMaxFactFloat);
  mpf_set_z(rankMaxFactFloat, locCurrFact);

  mpf_div(locSumFloat, locSumFloat, rankMaxFactFloat);

  mpz_clears(locCurrFact, locSum, NULL);
  mpf_clear(rankMaxFactFloat);

  if (rank != 0) {
    MPI::COMM_WORLD.Send(&locSumFloat->_mp_prec, 1, MPI::INT, 0, 0);
    MPI::COMM_WORLD.Send(&locSumFloat->_mp_size, 1, MPI::INT, 0, 0);
    MPI::COMM_WORLD.Send(&locSumFloat->_mp_exp, 1, MPI::LONG, 0, 0);

    int tmpLimbsSize = sizeof(locSumFloat->_mp_d[0]) * locSumFloat->_mp_size;

    MPI::COMM_WORLD.Send(&tmpLimbsSize, 1, MPI::INT, 0, 0);
    MPI::COMM_WORLD.Send(reinterpret_cast<char *>(locSumFloat->_mp_d),
                         tmpLimbsSize, MPI::CHAR, 0, 0);

    mpf_clear(locSumFloat);
  } else {
    mpf_add_ui(locSumFloat, locSumFloat, 1);

    mpf_t sum_i;
    for (int i = 1; i < commsize; i++) {

      MPI::COMM_WORLD.Recv(&sum_i->_mp_prec, 1, MPI::INT, i, 0);
      MPI::COMM_WORLD.Recv(&sum_i->_mp_size, 1, MPI::INT, i, 0);
      MPI::COMM_WORLD.Recv(&sum_i->_mp_exp, 1, MPI::LONG, i, 0);

      int tmpSize{};
      MPI::COMM_WORLD.Recv(&tmpSize, 1, MPI::INT, i, 0);

      auto *tmpLimbs =
          reinterpret_cast<mp_limb_t *>(calloc(tmpSize, sizeof(char)));

      MPI::COMM_WORLD.Recv(reinterpret_cast<char *>(tmpLimbs), tmpSize,
                           MPI::CHAR, i, 0);

      sum_i->_mp_d = tmpLimbs;

      mpf_add(locSumFloat, locSumFloat, sum_i);
      free(tmpLimbs);
    }

    auto *formatStr =
        reinterpret_cast<char *>(calloc(14 + strlen(argv[1]), sizeof(char)));

    snprintf(formatStr, 13 + strlen(argv[1]), "%%.%dFf\b \b\n", N + 1);
    gmp_printf(formatStr, locSumFloat);

    free(formatStr);
    mpf_clear(locSumFloat);
  }

  // Finalizing MPI
  MPI::Finalize();

  return 0;
}
