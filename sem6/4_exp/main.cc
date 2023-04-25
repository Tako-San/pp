#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <gmpxx.h>
#include <mpi.h>

int calculateMaxN(int N);
mpz_class getPrevFact(int rank);
void sendLocSum(mpf_class &locSumFloat);
void mpz_set_ull(mpz_class &num, int_fast64_t ull);
void collect(int commSize, mpf_class &locSumFloat);
void print(int N, std::string_view sv, mpf_class &locSumFloat);
std::pair<int, int> getInterval(int termNum, int commSize, int rank);

int main(int ac, char **av) {
  MPI::Init(ac, av);

  auto commSize = MPI::COMM_WORLD.Get_size();
  auto rank = MPI::COMM_WORLD.Get_rank();

  if (ac < 2) {
    if (rank == 0)
      std::cout << "Usage: " << av[0] << " [N]" << std::endl;

    MPI::Finalize();
    return 0;
  }

  const auto N = std::atoi(av[1]);

  // Calc the number of terms we must calculate
  int termNum{};
  if (rank == 0)
    termNum = calculateMaxN(N);

  MPI::COMM_WORLD.Bcast(&termNum, 1, MPI::INT, 0);

  // Calc starts and ends of summing among processes
  auto [start, end] = getInterval(termNum, commSize, rank);

  mpz_class a = end - 1;
  mpz_class s = end;

  mpz_class locCurrFact = 1_mpz;
  mpz_class locSum = 0_mpz;

  for (auto i = end - 2; i > start; i--) {
    if (i % (1u << 13) == 0) {
      locSum += locCurrFact * s;
      locCurrFact *= a;

      mpz_set_ull(a, i);
      s = a;
    } else {
      a *= i;
      s += a;
    }
  }

  locSum += locCurrFact * s;
  locCurrFact *= a;

  // locCurrFact = start * (start + 1) * ... * (end - 2) * (end - 1)
  locCurrFact *= start;

  if (rank != 0)
    locCurrFact *= getPrevFact(rank);

  // Send next process largest factorial of THIS process
  // Still valid if we have only 1 process
  if (rank < commSize - 1) {
    auto factStrSend = locCurrFact.get_str(32);
    MPI::COMM_WORLD.Isend(factStrSend.data(), factStrSend.size() + 1, MPI::CHAR,
                          rank + 1, 0);
  }

  // By now all processes have value of their maximum factorial stored in
  // LocCurrFact Convert all integer sums to floating point ones and perform
  // division by largest factorial to get true sum of THIS process Precision
  // chosen to be 64 + [ln(10)/ln(2) * N] bits
  mpf_set_default_prec(64 + std::ceil(3.33 * N));

  mpf_class rankMaxFactFloat = locCurrFact;
  mpf_class locSumFloat = locSum / rankMaxFactFloat;

  if (rank != 0) {
    sendLocSum(locSumFloat);
  } else {
    collect(commSize, locSumFloat);
    print(N, av[1], locSumFloat);
  }

  MPI::Finalize();
  return 0;
}

int calculateMaxN(int N) {
  auto x_curr = 3.0;
  auto x_prev = x_curr;

  // x_{n + 1} = x_n - f(x_n) / f'(x_n),
  // where f(x) = x * ln(x) - N * ln(10) - x
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

std::pair<int, int> getInterval(int termNum, int commSize, int rank) {
  const auto diff = termNum / commSize;
  auto start = diff * rank + 1;
  auto end = start + diff;

  if (auto remainder = termNum % commSize; remainder != 0) {
    auto expand = (rank < remainder);
    start += expand ? rank : remainder;
    end += expand ? rank + 1 : remainder;
  }

  return {start, end};
}

void print(int N, std::string_view sv, mpf_class &locSumFloat) {
  std::string formatStr{};
  formatStr.resize(14 + sv.size());
  std::snprintf(formatStr.data(), 13 + sv.size(), "%%.%dFf\b \b\n", N + 1);
  gmp_printf(formatStr.data(), locSumFloat.get_mpf_t());
}

mpz_class getPrevFact(int rank) {
  // All processes except first receive largest factorial from prev process
  MPI::Status status{};

  // Get length of passed number
  MPI::COMM_WORLD.Probe(rank - 1, 0, status);
  auto recvLength = status.Get_count(MPI::CHAR);

  std::string factStrRecv{};
  factStrRecv.resize(recvLength);

  MPI::COMM_WORLD.Recv(factStrRecv.data(), recvLength, MPI::CHAR, rank - 1, 0);

  mpz_class rankFactFromStr{factStrRecv.data(), 32};

  // If cur process is not last, multiply largest factorial of prev
  // proc by [start * (start + 1) * ... * (end - 2) * (end - 1)] of this
  // proc to obtain largest factorial of this proc
  return rankFactFromStr;
}

void sendLocSum(mpf_class &locSumFloat) {
  auto *tmp = locSumFloat.get_mpf_t();
  MPI::COMM_WORLD.Send(&tmp->_mp_prec, 1, MPI::INT, 0, 0);
  MPI::COMM_WORLD.Send(&tmp->_mp_size, 1, MPI::INT, 0, 0);
  MPI::COMM_WORLD.Send(&tmp->_mp_exp, 1, MPI::LONG, 0, 0);

  int tmpLimbsSize = sizeof(tmp->_mp_d[0]) * tmp->_mp_size;

  MPI::COMM_WORLD.Send(&tmpLimbsSize, 1, MPI::INT, 0, 0);
  MPI::COMM_WORLD.Send(reinterpret_cast<char *>(tmp->_mp_d), tmpLimbsSize,
                       MPI::CHAR, 0, 0);
}

void collect(int commSize, mpf_class &locSumFloat) {
  locSumFloat += 1;

  mpf_class sum_i{};
  for (int i = 1; i < commSize; i++) {
    auto *tmp = sum_i.get_mpf_t();
    MPI::COMM_WORLD.Recv(&tmp->_mp_prec, 1, MPI::INT, i, 0);
    MPI::COMM_WORLD.Recv(&tmp->_mp_size, 1, MPI::INT, i, 0);
    MPI::COMM_WORLD.Recv(&tmp->_mp_exp, 1, MPI::LONG, i, 0);

    int tmpSize{};
    MPI::COMM_WORLD.Recv(&tmpSize, 1, MPI::INT, i, 0);

    auto *tmpLimbs =
        reinterpret_cast<mp_limb_t *>(calloc(tmpSize, sizeof(char)));

    MPI::COMM_WORLD.Recv(reinterpret_cast<char *>(tmpLimbs), tmpSize, MPI::CHAR,
                         i, 0);

    free(tmp->_mp_d);
    tmp->_mp_d = tmpLimbs;

    locSumFloat += sum_i;
  }
}
