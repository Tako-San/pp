#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <numeric>
#include <vector>

#include <future>
#include <thread>

namespace chr = std::chrono;

using ldbl = long double;

constexpr ldbl k2Pi = 2 * std::numbers::pi_v<ldbl>;

constexpr ldbl kFrom = 0.001;
constexpr ldbl kTo = 5;

constexpr ldbl kPerNumLBDL = (1 / kFrom - 1 / kTo) / k2Pi;
constexpr auto kPerNum = static_cast<std::uint64_t>(kPerNumLBDL);
constexpr ldbl kPerDiff = kPerNumLBDL - kPerNum;

auto f(ldbl x) { return std::sin(1 / x); };

auto doSimpsonIter(ldbl a, ldbl b) {
  return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b));
};

auto runSimpson(ldbl from, ldbl to, ldbl step) {
  ldbl res = 0;
  for (auto start = from; start < to; start += step)
    res += doSimpsonIter(start, start + step);
  return res;
}

auto integrate(ldbl from, ldbl to, ldbl accuracy) {
  auto step = std::cbrt(accuracy);

  ldbl delta = 0;
  ldbl intVal = 0;
  auto prevIntVal = intVal;

  do {
    intVal = runSimpson(from, to, step);
    delta = intVal - prevIntVal;
    prevIntVal = intVal;
    step /= 2;
  } while (std::abs(delta) > accuracy);

  return intVal;
}

int main(int ac, char **av) {
  if (ac != 3) {
    std::cerr << "USAGE: " << av[0] << " [THREAD_NUM] [ACCURACY]" << std::endl;
    return 1;
  }

  auto threadNum = std::atoi(av[1]);
  if (threadNum < 1) {
    std::cerr << "Incorrect threads amount: " << threadNum << std::endl;
    return 1;
  }

  ldbl accuracy = std::abs(std::atof(av[2]));

  std::vector<std::future<ldbl>> terms(threadNum);

  auto perForThread = kPerNum / threadNum;
  ldbl invFrom = 1 / kFrom;
  ldbl invTo = invFrom - k2Pi * (perForThread + kPerNum % threadNum + kPerDiff);
  invTo = std::max(invTo, 1 / kTo);

  auto start = chr::high_resolution_clock::now();
  for (auto &&term : terms) {
    term = std::async(std::launch::async, integrate, 1 / invFrom, 1 / invTo,
                      accuracy / (threadNum * threadNum));

    invFrom = invTo;
    invTo -= k2Pi * perForThread;
    invTo = std::max(invTo, 1 / kTo);
  }

  ldbl res = 0;
  for (auto &&term : terms)
    res += term.get();

  auto finish = chr::high_resolution_clock::now();

  std::cout.precision(std::numeric_limits<ldbl>::max_digits10);
  std::cout << "I = " << res << std::endl;

  auto ms = chr::duration_cast<chr::milliseconds>(finish - start).count();
  std::cout << "Elapsed time " << ms << "ms" << std::endl;

  return 0;
}
