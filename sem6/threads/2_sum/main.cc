#include <cstddef>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

using ldbl = long double;

auto calcPartSum(size_t termNum, size_t threadNum, size_t threadID) {
  ldbl res = 0;
  for (auto i = threadID; i <= termNum; i += threadNum)
    res += 1 / static_cast<ldbl>(i);
  return res;
}

int main(int ac, char **av) {
  if (ac != 3) {
    std::cerr << "USAGE: " << av[0] << " [THREAD_NUM] [TERM_NUM]" << std::endl;
    return 1;
  }

  auto threadNum = std::atol(av[1]);
  if (threadNum < 1) {
    std::cerr << "Incorrect thread amount: " << threadNum << std::endl;
    return 1;
  }

  auto termNum = std::atol(av[2]);
  if (termNum < 1) {
    std::cerr << "Incorrect term amount: " << termNum << std::endl;
    return 1;
  }

  std::vector<std::future<ldbl>> threads{};
  std::generate_n(std::back_inserter(threads), threadNum, [&, i = 1]() mutable {
    return std::async(std::launch::async, calcPartSum, termNum, threadNum, i++);
  });

  ldbl res = 0;
  for (auto &&thread : threads)
    res += thread.get();

  std::cout << res << std::endl;
  return 0;
}