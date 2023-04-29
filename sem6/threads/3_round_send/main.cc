#include <cstddef>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

int msg = 0;
std::mutex msgMutex{};

void runThread(size_t tid) {
  std::lock_guard guard{msgMutex};
  std::cout << "#" << tid << ", msg: " << msg++ << std::endl;
}

int main(int ac, char **av) {
  if (ac != 2) {
    std::cerr << "USAGE: " << av[0] << " [THREAD_NUM]" << std::endl;
    return 1;
  }

  auto tnum = std::atoi(av[1]);
  if (tnum < 1) {
    std::cout << "Incorrect thread amount: " << tnum << std::endl;
    return 1;
  }

  std::vector<std::thread> threads{};
  std::generate_n(std::back_inserter(threads), tnum, [tid = 0]() mutable {
    return std::thread{runThread, tid++};
  });

  for (auto &&thread : threads)
    thread.join();

  return 0;
}
