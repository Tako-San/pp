#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#include <vector>

void sayHello(size_t id) {
  std::stringstream ss{};
  ss << "Hello world from #" << id << "!" << std::endl;
  std::cout << ss.str();
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
  std::generate_n(std::back_inserter(threads), tnum, [i = 0]() mutable {
    return std::thread{sayHello, i++};
  });

  for (auto &&thread : threads)
    thread.join();

  return 0;
}
