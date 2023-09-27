#include <iostream>

#include <omp.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "USAGE: " << argv[0] << " THREAD_NUM" << std::endl;
    return -1;
  }

  auto termNum = std::stoi(argv[1]);
  std::cout << "Term number: " << termNum << std::endl;

  auto threadNum = omp_get_max_threads();
  std::cout << "Thread number: " << threadNum << std::endl;

  double sum = 0;

#pragma omp parallel reduction(+ : sum)
  {
    for (auto n = omp_get_thread_num() + 1; n <= termNum; n += threadNum) {
      sum += 1.0 / n;
    }
  }

  std::cout << "Sum: " << sum << std::endl;

  return 0;
}
