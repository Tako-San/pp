#include <iostream>
#include <sstream>

#include <omp.h>

int main() {
  int msg = 0;
  auto tnum = omp_get_max_threads();

#pragma omp parallel for ordered schedule(dynamic)
  for (int tid = 0; tid < tnum; ++tid) {
    std::stringstream ss{};
    ss << "T" << tid << ": ";

#pragma omp ordered
    { ss << msg++; }

    ss << std::endl;
    std::cout << ss.rdbuf();
  }
  return 0;
}
