#include <iostream>
#include <omp.h>
#include <sstream>

int main() {
  omp_set_nested(1);
#pragma omp parallel num_threads(3)
  {
    auto tnum1 = omp_get_num_threads();
    auto tid1 = omp_get_thread_num();

#pragma omp parallel num_threads(2)
    {
      auto tnum2 = omp_get_num_threads();
      auto tid2 = omp_get_thread_num();

#pragma omp parallel num_threads(2)
      {
        auto tnum3 = omp_get_num_threads();
        auto tid3 = omp_get_thread_num();

        std::ostringstream ss;
        ss << "[L1] tnum: " << tnum1 << ", tid: " << tid1 << std::endl;
        ss << "[L2] tnum: " << tnum2 << ", tid: " << tid2 << std::endl;
        ss << "[L3] tnum: " << tnum3 << ", tid: " << tid3 << std::endl;
        ss << "=========================\n";

        std::cout << ss.str();
      }
    }
  }
}
