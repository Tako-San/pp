#include <iostream>
#include <sstream>

#include <omp.h>

int main() {
#pragma omp parallel
  {
    std::stringstream ss;
    ss << "Hello world! #" << omp_get_thread_num() << "\n";
    std::cout << ss.rdbuf();
  }
  return 0;
}
