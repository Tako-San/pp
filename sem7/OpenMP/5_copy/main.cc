#include <iostream>
#include <random>
#include <sstream>

#include <omp.h>

int base = -1;
int cin = -1;
int cprivate = -1;

#pragma omp threadprivate(cin)
#pragma omp threadprivate(cprivate)

void printThread(int cnt) {
  std::ostringstream ss;
  ss << "Thread " << omp_get_thread_num() << ", counter = " << cnt << std::endl;
  std::cout << ss.str();
}

int main() {
  std::cout << "Base:" << std::endl;
#pragma omp parallel
  {
    base = omp_get_thread_num();
    printThread(base);
  }

  std::cout << std::endl << "Copyin:" << std::endl;
#pragma omp parallel copyin(cin)
  {
    cin = omp_get_thread_num();
    printThread(cin);
  }

  std::cout << std::endl << "Copyprivate:" << std::endl;
#pragma omp parallel
  {
#pragma omp single copyprivate(cprivate)
    cprivate = omp_get_thread_num();
    printThread(cprivate);
  }

  std::cout << std::endl;
  std::cout << "base = " << base << std::endl;
  std::cout << "cin = " << cin << std::endl;
  std::cout << "cprivate = " << cprivate << std::endl;
}
