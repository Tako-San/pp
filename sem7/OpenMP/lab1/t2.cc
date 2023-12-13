#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <omp.h>
#include <string_view>
#include <vector>

constexpr std::size_t ISIZE = 5000;
constexpr std::size_t JSIZE = 5000;

using ArrTy = std::vector<std::vector<double>>;

void printArr(const ArrTy &arr, std::ostream &ost)
{
  for (auto &&row : arr)
  {
    for (auto elem : row)
      ost << elem << " ";
    ost << std::endl;
  }
}

using ProcFunc = std::function<void(ArrTy &)>;

void ethalon(ArrTy &arr)
{
#pragma omp parallel for
  for (std::size_t i = 0; i < ISIZE; i++)
    for (std::size_t j = 0; j < JSIZE; j++)
      arr[i][j] = std::sin(2 * arr[i][j]);
}

void processArr(ArrTy &arr)
{
  // Original cycle
  // for (std::size_t i = 0; i < ISIZE - 3; i++)
  //   for (std::size_t j = 4; j < JSIZE; j++)
  //     arr[i][j] = std::sin(0.2 * arr[i + 3][j - 4]);
  // Normalized version
  for (std::size_t i = 0; i < ISIZE - 3; i++)
    for (std::size_t j = 0; j < JSIZE - 4; j++)
      arr[i][j + 4] = std::sin(0.2 * arr[i + 3][j]);
}
// Check bernstein condition:
// F(k1, k2) = (k1, k2 + 3)
// G(l1, l2) = (l1 + 4, l2)
// Have a system
// k1 = l1 + 3
// k2 + 4 = l2
// => l = (k1, k2 + 6)
// D = l - k = (-3, 4) => d = (>, <)
// > ==> i anti-dependency
// < ==> j true-dependency

void processArrPar(ArrTy &arr)
{
  auto oldArr = arr;
#pragma omp parallel for schedule(static, 3)
  for (std::size_t i = 0; i < ISIZE - 3; i++)
  {
    auto arrnxt = oldArr[i + 3];
#pragma omp parallel for schedule(static, 4)
    for (std::size_t j = 0; j < JSIZE - 4; j++)
      arr[i][j + 4] = std::sin(0.2 * arrnxt[j]);
  }
}

void measureDump(ProcFunc f, ArrTy &arr, std::string_view filename)
{
  auto time = omp_get_wtime();
  f(arr);
  auto elapsed_ms = (omp_get_wtime() - time) * 1000;

  std::cout << "Elapsed time " << elapsed_ms  << " ms" << std::endl;

  std::ofstream of(filename.data());
  if (!of)
  {
    std::cerr << "Cannot open file for results" << std::endl;
    return;
  }
  printArr(arr, of);
}

void initArr(ArrTy &a)
{
  a.resize(ISIZE);
  // Fill array with data
  for (std::size_t i = 0; i < a.size(); i++)
  {
    auto &ai = a[i];
    ai.resize(JSIZE);
    for (std::size_t j = 0; j < ai.size(); j++)
      ai[j] = 10 * i + j;
  }
}

int main()
{
  ArrTy a{};
  initArr(a);

  std::cout << "Sequential:" << std::endl;
  measureDump(processArr, a, "seq.txt");

  initArr(a);
  std::cout << "Parallel:" << std::endl;
  measureDump(processArrPar, a, "par.txt");

  initArr(a);
  std::cout << "Ethalon:" << std::endl;
  measureDump(ethalon, a, "eth.txt");
}
