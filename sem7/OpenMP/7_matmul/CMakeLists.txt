ADD_OMP_TARGET(07_matmul main.cc)
target_compile_options(omp_07_matmul PRIVATE -O0 -mavx2)

ADD_OMP_TARGET(07_matmul_var main.cc)
target_compile_options(omp_07_matmul_var PRIVATE -O0 -mavx2)
target_compile_definitions(omp_07_matmul_var PRIVATE CMP_WAYS)
