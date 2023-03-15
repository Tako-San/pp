set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GMP REQUIRED)

ADD_MPI_TARGET(4_1_exp v1.cc)
ADD_MPI_TARGET(4_2_exp v2.cc)
# ADD_MPI_TARGET(4_3_exp v3.cc)
# ADD_MPI_TARGET(4_4_exp v4.cc mpi_gmp.c)

# target_compile_definitions(4_4_exp PRIVATE USE_GMP)
# target_link_libraries(4_4_exp PRIVATE gmp)

ADD_MPI_TARGET(4_5_exp v5.cc)
target_link_libraries(4_5_exp PRIVATE gmp)