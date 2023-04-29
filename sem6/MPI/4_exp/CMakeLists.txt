set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GMP REQUIRED)

ADD_MPI_TARGET(04_exp main.cc)
target_link_libraries(mpi_04_exp PRIVATE gmp)
