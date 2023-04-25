set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GMP REQUIRED)

ADD_MPI_TARGET(4_exp main.cc)
target_link_libraries(4_exp PRIVATE gmp)
