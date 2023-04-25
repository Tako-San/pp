cmake_minimum_required(VERSION 3.22)
project(PP)

set(DIRS sem6)

include(cmake/AddTarget.cmake)
include(cmake/SubDirList.cmake)

set(TARGETS)
foreach(DIR ${DIRS})
  add_subdirectory(${DIR})
endforeach()

message(STATUS "Collected targets: ${TARGETS}")

foreach(TARGET IN LISTS TARGETS)
  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_features(${TARGET} PRIVATE cxx_std_20)

  set_target_properties(${TARGET}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
endforeach()
