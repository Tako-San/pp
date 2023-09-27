macro(UPDATE_TARGET_LIST NEW_TAR)
  # Append target name to targets list
  # & propagate updated list to parent scope
  list(APPEND TARGETS ${NEW_TAR})
  set(TARGETS ${TARGETS} PARENT_SCOPE)
endmacro()

macro(ADD_TARGET NAME)
  # Define target name as variable
  set(TAR ${NAME})

  UPDATE_TARGET_LIST(${NAME})

  # Create new target w/ given name
  if(${ARGC} GREATER 1)
    add_executable(${TAR} ${ARGN})
  else()
    add_executable(${TAR} ${TAR}.cc)
  endif()
endmacro()

macro(ADD_MPI_TARGET NAME)
  set(TNAME "mpi_${NAME}")
  ADD_TARGET(${TNAME} ${ARGN})
  target_include_directories(${TNAME} SYSTEM PRIVATE ${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
  target_link_libraries(${TNAME} PRIVATE ${MPI_CXX_LIBRARIES})
endmacro()

macro(ADD_OMP_TARGET NAME)
  set(TNAME "omp_${NAME}")
  ADD_TARGET(${TNAME} ${ARGN})
  target_link_libraries(${TNAME} PRIVATE OpenMP::OpenMP_CXX)
endmacro()

macro(ADD_STDTHREAD_TARGET NAME)
  set(TNAME "thread_${NAME}")
  ADD_TARGET(${TNAME} ${ARGN})
  target_link_libraries(${TNAME} PRIVATE Threads::Threads)
endmacro()
