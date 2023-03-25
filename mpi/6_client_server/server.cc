#include <iostream>

#include <mpi.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);

  // Open port.
  char port_name[MPI_MAX_PORT_NAME];
  MPI::Open_port(MPI::INFO_NULL, port_name);

  // Publish port name and accept client.
  MPI::Publish_name("name", MPI::INFO_NULL, port_name);
  auto client = MPI::COMM_WORLD.Accept(port_name, MPI::INFO_NULL, 0);

  // Receive data from client.
  int recvbuf{};
  client.Recv(&recvbuf, 1, MPI::INT, 0, 0);
  std::cout << "server: recvbuf equals to " << recvbuf << std::endl;

  MPI::Unpublish_name("name", MPI::INFO_NULL, port_name);

  MPI::Finalize();
  return 0;
}
