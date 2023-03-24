#include <mpi.h>

int main(int ac, char **av) {
  MPI::Init(ac, av);

  // Look up for server's port name.
  char port_name[MPI::MAX_PORT_NAME];
  MPI::Lookup_name("name", MPI::INFO_NULL, port_name);

  // Connect to server.
  auto server = MPI::COMM_SELF.Connect(port_name, MPI::INFO_NULL, 0);

  // Send data to server.
  int sendbuf = 42;
  std::cout << "client: sendbuf equals to " << sendbuf << std::endl;
  server.Send(&sendbuf, 1, MPI::INT, 0, 0);

  MPI::Finalize();
  return 0;
}
