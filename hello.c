#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){

  MPI_Init(&argc,&argv);

  int nprocs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  printf("Hello from rank %d of %d\n",rank,nprocs);

  int N = params.ny;
  int work = N / nprocs;
  int start = rank * work;
  int end = start + work;

  //Find the neigbours
  int right = (rank + 1) % nprocs;
  int left = (rank == 0) ? (rank + nprocs - 1) : (rank - 1);

  //Get the right data
  int numElements = (end - start + 1)
  int initAddress = cells + start;
  MPI_Sendrecv(cells + start, params.nx , MPI_FLOAT, left, tag,
      cells+end, params.nx,  MPI_FLOAT right, tag, MPI_COMM_WORLD, &status);
 MPI_Sendrecv(cells + end, params.nx , MPI_FLOAT, right, tag,
      cells+start, params.nx,  MPI_FLOAT left, tag, MPI_COMM_WORLD, &status);



  MPI_Finalize();
}
