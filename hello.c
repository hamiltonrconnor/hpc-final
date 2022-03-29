#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){

  MPI_Init(&argc,&argv);

  int nprocs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  printf("Hello from rank %d of %d\n",rank,nprocs);

  int N = 1024;
  int work = N / nprocs;
  int start = rank * work;
  int end = start + work;
  for (int i = start; i < end; ++i) { A[i] = B[i] + C[i];
  }
  MPI_Finalize();
}
