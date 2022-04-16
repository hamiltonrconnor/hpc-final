/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

int nprocs,rank;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float fusion(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
float halo_fusion(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
float halo_timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
int halo_accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
void print_fushion(const t_param params,t_speed* cells);
void print_halo_fushion(const t_param params,t_speed* cells,int work);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

int findWork(int n,int procs,int rank){
  return (int)(round(n/procs*(rank+1)) -round(n/procs*(rank))) ;

}
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc,&argv);


  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  printf("Hello from rank %d of %d\n",rank,nprocs);
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */
  t_speed** cells_ptr = &cells;
  t_speed** tmp_cells_ptr= &tmp_cells;
  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }



  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);


  //initialise(paramfile, obstaclefile, &params, &test_cells, &test_tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;
  int N = params.ny;
  int work =findWork(N,nprocs,rank);
  int start = rank * work;
  int end = start + work;
  int flag;
  // for (int jj =0; jj < params.ny; jj++)
  // {
  //
  //   for (int ii = 0; ii < params.nx; ii++)
  //   {
  //     cells[ii+jj*params.nx].speeds[0] = 0.1f;
  //   }
  // }

  t_speed* local_cells  =(t_speed*)malloc(sizeof(t_speed) * ((work+2) * params.nx));
  t_speed* local_tmp_cells  =(t_speed*)malloc(sizeof(t_speed) * ((work+2) * params.nx));
  t_speed** local_cells_ptr = &local_cells;
  t_speed** local_tmp_cells_ptr= &local_tmp_cells;

  memcpy(&local_cells[1* params.nx],&cells[start*params.nx],sizeof(t_speed) * (work * params.nx));
  int* local_obstacles = malloc(sizeof(int) * (work * params.nx));

  memcpy(&local_obstacles[0],&obstacles[start*params.nx],sizeof(int) * (work * params.nx));
  // for(int i = 0;i<(work+2) ;i++){
  //   local_cells[i* params.nx].speeds[0] = i ;
  //
  //
  // }
  //printf("%d\n",work );


  for (int tt = 0; tt < params.maxIters; tt++)
  {
  // for (int tt = 0; tt < 10; tt++)
  // {


    //print_halo_fushion(params,local_cells,work);
    //print_halo_fushion(params,local_cells,work);
    //printf("rank: %d tt:%d 1\n",rank,tt);
    //Init local regions
    int tag = 0;
    MPI_Status status;

    int buffSize = params.nx *NSPEEDS;
    //Find the neigbours
    int right = (rank + 1) % nprocs;
    int left = (rank == 0) ? (rank + nprocs - 1) : (rank - 1);
    int posLeft = (start-1);
    if(rank==0){
      posLeft=(params.ny-1);
    }
    int posRight = (end);
   if(rank == nprocs-1){
     posRight = 0;
   }
    //printf("%d",work);
      // printf("rank: %d tt:%d 2\n",rank,tt);
    // printf("rank: %d tt:%d send:%d 2\n",rank,tt,local_cells[1*params.nx].speeds[0]);
    // printf("rank: %d tt:%d recv: %d2\n",rank,tt,local_cells[work*params.nx].speeds[0]);

        // printf("before Memcompare left Rank:%d result: %d\n",rank,memcmp(&local_cells[0],&cells[(posLeft)*params.nx],buffSize*sizeof(float)));
        // printf("before Memcompare mid Rank:%d result: %d\n",rank,memcmp(&local_cells[1*params.nx],&cells[start*params.nx],buffSize*sizeof(float)*work));
        // printf("before Memcompare right Rank:%d result: %d\n",rank,memcmp(&local_cells[(work+1)*params.nx],&cells[(posRight)*params.nx],buffSize*sizeof(float)));

    //printf("rank: %d tt:%d local_cells: %d end:%d buffSize:%d\n",rank,tt,sizeof(t_speed) * ((work+2) * params.nx),end*params.nx,buffSize*sizeof(float));
    // MPI_Sendrecv(&local_cells[1*params.nx],buffSize , MPI_FLOAT, left, tag,
    //     &local_cells[(129)*params.nx],  buffSize ,  MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);
    // //printf("rank: %d tt:%d 3\n",rank,tt);
    // MPI_Sendrecv(&local_cells[(128)*params.nx],buffSize , MPI_FLOAT, right, tag,
    //     &local_cells[0],  buffSize ,  MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);

    // for(int i = 0 ;i<work+2;i++){
    //   printf("Rank%d result: %d %f \n",rank,i,local_cells[i*params.nx].speeds[0]);
    // }


    // printf("mid tt :%d Memcompare left Rank:%d result: %d\n",tt,rank,memcmp(&local_cells[0],&cells[(posLeft)*params.nx],buffSize*sizeof(float)));
    // // printf("mid Memcompare mid Rank:%d result: %d\n",rank,memcmp(&local_cells[1*params.nx],&cells[start*params.nx],buffSize*sizeof(float)*work));
    //  printf("mid tt:%d Memcompare right Rank:%d result: %d\n",tt,rank,memcmp(&local_cells[(work+1)*params.nx],&cells[(posRight)*params.nx],buffSize*sizeof(float)));
    //
    //printf("%d",work);

    // for (int ii = 0; ii < params.nx; ii++)
    // {
    //   for (int kk = 0; kk < NSPEEDS; kk++)
    //   {
    //   local_cells[ii+0*params.nx].speeds[kk] = local_cells[ii+128*params.nx].speeds[kk] ;
    //   local_cells[ii+129*params.nx].speeds[kk] = local_cells[ii+1*params.nx].speeds[kk] ;
    //   }
    // }


    //printf("\n BEFORE \n");
    //print_fushion(params,cells);
    //print_fushion(params,*cells_ptr);
    //print_halo_fushion(params,local_cells,work);
    //print_halo_fushion(params,*local_cells_ptr,work);
    //av_vels[tt] = timestep(params, cells_ptr, tmp_cells_ptr, obstacles);


    av_vels[tt] = halo_timestep(params, local_cells_ptr, local_tmp_cells_ptr, local_obstacles);

    // printf("After Memcompare left Rank:%d result: %d\n",rank,memcmp(&local_tmp_cells[0],&cells[(posLeft)*params.nx],buffSize*sizeof(float)));
    //printf("After Memcompare mid Rank:%d result: %d\n",rank,memcmp(&local_tmp_cells[1*params.nx],&cells[start*params.nx],buffSize*sizeof(float)*work));
    // printf("After Memcompare right Rank:%d result: %d\n",rank,memcmp(&local_tmp_cells[(work+1)*params.nx],&cells[(posRight)*params.nx],buffSize*sizeof(float)));

    // //printf("After Memcompare left Rank:%d result: %d\n",rank,memcmp(&local_cells[0],&cells[posLeft*params.nx],buffSize*sizeof(float)));
    //printf("After Memcompare mid Rank:%d result: %d\n",rank,memcmp(&local_tmp_cells[1*params.nx],&tmp_cells[start*params.nx],buffSize*sizeof(float)*work));
    // //printf("After Memcompare right Rank:%d result: %d\n",rank,memcmp(&test_cells[work*params.nx],&cells[posRight*params.nx],buffSize*sizeof(float)));
    t_speed** local_temp = local_cells_ptr;
    local_cells_ptr= local_tmp_cells_ptr;
    local_tmp_cells_ptr= local_temp;

    // t_speed** temp = cells_ptr;
    // cells_ptr= tmp_cells_ptr;
    // tmp_cells_ptr= temp;
    //printf("rank: %d tt:%d 5\n",rank,tt);
    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("\n AFTER \n");
    //print_fushion(params,cells);
    //print_fushion(params,*cells_ptr);
    //print_halo_fushion(params,local_cells,work);
    //print_halo_fushion(params,*local_cells_ptr,work);

    //printf("After Memcompare mid Rank:%d result: %d\n",rank,memcmp(&local_cells[1*params.nx],&cells[start*params.nx],buffSize*sizeof(float)*work));




    //int flag = 0;



    // flag = 0;
    // for (int ii = 0; ii < params.nx; ii++)
    // {
    //   for (int kk = 0; kk < NSPEEDS; kk++)
    //   {
    //     if(cells[ii + posRight].speeds[kk] !=test_cells[ii + posRight].speeds[kk] ){
    //       flag =1;
    //
    //     }
    //   }
    //
    // }
    // if(flag==1){
    //   printf("posRight Rank: %d jj: %d\n",rank,posRight);
    // }
    //
    //
    //
    //
    // int flag = 0;
    // for (int ii = 0; ii < params.nx; ii++)
    // {
    //   for (int kk = 0; kk < NSPEEDS; kk++)
    //   {
    //     if(cells[ii + (params.ny-1)*params.nx].speeds[kk] !=local_cells[ii+0*params.nx].speeds[kk] ){
    //       flag =1;
    //
    //     }
    //   }
    //
    // }
    // if(flag==1){
    //   printf(" Rank: %d 127\n",rank);
    // }
    // flag = 0;
    // for (int ii = 0; ii < params.nx; ii++)
    // {
    //   for (int kk = 0; kk < NSPEEDS; kk++)
    //   {
    //     if(cells[ii + 0*params.nx].speeds[kk] !=local_cells[ii+(work+1)*params.nx].speeds[kk] ){
    //       flag =1;
    //
    //     }
    //   }
    //
    // }
    // if(flag==1){
    //   printf(" Rank: %d 0\n",rank);
    // }

    //
    //
    //
    // int flag;
    // for (int jj =1; jj < work+1; jj++)
    // {
    //   flag = 0;
    //   for (int ii = 0; ii < params.nx; ii++)
    //   {
    //     // if(ii == 2 &&jj ==2){
    //     //   local_cells[ii + jj*params.nx].speeds[0] = 0;
    //     //   cells[ii + (jj-1)*params.nx +start*params.nx].speeds[0] = 0;
    //     // }
    //     for (int kk = 0; kk < NSPEEDS; kk++)
    //     {
    //       if(cells[ii + (jj-1)*params.nx +start*params.nx].speeds[kk] !=local_cells[ii + jj*params.nx].speeds[kk] ){
    //         flag =1;
    //
    //       }
    //     }
    //
    //
    //   }
    //   if(flag==1){
    //     printf("Rank: %d jj: %d\n",rank,jj);
    //   }
    // }


    //av_vels[tt] = av_velocity(params, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  //printf("\n AFTER \n");

  //print_fushion(params,*cells_ptr);
  //print_halo_fushion(params,*local_cells_ptr,work);

  MPI_Barrier(MPI_COMM_WORLD);
  t_speed* output= (t_speed*)malloc(sizeof(float) * 20);
  print_halo_fushion(params,*local_cells_ptr,work);



  MPI_Gather(local_cells,10,MPI_FLOAT,output,10,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(rank==0){
  printf("\n");
  for(int i = 0;i<20;i++){
    printf("%f  ",output[i].speeds[0]);
  }
  printf("\n");
  }
  //print_fushion(params,output);
  //
  // int tag = 0;
  // MPI_Status status;
  // if (rank != 0) {
  //  int dest = 0;
  //
  //  MPI_Send(&*local_cells_ptr[1*params.nx],  NSPEEDS*params.nx*(work), MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
  //
  // }
  // else {             /* i.e. this is the master process */
  //
  //  for (int i=1; i<nprocs; i++) {
  //    int size = findWork(N,nprocs,i);
  //    int mystart = size*i;
  //    /* recieving their messages.. */
  //    MPI_Recv(&cells[mystart*params.nx], NSPEEDS*params.nx*(size), MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
  //
  //  }
  //  //printf("\n OUTPUT \n");
  //  //print_fushion(params,output);
  // }




  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
float halo_timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles)
{
    halo_accelerate_flow(params, *cells_ptr, obstacles);
    return halo_fusion(params, cells_ptr,tmp_cells_ptr, obstacles);
}
float timestep(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles)
{

  accelerate_flow(params, *cells_ptr, obstacles);

  return fusion(params, cells_ptr,tmp_cells_ptr, obstacles);


}

void print_fushion(const t_param params,t_speed* cells){
  char matrix[200000] ={0};
  for (int jj = 0; jj < params.ny; jj++)
  {
  for (int ii = 0; ii < params.nx; ii++)
  {

    char buf[20];
    float x =cells[ii+jj*params.nx].speeds[0];
    snprintf(buf,12,"%f   ",x);
    //printf("%s", buf);
    strcat(matrix,buf);
    // char space[2] ="  ";
    // strcat(matrix,space);
  }
  char newline[1] ="\n";
  strcat(matrix,newline);

  }
  printf("CELLS\n%s", matrix);
  //matrix = " ";

}

void print_halo_fushion(const t_param params,t_speed* local_cells,int work){
  char local_matrix[200000] ={0};

  for (int jj = 0; jj < work+2; jj++)
  {
  for (int ii = 0; ii < params.nx; ii++)
  {

    char buf[20];
    float x =local_cells[ii+jj*params.nx].speeds[0];
    snprintf(buf,12,"%f   ",x);
    //printf("%s", buf);
    strcat(local_matrix,buf);
    // char space[2] ="  ";
    // strcat(matrix,space);
  }
  char newline[1] ="\n";
  strcat(local_matrix,newline);

  }
  printf("LOCAL CELLS\n%s", local_matrix);
}

int halo_accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  // /printf("\n%d\n",(params.ny-2)/params.ny*nprocs);
  if(rank != nprocs-1){
    return EXIT_SUCCESS;
  }
  int work = findWork(params.ny,nprocs,rank);
  // /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = ((params.ny - 2)%work)+1;
  //int jj = params.ny - 2 +1;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + (jj-1)*params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }





  return EXIT_SUCCESS;
}
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{


  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  //int jj = (params.ny - 2%work)+1;
  int jj = params.ny - 2 ;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }


  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        float temp = (u_x * u_x) + (u_y * u_y);
        tot_u += sqrtf(temp);
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}
float halo_fusion(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles)
{
  //CONSTS FROM COLLISION
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float c_a = (2.f*c_sq*c_sq);
  const float c_c = 2.f*c_sq;

  t_speed* cells = *cells_ptr;
  t_speed* tmp_cells = *tmp_cells_ptr;
  //t_speed* output = *output_ptr;

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

	//Comment asdas
  /* loop over _all_ cells */
  //

    //#pragma omp parallel for collapse(2) reduction(+:tot_u,tot_cells)
    // //Init local regions
    // int tag = 0;
    // MPI_Status status;
    int N = params.ny;
    int work =findWork(N,nprocs,rank);
    int start = rank * work;
    int end = start + work;


    //Intialiase local cells
    //printf("work %d",work);
    // int flag =0;
    // for (int ii = 0; ii < params.nx; ii++)
    // {
    //   for (int kk = 0; kk < NSPEEDS; kk++)
    //   {
    //     cells[ii + 0*params.nx].speeds[kk] = cells[ii + 128*params.nx].speeds[kk];
    //     cells[ii+129*params.nx].speeds[kk] = cells[ii+1*params.nx].speeds[kk] ;
    //
    //
    //   }
    // }
    //cells[5 + 5*params.nx].speeds[0] = 0;


    // memcpy(&cells[0],&cells[128*params.nx],sizeof(t_speed) *  params.nx);
    // memcpy(&cells[129*params.nx],&cells[1*params.nx],sizeof(t_speed) *  params.nx);
    //print_halo_fushion(params,cells,work);
    int tag = 0;
    MPI_Status status;
    int buffSize = params.nx *NSPEEDS;
    int right = (rank + 1) % nprocs;
    int left = (rank == 0) ? (rank + nprocs - 1) : (rank - 1);
    MPI_Sendrecv(&cells[1*params.nx],buffSize , MPI_FLOAT, left, tag,
        &cells[(work+1)*params.nx],  buffSize ,  MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);
    //printf("rank: %d tt:%d 3\n",rank,tt);
    MPI_Sendrecv(&cells[(work)*params.nx],buffSize , MPI_FLOAT, right, tag,
        &cells[0],  buffSize ,  MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);

    //printf("\n AFTER SENDRECV \n");
    //print_halo_fushion(params,cells,work);

    //print_halo_fushion(params,cells,work);
    //cells[5+1*params.nx+1*params.nx].speeds[0] = 0;

    for (int jj =1; jj < work+1; jj++)
    {
      //printf("%d\n",jj);
      for (int ii = 0; ii < params.nx; ii++)
      {

      //printf("%d\n",omp_get_num_threads());
      //propagate(params,cells,tmp_cells,ii,jj);
      //PROPAGATE
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */

      short y_n = (jj + 1) ;
      // if(jj ==128){
      //   y_n = 1;
      // }
      const short x_e = (ii + 1) % params.nx;
       short y_s = (jj - 1);
      // if(jj==1){
      //   y_s = 128;
      // }
      const short x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */

      tmp_cells[ii + jj*params.nx].speeds[0] =cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
      //printf("%d 3\n",jj);


      // //REBOUND
      // /* if the cell contains an obstacle */
      if (obstacles[(jj-1)*params.nx + ii])
      {
        //.rebound(params, output,tmp_cells,ii, jj );
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        const float c1 = tmp_cells[ii + jj*params.nx].speeds[1];
        const float c2 = tmp_cells[ii + jj*params.nx].speeds[2];
        //const float c3 = tmp_cells[ii + jj*params.nx].speeds[3];
        //const float c4 = tmp_cells[ii + jj*params.nx].speeds[4];
        const float c5 = tmp_cells[ii + jj*params.nx].speeds[5];
        const float c6 = tmp_cells[ii + jj*params.nx].speeds[6];
        //const float c7 = tmp_cells[ii + jj*params.nx].speeds[7];
        //const float c8 = tmp_cells[ii + jj*params.nx].speeds[8];
        tmp_cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        tmp_cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        tmp_cells[ii + jj*params.nx].speeds[3] = c1;
        tmp_cells[ii + jj*params.nx].speeds[4] = c2;
        tmp_cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        tmp_cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        tmp_cells[ii + jj*params.nx].speeds[7] = c5;
        tmp_cells[ii + jj*params.nx].speeds[8] = c6;






      }


      //COLLISION
      /* don't consider occupied cells */
      else
      {
        //collision(params, output, tmp_cells,ii,jj);
        //Cooment
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        const float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        const float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */

        const float c_w1 = w1*local_density;
        const float c_w2 = w2*local_density;

        const float c_b = (u_sq*c_sq);





        /* local density total */

        /* relaxation step */
        //t_speed temp;
        float av_local_density =0.0f;
        float outVal;
        float diffVal;
        for(int i = 0; i<NSPEEDS;i++){
          if(i==0){
            diffVal = w0 * local_density* (1.f - u_sq / (2.f * c_sq));
          }else if(i<5){
            diffVal = c_w1 *(c_a+(c_c*u[i])+(u[i]*u[i])-c_b)/c_a;
          }else{
            diffVal = c_w2 *(c_a+(c_c*u[i])+(u[i]*u[i])-c_b)/c_a;
          }
          outVal = tmp_cells[ii + jj*params.nx].speeds[i]
                                                    + params.omega
                                                    * (diffVal - tmp_cells[ii + jj*params.nx].speeds[i]);
          av_local_density += outVal;
          tmp_cells[ii + jj*params.nx].speeds[i] =outVal;

        }


        //mp_cells[ii + jj*params.nx] = temp;
        const float inv_av_local_density = 1/av_local_density;
        /* x-component of velocity */
        const float av_u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                         *inv_av_local_density;

        /* compute y velocity component */
        const float av_u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     *inv_av_local_density;



        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf(((av_u_x * av_u_x) + (av_u_y * av_u_y)));
        /* increase counter of inspected cells */
        ++tot_cells;


      }
    }
    }


    return tot_u / (float)tot_cells;





}

float fusion(const t_param params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, int* obstacles)
{
  //CONSTS FROM COLLISION
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float c_a = (2.f*c_sq*c_sq);
  const float c_c = 2.f*c_sq;

  t_speed* cells = *cells_ptr;
  t_speed* tmp_cells = *tmp_cells_ptr;


  //t_speed* output = *output_ptr;

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

	//Comment asdas
  /* loop over _all_ cells */
  //

    //#pragma omp parallel for collapse(2) reduction(+:tot_u,tot_cells)
    // //Init local regions
    // int tag = 0;
    // MPI_Status status;
    int N = params.ny;
    int work =findWork(N,nprocs,rank);
    int start = rank * work;
    int end = start + work;

    //cells[5+1*params.nx].speeds[0] = 0;
    //cells[5 + 4*params.nx].speeds[0] = 0;
    //printf("\n BEFORE \n");
    // printf("\n 5 \n");
    // print_fushion(params,cells);


    //print_fushion(params,cells);
    for (int jj = 0; jj < params.ny; jj++)
    {
      for (int ii = 0; ii < params.nx; ii++)
      {
      //printf("%d\n",omp_get_num_threads());
      //propagate(params,cells,tmp_cells,ii,jj);
      //PROPAGATE
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const short y_n = (jj + 1) % params.ny;
      const short x_e = (ii + 1) % params.nx;
      const short y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const short x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */


      //REBOUND
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        //.rebound(params, output,tmp_cells,ii, jj );
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        const float c1 = tmp_cells[ii + jj*params.nx].speeds[1];
        const float c2 = tmp_cells[ii + jj*params.nx].speeds[2];
        //const float c3 = tmp_cells[ii + jj*params.nx].speeds[3];
        //const float c4 = tmp_cells[ii + jj*params.nx].speeds[4];
        const float c5 = tmp_cells[ii + jj*params.nx].speeds[5];
        const float c6 = tmp_cells[ii + jj*params.nx].speeds[6];
        //const float c7 = tmp_cells[ii + jj*params.nx].speeds[7];
        //const float c8 = tmp_cells[ii + jj*params.nx].speeds[8];
        tmp_cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        tmp_cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        tmp_cells[ii + jj*params.nx].speeds[3] = c1;
        tmp_cells[ii + jj*params.nx].speeds[4] = c2;
        tmp_cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        tmp_cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        tmp_cells[ii + jj*params.nx].speeds[7] = c5;
        tmp_cells[ii + jj*params.nx].speeds[8] = c6;






      }


      //COLLISION
      /* don't consider occupied cells */
      else
      {
        //collision(params, output, tmp_cells,ii,jj);
        //Cooment
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        const float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        const float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        //float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        // d_equ[0] = w0 * local_density
        //            * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        // d_equ[1] = w1 * local_density *
        // (1.f + (u[1] / c_sq )+ ((u[1] * u[1]) / (2.f * c_sq * c_sq)) - (u_sq / (2.f * c_sq)));

        //printf("%f\n",w1 *local_density *((2.f*c_sq*c_sq)+(2.f*c_sq*u[1])+(u[1]*u[1])-(u_sq*c_sq))/(2.f*c_sq*c_sq));
        const float c_w1 = w1*local_density;
        const float c_w2 = w2*local_density;

        const float c_b = (u_sq*c_sq);





        /* local density total */

        /* relaxation step */
        //t_speed temp;
        float av_local_density =0.0f;
        float outVal;
        float diffVal;
        for(int i = 0; i<NSPEEDS;i++){
          if(i==0){
            diffVal = w0 * local_density* (1.f - u_sq / (2.f * c_sq));
          }else if(i<5){
            diffVal = c_w1 *(c_a+(c_c*u[i])+(u[i]*u[i])-c_b)/c_a;
          }else{
            diffVal = c_w2 *(c_a+(c_c*u[i])+(u[i]*u[i])-c_b)/c_a;
          }
          outVal = tmp_cells[ii + jj*params.nx].speeds[i]
                                                    + params.omega
                                                    * (diffVal - tmp_cells[ii + jj*params.nx].speeds[i]);
          av_local_density += outVal;
          tmp_cells[ii + jj*params.nx].speeds[i] =outVal;

        }


        //mp_cells[ii + jj*params.nx] = temp;
        const float inv_av_local_density = 1/av_local_density;
        /* x-component of velocity */
        const float av_u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                         *inv_av_local_density;

        /* compute y velocity component */
        const float av_u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     *inv_av_local_density;



        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf(((av_u_x * av_u_x) + (av_u_y * av_u_y)));
        /* increase counter of inspected cells */
        ++tot_cells;


      }
    }
    }


    return tot_u / (float)tot_cells;





}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;
  //#pragma omp parallel for collapse(2)
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
