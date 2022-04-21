# Add any `module load` or `export` commands that your code needs to
module load languages/intel/2020-u4
module load languages/gcc/9.3.0
module load tools/tau/2.28.2-intel-mpi
export OMP_NUM_THREADS=28
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
# compile and run to this file.
