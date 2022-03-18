# Add any `module load` or `export` commands that your code needs to
module load languages/intel/2020-u4
module load languages/gcc/9.3.0
export OMP_NUM_THREADS=28
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
# compile and run to this file.
