# Add any `module load` or `export` commands that your code needs to
export OMP_PROC_BIND = "close"
export OMP_PLACES = cores
export OMP_SCHEDULE = static
# compile and run to this file.
