#PBS -N  project2
#PBS -l nodes=7:ppn=20
#PBS -q q200p48h 
#PBS -l walltime=01:30:00
#PBS -r n
#PBS -V

cd $PBS_O_WORKDIR
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 2 ./project2  > result_naive_wraparound_alltoallprsnl_2
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 4 ./project2  > result_naive_wraparound_alltoallprsnl_4
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 8 ./project2  > result_naive_wraparound_alltoallprsnl_8
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 16 ./project2  > result_naive_wraparound_alltoallprsnl_16
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 32 ./project2  > result_naive_wraparound_alltoallprsnl_32
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 64 ./project2  > result_naive_wraparound_alltoallprsnl_64
/usr/local/intel-2017/impi/2017.0.098/bin64/mpirun  -np 128 ./project2  > result_naive_wraparound_alltoallprsnl_128

