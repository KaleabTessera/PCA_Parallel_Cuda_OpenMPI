# PCA_Parallel_Cuda_OpenMPI

PCA implemented in serial and parallel in order to compare the performance of serial and parallel implementations.

PCA implemented in:
  -Serial
  -Parallel
    :Cuda C
    :OpenMPI
    :OpenMP

How to Run:
	Downlaod mnist(retrieved from https://pjreddie.com/projects/mnist-in-csv/), rename it mnist.csv and save it in the data folder. (if not already there).		
		serial and openMP:
			-Install GNU library : sudo apt-get install libgsl0-dev
			-go to relevant folder e.g. code/serial, code/openMP 
			-create folder named "results", if not already there.
			-type "make"
			-run program e.g. ./pcaOpenmp or ./pcaSerial 
		cuda:
			-create folder named "results", if not already there.
			-type "make"
			-run program e.g. ./pcaParallel

