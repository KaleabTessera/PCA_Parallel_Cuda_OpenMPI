# PCA_Parallel_Cuda_OpenMPI

PCA implemented in serial and parallel in order to compare the performance of serial and parallel implementations.

PCA implemented in:
  -Serial
  -Parallel
    :Cuda C
    :OpenMPI
    :OpenMP

How to Run:
	Downlaod mnist(retrieved from https://pjreddie.com/projects/mnist-in-csv/)
	Install GNU library : sudo apt-get install libgsl0-dev
	go to relevant folder, i.e. serial, 
	make
	./pcaSerial

