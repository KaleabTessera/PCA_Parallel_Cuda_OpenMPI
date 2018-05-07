pcaSerial : pca.c pca_serial.h
	nvcc pca.c -o pcaSerial -lgsl -lgslcblas -lm

pcaParallel : pca.c pca_parallel.h
	nvcc pca.cu -o pcaParallel

clean:
	rm pcaSerial
	rm pcaParallel
