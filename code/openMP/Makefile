pcaOpenMP : pca.c pca_parallel_openmp.h
	gcc pca.c -o pcaOpenmp -lgsl -lgslcblas -lm -fopenmp

clean:
	rm pcaOpenMP
