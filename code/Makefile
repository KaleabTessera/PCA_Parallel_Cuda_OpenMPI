pca.o: pca.cu
	nvcc -o pca pca.cu 

pcaSerial.o : pca.c
	gcc pca.c -lgsl -lgslcblas -lm
