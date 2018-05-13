#ifndef PCAPARALLEL_H_
#define PCAPARALLEL_H_
#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float *computeMeanVector(float *data, int amountOfElements, int dimension);
float *calculateCovarianceMatrix(float *data, int amountOfElements, int dimension, float *meanVectors);
void getCovarianceMatrix(float **dataTranspose, float **data, float *meanVectors, int dimension, int amountOfElements, float **covarianceMatrix);
float **computeEigenValues(float **covarianceMatrix, int dimension, int dimensionToMapTo);
float **transformSamplesToNewSubspace(float **allData, float **eigenVectorMatrix, int r1, int c1, int r2, int c2);
float **matrixMultiplication(float **matrixA, float **matrixB, int r1, int c1, int r2, int c2);

__constant__ int d_amountOfElements;
__constant__ int d_dimension;
__constant__ float d_meanVectors[4];
__global__ void meanVectorParallel(float *matrix, float *meanVector);
__global__ void covarianceMatrixParallel(float *dataTranspose);
__global__ void test(float *dataTranspose, float *data, float *covarianceMatrix);

void pca_parallel(int amountOfElements, int dimension, int dimensionToMapTo)
{
    FILE *fp = fopen("../data/data.txt", "r");
    if (fp == NULL)
    {
        perror("Error in opening file");
        exit(EXIT_FAILURE);
    }

    float **allData = (float **)malloc(sizeof(float *) * amountOfElements);
    for (int i = 0; i < amountOfElements; i++)
    {
        allData[i] = (float *)malloc(sizeof(float) * dimension);
    }
    loadAllData(allData, fp);

    float *allData1D = (float *)malloc(sizeof(float *) * amountOfElements * dimension);
    for (int i = 0; i < amountOfElements; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            int position = i * dimension + j;
            allData1D[position] = allData[i][j];
        }
    }
    printf("\nInitial Data \n");
    printAllData1D(allData1D, amountOfElements, dimension);

    printf("\nTimes taken for %d elements dimension of %d, mapping to %d principal compoments. \n", amountOfElements, dimension, dimensionToMapTo);
    clock_t beginMV = clock();
    float *meanVectors = computeMeanVector(allData1D, amountOfElements, dimension);
    clock_t endMV = clock();
    double time_spentMV = (double)(endMV - beginMV) / CLOCKS_PER_SEC;
    printf("Time to calculate mean vector: %f \n", time_spentMV);

    // for(int i=0; i< dimension; i++){
    //   printf("%f \n", meanVectors[i]);
    // }
    //printAllData(meanVectors, amountOfElements, dimensionToMapTo);
    clock_t beginCV = clock();
    float *covarianceMatrix = calculateCovarianceMatrix(allData1D, amountOfElements, dimension, meanVectors);
    clock_t endCV = clock();
    double time_spentCV = (double)(endCV - beginCV) / CLOCKS_PER_SEC;
    printf("Time to calculate co variance matrix: %f \n", time_spentCV);
    //
    // clock_t beginEV = clock();
    // float **eigenVectorMatrix = computeEigenValues(covarianceMatrix, dimension, dimensionToMapTo);
    // clock_t endEV = clock();
    // double time_spentEV = (double)(endEV - beginEV) / CLOCKS_PER_SEC;
    // printf("Time to calculate eigenvalues: %f \n", time_spentEV);
    //
    // clock_t beginNV = clock();
    // float **newValues = transformSamplesToNewSubspace(allData, eigenVectorMatrix, amountOfElements, dimension, dimension, dimensionToMapTo);
    // clock_t endNV = clock();
    // double time_spentNV = (double)(endNV - beginNV) / CLOCKS_PER_SEC;
    // printf("Time to map inputs to %d principal components: %f \n", dimensionToMapTo, time_spentNV);
    //
    // printf("\nData after PCA to %d principal components \n", dimensionToMapTo);
    //printAllData(newValues, amountOfElements, dimensionToMapTo);
}

float *computeMeanVector(float *data, int amountOfElements, int dimension)
{
    float *d_data, *d_meanVectors;
    float *meanVectors = (float *)malloc(sizeof(float) * dimension);
    int sizeArrayMemory = sizeof(float) * amountOfElements * dimension;
    cudaMalloc((void **)&d_data, sizeArrayMemory);
    cudaMalloc((void **)&d_meanVectors, sizeof(float) * dimension);

    cudaMemcpyToSymbol(d_amountOfElements, &amountOfElements, sizeof(int));
    cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(int));
    cudaMemcpy(d_data, data, sizeArrayMemory, cudaMemcpyHostToDevice);
    meanVectorParallel<<<dimension, amountOfElements>>>(d_data, d_meanVectors);
    cudaMemcpy(meanVectors, d_meanVectors, sizeof(float) * dimension, cudaMemcpyDeviceToHost);
    return meanVectors;
}

__global__ void meanVectorParallel(float *matrix, float *meanVector)
{
    float sum = 0;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < d_amountOfElements; i++)
        sum += matrix[idx + i * d_dimension];
}
meanVector[idx] = sum / d_amountOfElements;
}

__global__ void covarianceMatrixParallel(float *dataTranspose, float *data, float *covarianceMatrix)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int r1 = d_dimension;
    int c2 = d_dimension;
    int c1 = d_amountOfElements;

    //if((row < r1) && (col< c2 )){
    float sum = 0.0;
    for (int k = 0; k < c1; ++k)
    {
        sum += (dataTranspose[row * c1 + k] - d_meanVectors[row]) * (data[k * c2 + col] - d_meanVectors[col]);
        // if((row ==2 && col == 2)){
        //       printf("(%f(%d) - %f) + (%f(%d) - %f) \n",dataTranspose[row*c1+k],row*c1+k,d_meanVectors[row],data[k*c2+col],row*c1+k , d_meanVectors[col]);
        //     }
    }
    // if((row ==2 && col == 2)){
    //       printf("sum: %f \n",sum / (d_amountOfElements - 1));
    //     }
    covarianceMatrix[row * d_dimension + col] = sum / (d_amountOfElements - 1);
    //  }
}

__global__ void covarianceMatrixParallelSharedMemory(float *dataTranspose, float *data, float *covarianceMatrix)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int r1 = d_dimension;
    int c2 = d_dimension;
    int c1 = d_amountOfElements;

    float sum = 0.0;
    for (int k = 0; k < c1; ++k)
    {
        sum += (dataTranspose[row * c1 + k] - d_meanVectors[row]) * (data[k * c2 + col] - d_meanVectors[col]);
    }

    covarianceMatrix[row * d_dimension + col] = sum / (d_amountOfElements - 1);
}

float *calculateCovarianceMatrix(float *data, int amountOfElements, int dimension, float *meanVectors)
{
    float *covarianceMatrix = (float *)malloc(sizeof(float) * dimension * dimension);
    float *dataTranspose = (float *)malloc(sizeof(float) * dimension * amountOfElements);
    int sizeArrayMemoryCovariancematrix = sizeof(float) * dimension * dimension;
    int sizeArrayMemoryData = sizeof(float) * dimension * amountOfElements;
    for (int j = 0; j < dimension; j++)
    {
        for (int i = 0; i < amountOfElements; i++)
        {
            dataTranspose[j * amountOfElements + i] = data[i * dimension + j];
        }
    }

    float *d_data, *d_covarianceMatrix, *d_dataTranspose;
    cudaMalloc((void **)&d_data, sizeof(float) * amountOfElements * dimension);
    cudaMalloc((void **)&d_dataTranspose, sizeof(float) * dimension * amountOfElements);
    cudaMalloc((void **)&d_covarianceMatrix, sizeof(float) * dimension * dimension);

    cudaMemcpy(d_data, data, sizeArrayMemoryData, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataTranspose, dataTranspose, sizeArrayMemoryData, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_meanVectors, meanVectors, sizeof(float) * dimension);

    int BLOCK_WIDTH = 2;
    int numBlocks = dimension / BLOCK_WIDTH;
    if (dimension % BLOCK_WIDTH)
        numBlocks++;
    dim3 dimGrid(numBlocks, numBlocks, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    covarianceMatrixParallel<<<dimGrid, dimBlock>>>(d_dataTranspose, d_data, d_covarianceMatrix);
    cudaMemcpy(covarianceMatrix, d_covarianceMatrix, sizeArrayMemoryCovariancematrix, cudaMemcpyDeviceToHost);
    // cudaError err = cudaGetLastError();
    // if ( cudaSuccess != err )
    // {
    //     printf("%s\n", cudaGetErrorString(err));
    // }
    return covarianceMatrix;
}

float **computeEigenValues(float **covarianceMatrix, int dimension, int dimensionToMapTo)
{
    float **eigenVectorMatrix;
    return eigenVectorMatrix;
}

float **transformSamplesToNewSubspace(float **allData, float **eigenVectorMatrix, int r1, int c1, int r2, int c2)
{
    float **newValues; //= matrixMultiplication(allData, eigenVectorMatrix, r1, c1, r2, c2);
    return newValues;
}

float **matrixMultiplication(float **matrixA, float **matrixB, int r1, int c1, int r2, int c2)
{
    float **result = (float **)malloc(sizeof(float *) * r1);
    float sum;
    for (int i = 0; i < r1; i++)
    {
        result[i] = (float *)malloc(sizeof(float) * c2);
    }
    for (int i = 0; i < r1; ++i)
    {
        for (int j = 0; j < c2; ++j)
        {
            sum = 0.0;
            for (int k = 0; k < c1; ++k)
            {
                sum += (matrixA[i][k]) * (matrixB[k][j]);
            }
            result[i][j] = sum;
        }
    }
    return result;
}
#endif