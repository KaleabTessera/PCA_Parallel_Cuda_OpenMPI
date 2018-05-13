#ifndef PCAPARALLEL_H_
#define PCAPARALLEL_H_
#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 2

float *computeMeanVector(float *data, int amountOfElements, int dimension);
float *calculateCovarianceMatrix(float *data, int amountOfElements, int dimension, float *meanVectors);
void getCovarianceMatrix(float **dataTranspose, float **data, float *meanVectors, int dimension, int amountOfElements, float **covarianceMatrix);
float *computeEigenValues(float *covarianceMatrix, int dimension, int dimensionToMapTo);
float **transformSamplesToNewSubspace(float **allData, float **eigenVectorMatrix, int r1, int c1, int r2, int c2);
float **matrixMultiplication(float **matrixA, float **matrixB, int r1, int c1, int r2, int c2);

__constant__ int d_amountOfElements;
__constant__ int d_dimension;
__constant__ float d_meanVectors[4];
__global__ void meanVectorParallel(float *matrix, float *meanVector);
__global__ void covarianceMatrixParallel(float *dataTranspose);
__global__ void test(float *dataTranspose, float *data, float *covarianceMatrix);
__global__ void covarianceMatrixParallelSharedMemory(float *matrixA, float *matrixB, float *outputMatrix, int r1, int c1, int r2, int c2);
__global__ void computeEigenValuesParallel(float *matrix, int dimensionToMapTo);

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

    //printAllData(meanVectors, amountOfElements, dimensionToMapTo);
    clock_t beginCV = clock();
    float *covarianceMatrix = calculateCovarianceMatrix(allData1D, amountOfElements, dimension, meanVectors);
    clock_t endCV = clock();
    double time_spentCV = (double)(endCV - beginCV) / CLOCKS_PER_SEC;
    printf("Time to calculate co variance matrix: %f \n", time_spentCV);
    //
    clock_t beginEV = clock();
    float *eigenVectorMatrix = computeEigenValues(covarianceMatrix, dimension, dimensionToMapTo);
    clock_t endEV = clock();
    double time_spentEV = (double)(endEV - beginEV) / CLOCKS_PER_SEC;
    printf("Time to calculate eigenvalues: %f \n", time_spentEV);
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

    meanVector[idx] = sum / d_amountOfElements;
}

__global__ void covarianceMatrixParallel(float *dataTranspose, float *data, float *covarianceMatrix)
{
    //unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

//Adapted from https://gist.github.com/ironmanMA/41f6edaab6389b5f50bb
__global__ void covarianceMatrixParallelSharedMemory(float *matrixA, float *matrixB, float *outputMatrix, int r1, int c1, int r2, int c2)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
        tx = threadIdx.x, ty = threadIdx.y,
        Row = by * TILE_WIDTH + ty,
        Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    for (int m = 0; m < (c1 - 1) / TILE_WIDTH + 1; m++)
    {
        //float meanV_M = d_meanVectors[ty];
        //float meanV_N = d_meanVectors[tx];
        if (Row < r1 && m * TILE_WIDTH + tx < c1)
            ds_M[ty][tx] = matrixA[Row * c1 + m * TILE_WIDTH + tx];
        else
        {
            ds_M[ty][tx] = 0;
        }
        if (Col < c2 && m * TILE_WIDTH + ty < r2)
            ds_N[ty][tx] = matrixB[(m * TILE_WIDTH + ty) * c2 + Col];
        else
        {
            ds_N[ty][tx] = 0;
        }

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            if (ds_M[ty][k] != 0.0 && ds_N[k][tx] != 0.0)
            {

                Pvalue += (ds_M[ty][k] - d_meanVectors[Row]) * (ds_N[k][tx] - d_meanVectors[Col]);
                // if ((Row == 2 && Col == 2))
                // {
                //     printf("(%f(%d) - %f) + (%f(%d) - %f)  sum: Pvalue: %f\n", ds_M[ty][k], ty * r1 + k, meanV_M, ds_N[k][tx], k * r2 + tx, meanV_N,Pvalue);
                // }
            }
        }
        __syncthreads();
    }
    if (Row < r1 && Col < c2)
        outputMatrix[Row * c2 + Col] = Pvalue / (d_amountOfElements - 1);
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
    printAllData1D(dataTranspose, dimension, amountOfElements);

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

    covarianceMatrixParallelSharedMemory<<<dimGrid, dimBlock>>>(d_dataTranspose, d_data, d_covarianceMatrix,
                                                                4, 5,
                                                                5, 4);

    // matrixMultiply<<<dimGrid, dimBlock>>>(d_data, d_dataTranspose, d_covarianceMatrix,
    //                                       4, 5,
    //                                       5, 4,
    //                                       4, 4);
    //covarianceMatrixParallelSharedMemory<<<dimGrid, dimBlock>>>(d_dataTranspose, d_data, d_covarianceMatrix);
    cudaMemcpy(covarianceMatrix, d_covarianceMatrix, sizeArrayMemoryCovariancematrix, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 16; i++)
    // {
    //     printf("%f \n", covarianceMatrix[i]);
    // }
    //printAllData1D(covarianceMatrix,4,4);
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("%s\n", cudaGetErrorString(err));
    }
    return covarianceMatrix;
}

float *computeEigenValues(float *covarianceMatrix, int dimension, int dimensionToMapTo)
{
     // float *eigenVectorMatrix, *d_eigenVectorMatrix;

    // cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    // cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // cusolver_status = cusolverDnCreate(&cusolverH);
    // cudaMalloc((void **)&d_eigenVectorMatrix, sizeof(float) * dimension * dimensionToMapTo);
    // cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    // cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;

    const int m = dimension;
    const int lda = m;
    double lambda[3] = {2.0, 3.0, 4.0};

    double V[lda * m]; // eigenvectors
    double W[m];       // eigenvalues

    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;

    int info_gpu = 0;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cudaStat1 = cudaMalloc((void **)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc((void **)&d_W, sizeof(double) * m);
    cudaStat3 = cudaMalloc((void **)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, covarianceMatrix, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    cusolver_status = cusolverDnDsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    printf("after syevd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    printf("eigenvalue = (matlab base-1), ascending order\n");
    for(int i = 0 ; i < m ; i++){
        printf("W[%d] = %E\n", i+1, W[i]);
    }

    printf("V = (matlab base-1)\n");
    printMatrix(m, m, V, lda, "V");
    printf("=====\n");

    double lambda_sup = 0;
    for(int i = 0 ; i < m ; i++){
        double error = fabs( lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error)? lambda_sup : error;
    }
    printf("|lambda - W| = %E\n", lambda_sup);

}

__global__ void computeEigenValuesParallel(float *matrix, int dimensionToMapTo)
{
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