#ifndef PCASERIAL_H_
#define PCASERIAL_H_
#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>

void computeMeanVector(float **data, int amountOfElements, int dimension, float *meanVectors);
void calculateCovarianceMatrix(float **data, int amountOfElements, int dimension, float *meanVectors,float **covarianceMatrix);
void getCovarianceMatrix(float **dataTranspose, float **data,float * meanVectors, int dimension,int amountOfElements,float **covarianceMatrix);
void pca_serial(int amountOfElements, int dimension)
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
    float *meanVectors = (float *)malloc(sizeof(float) * dimension);
    float **covarianceMatrix = (float **)malloc(sizeof(float *) * dimension);
    for (int i = 0; i < dimension; i++)
    {
        covarianceMatrix[i] = (float *)malloc(sizeof(float) * dimension);
    }
    loadAllData(allData, fp);
    // printAllData(allData, amountOfElements,dimension);
    computeMeanVector(allData, amountOfElements, dimension, meanVectors);
    calculateCovarianceMatrix(allData, amountOfElements, dimension, meanVectors, covarianceMatrix);
}

void computeMeanVector(float **data, int amountOfElements, int dimension, float *meanVectors)
{
    for (int j = 0; j < dimension; j++)
    {
        float mean = 0.0;
        float sum = 0.0;

        for (int i = 0; i < amountOfElements; i++)
        {
            sum += data[i][j];
        }
        meanVectors[j] = sum / amountOfElements;
        printf("sum: %f \n", meanVectors[j]);
    }
}

void calculateCovarianceMatrix(float **data, int amountOfElements, int dimension, float *meanVectors, float **covarianceMatrix)
{
    float **dataTranspose = (float **)malloc(sizeof(float *) * dimension);
    for (int i = 0; i < dimension; i++)
    {
        dataTranspose[i] = (float *)malloc(sizeof(float) * amountOfElements);
    }
    for (int j = 0; j < dimension; j++)
    {
        for (int i = 0; i < amountOfElements; i++)
        {
            dataTranspose[j][i] = data[i][j];
        }
    }

    getCovarianceMatrix(dataTranspose,data,meanVectors,dimension,amountOfElements,covarianceMatrix);

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            printf("%f ,", covarianceMatrix[i][j]);
        }
        printf("\n");
    }
}

void getCovarianceMatrix(float **dataTranspose, float **data,float * meanVectors,int dimension,int amountOfElements, float **covarianceMatrix)
{
    float sum;
    for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension; ++j)
        {
            sum = 0.0;
            for (int k = 0; k < amountOfElements; ++k)
            {
                sum += (dataTranspose[i][k] - meanVectors[j]) * (data[k][j] - meanVectors[j]);
            }
            covarianceMatrix[i][j] = sum / (amountOfElements - 1);
        }
}
#endif