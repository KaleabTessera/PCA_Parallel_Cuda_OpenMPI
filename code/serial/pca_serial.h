#ifndef PCASERIAL_H_
#define PCASERIAL_H_
#include "../helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <time.h>
#define atoa(x) #x

float *computeMeanVector(float **data, int amountOfElements, int dimension);
float **calculateCovarianceMatrix(float **data, int amountOfElements, int dimension, float *meanVectors);
void getCovarianceMatrix(float **dataTranspose, float **data, float *meanVectors, int dimension, int amountOfElements, float **covarianceMatrix);
float **computeEigenValues(float **covarianceMatrix, int dimension, int dimensionToMapTo);
float **transformSamplesToNewSubspace(float **allData, float **eigenVectorMatrix, float *meanVectors, int r1, int c1, int r2, int c2);
float **matrixMultiplication(float **matrixA, float **matrixB, int r1, int c1, int r2, int c2);

void pca_serial(int amountOfElements, int dimension, int dimensionToMapTo, int numOfRuns)
{
    float *timeArray = (float *)malloc(sizeof(float) * numOfRuns * 4);
    int timeArrayIndex = 0;

    FILE *fp = fopen("../../data/mnist.csv", "r");
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

    loadAllData(allData, fp, amountOfElements, dimension);

    // printf("\nInitial Data \n");
    // printAllData(allData, amountOfElements, dimension);
    for (int r = 0; r < numOfRuns; r++)
    {
        printf("\nTimes taken for %d elements dimension of %d, mapping to %d principal compoments. \n", amountOfElements, dimension, dimensionToMapTo);
        clock_t beginMV = clock();
        float *meanVectors = computeMeanVector(allData, amountOfElements, dimension);
        clock_t endMV = clock();
        float time_spentMV = (float)(endMV - beginMV) / CLOCKS_PER_SEC;
        timeArray[timeArrayIndex++] = time_spentMV;
        printf("Time to calculate mean vector: %f \n", time_spentMV);

        clock_t beginCV = clock();
        float **covarianceMatrix = calculateCovarianceMatrix(allData, amountOfElements, dimension, meanVectors);
        clock_t endCV = clock();
        float time_spentCV = (float)(endCV - beginCV) / CLOCKS_PER_SEC;
        timeArray[timeArrayIndex++] = time_spentCV;
        printf("Time to calculate co variance matrix: %f \n", time_spentCV);
        // printAllData(covarianceMatrix, dimension, dimension);

        clock_t beginEV = clock();
        float **eigenVectorMatrix = computeEigenValues(covarianceMatrix, dimension, dimensionToMapTo);
        clock_t endEV = clock();
        float time_spentEV = (float)(endEV - beginEV) / CLOCKS_PER_SEC;
        timeArray[timeArrayIndex++] = time_spentEV;
        printf("Time to calculate eigenvalues: %f \n", time_spentEV);

        clock_t beginNV = clock();
        float **newValues = transformSamplesToNewSubspace(allData, eigenVectorMatrix, meanVectors, amountOfElements, dimension, dimension, dimensionToMapTo);
        clock_t endNV = clock();
        float time_spentNV = (float)(endNV - beginNV) / CLOCKS_PER_SEC;
        timeArray[timeArrayIndex++] = time_spentNV;
        printf("Time to map inputs to %d principal components: %f \n", dimensionToMapTo, time_spentNV);

        // printf("\nData after PCA to %d principal components \n", dimensionToMapTo);
        // printAllData(newValues, amountOfElements, dimensionToMapTo);
    }

    char fileNameDistance[200];
    sprintf(fileNameDistance, "results/results_%d_%d_%d.csv", amountOfElements, dimension, dimensionToMapTo);
    printf("%s", fileNameDistance);
    FILE *f = fopen(fileNameDistance, "w");
    if (f == NULL)
    {
        perror("Error in opening file");
        exit(EXIT_FAILURE);
    }
    fprintf(f, "MeanVector,CoVariance,Eigenvalues,Map inputs \n");
    printAllData1DToFile(timeArray, numOfRuns, 4, f);
    fclose(f);
    fclose(fp);
}

float *computeMeanVector(float **data, int amountOfElements, int dimension)
{
    float *meanVectors = (float *)malloc(sizeof(float) * dimension);
    for (int j = 0; j < dimension; j++)
    {
        float mean = 0.0;
        float sum = 0.0;

        for (int i = 0; i < amountOfElements; i++)
        {
            sum += data[i][j];
        }
        meanVectors[j] = sum / amountOfElements;
    }
    return meanVectors;
}

float **calculateCovarianceMatrix(float **data, int amountOfElements, int dimension, float *meanVectors)
{
    float **covarianceMatrix = (float **)malloc(sizeof(float *) * dimension);
    for (int i = 0; i < dimension; i++)
    {
        covarianceMatrix[i] = (float *)malloc(sizeof(float) * dimension);
    }
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

    getCovarianceMatrix(dataTranspose, data, meanVectors, dimension, amountOfElements, covarianceMatrix);
    return covarianceMatrix;
}

void getCovarianceMatrix(float **dataTranspose, float **data, float *meanVectors, int dimension, int amountOfElements, float **covarianceMatrix)
{
    float sum;
    int r1 = dimension;
    int c2 = dimension;
    int c1 = amountOfElements;
    for (int i = 0; i < r1; ++i)
    {
        for (int j = 0; j < c2; ++j)
        {
            sum = 0.0;
            for (int k = 0; k < c1; ++k)
            {
                sum += (dataTranspose[i][k] - meanVectors[i]) * (data[k][j] - meanVectors[j]);
            }
            covarianceMatrix[i][j] = sum / (amountOfElements - 1);
        }
    }
}

void transformSamples(float **dataTranspose, float **data, float *meanVectors, int r1, int c1, int r2, int c2, float **newValues)
{
    float sum;
    for (int i = 0; i < r1; ++i)
    {
        for (int j = 0; j < c2; ++j)
        {
            sum = 0.0;
            for (int k = 0; k < c1; ++k)
            {
                sum += ((dataTranspose[i][k] - meanVectors[k]) * (data[k][j])) * (data[k][j]);
            }
            newValues[i][j] = meanVectors[j] + sum;
        }
    }
}

float **computeEigenValues(float **covarianceMatrix, int dimension, int dimensionToMapTo)
{
    float **eigenVectorMatrix = (float **)malloc(sizeof(float *) * dimension);
    for (int i = 0; i < dimension; i++)
    {
        eigenVectorMatrix[i] = (float *)malloc(sizeof(float) * dimensionToMapTo);
    }

    double *data = (double *)malloc(sizeof(double) * dimension * dimension);
    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            data[i * dimension + j] = (double)covarianceMatrix[i][j];
        }
    }

    gsl_set_error_handler_off();
    gsl_matrix_view m = gsl_matrix_view_array(data, dimension, dimension);

    gsl_vector_complex *eval = gsl_vector_complex_alloc(dimension);
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc(dimension, dimension);

    gsl_eigen_nonsymmv_workspace *w =
        gsl_eigen_nonsymmv_alloc(dimension);

    gsl_eigen_nonsymmv(&m.matrix, eval, evec, w);

    gsl_eigen_nonsymmv_free(w);

    gsl_eigen_nonsymmv_sort(eval, evec,
                            GSL_EIGEN_SORT_ABS_DESC);

    {
        int i, j;
        for (i = 0; i < dimensionToMapTo; i++)
        {
            gsl_complex eval_i = gsl_vector_complex_get(eval, i);
            gsl_vector_complex_view evec_i = gsl_matrix_complex_column(evec, i);

            for (j = 0; j < dimension; ++j)
            {
                gsl_complex z =
                    gsl_vector_complex_get(&evec_i.vector, j);
                eigenVectorMatrix[j][i] = GSL_REAL(z);
            }
        }
    }

    gsl_vector_complex_free(eval);
    gsl_matrix_complex_free(evec);
    return eigenVectorMatrix;
}

float **transformSamplesToNewSubspace(float **allData, float **eigenVectorMatrix, float *meanVectors, int r1, int c1, int r2, int c2)
{

    float **newValues = (float **)malloc(sizeof(float *) * r1);
    for (int i = 0; i < r1; i++)
    {
        newValues[i] = (float *)malloc(sizeof(float) * c2);
    }
    transformSamples(allData, eigenVectorMatrix, meanVectors, r1, c1, r2, c2, newValues);
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
