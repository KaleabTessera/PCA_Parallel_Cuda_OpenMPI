#ifndef HELPERS_H_
#define HELPERS_H_
#include <string.h>
#include <math.h>
#include <omp.h>

void printAllData(float **array, int amountOfElements, int dimension);
void loadAllData(float **data, FILE *fp, int amountOfElements, int dimension);
void loadData(float *data, char buf[4096], int dimension);

void loadAllData(float **data, FILE *fp, int amountOfElements, int dimension)
{
    char buf[4096];
    int index = 0;
    int lineNumber = 0;
    int maxNumLinesRead = amountOfElements;

    while (fgets(buf, sizeof(buf), fp) != NULL && (lineNumber < maxNumLinesRead))
    {
        if (lineNumber > 0)
        {
            loadData(data[index], buf, dimension);
            index++;
        }
        lineNumber++;
    }
}

void loadAllData1D(float *data, FILE *fp, int amountOfElements, int dimension)
{
    char buf[4096];
    int index = 0;
    int lineNumber = 0;
    int maxNumLinesRead = amountOfElements;
    while (fgets(buf, sizeof(buf), fp) != NULL && (lineNumber < maxNumLinesRead))
    {
        int numColumns = dimension;
        int count = 0;
        int indexC = 0;
        buf[strlen(buf) - 1] = '\0';
        const char delimeter[2] = ",";
        char *token = strtok(buf, delimeter);
        float num;
        while (token != NULL && count < numColumns)
        {
            sscanf(token, "%f,", &num);
            if (count > 0)
            {
                if (isnan(num))
                {
                    num = 0;
                }
                data[indexC] = num;
                indexC++;
            }
            token = strtok(NULL, delimeter);
            count = count + 1;
        }
    }
}

void loadData(float *data, char buf[4096], int dimension)
{
    int numColumns = dimension;
    int count = 0;
    int index = 0;
    buf[strlen(buf) - 1] = '\0';
    const char delimeter[2] = ",";
    char *token = strtok(buf, delimeter);
    float num;
    while (token != NULL && count < numColumns)
    {
        sscanf(token, "%f,", &num);
        if (count > 0)
        {
            if (isnan(num))
            {
                num = 0;
            }
            data[index] = num;
            index++;
        }
        token = strtok(NULL, delimeter);
        count = count + 1;
    }
}

void printAllData(float **array, int amountOfElements, int dimension)
{
    for (int i = 0; i < amountOfElements; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            if(j == (dimension-1)){
                printf("%f", array[i][j]);
            }
            else printf("%f,", array[i][j]);
        }
        printf("\n");
    }
}

void printAllDataToFile(float **array, int amountOfElements, int dimension,FILE *file)
{
    for (int i = 0; i < amountOfElements; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            if(j == (dimension-1)){
                fprintf(file,"%f", array[i][j]);
            }
            else fprintf(file,"%f,", array[i][j]);
        }
        fprintf(file,"\n");
    }
}

void printAllData1D(float *array, int amountOfElements, int dimension)
{
    for (int i = 0; i < amountOfElements; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            if(j ==(dimension-1)){
                printf("%f", array[i * dimension + j]);
            }
            else printf("%f,", array[i * dimension + j]);
        }
        printf("\n");
    }
}

void printAllData1DToFile(float *array, int amountOfElements, int dimension,FILE *file)
{
    for (int i = 0; i < amountOfElements; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            if((j ==(dimension-1))){
                fprintf(file,"%f", array[i * dimension + j]);
            }
            else fprintf(file,"%f,", array[i * dimension + j]);
        }
        fprintf(file,"\n");
    }
}

// void addTimeArray(double *time, int)

#endif
