#include <stdio.h>
#include "pca_parallel_openmp.h"
#define NUMELEMENTCONFIGS 4
#define NUMDIMENSIONCONFIGS 3
#define NUMDIMENSIONTOMAPTOCONFIGS 3
#define NUM_RUNS 2

int main()
{
    //pca_parallel_openmp(AMOUNT_ELEMENTS,DIMENSION,DIMENSIONTOMAPTO,NUM_RUNS);

    int arrayAmountOfElementConfigs[NUMELEMENTCONFIGS] = {100, 1000, 10000, 30000};
    int arrayDimensionConfigs[NUMDIMENSIONCONFIGS] = {100, 1000, 1500};
    int dimensionsToMapTo[NUMDIMENSIONTOMAPTOCONFIGS] = {5, 10, 20};

    for (int i = 0; i < NUMELEMENTCONFIGS; i++)
    {
        for (int j = 0; j < NUMDIMENSIONCONFIGS; j++)
        {
            for (int k = 0; k < NUMDIMENSIONTOMAPTOCONFIGS; k++)
            {
                int amountElements = arrayAmountOfElementConfigs[i];
                int amountOfDimensions = arrayDimensionConfigs[j];
                int dimensionsToMap = dimensionsToMapTo[k];
                pca_parallel_openmp(arrayAmountOfElementConfigs[i], arrayDimensionConfigs[j], dimensionsToMapTo[k], NUM_RUNS);
            }
        }
    }
    return 0;
   return 0;
}

