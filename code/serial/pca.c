#include <stdio.h>
#include "pca_serial.h"
#define NUMELEMENTCONFIGS 5
#define NUMDIMENSIONCONFIGS 4
#define NUMDIMENSIONTOMAPTOCONFIGS 3
#define NUM_RUNS 2

int main()
{
    int arrayAmountOfElementConfigs[NUMELEMENTCONFIGS] = {25, 100, 1000, 10000, 30000};
    int arrayDimensionConfigs[NUMDIMENSIONCONFIGS] = {25,100, 1000, 1500};
    int dimensionsToMapTo[NUMDIMENSIONTOMAPTOCONFIGS] = {5, 10, 20};

    for (int i = 0; i < NUMELEMENTCONFIGS; i++)
    {
        for (int j = 0; j < NUMDIMENSIONCONFIGS; j++)
        {
            for (int k = 0; k < NUMDIMENSIONTOMAPTOCONFIGS; k++)
            {
                printf("%d,%d,%d, \n",i,j,k);
                int amountElements = arrayAmountOfElementConfigs[i];
                int amountOfDimensions = arrayDimensionConfigs[j];
                int dimensionsToMap = dimensionsToMapTo[k];
                pca_serial(arrayAmountOfElementConfigs[i], arrayDimensionConfigs[j], dimensionsToMapTo[k], NUM_RUNS);
            }
        }
    }
    return 0;
}
