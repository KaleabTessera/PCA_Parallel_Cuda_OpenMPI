#include <stdio.h>
#include "pca_serial.h"
#define AMOUNT_ELEMENTS 10000
#define DIMENSION 1000
#define DIMENSIONTOMAPTO 3
#define NUM_RUNS 10

int main()
{
   pca_serial(AMOUNT_ELEMENTS,DIMENSION,DIMENSIONTOMAPTO,NUM_RUNS);
   return 0;
}

