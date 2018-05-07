#include <stdio.h>
#include "pca_parallel.h"
#define AMOUNT_ELEMENTS 5
#define DIMENSION 4
#define DIMENSIONTOMAPTO 3

int main()
{
   pca_parallel(AMOUNT_ELEMENTS,DIMENSION,DIMENSIONTOMAPTO);
   return 0;
}

