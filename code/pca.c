#include <stdio.h>
#include "pca_serial.h"
#define AMOUNT_ELEMENTS 5
#define DIMENSION 4
#define DIMENSIONTOMAPTO 3

int main()
{
   pca_serial(AMOUNT_ELEMENTS,DIMENSION,DIMENSIONTOMAPTO);
   return 0;
}

