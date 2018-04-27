#ifndef HELPERS_H_
#define HELPERS_H_

void printAllData(float **array, int amountOfElements,int dimension);
void loadAllData(float **data, FILE *fp);
void loadData(float *data, char buf[512]);

void loadAllData(float **data, FILE *fp)
{
    char buf[512];
    int index = 0;

    while (fgets(buf, sizeof(buf), fp) != NULL)
    {
        loadData(data[index], buf);
        index++;
    }
}

void loadData(float *data, char buf[512])
{
    int count = 0;
    buf[strlen(buf) - 1] = '\0';
    const char delimeter[2] = ",";
    char *token = strtok(buf, delimeter);
    //printf(token);
    while (token != NULL)
    {
        sscanf(token, "%f,", &data[count]);
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
            printf("%f ", array[i][j]);
        }
        printf("\n");
    }
}

#endif