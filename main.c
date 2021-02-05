#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b)
{
    return (a + b);
};

/*
void get_K_element_matrix(double K_e[6][6], double node[2][2], double length, double A)
{
    K_e[]
}
*/
void print_matrix(double K[6][6])
{
    int i, j;
    int row = 6;
    int column = 6;
    /* 入力した行列の表示 */
    printf("\n");
    for (i = 0; i < row; ++i)
    {
        for (j = 0; j < column; ++j)
        {
            printf("%f  ", K[i][j]);
            if (j == column - 1)
                printf("\n");
        }
    }
}

void matrix_mul(double c[][6], double a[][6], double b[][6])
{
    int i, j, k;
    int N = 6;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = 0.0;
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
    }
    return;
}

int main(void)
{
    double K_e[6][6];
    double M_e[6][6];
    double c[6][6];
    double node[2][2];
    double length;
    double A;

    int i, j;
    // zero matrix
    for (i = 0; i < 6; i++)
    {
        for (j = 0; j < 6; j++)
        {
            K_e[i][j] = i;
        }
    }

    for (i = 0; i < 6; i++)
    {
        for (j = 0; j < 6; j++)
        {
            M_e[i][j] = j;
        }
    }

    print_matrix(K_e);
    print_matrix(M_e);

    matrix_mul(c, K_e, M_e);
    print_matrix(c);

    return 0;
}