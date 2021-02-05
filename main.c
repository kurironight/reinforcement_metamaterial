#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void get_K_element_matrix(double K_e[6][6], double node[2][2], double A)
{
    double vector[2];
    double length, sin, cos;
    double T[6][6];

    int i, j;

    vector[0] = node[1][0] - node[0][0];
    vector[1] = node[1][1] - node[0][1];

    length = sqrt(pow(vector[0], 2) + pow(vector[1], 2));
    printf("%f,%f,%f\n", vector[0], vector[1], length);
    sin = vector[0] / length;
    cos = vector[1] / length;

    // Tを一旦零行列にする
    for (i = 0; i < 6; i++)
    {
        for (j = 0; j < 6; j++)
        {
            T[i][j] = 0;
        }
    }
    T[0][0] = cos;
    T[0][1] = sin;
    T[1][0] = -sin;
    T[1][1] = cos;
    T[2][2] = 1;
    T[3][3] = cos;
    T[3][4] = sin;
    T[4][3] = -sin;
    T[4][4] = cos;
    T[6][6] = 1;

    print_matrix(T);
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
    // double node[2][2];
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
    A = 2;

    double node[2][2] = {1.0, 3.0, 3.0, 4.0};
    get_K_element_matrix(K_e, node, A);

    return 0;
}