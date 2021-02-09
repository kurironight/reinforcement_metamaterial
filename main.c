#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void print_matrix(double K[6][6], int row, int column)
{
    int i, j;
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

void matrix_transpose(double c[][6], double a[][6])
{
    int i, j;
    int N = 6;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = a[j][i];
        }
    }
    return;
}

void get_K_element_matrix(double K_e[6][6], double node[2][2], double A)
{
    double vector[2];
    double L, sin, cos, coef;
    double T[6][6], K_zero[6][6], T_trans[6][6], K_e_ref[6][6];
    double E = 1.0; //弾性率

    int i, j;

    vector[0] = node[1][0] - node[0][0]; //x軸方面
    vector[1] = node[1][1] - node[0][1]; //y軸方面

    L = sqrt(pow(vector[0], 2) + pow(vector[1], 2));
    cos = vector[0] / L;
    sin = vector[1] / L;

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
    T[5][5] = 1;

    // K_zeroを一旦零行列にする
    for (i = 0; i < 6; i++)
    {
        for (j = 0; j < 6; j++)
        {
            K_zero[i][j] = 0;
        }
    }
    coef = E * A / (pow(L, 3));
    K_zero[0][0] = coef * L * L;
    K_zero[3][0] = -coef * L * L;
    K_zero[1][1] = coef * 12;
    K_zero[1][2] = coef * 6 * L;
    K_zero[1][4] = -coef * 12;
    K_zero[1][5] = coef * 6 * L;
    K_zero[2][2] = coef * 4 * L * L;
    K_zero[2][4] = -coef * 6 * L;
    K_zero[2][5] = coef * 2 * L * L;
    K_zero[3][3] = coef * L * L;
    K_zero[4][4] = coef * 12;
    K_zero[4][5] = -coef * 6 * L;
    K_zero[5][5] = coef * 4 * L * L;

    //assymetry
    K_zero[0][3] = -coef * L * L;
    K_zero[4][1] = -coef * 12;
    K_zero[5][1] = coef * 6 * L;
    K_zero[2][1] = coef * 6 * L;
    K_zero[4][2] = -coef * 6 * L;
    K_zero[5][2] = coef * 2 * L * L;
    K_zero[5][4] = -coef * 6 * L;

    matrix_transpose(T_trans, T);
    matrix_mul(K_e_ref, T_trans, K_zero);
    matrix_mul(K_e, K_e_ref, T); // K_e行列を作成
}

void kata(double **nodes_pos, int **edges_indices, double **edges_thickness, int node_num, int edge_num)
{
    int i, j, k;
    int node1, node2;
    int free_D = 3;
    double K[node_num * free_D][node_num * free_D];
    double K_e[6][6];
    double node[2][2];

    // K行列の初期化
    for (i = 0; i < node_num * free_D; ++i)
    {
        for (j = 0; j < node_num * free_D; ++j)
        {
            K[i][j] = 0;
        }
    }

    for (i = 0; i < edge_num; ++i)
    {
        node1 = edges_indices[i][0];
        node2 = edges_indices[i][1];
        node[0][0] = nodes_pos[node1][0];
        node[0][1] = nodes_pos[node1][1];
        node[1][0] = nodes_pos[node2][0];
        node[1][1] = nodes_pos[node2][1];
        /* 入力した行列の表示 */
        printf("%d回目  ", i);
        printf("\n");
        /* 入力した行列の表示 */
        get_K_element_matrix(K_e, node, edges_thickness[i][0]);
        print_matrix(K_e, 6, 6);

        // K行列に代入
        // K11をK[node1*3:(node1+1)*3,node1*3:(node1+1)*3]に代入
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < 3; ++k)
            {
                K[node1 * 3 + j][node1 * 3 + k] += K_e[j][k];
            }
        }
        // K12をK[node1*3:(node1+1)*3,node2*3:(node2+1)*3]に代入
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < 3; ++k)
            {
                K[node1 * 3 + j][node2 * 3 + k] += K_e[j][k + 3];
            }
        }
        // K21をK[node2*3:(node2+1)*3,node1*3:(node1+1)*3]に代入
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < 3; ++k)
            {
                K[node2 * 3 + j][node1 * 3 + k] += K_e[j + 3][k];
            }
        }
        // K22をK[node2*3:(node2+1)*3,node2*3:(node2+1)*3]に代入
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < 3; ++k)
            {
                K[node2 * 3 + j][node2 * 3 + k] += K_e[j + 3][k + 3];
            }
        }
    }
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

    double node[2][2] = {1.0, 3.5, 3.0, 4.0};
    clock_t start, end;
    start = clock();

    get_K_element_matrix(K_e, node, A);

    end = clock();
    printf("%.2f秒かかりました\n", (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}