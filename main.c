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

void kata(double **nodes_pos, int **edges_indices, double **edges_thickness, int node_num, int edge_num, int *input_nodes, double **input_vectors, int *frozen_nodes)
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

    int input_node_num = sizeof input_nodes / sizeof input_nodes[0];
    int frozen_node_num = sizeof frozen_nodes / sizeof frozen_nodes[0];

    // 各条件の要素（num_node*3中）と，その要素の強制変位や固定変位を収納
    int *indexes_array = malloc((input_node_num * 2 + frozen_node_num * 3) * sizeof(int));
    double *f_array = malloc((input_node_num * 2 + frozen_node_num * 3) * sizeof(double));
    for (int i = 0; i < input_node_num; i++)
    {
        indexes_array[i * 2 + 0] = input_nodes[i] * 3 + 0;
        indexes_array[i * 2 + 1] = input_nodes[i] * 3 + 1;

        f_array[i * 2 + 0] = input_vectors[i][0];
        f_array[i * 2 + 1] = input_vectors[i][1];
    }
    for (int i = 0; i < frozen_node_num; i++)
    {
        indexes_array[input_node_num * 2 + i * 3 + 0] = frozen_nodes[i] * 3 + 0;
        indexes_array[input_node_num * 2 + i * 3 + 1] = frozen_nodes[i] * 3 + 1;
        indexes_array[input_node_num * 2 + i * 3 + 2] = frozen_nodes[i] * 3 + 2;

        f_array[input_node_num * 2 + i * 3 + 0] = 0.0;
        f_array[input_node_num * 2 + i * 3 + 1] = 0.0;
        f_array[input_node_num * 2 + i * 3 + 2] = 0.0;
    }

    /* 入力した行列の表示 */
    printf("\n");
    for (i = 0; i < input_node_num * 2 + frozen_node_num * 3; ++i)
    {
        printf("%f  ", f_array[i]);
    }

    free(indexes_array);
    free(f_array);
}

/*==================================================*/
// CG法テストプログラム
/*==================================================*/

#define N 10
#define TMAX 100
#define EPS (1.0e-6)

// ベクトルに行列を作用 y = Ax
void vector_x_matrix(double *y, double **a, double *x, int size)
{
    int i, j;
    double vxm;
    for (i = 0; i < size; i++)
    {
        vxm = 0;
        for (j = 0; j < size; j++)
        {
            vxm += a[i][j] * x[j];
        }
        y[i] = vxm;
    }
}

// 内積を計算
double dot_product(double *x, double *y, int size)
{
    int i;
    double dot_p = 0;
    for (i = 0; i < size; i++)
    {
        dot_p += x[i] * y[i];
    }
    return dot_p;
}

// ベクトルノルムを計算
// ベクトルノルム := sgm(0〜N-1)|x[i]|
double vector_norm(double *x, int size)
{
    int i;
    double norm = 0;
    for (i = 0; i < N; i++)
    {
        norm += fabs(x[i]);
    }
    return norm;
}

// CG法
void cg_method(double **a, double *x, double *b, int size)
{
    int i, iter;
    double *p = malloc(size * sizeof(double));
    double *r = malloc(size * sizeof(double));
    double *ax = malloc(size * sizeof(double));
    double *ap = malloc(size * sizeof(double));

    // Axを計算
    vector_x_matrix(ax, a, x, 10);

    // pとrを計算 p = r := b - Ax
    for (i = 0; i < size; i++)
    {
        p[i] = b[i] - ax[i];
        r[i] = p[i];
    }

    // 反復計算
    for (iter = 1; iter < TMAX; iter++)
    {
        double alpha, beta, err = 0;

        // alphaを計算
        vector_x_matrix(ap, a, p, 10);
        alpha = dot_product(p, r, size) / dot_product(p, ap, size);

        for (i = 0; i < size; i++)
        {
            x[i] += +alpha * p[i];
            r[i] += -alpha * ap[i];
        }

        err = vector_norm(r, size); // 誤差を計算
        printf("LOOP : %d\t Error : %g\n", iter, err);
        if (EPS > err)
            break;

        // EPS < err ならbetaとpを計算してループを継続
        beta = -dot_product(r, ap, size) / dot_product(p, ap, size);
        for (i = 0; i < size; i++)
        {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(p);
    free(r);
    free(ax);
    free(ap);
}

int main(void)
{
    // 連立方程式 Ax = b
    // 行列Aは正定値対象行列
    double a[N][N] = {{5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      {2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      {0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      {0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0},
                      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0},
                      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0}};
    double b[N] = {3.0, 1.0, 4.0, 0.0, 5.0, -1.0, 6.0, -2.0, 7.0, -15.0};
    // 初期値は適当
    double x[N] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int i;
    int size = 10;

    double **A = malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
    {
        A[i] = malloc(size * sizeof(double));
        for (int j = 0; j < size; j++)
        {
            A[i][j] = a[i][j];
        }
    }

    // CG法でAx=bを解く
    cg_method(A, x, b, size);

    printf("###Calc End.###\n");
    for (i = 0; i < size; i++)
    {
        printf("x[%d] = %2g\n", i, x[i]);
    }

    // メモリの解放
    for (int i = 0; i < size; i++)
    {
        free(A[i]); //各行のメモリを解放
    }
    free(A);

    return 0;
}
