#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

// y* = Ax とyとで誤差がどれぐらいあるのかを確認．そして，CGの精度を測定する．
double confirm_acurracy_of_cg(double **a, double *x, double *F, int size)
{
    double F_ref[size];
    double gosa = 0.0;
    int i;
    vector_x_matrix(F_ref, a, x, size);
    for (i = 0; i < size; i++)
    {
        gosa += fabs(F_ref[i] - F[i]);
    }
    return gosa;
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
double vector_norm(double *x, int size)
{
    int i;
    double norm = 0;
    for (i = 0; i < size; i++)
    {
        norm += fabs(x[i]);
    }
    return norm;
}

// CG法
void cg_method(double **a, double *x, double *b, int size, int TMAX, double EPS)
{
    int i, iter;
    double *p = malloc(size * sizeof(double));
    double *r = malloc(size * sizeof(double));
    double *ax = malloc(size * sizeof(double));
    double *ap = malloc(size * sizeof(double));

    // Axを計算
    vector_x_matrix(ax, a, x, size);

    // pとrを計算 p = r := b - Ax
    for (i = 0; i < size; i++)
    {
        p[i] = b[i] - ax[i];
        r[i] = p[i];
    }

    // 反復計算
    for (iter = 1; iter < TMAX + 1; iter++)
    {
        double alpha, beta, err = 0;

        // alphaを計算
        vector_x_matrix(ap, a, p, size);
        alpha = dot_product(p, r, size) / dot_product(p, ap, size);

        for (i = 0; i < size; i++)
        {
            x[i] += +alpha * p[i];
            r[i] += -alpha * ap[i];
        }

        err = vector_norm(r, size); // 誤差を計算
        if (EPS > err)
            break;

        // EPS < err ならbetaとpを計算してループを継続
        beta = -dot_product(r, ap, size) / dot_product(p, ap, size);
        for (i = 0; i < size; i++)
        {
            p[i] = r[i] + beta * p[i];
        }
        if (iter == TMAX && EPS < err)
        {
            printf("failed \n");
        }
        // printf("%d  ", iter);
        // printf("%f  \n", err);
    }

    free(p);
    free(r);
    free(ax);
    free(ap);
}

void print_matrix(double **K, int row_size, int column_size)
{
    int i, j;
    /* 入力した行列の表示 */
    printf("\n");
    for (i = 0; i < row_size; ++i)
    {
        for (j = 0; j < column_size; ++j)
        {
            printf("%f  ", K[i][j]);
            if (j == column_size - 1)
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

void get_K_element_matrix(double K_e[6][6], double node[2][2], double h)
{
    double vector[2];
    double L, sin, cos, coef;
    double T[6][6], K_zero[6][6], T_trans[6][6], K_e_ref[6][6];
    double E = 1.0; //弾性率
    double b=0.2; //奥行
    double A=b*h;
    double Iz_A = h * h / 12;

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
    K_zero[1][1] = coef * 12 * Iz_A;
    K_zero[1][2] = coef * 6 * L * Iz_A;
    K_zero[1][4] = -coef * 12 * Iz_A;
    K_zero[1][5] = coef * 6 * L * Iz_A;
    K_zero[2][2] = coef * 4 * L * L * Iz_A;
    K_zero[2][4] = -coef * 6 * L * Iz_A;
    K_zero[2][5] = coef * 2 * L * L * Iz_A;
    K_zero[3][3] = coef * L * L;
    K_zero[4][4] = coef * 12 * Iz_A;
    K_zero[4][5] = -coef * 6 * L * Iz_A;
    K_zero[5][5] = coef * 4 * L * L * Iz_A;

    //assymetry
    K_zero[0][3] = -coef * L * L;
    K_zero[4][1] = -coef * 12 * Iz_A;
    K_zero[5][1] = coef * 6 * L * Iz_A;
    K_zero[2][1] = coef * 6 * L * Iz_A;
    K_zero[4][2] = -coef * 6 * L * Iz_A;
    K_zero[5][2] = coef * 2 * L * L * Iz_A;
    K_zero[5][4] = -coef * 6 * L * Iz_A;

    matrix_transpose(T_trans, T);
    matrix_mul(K_e_ref, T_trans, K_zero);
    matrix_mul(K_e, K_e_ref, T); // K_e行列を作成
}

void bar_fem(double **nodes_pos, int **edges_indices, double **edges_thickness, int node_num, int edge_num, int input_node_num, int *input_nodes, double **input_vectors, int frozen_node_num, int *frozen_nodes, double **displacement, int tmax, double eps)
{
    int i, j, k;
    int node1, node2;
    int free_D = 3;
    int all_element_size = node_num * free_D;
    double K[all_element_size][all_element_size];
    double K_e[6][6];
    double F[all_element_size];
    double node[2][2];

    // K行列の初期化
    for (i = 0; i < all_element_size; ++i)
    {
        for (j = 0; j < all_element_size; ++j)
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

    //ひずみ角の条件を加えない場合
    int condition_element_num = input_node_num * 2 + frozen_node_num * 3;

    // 各条件の要素（num_node*3中）と，その要素の強制変位や固定変位を収納
    int *indexes_array = malloc(condition_element_num * sizeof(int));
    double *f_array = malloc(condition_element_num * sizeof(double));
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

    /* ひずみ角の条件を加える場合
    int condition_element_num = input_node_num * 3 + frozen_node_num * 3; //モーメントの条件を加える場合

    // 各条件の要素（num_node*3中）と，その要素の強制変位や固定変位を収納
    int *indexes_array = malloc(condition_element_num * sizeof(int));
    double *f_array = malloc(condition_element_num * sizeof(double));
    for (int i = 0; i < input_node_num; i++)
    {
        indexes_array[i * 2 + 0] = input_nodes[i] * 3 + 0;
        indexes_array[i * 2 + 1] = input_nodes[i] * 3 + 1;
        indexes_array[i * 2 + 2] = input_nodes[i] * 3 + 2; //モーメントの条件を加える場合

        f_array[i * 2 + 0] = input_vectors[i][0];
        f_array[i * 2 + 1] = input_vectors[i][1];
        f_array[i * 2 + 2] = input_vectors[i][2]; //モーメントの条件を加える場合
    }
    for (int i = 0; i < frozen_node_num; i++)
    {
        indexes_array[input_node_num * 3 + i * 3 + 0] = frozen_nodes[i] * 3 + 0;
        indexes_array[input_node_num * 3 + i * 3 + 1] = frozen_nodes[i] * 3 + 1;
        indexes_array[input_node_num * 3 + i * 3 + 2] = frozen_nodes[i] * 3 + 2;

        f_array[input_node_num * 3 + i * 3 + 0] = 0.0;
        f_array[input_node_num * 3 + i * 3 + 1] = 0.0;
        f_array[input_node_num * 3 + i * 3 + 2] = 0.0;
    }
    */

    //F行列の初期化
    for (i = 0; i < all_element_size; ++i)
    {
        F[i] = 0;
    }

    //F,K行列に条件を適用
    for (i = 0; i < condition_element_num; ++i)
    {
        int target_index = indexes_array[i];
        //makinf F matrix
        for (j = 0; j < all_element_size; ++j)
        {
            F[j] -= K[j][target_index] * f_array[i];
        }
        F[target_index] = f_array[i];

        //makinf K matrix
        for (j = 0; j < all_element_size; ++j)
        {
            K[target_index][j] = 0;
            K[j][target_index] = 0;
        }
        K[target_index][target_index] = 1;
    }

    // CGをかける為の形にする
    double **A = malloc(all_element_size * sizeof(double *));
    for (int i = 0; i < all_element_size; i++)
    {
        A[i] = malloc(all_element_size * sizeof(double));
        for (int j = 0; j < all_element_size; j++)
        {
            A[i][j] = K[i][j];
        }
    }
    // xの初期値は適当
    double x[all_element_size];
    for (i = 0; i < all_element_size; ++i)
    {
        x[i] = 0;
    }

    // CG法でAx=bを解く
    cg_method(A, x, F, all_element_size, tmax, eps);

    // メモリの解放
    for (int i = 0; i < all_element_size; i++)
    {
        free(A[i]); //各行のメモリを解放
    }

    for (i = 0; i < all_element_size; ++i)
    {
        displacement[i][0] = x[i];
    }
    free(A);
    free(indexes_array);
    free(f_array);
}

void bar_fem_force(double **nodes_pos, int **edges_indices, double **edges_thickness, int node_num, int edge_num, int input_node_num, int *input_nodes, double **input_forces, int frozen_node_num, int *frozen_nodes, double **displacement, int tmax, double eps)
{
    int i, j, k;
    int node1, node2;
    int free_D = 3;
    int all_element_size = node_num * free_D;
    double K[all_element_size][all_element_size];
    double K_e[6][6];
    double F[all_element_size];
    double node[2][2];

    // K行列の初期化
    for (i = 0; i < all_element_size; ++i)
    {
        for (j = 0; j < all_element_size; ++j)
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

    //ひずみ角の条件を加えない場合
    int condition_element_num = frozen_node_num * 3;

    // 各条件の要素（num_node*3中）と，その要素の強制変位や固定変位を収納
    int *indexes_array = malloc(condition_element_num * sizeof(int));
    double *f_array = malloc(condition_element_num * sizeof(double));
    for (int i = 0; i < frozen_node_num; i++)
    {
        indexes_array[i * 3 + 0] = frozen_nodes[i] * 3 + 0;
        indexes_array[i * 3 + 1] = frozen_nodes[i] * 3 + 1;
        indexes_array[i * 3 + 2] = frozen_nodes[i] * 3 + 2;

        f_array[i * 3 + 0] = 0.0;
        f_array[i * 3 + 1] = 0.0;
        f_array[i * 3 + 2] = 0.0;
    }

    //F行列の初期化
    for (i = 0; i < all_element_size; ++i)
    {
        F[i] = 0;
    }
    // Fの外力の部分を設定
    for (int i = 0; i < input_node_num; i++)
    {
        F[input_nodes[i] * 3 + 0] = input_forces[i][0];
        F[input_nodes[i] * 3 + 1] = input_forces[i][1];
    }

    //F,K行列に条件を適用
    for (i = 0; i < condition_element_num; ++i)
    {
        int target_index = indexes_array[i];
        //makinf F matrix
        for (j = 0; j < all_element_size; ++j)
        {
            F[j] -= K[j][target_index] * f_array[i];
        }
        F[target_index] = f_array[i];

        //makinf K matrix
        for (j = 0; j < all_element_size; ++j)
        {
            K[target_index][j] = 0;
            K[j][target_index] = 0;
        }
        K[target_index][target_index] = 1;
    }

    // CGをかける為の形にする
    double **A = malloc(all_element_size * sizeof(double *));
    for (int i = 0; i < all_element_size; i++)
    {
        A[i] = malloc(all_element_size * sizeof(double));
        for (int j = 0; j < all_element_size; j++)
        {
            A[i][j] = K[i][j];
        }
    }
    // xの初期値は適当
    double x[all_element_size];
    for (i = 0; i < all_element_size; ++i)
    {
        x[i] = 0;
    }

    // CG法でAx=bを解く
    cg_method(A, x, F, all_element_size, tmax, eps);
    // double gosa = confirm_acurracy_of_cg(A, x, F, all_element_size);
    // printf("%f  ", gosa);

    // メモリの解放
    for (int i = 0; i < all_element_size; i++)
    {
        free(A[i]); //各行のメモリを解放
    }

    for (i = 0; i < all_element_size; ++i)
    {
        displacement[i][0] = x[i];
    }
    free(A);
    free(indexes_array);
    free(f_array);
}

//APDLにおける，ソルバー部分の精度を求める
void confirm_apdl_accuracy(double **nodes_pos, int **edges_indices, double **edges_thickness, int node_num, int edge_num, int input_node_num, int *input_nodes, double **input_forces, int frozen_node_num, int *frozen_nodes, double **displacement)
{
    int i, j, k;
    int node1, node2;
    int free_D = 3;
    int all_element_size = node_num * free_D;
    double K[all_element_size][all_element_size];
    double K_e[6][6];
    double F[all_element_size];
    double node[2][2];

    // K行列の初期化
    for (i = 0; i < all_element_size; ++i)
    {
        for (j = 0; j < all_element_size; ++j)
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

    //ひずみ角の条件を加えない場合
    int condition_element_num = frozen_node_num * 3;

    // 各条件の要素（num_node*3中）と，その要素の強制変位や固定変位を収納
    int *indexes_array = malloc(condition_element_num * sizeof(int));
    double *f_array = malloc(condition_element_num * sizeof(double));
    for (int i = 0; i < frozen_node_num; i++)
    {
        indexes_array[i * 3 + 0] = frozen_nodes[i] * 3 + 0;
        indexes_array[i * 3 + 1] = frozen_nodes[i] * 3 + 1;
        indexes_array[i * 3 + 2] = frozen_nodes[i] * 3 + 2;

        f_array[i * 3 + 0] = 0.0;
        f_array[i * 3 + 1] = 0.0;
        f_array[i * 3 + 2] = 0.0;
    }

    //F行列の初期化
    for (i = 0; i < all_element_size; ++i)
    {
        F[i] = 0;
    }
    // Fの外力の部分を設定
    for (int i = 0; i < input_node_num; i++)
    {
        F[input_nodes[i] * 3 + 0] = input_forces[i][0];
        F[input_nodes[i] * 3 + 1] = input_forces[i][1];
    }

    //F,K行列に条件を適用
    for (i = 0; i < condition_element_num; ++i)
    {
        int target_index = indexes_array[i];
        //makinf F matrix
        for (j = 0; j < all_element_size; ++j)
        {
            F[j] -= K[j][target_index] * f_array[i];
        }
        F[target_index] = f_array[i];

        //makinf K matrix
        for (j = 0; j < all_element_size; ++j)
        {
            K[target_index][j] = 0;
            K[j][target_index] = 0;
        }
        K[target_index][target_index] = 1;
    }

    // CGをかける為の形にする
    double **A = malloc(all_element_size * sizeof(double *));
    for (int i = 0; i < all_element_size; i++)
    {
        A[i] = malloc(all_element_size * sizeof(double));
        for (int j = 0; j < all_element_size; j++)
        {
            A[i][j] = K[i][j];
        }
    }

    double gosa = confirm_acurracy_of_cg(A, *displacement, F, all_element_size);
    printf("%f  ", gosa);

    // メモリの解放
    for (int i = 0; i < all_element_size; i++)
    {
        free(A[i]); //各行のメモリを解放
    }

    free(A);
    free(indexes_array);
    free(f_array);
}