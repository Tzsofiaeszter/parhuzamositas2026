#define TS 16  // tile size (egyezzen local_size-val): egy olyan paraméter, ami azt mondja meg, hogy a mátrixot mekkora kisebb blokkokra (tile-okra) bontod fel a GPU-n történő számítás során

__kernel void matrix_multiplication(__global float* A,
                                     __global float* B,
                                     __global float* C,
                                     int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    int localRow = get_local_id(0);
    int localCol = get_local_id(1);

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float sum = 0.0f;

    for (int t = 0; t < N; t += TS) {

        // globális -> lokális betöltés
        Asub[localRow][localCol] = A[row * N + (t + localCol)];
        Bsub[localRow][localCol] = B[(t + localRow) * N + col];

        barrier(CLK_LOCAL_MEM_FENCE);

        // részszorzás tile-on belül
        for (int k = 0; k < TS; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row * N + col] = sum;
}