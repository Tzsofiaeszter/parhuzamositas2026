__kernel void matrix_szorzas(__global float* A,
                             __global float* B,
                             __global float* C,
                             int N)
{
    int sor = get_global_id(0);
    int oszlop = get_global_id(1);

    float osszeg = 0.0f;

    for (int k = 0; k < N; k++) {
        osszeg += A[sor * N + k] * B[k * N + oszlop];
    }

    C[sor * N + oszlop] = osszeg;
}