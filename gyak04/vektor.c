#include "vektor.h"

void addVectors(float* A, float* B, float* C, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}