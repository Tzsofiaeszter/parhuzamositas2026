#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vektor.h"

#define N 1000
#define EPSILON 1e-5

int main() {
    //float A[N], B[N], C[N];

    float A[N], B[N];
    float resultParallel[N];   // addVectors eredménye
    float resultSequential[N]; // CPU ellenőrzés

    srand(time(NULL));   // Véletlenszám-generátor inicializálása

    for (int i = 0; i < N; i++) {     // Vektorok feltöltése 0 és 10 közötti véletlen valós számokkal
        A[i] = (float)rand() / RAND_MAX * 10.0f;
        B[i] = (float)rand() / RAND_MAX * 10.0f;
    }
/* 4/a feladat
    // Vektorok összeadása
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    // Eredmények kiírása
    printf("A vektor:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", A[i]);
    }

    printf("\n\nB vektor:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", B[i]);
    }

*/
// 4/b feladat
/*
    addVectors(A, B, C, N); // Itt már nem látszik, hogy milyen implementáció fut

    printf("\n\nC = A + B:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", C[i]);
    }

    printf("\n");
*/

// 4/c feladat
    // "Párhuzamos" (vagy OpenCL-es) számítás
    addVectors(A, B, resultParallel, N);

    // Szekvenciális ellenőrző számítás
    for (int i = 0; i < N; i++) {
        resultSequential[i] = A[i] + B[i];
    }

    // Összehasonlítás
    int correct = 1;

    for (int i = 0; i < N; i++) {
        if (fabs(resultParallel[i] - resultSequential[i]) > EPSILON) {
            correct = 0;
            printf("Hiba a(z) %d. indexen!\n", i);
            break;
        }
    }

    if (correct)
        printf("Az eredmeny HELYES.\n");
    else
        printf("Az eredmeny HIBAS.\n");

    return 0;
}