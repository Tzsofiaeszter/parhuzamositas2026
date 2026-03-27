#include "kernel_loader.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 200

// Soros mátrixszorzás
void soros_matrix_szorzas(float* A, float* B, float* C, int meret) {
    for(int i=0;i<meret;i++)
        for(int j=0;j<meret;j++){
            float osszeg=0;
            for(int k=0;k<meret;k++)
                osszeg += A[i*meret+k]*B[k*meret+j];
            C[i*meret+j] = osszeg;
        }
}

int main() {
    cl_int hiba;                    // egész szám típus, amit az OpenCL hibakódok tárolására használ
    cl_uint platformok_szama;       // nem-negatív egész, ami az elérhető OpenCL platformok számát tárolja
    cl_platform_id platform_azon;   // platform azonosító (ID), amit az OpenCL ad vissza: NVIDIA, Intel pl.; ezzel azonosítjuk, hogy melyik platformon akarunk számításokat végezni
    cl_device_id eszkoz_azon;       // eszköz azonosító, a konkrét GPU, amin futtatjuk a számításokat
    cl_context kontextus;           // egy OpenCL munkaterület, ami összeköti a platformot, eszközt és a memóriát; minden OpenCL programnak szüksége van kontextusra, hogy tudja, hol és mivel dolgozik.
    cl_command_queue parancs_sor;   // parancssor, ahol a CPU küldi az utasításokat a GPU-nak

// 1. Lekérdezi az elérhető OpenCL platformokat(pl.: NVIDIA, Intel); a platform_azon lesz az, amit később használunk
    hiba = clGetPlatformIDs(1, &platform_azon, &platformok_szama);
    if(hiba != CL_SUCCESS){ 
        printf("Platform hiba: %d\n", hiba); 
        return 1; 
    }

    cl_uint eszkozok_szama; // Arra használjuk, hogy megmondja, hány eszköz (GPU/CPU) érhető el egy platformon; a gépben van 2 GPU, akkor a eszkozok_szama = 2; tudni akarjuk, hány GPU közül lehet választani
// 2. A platformhoz tartozó eszközök (GPU) lekérdezése; eszkoz_azon → konkrét GPU
    hiba = clGetDeviceIDs(platform_azon, CL_DEVICE_TYPE_GPU, 1, &eszkoz_azon, &eszkozok_szama);
    if(hiba != CL_SUCCESS){ 
        printf("Device hiba: %d\n", hiba); 
        return 1; 
    }

// 3. OpenCL kontextus létrehozása 
    kontextus = clCreateContext(NULL, 1, &eszkoz_azon, NULL, NULL, &hiba);
 
// 4. Itt küldjük a parancsokat a GPU-nak; profiling bekapcsolva → időméréshez kell
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    parancs_sor = clCreateCommandQueueWithProperties(kontextus, eszkoz_azon, props, &hiba);

    // Kernel betöltése fájlból
    int err;
// 5.Betölti a .cl fájlt és lefordítja GPU-ra; ha hiba van kiírja
    char* kernel_src = load_kernel_source("matrix.cl", &err);
    if (err != 0) {
        printf("Kernel file betoltesi hiba!\n");
        return 1;
    }

// 6. Host buffer inicializálása;  CPU memóriában létrehozza a mátrixokat; feltölti őket véletlen számokkal
    float* A = (float*)malloc(N*N*sizeof(float));     // foglalunk egy N×N-es mátrixot az A-nak
    float* B = (float*)malloc(N*N*sizeof(float));
    float* C_soros = (float*)malloc(N*N*sizeof(float));
    float* C_gpu = (float*)malloc(N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        A[i] = (float)(rand()%10);                    // majd feltöltjük őket random számokkal : 0-9
        B[i] = (float)(rand()%10);
        C_soros[i]=C_gpu[i]=0;                        // nullázzuk mert még nincs értéke
    }

    // Soros futásidő mérés
    clock_t t1 = clock();
    soros_matrix_szorzas(A,B,C_soros,N);
    clock_t t2 = clock();
    double soros_ido = (double)(t2-t1)/CLOCKS_PER_SEC;

// 13. Eredmények megjelenítése
    printf("CPU: soros futasido: %.6f s\n", soros_ido);

// 7. Device buffer létrehozása: GPU memóriában buffer-ek létrehozása; A és B azonnal feltöltve (COPY_HOST_PTR)
    cl_mem bufA = clCreateBuffer(kontextus, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*N*sizeof(float), A, &hiba);
    cl_mem bufB = clCreateBuffer(kontextus, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*N*sizeof(float), B, &hiba);
    cl_mem bufC = clCreateBuffer(kontextus, CL_MEM_WRITE_ONLY, N*N*sizeof(float), NULL, &hiba);

    // Program létrehozása
    cl_program program = clCreateProgramWithSource(kontextus, 1, (const char**)&kernel_src, NULL, &hiba);
    hiba = clBuildProgram(program, 1, &eszkoz_azon, NULL, NULL, NULL);

    if(hiba != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program, eszkoz_azon, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size+1);
        clGetProgramBuildInfo(program, eszkoz_azon, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size]=0;
        printf("Build log:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_szorzas", &hiba);

// 8. Kernel argumentumok: Átadjuk a GPU-nak a bemeneti/kimeneti adatokat; paraméterek: A, B, C, méret
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    int n= N;
    clSetKernelArg(kernel, 3, sizeof(int), &n);

// 9. Problémaméret megadása: Meghatározza hány szál (work-item) fut; minden mátrix elemre egy szál
    size_t global[2] = {N, N};

    // OpenCL futás
    cl_event esemeny;
// 11. Kernelek futtatása: GPU-n fut a mátrixszorzás; 2 dimenziós végrehajtás
    hiba = clEnqueueNDRangeKernel(parancs_sor, kernel, 2, NULL, global, NULL, 0, NULL, &esemeny);
    clWaitForEvents(1,&esemeny);

    // Időmérés
    cl_ulong kezdet, veg;
    clGetEventProfilingInfo(esemeny, CL_PROFILING_COMMAND_START, sizeof(kezdet), &kezdet, NULL);
    clGetEventProfilingInfo(esemeny, CL_PROFILING_COMMAND_END, sizeof(veg), &veg, NULL);
    double gpu_ido = (double)(veg-kezdet)*1e-9;

 // 13. Eredmények megjelenítése
    printf("GPU: OpenCL futasido: %.6f s\n", gpu_ido);

// 12. Adatok visszaolvasása a Device buffer-ből: CPU → CPU másolás; Az eredmény bekerül C_gpu-ba
    clEnqueueReadBuffer(parancs_sor, bufC, CL_TRUE, 0, N*N*sizeof(float), C_gpu, 0, NULL, NULL);

// 13. Eredmények megjelenítése
    printf("Gyorsitas (CPU/GPU): %.2f\n", soros_ido/gpu_ido);

// 14. Erőforrások felszabadítása: Minden memória és OpenCL objektum felszabadítása; Nagyon fontos (memóriaszivárgás elkerülése)
    free(kernel_src);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(parancs_sor);
    clReleaseContext(kontextus);

    free(A); free(B); free(C_soros); free(C_gpu);

    return 0;
}