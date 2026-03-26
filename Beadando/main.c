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
    cl_int hiba;
    cl_uint platformok_szama;
    cl_platform_id platform_azon;
    cl_device_id eszkoz_azon;
    cl_context kontextus;
    cl_command_queue parancs_sor;

    // OpenCL platform
    hiba = clGetPlatformIDs(1, &platform_azon, &platformok_szama);
    if(hiba != CL_SUCCESS){ printf("Platform hiba: %d\n", hiba); return 1; }

    // OpenCL GPU device
    cl_uint eszkozok_szama;
    hiba = clGetDeviceIDs(platform_azon, CL_DEVICE_TYPE_GPU, 1, &eszkoz_azon, &eszkozok_szama);
    if(hiba != CL_SUCCESS){ printf("Device hiba: %d\n", hiba); return 1; }

    // Kontextus és parancssor
    kontextus = clCreateContext(NULL, 1, &eszkoz_azon, NULL, NULL, &hiba);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    parancs_sor = clCreateCommandQueueWithProperties(kontextus, eszkoz_azon, props, &hiba);

    // Kernel betöltése fájlból
    int err;
    char* kernel_src = load_kernel_source("matrix.cl", &err);
    if (err != 0) {
        printf("Kernel file betoltesi hiba!\n");
        return 1;
    }

    // Mátrixok létrehozása
    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));
    float* C_soros = (float*)malloc(N*N*sizeof(float));
    float* C_gpu = (float*)malloc(N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        A[i] = (float)(rand()%10);
        B[i] = (float)(rand()%10);
        C_soros[i]=C_gpu[i]=0;
    }

    // Soros futásidő mérés
    clock_t t1 = clock();
    soros_matrix_szorzas(A,B,C_soros,N);
    clock_t t2 = clock();
    double soros_ido = (double)(t2-t1)/CLOCKS_PER_SEC;
    printf("CPU: soros futasido: %.6f s\n", soros_ido);

    // OpenCL memóriák
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

    // Kernel argumentumok
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    int n= N;
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Méretezés
    size_t global[2] = {N, N};

    // OpenCL futás
    cl_event esemeny;
    hiba = clEnqueueNDRangeKernel(parancs_sor, kernel, 2, NULL, global, NULL, 0, NULL, &esemeny);
    clWaitForEvents(1,&esemeny);

    // Időmérés
    cl_ulong kezdet, veg;
    clGetEventProfilingInfo(esemeny, CL_PROFILING_COMMAND_START, sizeof(kezdet), &kezdet, NULL);
    clGetEventProfilingInfo(esemeny, CL_PROFILING_COMMAND_END, sizeof(veg), &veg, NULL);
    double gpu_ido = (double)(veg-kezdet)*1e-9;
    printf("GPU: OpenCL futasido: %.6f s\n", gpu_ido);

    // Eredmény visszaolvasása
    clEnqueueReadBuffer(parancs_sor, bufC, CL_TRUE, 0, N*N*sizeof(float), C_gpu, 0, NULL, NULL);

    printf("Gyorsitas (CPU/GPU): %.2f\n", soros_ido/gpu_ido);

    // Felszabadítás
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