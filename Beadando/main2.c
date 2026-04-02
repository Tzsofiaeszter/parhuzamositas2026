#include "kernel_loader.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

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

// 1. Lekérdezi az elérhető OpenCL platformokat
    hiba = clGetPlatformIDs(1, &platform_azon, &platformok_szama);
    if(hiba != CL_SUCCESS){
        printf("Platform hiba: %d\n", hiba);
        return 1;
    }

// 2. GPU eszköz lekérdezése
    cl_uint eszkozok_szama;
    hiba = clGetDeviceIDs(platform_azon, CL_DEVICE_TYPE_GPU, 1, &eszkoz_azon, &eszkozok_szama);
    if(hiba != CL_SUCCESS){
        printf("Device hiba: %d\n", hiba);
        return 1;
    }

// 3. OpenCL kontextus létrehozása
    kontextus = clCreateContext(NULL, 1, &eszkoz_azon, NULL, NULL, &hiba);
    if(hiba != CL_SUCCESS){
        printf("Context hiba: %d\n", hiba);
        return 1;
    }

// 4. Parancssor profilinggal
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    parancs_sor = clCreateCommandQueueWithProperties(kontextus, eszkoz_azon, props, &hiba);

    // Kernel betöltése
    int err;
    char* kernel_src = load_kernel_source("matrix.cl", &err);
    if (err != 0) {
        printf("Kernel file betoltesi hiba!\n");
        return 1;
    }

// 6. Host buffer inicializálása
    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));
    float* C_soros = (float*)malloc(N*N*sizeof(float));
    float* C_gpu = (float*)malloc(N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        A[i] = (float)(rand()%10);
        B[i] = (float)(rand()%10);
        C_soros[i]=C_gpu[i]=0;
    }

    // CPU futásidő mérés
    clock_t t1 = clock();
    soros_matrix_szorzas(A,B,C_soros,N);
    clock_t t2 = clock();
    double soros_ido = (double)(t2-t1)/CLOCKS_PER_SEC;

    printf("CPU: soros futasido: %.6f s\n", soros_ido);

// 7. Device buffer létrehozása 
    cl_mem bufA = clCreateBuffer(kontextus, CL_MEM_READ_ONLY, N*N*sizeof(float), NULL, &hiba);
    cl_mem bufB = clCreateBuffer(kontextus, CL_MEM_READ_ONLY, N*N*sizeof(float), NULL, &hiba);
    cl_mem bufC = clCreateBuffer(kontextus, CL_MEM_WRITE_ONLY, N*N*sizeof(float), NULL, &hiba);

// --- Host -> Device másolás mérése ---
    cl_event write_eventA, write_eventB;

    clEnqueueWriteBuffer(parancs_sor, bufA, CL_TRUE, 0, N*N*sizeof(float), A, 0, NULL, &write_eventA);
    clEnqueueWriteBuffer(parancs_sor, bufB, CL_TRUE, 0, N*N*sizeof(float), B, 0, NULL, &write_eventB);

    cl_ulong kezdet, veg;

    // A másolás ideje
    clGetEventProfilingInfo(write_eventA, CL_PROFILING_COMMAND_START, sizeof(kezdet), &kezdet, NULL);
    clGetEventProfilingInfo(write_eventA, CL_PROFILING_COMMAND_END, sizeof(veg), &veg, NULL);
    double write_time_A = (double)(veg-kezdet)*1e-9;

    clGetEventProfilingInfo(write_eventB, CL_PROFILING_COMMAND_START, sizeof(kezdet), &kezdet, NULL);
    clGetEventProfilingInfo(write_eventB, CL_PROFILING_COMMAND_END, sizeof(veg), &veg, NULL);
    double write_time_B = (double)(veg-kezdet)*1e-9;

    double total_write_time = write_time_A + write_time_B;
    printf("Host -> Device masolas ideje: %.6f s\n", total_write_time);

// Program build
    cl_program program = clCreateProgramWithSource(kontextus, 1, (const char**)&kernel_src, NULL, &hiba);
    hiba = clBuildProgram(program, 1, &eszkoz_azon, NULL, NULL, NULL);

    if(hiba != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program, eszkoz_azon, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size+1);
        clGetProgramBuildInfo(program, eszkoz_azon, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size]=0;
        printf("Build log:\n%s\n", log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_szorzas", &hiba);

// 8. Kernel argumentumok
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    int n= N;
    clSetKernelArg(kernel, 3, sizeof(int), &n);

// 9. Problémaméret ELTÉRÉS A main.c-hez KÉPEST
    size_t global[2] = {N, N};
    size_t local[2] = {16, 16};

    if (N % local[0] != 0 || N % local[1] != 0) {
        printf("Hiba: global nem oszthato local-lal!\n");
        return 1;
    }

    size_t total_threads = global[0] * global[1];
    printf("GPU szalak: %lu x %lu = %lu\n",
           (unsigned long)global[0],
           (unsigned long)global[1],
           (unsigned long)total_threads);

// 11. Kernel futtatás
    cl_event esemeny;
    hiba = clEnqueueNDRangeKernel(parancs_sor, kernel, 2, NULL, global, local, 0, NULL, &esemeny);
    if (hiba != CL_SUCCESS) {
        printf("Kernel hiba: %d\n", hiba);
        return 1;
    }

    clFinish(parancs_sor);

// Kernel idő
    clGetEventProfilingInfo(esemeny, CL_PROFILING_COMMAND_START, sizeof(kezdet), &kezdet, NULL);
    clGetEventProfilingInfo(esemeny, CL_PROFILING_COMMAND_END, sizeof(veg), &veg, NULL);
    double gpu_ido = (double)(veg-kezdet)*1e-9;

    printf("GPU kernel ido: %.6f s\n", gpu_ido);

// Egy szál idejének becslése
    double time_per_thread = gpu_ido / (double)total_threads;
    printf("Egy szal ideje: %.12f s\n", time_per_thread);

// 12. Device -> Host másolás mérése
    cl_event read_event;
    clEnqueueReadBuffer(parancs_sor, bufC, CL_TRUE, 0, N*N*sizeof(float), C_gpu, 0, NULL, &read_event);

    clFinish(parancs_sor);

    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(kezdet), &kezdet, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(veg), &veg, NULL);
    double read_time = (double)(veg-kezdet)*1e-9;

    printf("Device -> Host ido: %.6f s\n", read_time);

    
// Teljes GPU idő
    double total_gpu_time = gpu_ido + total_write_time + read_time;
    printf("Teljes GPU ido: %.6f s\n", total_gpu_time);

// Gyorsítások    
    printf("Gyorsitas (kernel): %.2f\n", soros_ido/gpu_ido);
    printf("Gyorsitas (teljes): %.2f\n", soros_ido/total_gpu_time);

// 14. Felszabadítás
    free(kernel_src);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseEvent(esemeny);
    clReleaseProgram(program);
    clReleaseCommandQueue(parancs_sor);
    clReleaseContext(kontextus);
    clReleaseEvent(read_event);
    clReleaseEvent(write_eventA);
    clReleaseEvent(write_eventB);

    free(A); free(B); free(C_soros); free(C_gpu);

    return 0;
}