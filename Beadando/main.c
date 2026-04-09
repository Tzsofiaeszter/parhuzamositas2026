#include "kernel_loader.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048

// Soros mátrixszorzás
void serial_matrix_multiplication(float* A, float* B, float* C, int size) {
    for(int i=0;i<size;i++)
        for(int j=0;j<size;j++){
            float sum=0;
            for(int k=0;k<size;k++)
                sum += A[i*size+k]*B[k*size+j];
            C[i*size+j] = sum;
        }
}

// OpenCL inicializálás
int init_opencl(cl_platform_id* platform, cl_device_id* device, cl_context* context, cl_command_queue* queue) {

    cl_int error;
    cl_uint platform_count;
// 1. Lekérdezi az elérhető OpenCL platformokat
    error = clGetPlatformIDs(1, platform, &platform_count);
    if(error != CL_SUCCESS){
        printf("Platform error: %d\n", error);
        return 1;
    }

// 2. GPU eszköz lekérdezése, device
    error = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
    if(error != CL_SUCCESS){
        printf("Device error: %d\n", error);
        return 1;
    }

// 3. OpenCL kontextus létrehozása 
    *context = clCreateContext(NULL, 1, device, NULL, NULL, &error);

// 4.-5. Parancssor profilinggal
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    *queue = clCreateCommandQueueWithProperties(*context, *device, props, &error);

    return 0;
}

// 7. Device buffer létrehozása 
void create_buffers(cl_context context, cl_mem* bufferA, cl_mem* bufferB, cl_mem* bufferC) {
    cl_int error;

    *bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, N*N*sizeof(float), NULL, &error);
    *bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, N*N*sizeof(float), NULL, &error);
    *bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*N*sizeof(float), NULL, &error);
}

//a.) Host -> Device másolás mérése  :  CPU->GPU
double write_to_device(cl_command_queue queue, cl_mem bufferA, cl_mem bufferB, float* A, float* B) {

    cl_event eventA, eventB;
    cl_ulong start, end;

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, N*N*sizeof(float), A, 0, NULL, &eventA);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, N*N*sizeof(float), B, 0, NULL, &eventB);

    clGetEventProfilingInfo(eventA, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(eventA, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double timeA = (double)(end-start)*1e-9;

    clGetEventProfilingInfo(eventB, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(eventB, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double timeB = (double)(end-start)*1e-9;

    return timeA + timeB;
}

// b.) Kernel futás
double run_kernel(cl_context context, cl_device_id device, cl_command_queue queue,cl_mem bufferA, cl_mem bufferB, cl_mem bufferC, char* kernel_source, size_t* total_threads) {

    cl_int error;
// Program build
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &error);
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if(error != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size+1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size]=0;
        printf("Build log:\n%s\n", log);
        free(log);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication", &error);

    // Kernel argumentumok
    int n = N;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t global[2] = {N, N};
    size_t local[2] = {16,16};

// Szálak számának kiírása 
    *total_threads = global[0] * global[1];
    printf("GPU threads: %lu x %lu = %lu\n", (unsigned long)global[0], (unsigned long)global[1], (unsigned long)(*total_threads));

    cl_event event;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1,&event);

    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    double kernel_time = (double)(end-start)*1e-9;

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseEvent(event);

    return kernel_time;
}

//12. Device -> Host másolás mérése  : GPU->CPU
double read_from_device(cl_command_queue queue, cl_mem bufferC, float* C) {

    cl_event event;
    cl_ulong start, end;

    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, N*N*sizeof(float), C, 0, NULL, &event);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    return (double)(end-start)*1e-9;
}


int main() {
// 6. Host buffer inicializálása  : CPU
    float* A = malloc(N*N*sizeof(float));
    float* B = malloc(N*N*sizeof(float));
    float* C_cpu = malloc(N*N*sizeof(float));
    float* C_gpu = malloc(N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        A[i] = rand()%10;
        B[i] = rand()%10;
        C_cpu[i]=C_gpu[i]=0;
    }

    // CPU CPU futásidő mérés
    clock_t t1 = clock();
    serial_matrix_multiplication(A,B,C_cpu,N);
    clock_t t2 = clock();
    double cpu_time = (double)(t2-t1)/CLOCKS_PER_SEC;

    printf("CPU time: %.6f s\n", cpu_time);

//1.-5. OpenCL inicializálás
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    if(init_opencl(&platform, &device, &context, &queue) != 0)
        return 1;

// 7. Device buffer létrehozása 
    cl_mem bufferA, bufferB, bufferC;
    create_buffers(context, &bufferA, &bufferB, &bufferC);

    // Kernel betöltése
    int err;
    char* kernel_source = load_kernel_source("matrix.cl", &err);

    // CPU->GPU
    double write_time = write_to_device(queue, bufferA, bufferB, A, B);

    // Kernel futás
    size_t total_threads;
    double kernel_time = run_kernel(context, device, queue, bufferA, bufferB, bufferC, kernel_source, &total_threads);

    // GPU->CPU
    double read_time = read_from_device(queue, bufferC, C_gpu);

    double total_gpu_time = kernel_time + write_time + read_time;

// 13. Eredmények kiírása
    printf("GPU kernel time: %.6f s\n", kernel_time);
    printf("Total GPU time: %.6f s\n", total_gpu_time);
    printf("Speedup (CPU / kernel): %.2f\n", cpu_time/kernel_time);
    printf("Speedup (CPU / total GPU): %.2f\n", cpu_time/total_gpu_time);
    printf("Time per thread: %.12f s\n", kernel_time / total_threads);

// 14. Felszabadítás
    free(kernel_source);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A); free(B); free(C_cpu); free(C_gpu);

    return 0;
}