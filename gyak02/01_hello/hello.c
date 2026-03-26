#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define CL_TARGET_OPENCL_VERSION 220
#define SAMPLE_SIZE 1000

// Kernel betöltése fájlból
char* load_kernel_source(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("[ERROR] Cannot open kernel file: %s\n", filename);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char* source = (char*)malloc(size + 1);
    if (!source) {
        fclose(f);
        printf("[ERROR] Cannot allocate memory for kernel source\n");
        return NULL;
    }
    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);
    return source;
}

int main(void) {
    cl_int err;
    cl_uint n_platforms;
    cl_platform_id platform_id;
    cl_uint n_devices;
    cl_device_id device_id;

    // Platform
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetPlatformIDs failed: %d\n", err);
        return 1;
    }

    // Device
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetDeviceIDs failed: %d\n", err);
        return 1;
    }

    // Context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateContext failed: %d\n", err);
        return 1;
    }

    // Kernel betöltése fájlból
    char* kernel_source = load_kernel_source("hello_kernel.cl");
    if (!kernel_source) return 1;

    // Program létrehozása és build
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);
    free(kernel_source);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateProgramWithSource failed: %d\n", err);
        return 1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Build error log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("[BUILD ERROR]\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "hello_kernel", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateKernel failed: %d\n", err);
        return 1;
    }

    // Host buffer
    int* host_buffer = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    for (int i = 0; i < SAMPLE_SIZE; ++i) host_buffer[i] = i;

    // Device buffer
    cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateBuffer failed: %d\n", err);
        return 1;
    }

    // Kernel argumentek
    int n = SAMPLE_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &n);

    // Command queue (modern OpenCL 2.2)
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clCreateCommandQueueWithProperties failed: %d\n", err);
        return 1;
    }

    // Másolás host -> device
    clEnqueueWriteBuffer(queue, device_buffer, CL_TRUE, 0, SAMPLE_SIZE * sizeof(int), host_buffer, 0, NULL, NULL);

    // Work sizes
    size_t local_work_size = 256;
    size_t global_work_size = ((SAMPLE_SIZE + local_work_size - 1) / local_work_size) * local_work_size;

    // Kernel futtatása
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    // Másolás device -> host
    clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, SAMPLE_SIZE * sizeof(int), host_buffer, 0, NULL, NULL);

    // Eredmények kiírása
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        printf("[%d] = %d\n", i, host_buffer[i]);
    }

    // Erőforrások felszabadítása
    clReleaseMemObject(device_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(host_buffer);

    return 0;
}