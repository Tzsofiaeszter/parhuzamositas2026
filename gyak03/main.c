#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include "kernel_loader.h"

#define N 16

int main() {
    cl_int err;

    // Platform és device
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Context és Queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_queue_properties props[] = {0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);

    size_t kernel_length;
    char* kernel_src = load_kernel_source("kernels.cl", &kernel_length);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_src, &kernel_length, &err);
    free(kernel_src);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "map_reverse_and_swap", &err);

    // Memória allokálás
    int* data = (int*)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) data[i] = 0;

    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, data, &err);

    // Kernel argumentumok
    int n = N;  // hibát javítja (#define N nem lehet &-el)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(kernel, 1, sizeof(int), &n);

    // Kernel futtatás
    size_t global = (N + 1) / 2;  // minden második párhoz
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Eredmény visszaolvasása
    clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(int) * N, data, 0, NULL, NULL);

    // Eredmény kiírása
    printf("Eredmeny: ");
    for (int i = 0; i < N; i++) printf("%d ", data[i]);
    printf("\n");

    // Takarítás
    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(data);

    return 0;
}