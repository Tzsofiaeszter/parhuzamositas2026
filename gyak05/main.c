#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define N 7

// kernel fájl betöltése
char* load_kernel(const char* filename, size_t* len) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* src = (char*)malloc(size+1);
    fread(src, 1, size, f);
    src[size]='\0';
    fclose(f);
    *len = size;
    return src;
}

int main() {
    int data[N] = {5,0,0,10,7,0,12};  // hiányzó értékek 0-val jelölve

    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    cl_context context = clCreateContext(NULL,1,&device,NULL,NULL,&err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

    size_t len;
    char* source = load_kernel("gapfill.cl", &len);

    cl_program program = clCreateProgramWithSource(context,1,(const char**)&source,&len,&err);
    free(source);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "gapfill", &err);

    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(int)*N, data, &err);

    int n = N;  // hibát javítja
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(kernel, 1, sizeof(int), &n);

    size_t global = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(int)*N, data, 0, NULL, NULL);

    printf("Kitoltott tomb: ");
    for (int i=0; i<N; i++) printf("%d ", data[i]);
    printf("\n");

    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}