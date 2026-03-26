#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

int main(void) {
    cl_int err;

    // Platformok száma
    cl_uint n_platforms;
    err = clGetPlatformIDs(0, NULL, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetPlatformIDs failed. Error code: %d\n", err);
        return 1;
    }
    printf("Number of platforms: %u\n\n", n_platforms);

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platforms);
    clGetPlatformIDs(n_platforms, platforms, NULL);

    for (cl_uint i = 0; i < n_platforms; i++) {
        char platform_name[128];
        char platform_vendor[128];
        char platform_version[128];

        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);

        printf("Platform %u: %s\n", i, platform_name);
        printf("  Vendor: %s\n", platform_vendor);
        printf("  Version: %s\n", platform_version);

        // Eszközök száma a platformon
        cl_uint n_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
        if (err != CL_SUCCESS) {
            printf("  [ERROR] clGetDeviceIDs failed. Error code: %d\n\n", err);
            continue;
        }
        printf("  Number of devices: %u\n", n_devices);

        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * n_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, n_devices, devices, NULL);

        for (cl_uint j = 0; j < n_devices; j++) {
            char device_name[128];
            cl_uint compute_units;
            cl_ulong global_mem;
            cl_ulong local_mem;
            cl_uint clock_freq;
            size_t max_workgroup;

            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_freq), &clock_freq, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_workgroup), &max_workgroup, NULL);

            printf("    Device %u: %s\n", j, device_name);
            printf("      Compute Units: %u\n", compute_units);
            printf("      Max Clock Frequency: %u MHz\n", clock_freq);
            printf("      Global Memory: %llu MB\n", (unsigned long long)(global_mem / (1024*1024)));
            printf("      Local Memory: %llu KB\n", (unsigned long long)(local_mem / 1024));
            printf("      Max Work Group Size: %zu\n\n", max_workgroup);
        }

        free(devices);
        printf("\n");
    }

    free(platforms);
    return 0;
}