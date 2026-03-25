#include <stdio.h>
#include <CL/cl.h>

int main(void)
{
    cl_int err;
    cl_platform_id platform_id;
    cl_uint n_platforms;
	cl_device_id device_id;
	cl_uint n_devices;
	size_t real_size;
	char param_value[1000];

    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
		return 0;
	}
	printf("Detected OpenCL platforms: %d\n", n_platforms);

	err = clGetPlatformInfo(
		platform_id,
		CL_PLATFORM_NAME,
		1000,
		param_value,
		&real_size
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetPlatformInfo. Error code: %d\n", err);
		return 0;
	}
	printf("Platform: %s\n", param_value);

	err = clGetDeviceIDs(
		platform_id,
		CL_DEVICE_TYPE_GPU,
		1,
		&device_id,
		&n_devices
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
		return 0;
	}
	printf("Number of devices: %d\n", n_devices);

	err = clGetDeviceInfo(
		device_id,
		CL_DEVICE_NAME,
		1000,
		param_value,
		&real_size
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetDeviceInfo. Error code: %d\n", err);
		return 0;
	}
	printf("Device name: %s\n", param_value);

    return 0;
}
