#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <CL/cl.h>

char* load_kernel_source(const char* filename, size_t* length);

#endif