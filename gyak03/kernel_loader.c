#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* filename, size_t* length) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(1); }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* src = (char*)malloc(len + 1);
    if (!src) { perror("malloc"); exit(1); }

    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);

    if (length) *length = len;
    return src;
}