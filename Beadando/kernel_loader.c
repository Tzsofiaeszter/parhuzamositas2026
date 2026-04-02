#include "kernel_loader.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* const path, int* error_code)  // matrix.cl elérési útvonala
{
    FILE* source_file;
    char* source_code;
    int file_size;

    source_file = fopen(path, "rb");      // fájl megnyitása
    if (source_file == NULL) {
        *error_code = -1;
        return NULL;
    }

    fseek(source_file, 0, SEEK_END);         // fájl méretének meghatározása
    file_size = ftell(source_file);
    rewind(source_file);

    source_code = (char*)malloc(file_size + 1);    // memóriafoglalás

    fread(source_code, sizeof(char), file_size, source_file);  //beolvassa a teljes fájlt a source_code-ba
    source_code[file_size] = 0;   // \0 a lezárásra, így már szöveges sztring a fájl

    *error_code = 0;    // ha minden rendben, a hibakód 0
    return source_code;  // visszaadja a fájl tartalmát karakterláncként
}
