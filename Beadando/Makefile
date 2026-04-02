CC = gcc
CFLAGS = -Wall -O2 -DCL_TARGET_OPENCL_VERSION=220
LIBS = -lOpenCL -lm

# A célnév
TARGET = matrix.exe

# Forrásfájlok
SRCS = main2.c kernel_loader.c

# Alap build target
all: $(TARGET)

# Linkelés
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o "$(TARGET)" $(LIBS)

# Tisztítás
clean:
	if exist "$(TARGET)" del "$(TARGET)"