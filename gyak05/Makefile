CC = gcc
CFLAGS = -std=c99 -Wall
LDFLAGS = -lOpenCL
TARGET = main.exe
SRC = main.c 

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	del $(TARGET)