CC = gcc

OPTS = -pthread -O2 -fPIC -fopenmp -funroll-loops -mavx -march=corei7-avx
CFLAGS = -Wall -shared -std=gnu99 $(OPTS) -I/usr/local/cuda-7.5/include
LFLAGS = -shared -Bsymbolic-functions -L/usr/local/cuda-7.5/lib64 -lOpenCL

targets : nms.so

.PHONY : default
default : all
all: clean $(targets)


nms.so : nmsModule.c
	$(CC) $(CFLAGS) $(LFLAGS) $< -o $@

clean:
	rm -f *.so *.o
