CC = gcc

OPTS = -pthread -O2 -fPIC -fopenmp -funroll-loops -mavx -march=corei7-avx
CFLAGS = -Wall -shared -std=gnu99 $(OPTS)
LFLAGS = -shared -Bsymbolic-functions 

targets : nms.so

.PHONY : default
default : all
all: clean $(targets)


nms.so : nmsModule.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.so *.o
