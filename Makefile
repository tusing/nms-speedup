CC = gcc

OPTS = -pthread -O3 -fPIC -fopenmp -funroll-loops
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
