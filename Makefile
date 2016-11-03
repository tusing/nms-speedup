CC = gcc

OPTS = -pthread -O0 -fPIC
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
