CC=gcc

LIBNAME=libcorrections

all: src/corrections.c
        #$(CC) -shared -fPIC -DDEBUG -O3 -o $(LIBNAME).so $?
        #$(CC) -shared -fPIC -O3 $(CFLAGS) -o $(LIBNAME).so $?
	$(CC) --std=c99 -mtune=core-avx2 -shared -fPIC -O2 -o $(LIBNAME).so $? -Wall
