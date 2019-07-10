CC=gcc

LIBNAME=libcorrections

all: src/corrections.c
	$(CC) --std=c99 -mtune=core-avx2 -shared -fPIC -O3 -o $(LIBNAME).so $? -Wall
