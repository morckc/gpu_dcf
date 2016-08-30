OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

all: main viewer 

main: main.c
	gcc -Wall -g main.c -o main $(OPTIONS)
viewer: viewer.c 
	gcc -Wall -g viewer.c -o viewer 
clean:
	rm -rf main
