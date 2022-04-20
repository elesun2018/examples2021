main:main.o fun.o
	gcc main.o fun.o -o main
main.o:main.c
	gcc -c main.c
fun.o:fun.c
	gcc -c fun.c
.PHONY:clean
clean:
	rm -f main main.o fun.o