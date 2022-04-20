#!/bin/sh
file_1 := test_a.c
file_2 := test_b.c
main_file := test.c
out_file := test
daymic_lib := .so 
out_so_name := lib$(out_file)$(daymic_lib)
 
.PHONY:all
all:
	gcc ${file_1} ${file_2} -fPIC -shared -o $(out_so_name)
	gcc ${main_file} -o ${out_file} -L. -l${out_file}
clean:
	rm -f ${out_file} $(out_so_name)