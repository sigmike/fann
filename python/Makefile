# This makefile is on purpose not made with configure, to show how to use the library
# The make file requires that the fann library is installed (see ../README)

TARGETS = _fann.so 

all: $(TARGETS)

_%.so: %_wrap.o
	gcc -shared $< -lfann -o $@ 

%.o: %.c 
	gcc -c -fpic $< -I/usr/include/python2.3/

%_wrap.c: %.i 
	swig -python $<



clean:
	rm -f $(TARGETS) *_wrap.* fann.py fann.pyc xor_float.net *~
