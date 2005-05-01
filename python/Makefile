# This makefile was written to compile a distribution of pyfann for
# GNU platforms (cygwin included.)

ROOT=/

all: install

linux:
	@(cd pyfann && make)

clean:
	@(cd pyfann && make clean)
	@rm -fr dist build setup.pyc

msvc:
	@(cd pyfann && make -f makefile.msvc)

build: ../src/fann.o
	@python setup.py build

install: build
	@python setup.py install --root $(ROOT)

../src/fann.o: ../Makefile
	@(cd .. && make )

../Makefile:
	@(cd .. && ./configure)

