# This makefile was written to compile a distribution of pyfann for
# GNU platforms (cygwin included.)

ROOT=/

all: build

linux:
	@(cd pyfann && make)

clean:
	@(cd pyfann && make clean)
	@rm -fr dist build setup.pyc
	@rm -fr examples/{nets,pyfann/}

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

test: build
	@-mkdir examples/nets
	@python setup.py install --install-lib examples 
	@cd examples && python simple_train.py && python mushroom.py
