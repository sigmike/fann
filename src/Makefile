CFLAGS = -pipe -fsigned-char -ansi -pedantic -Wall -O3 -finline-functions -funroll-loops -Werror -Iinclude/ # -ggdb

ARM-CFLAGS = -pipe -fsigned-char -ansi -pedantic -Wall -O3 -finline-functions -funroll-loops -Werror -Iinclude/ -Xlinker --strip-all -Xlinker -dy

LFLAGS = -lm
H_FILES = $(wildcard include/*.h)
C_FILES = $(wildcard *.c)

GCC = gcc
G++ = g++
ARM-GCC = arm-linux-gcc
ARM-G++ = arm-linux-g++

DEBUGDOUBLE_O_FILES = $(patsubst %.c, %_debug_double.o, $(C_FILES))
DOUBLE_O_FILES = $(patsubst %.c, %_double.o, $(C_FILES))
DEBUGFLOAT_O_FILES = $(patsubst %.c, %_debug_float.o, $(C_FILES))
FLOAT_O_FILES = $(patsubst %.c, %_float.o, $(C_FILES))
DEBUGFIXED_O_FILES = $(patsubst %.c, %_debug_fixed.o, $(C_FILES))
FIXED_O_FILES = $(patsubst %.c, %_fixed.o, $(C_FILES))
ARM_FIXED_O_FILES = $(patsubst %.c, %_arm_fixed.o, $(C_FILES))
ARM_FLOAT_O_FILES = $(patsubst %.c, %_arm_float.o, $(C_FILES))
ARM_DOUBLE_O_FILES = $(patsubst %.c, %_arm_double.o, $(C_FILES))

O_FILES = $(FIXED_O_FILES) $(DEBUGFIXED_O_FILES) $(FLOAT_O_FILES) $(DEBUGFLOAT_O_FILES) $(DOUBLE_O_FILES) $(DEBUGDOUBLE_O_FILES)

LIBRARIES = libfloatfann.a libdoublefann.a libfixedfann.a libdebugfloatfann.a libdebugdoublefann.a libdebugfixedfann.a

TRAIN_C_FILES = $(wildcard test/*_train.c)
TEST_C_FILES = $(wildcard test/*_test.c)
JNEURAL_CC_FILES = $(wildcard test/*_jneural.cc)
LWNN_C_FILES = $(wildcard test/*_lwnn.c)

TEST_RUN_FILES = $(patsubst %_train.c, %_train_float, $(TRAIN_C_FILES)) \
		 $(patsubst %_train.c, %_train_double, $(TRAIN_C_FILES)) \
		 $(patsubst %_test.c, %_test_float, $(TEST_C_FILES)) \
		 $(patsubst %_test.c, %_test_double, $(TEST_C_FILES)) \
		 $(patsubst %_test.c, %_test_fixed, $(TEST_C_FILES)) \
		 $(patsubst %_jneural.cc, %_jneural, $(JNEURAL_CC_FILES)) \
		 $(patsubst %_lwnn.c, %_lwnn, $(LWNN_C_FILES))

BENCH_FILES =	 test/quality test/performance test/performance_fixed \
		 test/quality_fixed

ARM_FILES = $(ARM_FIXED_O_FILES) $(ARM_FLOAT_O_FILES) $(ARM_DOUBLE_O_FILES) \
	    $(patsubst %_test.c, %_test_arm_float, $(TEST_C_FILES)) \
	    $(patsubst %_test.c, %_test_arm_double, $(TEST_C_FILES)) \
	    $(patsubst %_test.c, %_test_arm_fixed, $(TEST_C_FILES)) \
	    test/performance_arm test/performance_fixed_arm
#	    $(patsubst %_jneural.cc, %_jneural_arm, $(JNEURAL_CC_FILES)) \

ARM_LIBRARIES = libarmfixedfann.a libarmfloatfann.a libarmdoublefann.a

all: $(LIBRARIES)

arm: $(ARM_FILES)

bench: $(BENCH_FILES)

%_debug_float.o: %.c $(H_FILES) Makefile
	$(GCC) -c $(CFLAGS) -ggdb -DDEBUG -include include/floatfann.h $< -o $@

%_float.o: %.c $(H_FILES) Makefile
	$(GCC) -c $(CFLAGS) -include include/floatfann.h $< -o $@

%_debug_double.o: %.c $(H_FILES) Makefile
	$(GCC) -c $(CFLAGS) -ggdb -DDEBUG -include include/doublefann.h $< -o $@

%_double.o: %.c $(H_FILES) Makefile
	$(GCC) -c $(CFLAGS) -include include/doublefann.h $< -o $@

%_debug_fixed.o: %.c $(H_FILES) Makefile
	$(GCC) -c $(CFLAGS) -ggdb -DDEBUG -include include/fixedfann.h $< -o $@

%_fixed.o: %.c $(H_FILES) Makefile
	$(GCC) -c $(CFLAGS) -include include/fixedfann.h $< -o $@

%_arm_float.o: %.c $(H_FILES) Makefile
	$(ARM-GCC) -c $(ARM-CFLAGS) -include include/floatfann.h $< -o $@

%_arm_fixed.o: %.c $(H_FILES) Makefile
	$(ARM-GCC) -c $(ARM-CFLAGS) -include include/fixedfann.h $< -o $@

%_arm_double.o: %.c $(H_FILES) Makefile
	$(ARM-GCC) -c $(ARM-CFLAGS) -include include/doublefann.h $< -o $@

libfloatfann.a: $(FLOAT_O_FILES)
	ar src libfloatfann.a $(FLOAT_O_FILES)

libdebugfloatfann.a: $(DEBUGFLOAT_O_FILES)
	ar src libdebugfloatfann.a $(DEBUGFLOAT_O_FILES)

libdoublefann.a: $(DOUBLE_O_FILES)
	ar src libdoublefann.a $(DOUBLE_O_FILES)

libdebugdoublefann.a: $(DEBUGDOUBLE_O_FILES)
	ar src libdebugdoublefann.a $(DEBUGDOUBLE_O_FILES)

libfixedfann.a: $(FIXED_O_FILES)
	ar src libfixedfann.a $(FIXED_O_FILES)

libdebugfixedfann.a: $(DEBUGFIXED_O_FILES)
	ar src libdebugfixedfann.a $(DEBUGFIXED_O_FILES)

libarmfixedfann.a: $(ARM_FIXED_O_FILES)
	arm-linux-ar src libarmfixedfann.a $(ARM_FIXED_O_FILES)

libarmfloatfann.a: $(ARM_FLOAT_O_FILES)
	arm-linux-ar src libarmfloatfann.a $(ARM_FLOAT_O_FILES)

libarmdoublefann.a: $(ARM_DOUBLE_O_FILES)
	arm-linux-ar src libarmdoublefann.a $(ARM_DOUBLE_O_FILES)

test/quality: test/quality.cc libfloatfann.a libdebugfloatfann.a
	$(G++) $(CFLAGS) $(LFLAGS) -I../libraries/jneural/include/ -I../libraries/lwneuralnet-0.6/source/ -I. -L. -L../libraries/lwneuralnet-0.6/source/ -L../libraries/jneural/  -include include/floatfann.h $< -o $@ -lfloatfann -llwneuralnet -ljneural -lfl

test/quality_fixed: test/quality_fixed.c libfixedfann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I. -L. $< -o $@ -lfixedfann

test/performance: test/performance.cc libfloatfann.a
	$(G++) $(CFLAGS) $(LFLAGS) -I../libraries/jneural/include/ -I../libraries/lwneuralnet-0.6/source/ -I. -L. -L../libraries/lwneuralnet-0.6/source/ -L../libraries/jneural/  -include include/floatfann.h $< -o $@ -lfloatfann -llwneuralnet -ljneural -lfl

test/performance_arm: test/performance.cc libfloatfann.a
	$(ARM-G++) $(ARM-CFLAGS) $(LFLAGS) -I../libraries/jneural-arm/include/ -I../libraries/lwneuralnet-0.6-arm/source/ -I. -L. -L../libraries/lwneuralnet-0.6-arm/source/ -L../libraries/jneural-arm/  -include include/floatfann.h $< -o $@ -larmfloatfann -llwneuralnet -ljneural -lfl

test/performance_fixed: test/performance.cc libfixedfann.a
	$(G++) $(CFLAGS) $(LFLAGS) -I. -L. -include include/fixedfann.h $< -o $@ -lfixedfann

test/performance_fixed_arm: test/performance.cc libarmfixedfann.a
	$(ARM-G++) $(ARMCFLAGS) $(LFLAGS) -I. -L. -include include/fixedfann.h $< -o $@ -larmfixedfann

test/%_train_float: test/%_train.c libfloatfann.a libdebugfloatfann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I. -L. -include include/floatfann.h $< -o $@ -lfloatfann

test/%_train_double: test/%_train.c libdoublefann.a libdebugdoublefann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I. -L. -include include/doublefann.h $< -o $@ -ldebugdoublefann

test/%_test_fixed: test/%_test.c libfixedfann.a libdebugfixedfann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I. -L. -include include/fixedfann.h $< -o $@ -ldebugfixedfann

test/%_test_float: test/%_test.c libfloatfann.a libdebugfloatfann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I. -L. -include include/floatfann.h $< -o $@ -ldebugfloatfann

test/%_test_double: test/%_test.c libdoublefann.a libdebugdoublefann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I. -L. -include include/doublefann.h $< -o $@ -ldebugdoublefann

test/%_test_arm_fixed: test/%_test.c libarmfixedfann.a libarmfixedfann.a
	$(ARM-GCC) $(ARM-CFLAGS) $(LFLAGS) -I. -L. -include include/fixedfann.h $< -o $@ -larmfixedfann

test/%_test_arm_float: test/%_test.c libarmfloatfann.a libarmfloatfann.a
	$(ARM-GCC) $(ARM-CFLAGS) $(LFLAGS) -I. -L. -include include/floatfann.h $< -o $@ -larmfloatfann

test/%_test_arm_double: test/%_test.c libarmdoublefann.a libarmdoublefann.a
	$(ARM-GCC) $(ARM-CFLAGS) $(LFLAGS) -I. -L. -include include/doublefann.h $< -o $@ -larmdoublefann

test/%_jneural: test/%_jneural.cc libfloatfann.a
	$(G++) $(CFLAGS) -I../libraries/jneural/include/ -I. -L. -L../libraries/jneural/ $< -o $@ -ljneural -lfloatfann -lfl

test/%_jneural_arm: test/%_jneural.cc libarmfloatfann.a
	$(ARM-G++) $(CFLAGS) -I../libraries/jneural/include/ -I. -L. -L../libraries/jneural-arm/ $< -o $@ -ljneural -larmfloatfann -lfl

test/%_lwnn: test/%_lwnn.c libfloatfann.a
	$(GCC) $(CFLAGS) $(LFLAGS) -I../libraries/lwneuralnet-0.3/source/ -I. -L. -L../libraries/lwneuralnet-0.3/source/ $< -o $@ -lnetwork -lfloatfann

test/train_robot: test/train_robot.cc libfloatfann.a libdebugfloatfann.a
	$(G++) $(CFLAGS) $(LFLAGS) -I../libraries/jneural/include/ -I. -L. -L../libraries/jneural/  $< -o $@ -lfloatfann -ljneural -lfl

test: $(TEST_RUN_FILES) Makefile

runtest: test
	@echo Training network
	./test/xor_train_float
	@echo Testing network with floats
	./test/xor_test_float
	@echo Testing network with fixed points
	./test/xor_test_fixed

clean:
	rm -rf $(TEST_RUN_FILES) $(ARM_FILES) $(O_FILES) $(LIBRARIES) $(ARM_LIBRARIES) *~ test/*~ include/*~ test/*.net test/*fixed*data test/*float*data
