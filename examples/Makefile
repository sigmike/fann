# This makefile is on purpose not made with configure, to show how to use the library
# The make file requires that the fann library is installed (see ../README)

TARGETS = xor_train xor_test xor_test_fixed simple_train steepness_train simple_test 
DEBUG_TARGETS = xor_train_debug xor_test_debug xor_test_fixed_debug

all: $(TARGETS)

%: %.c Makefile
	gcc -O3 -lm -lfann $< -o $@

%_fixed: %.c Makefile
	gcc -O3 -lm -lfixedfann -DFIXEDFANN $< -o $@

clean:
	rm -f $(TARGETS) $(DEBUG_TARGETS) xor_fixed.data xor_float.net xor_fixed.net

runtest: $(TARGETS)
	@echo
	@echo Training network
	./xor_train

	@echo
	@echo Testing network with floats
	./xor_test

	@echo
	@echo Testing network with fixed points
	./xor_test_fixed

#below this line is only for debugging the fann library

rundebugtest: $(DEBUG_TARGETS)
	@echo
	@echo Training network
	./xor_train_debug

	@echo
	@echo Testing network with floats
	./xor_test_debug

	@echo
	@echo Testing network with fixed points
	./xor_test_fixed_debug

debug: $(DEBUG_TARGETS)

%_debug: %.c Makefile
	gcc -O -ggdb -lm -DDEBUG -I../src/ -I../src/include/ ../src/floatfann.c $< -o $@

%_fixed_debug: %.c Makefile
	gcc -O -ggdb -lm -DDEBUG -DFIXEDFANN -I../src/ -I../src/include/ ../src/fixedfann.c $< -o $@

rundebug: $(TARGETS)
	@echo
	@echo Training network
	valgrind --leak-check=yes --show-reachable=yes --leak-resolution=high ./xor_train

	@echo
	@echo Testing network with floats
	valgrind --leak-check=yes --show-reachable=yes --leak-resolution=high ./xor_test

	@echo
	@echo Testing network with fixed points
	valgrind --leak-check=yes --show-reachable=yes --leak-resolution=high ./xor_test_fixed


