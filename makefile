COMPILED_C_FILE := trees/c/tree.cpython-38-darwin.so

build:
	-rm $(COMPILED_C_FILE) # TODO make properly
	python setup.py build_ext --build-lib trees/c --build-temp trees/c

test: build
	python -m pytest

run: test
	python -m trees.benchmarks

disassemble:
	objdump -S --disassemble $(COMPILED_C_FILE) | subl