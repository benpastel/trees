COMPILED_C_FILE := trees/c/split.cpython-38-darwin.so

build:
	-rm $(COMPILED_C_FILE) # TODO make properly
	cd trees/c && python setup.py build_ext --inplace

test: build
	python -m pytest

run: test
	python -m pytest
	python -m trees.benchmarks

disassemble:
	objdump -S --disassemble $(COMPILED_C_FILE) | subl