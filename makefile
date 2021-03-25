COMPILED_C_FILE := trees/c/tree.cpython-38-darwin.so
TREE_COUNT ?= 10

lint:
	mypy trees

build: lint
	-rm $(COMPILED_C_FILE) # TODO make properly
	python setup.py build_ext --build-lib trees/c --build-temp trees/c

test: build
	python -m pytest

run: build test
	TREE_COUNT=$(TREE_COUNT) python -m trees.benchmarks

disassemble:
	objdump -S --disassemble $(COMPILED_C_FILE) | subl