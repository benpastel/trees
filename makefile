COMPILED_C_FILE := trees/c/bfs_tree.cpython-38-darwin.so
TREE_COUNT ?= 10

.PHONY: lint
lint:
	mypy trees

.PHONY: build
build: lint
	-rm $(COMPILED_C_FILE) # TODO make properly
	python setup.py build_ext --build-lib trees/c --build-temp trees/c

.PHONY: test
test: build
	python -m pytest

.PHONY: run
run: build test
	TREE_COUNT=$(TREE_COUNT) python -m trees.benchmarks

.PHONY: disassemble
disassemble:
	objdump -S --disassemble $(COMPILED_C_FILE) | subl