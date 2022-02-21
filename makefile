COMPILED_BFS := trees/c/bfs_tree.cpython-38-darwin.so
COMPILED_DFS := trees/c/dfs_tree.cpython-38-darwin.so
TREE_COUNT ?= 10

.PHONY: lint
lint:
	mypy trees

.PHONY: build
build: lint
	-rm $(COMPILED_BFS) # TODO make properly
	-rm $(COMPILED_DFS)
	python setup.py build_ext --build-lib trees/c --build-temp trees/c

.PHONY: test
test: build
	python -m pytest

.PHONY: run
run: build test
	TREE_COUNT=$(TREE_COUNT) python -m trees.benchmarks

.PHONY: disassemble
disassemble:
	objdump -S --disassemble $(COMPILED_DFS) | subl