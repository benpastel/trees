COMPILED_BFS := trees/c/bfs_tree.cpython-38-darwin.so
COMPILED_DFS := trees/c/dfs_tree.cpython-38-darwin.so

.PHONY: install-deps
install-deps:
	brew install llvm # update clang

	# link to brew's open mp location - from `brew info llvm`
	rm /usr/local/lib/libomp.dylib
	ln -s /usr/local/Cellar/llvm/16.0.6/lib/libomp.dylib /usr/local/lib/libomp.dylib

	pip install -r requirements.txt

.PHONY: lint
lint:
	mypy trees

.PHONY: build
build: lint
	-rm $(COMPILED_BFS) # TODO make properly
	-rm $(COMPILED_DFS)
	python3 setup.py build_ext --build-lib trees/c --build-temp trees/c

.PHONY: test
test: build
	python3 -m pytest

TREE_COUNT ?= 10
OURS_ONLY ?= 0
M5_ONLY ?= 0

.PHONY: run
run: build test
	TREE_COUNT=$(TREE_COUNT) OURS_ONLY=$(OURS_ONLY) M5_ONLY=$(M5_ONLY) \
		python3 -m trees.benchmarks

.PHONY: disassemble
disassemble:
	objdump -S --disassemble $(COMPILED_DFS) | subl