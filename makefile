build:
	-rm trees/c/bucket_stats.cpython-38-darwin.son # TODO make properly
	cd trees/c && python setup.py build_ext --inplace

test: build
	python -m pytest

run: test
	python -m pytest
	python -m trees.benchmarks
