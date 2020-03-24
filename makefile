.PHONY:build
build:
	-rm trees/c/bucket_stats.cpython-38-darwin.so
	cd trees/c && python setup.py build_ext --inplace
	python -m trees.c.test_bucket_stats

run:
	python -m pytest
	python -m trees.benchmarks