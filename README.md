# trees

## Summary
A Gradient Boosting Machines implementation with a couple of experimental tricks.

The GBM implementation is super bare-bones, only providing `fit()` and `predict()` with a hard-coded MSE loss function.

There are 3 tricks related to regularization:
(1) make the split penalty a function of the root variance
(2) use Laplacian smoothing on the leaf weights
(3) re-bin frequently and semi-randomly

With better regularization, it becomes practical to use TRINARY trees.  This has a bit better accuracy XGBoost and LightGBM on most of my (extremely crude) benchmarks.

See #implementation-tricks below for more details.

## Installation
TODO:
  - make this easier
  - make the files / package installed by setup.py cleaner
  - test & document with ubuntu via Dockerfile
  - check in Agaricus and any of the smaller benchmarks that we can

Needs OpenMP to build the C code.  On my machine (Mac OS X 10.15) I got that working by switching to the LLVM clang installed by `brew install llvm`.

The benchmarks require data from various kaggle competitions.  Downloading each one requires clicking through some TOS with Kaggle.

The most useful one so far is the M5 competition from https://www.kaggle.com/c/m5-forecasting-accuracy.  Download the data files to data/m5.  It takes 2-3 minutes to train a 100-tree model, which is short enough to iterate quickly, but long enough to measure performance bottlenecks.

## Usage
TODO:
 - separate the library from the benchmarks so it's easy to import the library into a custom project

```
make run TREE_COUNT=100
```
This builds the C code, runs tests, and then runs a suite of benchmarks against XGBoost.  Smaller tree counts are faster when iterating quickly on new ideas.

## Implementation Tricks

