set -euo pipefail

for NODES in 32 64 128; do
    TREE_COUNT=100 \
    NODES=$NODES \
        python -m trees.benchmarks
done
