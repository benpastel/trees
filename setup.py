from distutils.core import setup, Extension

bfs_module = Extension('bfs_tree', sources=['trees/c/bfs_tree.c'],
  include_dirs=[
    '/usr/local/lib',
    '/usr/local/lib/python3.11/site-packages/numpy/core/include/numpy/'],
    extra_compile_args = [
      '-fopenmp',
      '-ffast-math',
      # '-Rpass=loop-vectorize',
      # '-Rpass-missed=loop-vectorize',
      # '-Rpass-analysis=loop-vectorize',
    ],
    extra_link_args = ['-L/usr/local/lib/', '-lomp']
  )

dfs_module = Extension('dfs_tree', sources=['trees/c/dfs_tree.c'],
  include_dirs=[
    '/usr/local/lib',
    '/usr/local/lib/python3.11/site-packages/numpy/core/include/numpy/'],
    extra_compile_args = [
      '-fopenmp',
      '-ffast-math',
      # '-Rpass=loop-vectorize',
      # '-Rpass-missed=loop-vectorize',
      # '-Rpass-analysis=loop-vectorize',
    ],
    extra_link_args = ['-L/usr/local/lib/', '-lomp']
  )

setup(name='bfs_tree', ext_modules=[bfs_module, dfs_module])

