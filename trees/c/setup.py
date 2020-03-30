from distutils.core import setup, Extension

module = Extension('build_tree', sources=['build_tree.c'],
  include_dirs=[
    '/usr/local/lib',
    '/usr/local/lib/python3.8/site-packages/numpy/core/include/numpy/'],
    extra_compile_args = ['-fopenmp'],
    extra_link_args = ['-lomp']
  )

setup(name='build_tree', ext_modules=[module])

