from distutils.core import setup, Extension

module = Extension('tree', sources=['tree.c'],
  include_dirs=[
    '/usr/local/lib',
    '/usr/local/lib/python3.8/site-packages/numpy/core/include/numpy/'],
    extra_compile_args = [
      '-fopenmp',
      '-ffast-math',
      '-Rpass=loop-vectorize',
      '-Rpass-missed=loop-vectorize',
      '-Rpass-analysis=loop-vectorize'
    ],
    extra_link_args = ['-lomp']
  )

setup(name='tree', ext_modules=[module])

