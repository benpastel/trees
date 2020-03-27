from distutils.core import setup, Extension

module = Extension('split', sources=['split.c'],
  include_dirs=[
    '/usr/local/lib',
    '/usr/local/lib/python3.8/site-packages/numpy/core/include/numpy/'],
    extra_compile_args = ['-fopenmp'],
    extra_link_args = ['-lomp']
  )

setup(name='split', ext_modules=[module])

