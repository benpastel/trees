'''
  run with:
    python setup.py build_ext --inplace
'''
from distutils.core import setup, Extension

module = Extension('bucket_stats', sources=['bucket_stats.c'], include_dirs=['/usr/local/lib'])

setup(
  name='bucket_stats',
  version='1.0',
  description='calculate count, sum, sum_of_squares across y for each feature and value bucket',
  ext_modules = [module])

