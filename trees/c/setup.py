from distutils.core import setup, Extension

module = Extension('bucket_stats', sources=['bucket_stats.c'],
  include_dirs=[
    '/usr/local/lib',
    '/usr/local/lib/python3.8/site-packages/numpy/core/include/numpy/'])

setup(name='bucket_stats', ext_modules=[module])

