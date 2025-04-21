from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('cython_exts/cython_extensions.pyx', compiler_directives={"language_level": "3"})
)