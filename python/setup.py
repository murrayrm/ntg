from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(
        [Extension('ntg', ['ntg.pyx'],
                   include_dirs=[numpy.get_include()],
                   libraries=['ntg', 'pgs', 'npsol', 'gfortran'],
                   library_dirs=['../lib']),
        
         ],
        compiler_directives={'language_level' : '3'}
    )
)
