from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        [Extension("ntg", ["ntg.pyx"],
                   libraries=["ntg", "pgs", "npsol", "gfortran"],
                   library_dirs=['../lib']),
        
         ],
        compiler_directives={'language_level' : '3'}
    )
)
