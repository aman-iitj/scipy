
from distutils.core import setup, Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("_ni_measurement", ["_ni_measurement.pyx"], gdb_debug=True)]
)
