name = 'PyLJFluid'
version = '0.0.1'


import os

from distutils.core import setup, Extension

try:
    import numpy as np
    from numpy.distutils.misc_util import get_numpy_include_dirs
except ImportError:
    print name + ' require numpy'
    exit(1)

try:
    import Cython.Build
except ImportError:
    Cython = None

def cython_extension(name):
    base_path = name.replace('.','/')
    cython_path = base_path + '.pyx'
    assert os.path.exists(cython_path)
    cpath = base_path + '.c'
    assert os.path.exists(cpath)
    return Extension(name=name,
                     sources=[cpath],
                     include_dirs=get_numpy_include_dirs())

extensions = [cython_extension('pyljfluid.util'),
              cython_extension('pyljfluid.components')]

setup(
    name=name,
    version=version,
    url='https://github.com/matthagy/PyLJFluid',
    author='Matt Hagy',
    author_email='hagy@gatech,.edu',
    description='Classical fluids simulations using Python',
    long_description='''
    Perform the famous computer "expriments" on classical fluids by
    Verlet et al. using Python.
    ''',
    classifiers = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Cython",
    'Topic :: Education',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Utilities'
    ],
    package = ['pyljfluid'],
    ext_modules = extensions
    )
