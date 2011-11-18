name = 'PyLJFluid'
version = '0.0.1'

import sys
import os

from distutils.core import setup, Extension

try:
    import numpy as np
    from numpy.distutils.misc_util import get_numpy_include_dirs
except ImportError:
    print name + ' require numpy'
    exit(1)

try:
    from Cython.Compiler.Main import (compile as cython_compile,
                                      CompilationOptions as CythonCompilationOptions,
                                      default_options as cython_default_options)
    from Cython.Compiler.Errors import PyrexError
    have_cython = True
except ImportError,e:
    have_cython = False

os.chdir(os.path.dirname(__file__))

def msg(s, *args):
    sys.stderr.write((s % args if args else s) + '\n')
    sys.stderr.flush()

def warn_msg(*args):
    msg(*args)

def error_msg(*args):
    msg(*args)
    error_exit()

def error_exit():
    exit(1)

def ensure_file(path):
    if not os.path.isfile(path):
        error_msg('missing path %s', path)
    return path

def run_cython(cython_file, c_file):
    assert have_cython
    msg('Cythonizing %s -> %s', cython_file, c_file)
    options = CythonCompilationOptions(cython_default_options)
    options.output_file = c_file
    try:
        result = cython_compile([cython_file], options)
    except (EnvironmentError, PyrexError), e:
        error_msg(str(e))
    else:
        if result.num_errors > 0:
            error_exit()

def cython_extension(name):
    base_path = name.replace('.','/')
    cython_file = ensure_file(base_path + '.pyx')
    cython_def_file = ensure_file(base_path + '.pxd')
    c_file = base_path + '.c'

    if (not os.path.exists(c_file) or
        os.path.getmtime(c_file) <
        max(os.path.getmtime(path) for path in [cython_file, cython_def_file]
            if os.path.exists(path))):
        if have_cython:
            run_cython(cython_file, c_file)
        else:
            ensure_file(c_file)
            warn_msg('%s stale : %s has been updated and cython not available',
                     c_file, cython_file)

    return Extension(name=name,
                     sources=[c_file],
                     include_dirs=get_numpy_include_dirs())


extensions = [cython_extension('pyljfluid.util'),
              cython_extension('pyljfluid.base_components')]

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
    packages = ['pyljfluid'],
    ext_modules = extensions
    )
