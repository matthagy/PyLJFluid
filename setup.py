name = 'PyLJFluid'
version = '0.0.1'

from distutils.core import setup

try:
    import numpy as np
except ImportError:
    print name + ' require numpy'
    exit(1)

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
    )
