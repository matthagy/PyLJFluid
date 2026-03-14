#  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

name = "PyLJFluid"
version = "0.0.2"

try:
    import numpy as np
except ImportError:
    print(f"{name} requires numpy", file=sys.stderr)
    raise SystemExit(1)

try:
    from Cython.Compiler.Errors import PyrexError
    from Cython.Compiler.Main import (
        CompilationOptions as CythonCompilationOptions,
        compile as cython_compile,
        default_options as cython_default_options,
    )

    have_cython = True
except ImportError:
    have_cython = False

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


def msg(s: str, *args: object) -> None:
    sys.stderr.write(((s % args) if args else s) + "\n")
    sys.stderr.flush()


def warn_msg(s: str, *args: object) -> None:
    msg(s, *args)


def error_exit() -> None:
    raise SystemExit(1)


def error_msg(s: str, *args: object) -> None:
    msg(s, *args)
    error_exit()


def ensure_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        error_msg("missing path %s", path)
    return path


def run_cython(cython_file: Path, c_file: Path) -> None:
    assert have_cython
    msg("Cythonizing %s -> %s", cython_file, c_file)
    options = CythonCompilationOptions(cython_default_options)
    options.output_file = str(c_file)
    try:
        result = cython_compile([str(cython_file)], options)
    except (EnvironmentError, PyrexError) as e:
        error_msg(str(e))
    else:
        if result.num_errors > 0:
            error_exit()


def prepare_extra_c_file(info: dict) -> str:
    c_file = info["filename"]
    compile_args = info.get("compile_args", [])

    cc = new_compiler(verbose=3)
    customize_compiler(cc)

    objects = cc.compile(
        [c_file],
        output_dir=".",
        extra_postargs=compile_args,
    )
    [o_file] = objects
    return o_file


def cython_extension(module_name: str, extra_c_files: list[dict] | None = None) -> Extension:
    if extra_c_files is None:
        extra_c_files = []

    base_path = Path(*module_name.split("."))
    cython_file = ensure_file(base_path.with_suffix(".pyx"))
    cython_def_file = ensure_file(base_path.with_suffix(".pxd"))
    c_file = base_path.with_suffix(".c")

    source_mtime = max(
        path.stat().st_mtime
        for path in (cython_file, cython_def_file)
        if path.exists()
    )

    if (not c_file.exists()) or (c_file.stat().st_mtime < source_mtime):
        if have_cython:
            run_cython(cython_file, c_file)
        else:
            ensure_file(c_file)
            warn_msg(
                "%s stale: %s has been updated and Cython is not available",
                c_file,
                cython_file,
            )

    return Extension(
        name=module_name,
        sources=[str(c_file)],
        extra_objects=[prepare_extra_c_file(info) for info in extra_c_files],
        include_dirs=[np.get_include()],
    )


extensions = [
    cython_extension("pyljfluid.util"),
    cython_extension(
        "pyljfluid.base_components",
        extra_c_files=[
            {
                "filename": "pyljfluid/lj_forces.c",
                "compile_args": ["-std=c99", "-O3"],
            }
        ],
    ),
]

setup(
    name=name,
    version=version,
    url="https://github.com/matthagy/PyLJFluid",
    author="Matt Hagy",
    author_email="matthew.hagy@gmail.com",
    description="Classical fluids simulations using Python",
    long_description=(
        'Perform the famous computer "experiments" on classical fluids by '
        'Verlet et al. using Python.'
    ),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Utilities",
    ],
    packages=["pyljfluid"],
    ext_modules=extensions,
)
