## PyLJFluid - Classical Fluids Simulations Using Python
==================================================================

Classical fluids simulations using Python.  
Reproduces the classic Lennard-Jones fluid experiments (Verlet-style molecular dynamics).

---

# Building and Installing (macOS + Homebrew)

These instructions describe how to build and install the project locally using **Homebrew Python** on macOS.

The package contains **C extensions** and optionally **Cython**, so a working compiler toolchain is required.

---

# 1. Install system dependencies

Install Apple’s compiler tools and Homebrew Python.

```bash
xcode-select --install
brew update
brew install python@3.14
```

Create a virtual environment and activate it in the project directory.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python dependencies for building and installing the package.
```
python -m pip install --upgrade pip setuptools wheel build numpy cython matplotlib
```

Build and install the package
```
python -m pip install .
```

# License
------------------------------------------------------------------
PyLJFluid is Licensed under the permissive Apache Licensed.
See included LICENSE file.
