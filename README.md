pyOFM
=====

[![pyOFM](https://github.com/mdolab/pyofm/actions/workflows/reg_tests.yml/badge.svg)](https://github.com/mdolab/pyofm/actions/workflows/reg_tests.yml)

pyOFM is a Python wrapper for OpenFOAM meshes that provides a simple mesh reading class to parse OpenFOAM polyMesh directories and store mesh data in Python for user interaction.

## Features

- **Automatic Build System**: Modern `pyproject.toml`-based build system with automatic Cython extension compilation
- **Flexible Installation**: Works with or without OpenFOAM environment (falls back to pure Python)
- **OpenFOAM Integration**: Direct reading of OpenFOAM polyMesh format
- **Python 3.8+ Support**: Compatible with Python 3.8 through 3.12
- **MPI Support**: Includes MPI capabilities via mpi4py

## Requirements

- Python 3.8 or higher
- NumPy >= 1.16.4
- mpi4py >= 3.0.0
- OpenFOAM environment (optional, for Cython extensions)
- Cython >= 0.29.0 (for building from source)

## Installation

### Quick Installation (Recommended)

For most users, simply install via pip:

```bash
pip install -e .
```

The build system will automatically:
- Detect if OpenFOAM environment is available
- Build Cython extensions if OpenFOAM is present
- Fall back to pure Python package if OpenFOAM is not available


### Building from Source Distribution

When installing from a source distribution (e.g., from PyPI):

```bash
pip install pyofm
```

The package will automatically handle compilation during installation.

## Build System

The package uses a modern build system with:

- **pyproject.toml**: Modern Python packaging configuration
- **Automatic Extension Building**: Cython extensions are built automatically during `pip install`
- **Environment Detection**: Automatically detects OpenFOAM availability
- **Fallback Support**: Works as pure Python package when OpenFOAM is unavailable
- **Cross-Platform**: Supports Linux environments with proper OpenFOAM setup


## License

This project is licensed under the GNU General Public License v3.0. See [`LICENSE.md`](LICENSE.md) for details.
