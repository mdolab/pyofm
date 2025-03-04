"""

    pyOFM  : Python interface for OpenFOAM mesh
    Version : v1.2

    Description:
        Cython setup file for wrapping OpenFOAM libraries and solvers.
        One needs to set include dirs/files and flags according to the 
        information in Make/options and Make/files in OpenFOAM libraries 
        and solvers. Then, follow the detailed instructions below. The 
        python naming convention is to add "py" before the C++ class name
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import numpy

libName = "pyOFMesh"

os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpicxx"

# These setup should reproduce calling wmake to compile OpenFOAM libraries and solvers
ext = [
    Extension(
        "pyOFMesh",
        # All source files, taken from Make/files
        sources=["pyOFMesh.pyx", "OFMesh.C"],
        # All include dirs, refer to Make/options in OpenFOAM
        include_dirs=[
            # These are from Make/options:EXE_INC
            os.getenv("FOAM_SRC") + "/meshTools/lnInclude",
            os.getenv("FOAM_SRC") + "/finiteVolume/lnInclude",
            os.getenv("FOAM_SRC") + "/surfMesh/lnInclude",
            # These are common for all OpenFOAM executives
            os.getenv("FOAM_SRC") + "/OpenFOAM/lnInclude",
            os.getenv("FOAM_SRC") + "/OSspecific/POSIX/lnInclude",
            os.getenv("FOAM_LIBBIN"),
            os.getenv("FOAM_USER_LIBBIN"),
            numpy.get_include(),
        ],
        # These are from Make/options:EXE_LIBS
        libraries=["meshTools", "finiteVolume"],
        # These are pathes of linked libraries
        library_dirs=[os.getenv("FOAM_LIBBIN"), os.getenv("FOAM_USER_LIBBIN")],
        # All other flags for OpenFOAM, users don't need to touch this
        extra_compile_args=[
            # "-DFULLDEBUG -g -O0", # this is for debugging
            "-std=c++17",
            "-m64",
            "-pthread",
            "-DOPENFOAM=2112",
            "-DDAOF_AD_TOOL_CODI"
            "-DDAOF_AD_MODE_A1S",
            #"-Dlinux64",
            #"-DWM_ARCH_OPTION=64",
            "-DWM_DP",
            "-DWM_LABEL_SIZE=32",
            "-Wall",
            "-Wextra",
            "-Wno-deprecated-copy",
            "-Wnon-virtual-dtor",
            "-Wno-unused-parameter",
            "-Wno-invalid-offsetof",
            "-O3",
            "-DNoRepository",
            "-ftemplate-depth-100",
            "-fPIC",
            "-c",
        ],
        # Extra link flags for OpenFOAM, users don't need to touch this
        extra_link_args=["-shared", "-Xlinker", "--add-needed", "-Xlinker", "--no-as-needed"],
    )
]

setup(
    name=libName,
    packages=[libName],  # this must be the same as the name above
    description="Cython wrapper for OpenFOAM",
    long_description="Cython wrapper for OpenFOAM",
    ext_modules=cythonize(ext, language_level=3),
)  # languate_level=3 means python3
