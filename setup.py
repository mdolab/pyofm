import os
import sys
import numpy
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import re

# Try to import Cython
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# Get version from __init__.py
def get_version():
    with open(os.path.join('pyofm', '__init__.py'), 'r') as f:
        content = f.read()
    version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Get long description from README
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Custom build_ext command to handle OpenFOAM dependencies
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Check if OpenFOAM environment is available
        foam_src = os.getenv("FOAM_SRC")
        foam_libbin = os.getenv("FOAM_LIBBIN")
        
        if foam_src and foam_libbin:
            # Set up MPI compilers only if OpenFOAM is available
            os.environ["CC"] = os.getenv("CC", "mpicc")
            os.environ["CXX"] = os.getenv("CXX", "mpicxx")
        
        super().build_extensions()

# Build extensions only if building and OpenFOAM is available
ext_modules = []

def build_cython_extension():
    """Build Cython extension automatically during pip install"""
    foam_src = os.getenv("FOAM_SRC")
    foam_libbin = os.getenv("FOAM_LIBBIN")
    
    if not foam_src or not foam_libbin:
        print("OpenFOAM environment not detected - building pure Python package")
        return False
    
    if not CYTHON_AVAILABLE:
        print("Cython not available - building pure Python package")
        return False
    
    print(f"🔨 Auto-building Cython extension during pip install...")
    print(f"FOAM_SRC: {foam_src}")
    
    try:
        import subprocess
        import glob
        import shutil
        
        # Get current directory info for debugging
        build_dir = os.getcwd()
        print(f"🔍 Build directory: {build_dir}")
        print(f"🔍 Directory contents: {os.listdir('.')}")
        if os.path.exists('src'):
            print(f"🔍 src/ contents: {os.listdir('src')}")
        
        # Always clean first to ensure fresh build
        print("🧹 Cleaning previous build artifacts...")
        clean_result = subprocess.run(["make", "clean"], cwd=".", capture_output=True, text=True)
        print(f"🧹 Clean exit code: {clean_result.returncode}")
        if clean_result.stdout:
            print(f"🧹 Clean stdout: {clean_result.stdout}")
        if clean_result.stderr:
            print(f"🧹 Clean stderr: {clean_result.stderr}")
        
        # Remove any existing .so files to ensure fresh compilation
        old_so_files = glob.glob("src/*.so") + glob.glob("pyofm/*.so")
        for old_so in old_so_files:
            os.remove(old_so)
            print(f"🗑️ Removed old extension: {old_so}")
        
        # Check if we have Makefile or need to use src/Allmake directly
        compile_env = os.environ.copy()
        compile_env["PYTHONPATH"] = ""  # Clear to avoid conflicts
        
        print(f"🔍 FOAM_SRC: {compile_env.get('FOAM_SRC', 'NOT_SET')}")
        print(f"🔍 FOAM_LIBBIN: {compile_env.get('FOAM_LIBBIN', 'NOT_SET')}")
        print(f"🔍 Numpy include: {numpy.get_include()}")
        
        if os.path.exists("Makefile"):
            print("🔨 Found Makefile - using 'make compile'...")
            compile_result = subprocess.run(
                ["make"],
                cwd=".",
                capture_output=True,
                text=True,
                env=compile_env
            )
        else:
            print("🔨 No Makefile found - using 'src/Allmake' directly...")
            # We're in a sdist extraction - use Allmake directly
            compile_result = subprocess.run(
                ["./Allmake"],
                cwd="src",
                capture_output=True,
                text=True,
                env=compile_env
            )
        
        print(f"🔨 Compile exit code: {compile_result.returncode}")
        print(f"🔨 Compile stdout: {compile_result.stdout}")
        if compile_result.stderr:
            print(f"🔨 Compile stderr: {compile_result.stderr}")
        
        if compile_result.returncode != 0:
            print(f"❌ Make compilation failed!")
            print("Building pure Python package instead...")
            return False
        
        # Find the compiled .so file with debugging
        so_files = glob.glob("src/pyOFMesh.cpython-*.so")
        print(f"🔍 Looking for .so files in src/: {so_files}")
        
        # Also check if any .so files exist anywhere
        all_so_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".so"):
                    all_so_files.append(os.path.join(root, file))
        print(f"🔍 All .so files found: {all_so_files}")
        
        if not so_files:
            print("❌ No .so file found after make compile")
            print("🔍 Final src/ contents:", os.listdir('src') if os.path.exists('src') else 'src not found')
            return False
        
        so_file = so_files[0]
        print(f"✓ Found compiled extension: {os.path.basename(so_file)}")
        
        # Copy to package directory
        target_dir = "pyofm"
        target_file = os.path.join(target_dir, os.path.basename(so_file))
        
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(so_file, target_file)
        print(f"✓ Copied to package: {target_file}")
        
        # Verify the copy worked
        if os.path.exists(target_file):
            file_size = os.path.getsize(target_file)
            print(f"✓ Extension successfully integrated ({file_size:,} bytes)")
            return True
        else:
            print("❌ Failed to copy extension to package directory")
            return False
        
    except Exception as e:
        print(f"❌ Error in automatic build: {e}")
        import traceback
        traceback.print_exc()
        return False

# Try to build the Cython extension if we're installing/building
if any(cmd in sys.argv for cmd in ['build', 'build_ext', 'bdist_wheel', 'install']):
    build_success = build_cython_extension()
    if build_success:
        print("✓ Cython extension built successfully using OpenFOAM build system")
    else:
        print("⚠ Building pure Python package without Cython extensions")

# Always use empty ext_modules - we handle compilation separately
ext_modules = []

# Setup configuration
setup(
    name='pyofm',
    version=get_version(),
    description="pyOFM: Python wrapper for OpenFOAM meshes",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author='MDO Lab',
    url='https://github.com/mdolab/pyofm',
    license='GPL-3.0',
    packages=['pyofm'],
    package_data={
        'pyofm': ['*.so', '*.pyx', '*.pxd']
    },
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=[
        'numpy>=1.16.4',
        'mpi4py>=3.0.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords='openfoam mesh cfd computational-fluid-dynamics',
    zip_safe=False,
)
