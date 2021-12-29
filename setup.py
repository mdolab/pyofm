from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('pyofm/__init__.py').read(),
)[0]

setup(name='pyofm',
      version=__version__,


      description="pyOFM: Python wrapper for OpenFOAM meshes",
      long_description="""
      pyOFM: Python wrapper for OpenFOAM meshes
      """,
      long_description_content_type="text/markdown",
      keywords='',
      author='',
      author_email='',
      url='https://github.com/mdolab/pyofm',
      license='GPL version 3',
      packages=[
          'pyofm',
      ],
      package_data={
          'pyofm': ['*.so']
      },
      install_requires=[
            'numpy>=1.16.4',
            'mpi4py>=3.0.0',
      ],
      classifiers=[
        "Operating System :: Linux",
        "Programming Language :: Cython, C++"]
      )

