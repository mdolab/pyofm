default:
	(cd src && python setup.py build_ext --inplace)
	(mv src/*.so pyofm)

clean:
	(cd src && rm -rf *.cpp *.so build)
	(rm -rf */__pycache__ */*/__pycache__)
