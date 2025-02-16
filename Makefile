default:
	(cd src && ./Allmake && cd -)

clean:
	(cd src && ./Allclean && cd -)
	(rm -rf */__pycache__ */*/__pycache__)
