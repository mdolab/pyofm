
# distutils: language = c++
# distutils: sources = OFMesh.C

'''

    DAFoam  : Python interface for OpenFOAM mesh
    Version : v1.1

    Description:
        Cython wrapper functions that call OpenFOAM libraries defined
        in the *.C and *.H files. The python naming convention is to 
        add "py" before the C++ class name

'''

import numpy as np

from libcpp.string cimport string

# declear cpp functions
cdef extern from "OFMesh.H" namespace "Foam":
    cppclass OFMesh:
        OFMesh(char*) except +
        void readMesh()
        double getMeshPointCoord(int,int)
        void setMeshPointCoord(int,int,double)
        int getNLocalPoints()
        int getNLocalCells()
        int getNLocalFaces()
        int getNLocalInternalFaces()
        int getNFacePoints(int)
        int getMeshFacePointIndex(int,int)
        void writeMesh()
        void updateMesh()
        int getNLocalBoundaryPatches()
        string getLocalBoundaryName(int)
        string getLocalBoundaryType(int)
        int getLocalBoundaryStartFace(int)
        int getLocalBoundaryNFaces(int)
        int getLocalFaceOwner(int)
        int getLocalBoundaryFaceOwner(int,int)
        int getLocalFaceNeighbour(int)

# create python wrappers that call cpp functions
cdef class pyOFMesh:

    # define a class pointer for cpp functions
    cdef:
        OFMesh * _thisptr

    # initialize this class pointer with NULL
    def __cinit__(self):
        self._thisptr = NULL

    # deallocate the class pointer, and
    # make sure we don't have memory leak
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    # point the class pointer to the cpp class constructor
    def __init__(self, argsAll):
        '''
        argsAll: string that contains all the arguments
        for running OpenFOAM solvers, including
        the name of the solver.

        For example, in OpenFOAM, if we run the following:

        mpirun -np 2 simpleFoam -parallel

        Then, the corresponding call in pySimpleFoam is:

        pySimpleFoam(b"simpleFoam -parallel")
        '''
        self._thisptr = new OFMesh(argsAll)
    
    # wrap all the other memeber functions in the cpp class
    def readMesh(self):
        self._thisptr.readMesh()
    
    def getMeshPointCoord(self, pointI, compI):
        return self._thisptr.getMeshPointCoord(pointI,compI)
    
    def setMeshPointCoord(self, pointI, compI, value):
        self._thisptr.setMeshPointCoord(pointI,compI,value)

    def getNLocalPoints(self):
        return self._thisptr.getNLocalPoints()
    
    def getNLocalCells(self):
        return self._thisptr.getNLocalCells()
    
    def getNLocalFaces(self):
        return self._thisptr.getNLocalFaces()
    
    def getNLocalInternalFaces(self):
        return self._thisptr.getNLocalInternalFaces()

    def getNFacePoints(self, faceI):
        return self._thisptr.getNFacePoints(faceI)
    
    def getMeshFacePointIndex(self, faceI, pointI):
        return self._thisptr.getMeshFacePointIndex(faceI,pointI)
    
    def writeMesh(self):
        self._thisptr.writeMesh()
    
    def updateMesh(self):
        self._thisptr.updateMesh()
    
    def getNLocalBoundaryPatches(self):
        return self._thisptr.getNLocalBoundaryPatches()
    
    def getLocalBoundaryName(self, patchI):
        return self._thisptr.getLocalBoundaryName(patchI)
    
    def getLocalBoundaryType(self, patchI):
        return self._thisptr.getLocalBoundaryType(patchI)
    
    def getLocalBoundaryStartFace(self, patchI):
        return self._thisptr.getLocalBoundaryStartFace(patchI)
    
    def getLocalBoundaryNFaces(self, patchI):
        return self._thisptr.getLocalBoundaryNFaces(patchI)

    def getLocalFaceOwner(self, faceI):
        return self._thisptr.getLocalFaceOwner(faceI)
    
    def getLocalBoundaryFaceOwner(self, patchI, faceI):
        return self._thisptr.getLocalBoundaryFaceOwner(patchI,faceI)
    
    def getLocalFaceNeighbour(self, faceI):
        return self._thisptr.getLocalFaceNeighbour(faceI)