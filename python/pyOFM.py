#!/usr/bin/python
"""
OpenFoam Mesh Reader

The OpenFoam Mesh module is used for interacting with an
OpenFoam Mesh mesh - typically used in a 3D CFD program.

"""
# =============================================================================
# Imports
# =============================================================================
import os
import shutil
import numpy as np
from mpi4py import MPI
from .pyOFMesh import pyOFMesh
try:
    from collections import OrderedDict
except ImportError:
    try:
        from ordereddict import OrderedDict
    except ImportError:
        print('Could not find any OrderedDict class. For 2.6 and earlier, \
use:\n pip install ordereddict')

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """
    def __init__(self, message):
        msg = '\n+'+'-'*78+'+'+'\n' + '| OFMesh Error: '
        i = 22
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)
        Exception.__init__(self)
    

class PYOFM(object):
    """
    Create an instance of pyOFM to work with.

    Parameters
    ----------

    comm : mpi4py communicator
        An optional argument to pass in an external communicator.
        Note that this option has not been tested with openfoam.

    """

    def __init__(self, comm=None):

        if comm==None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        
        self.parRun=False
        parArg = ''
        if self.comm.size > 1:
            parArg = ' -parallel'
    
        solverName='pyOFMesh -python'+parArg

        self.ofMesh=pyOFMesh(solverName.encode())

        self.ofMesh.readMesh()

        self.nLocalPoints = self.ofMesh.getNLocalPoints()

        self.nLocalFaces = self.ofMesh.getNLocalFaces()

        self.nLocalInternalFaces = self.ofMesh.getNLocalInternalFaces()

        self.nLocalBoundaryPatches = self.ofMesh.getNLocalBoundaryPatches()

        dirName = os.getcwd()

        self.fileNames = self.getFileNames(dirName,self.comm)
    
    def getFileNames(self, caseDir, comm=None):
        """
        Generate the standard set of filename for an openFoam grid.
    
        Parameters
        ----------
        caseDir : str
            The folder where the files are stored.
    
        parallel : bool
            A logical describing whether or not this case is parallel
        """
        fileNames = {}

        self.checkMeshCompression()
    
        postfix=''
        if self.writeCompression():
            postfix='.gz'
        
        if self.comm.size > 1:
            fileNames['refPointsFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/points_orig%s'%(self.comm.rank,postfix))
            fileNames['pointsFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/points%s'%(self.comm.rank,postfix))
            fileNames['pointsFile0'] = os.path.join(caseDir, 'processor%d/0/polyMesh/points%s'%(self.comm.rank,postfix))
            fileNames['pointsFile0Dir'] = os.path.join(caseDir, 'processor%d/0/polyMesh'%(self.comm.rank))
            fileNames['boundaryFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/boundary'%self.comm.rank)
            fileNames['faceFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/faces%s'%(self.comm.rank,postfix))
            fileNames['ownerFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/owner%s'%(self.comm.rank,postfix))
            fileNames['neighbourFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/neighbour%s'%(self.comm.rank,postfix))
            fileNames['varBaseDir'] = os.path.join(caseDir,'processor%d'%self.comm.rank)
        else:
            fileNames['refPointsFile'] = os.path.join(caseDir, 'constant/polyMesh/points_orig%s'%postfix)
            fileNames['pointsFile'] = os.path.join(caseDir, 'constant/polyMesh/points%s'%postfix)
            fileNames['pointsFile0'] = os.path.join(caseDir, '0/polyMesh/points%s'%postfix)
            fileNames['pointsFile0Dir'] = os.path.join(caseDir, '0/polyMesh/')
            fileNames['boundaryFile'] = os.path.join(caseDir, 'constant/polyMesh/boundary')
            fileNames['faceFile'] = os.path.join(caseDir, 'constant/polyMesh/faces%s'%postfix)
            fileNames['ownerFile'] = os.path.join(caseDir, 'constant/polyMesh/owner%s'%postfix)
            fileNames['neighbourFile'] = os.path.join(caseDir, 'constant/polyMesh/neighbour%s'%postfix)
            fileNames['varBaseDir'] = caseDir
    
        return fileNames
    
    def readVolumeMeshPoints(self):
        """
        Return a numpy array of the mesh points

        Parameters
        ----------
        points : array, dimension NPoints * 3
            x, y, and z coordinates for all points in constant/polyMesh/points
        """

        points = np.zeros((self.nLocalPoints, 3),dtype='d')
    
        for pointI, tmp in enumerate(points):
            for comp in range(3):
                points[pointI][comp] = self.ofMesh.getMeshPointCoord(pointI,comp)
        
        return points
    
    def readFaceInfo(self):
        """
        Read the face info for this case.

        Parameters
        ----------
        faces : array, dimension NFaces * NPointsPerFace
            Point indices for all faces in constant/polyMesh/points
        """

        faces = [[] for k in range(self.nLocalFaces)]

        for faceI, _ in enumerate(faces):
            nFacePoints = self.ofMesh.getNFacePoints(faceI)
            for pointI in range(nFacePoints):
                val = self.ofMesh.getMeshFacePointIndex(faceI,pointI)
                faces[faceI].append(val)

        return faces

    def readBoundaryInfo(self,faceData):
        """
        read the boundary file information for this case and store in a dict.

        Parameters
        ----------
        boundaries : dict
            Dictionary contains all the boundary information in constant/polyMesh/boundary
        """

        boundaries = {}

        for patchI in range(self.nLocalBoundaryPatches):
            patchName = self.ofMesh.getLocalBoundaryName(patchI)
            patchName = patchName.decode()

            boundaries[patchName] = {}
            pathType = self.ofMesh.getLocalBoundaryType(patchI)
            pathType = pathType.decode()
            boundaries[patchName]['type']=pathType

            startFace = self.ofMesh.getLocalBoundaryStartFace(patchI)
            nFaces = self.ofMesh.getLocalBoundaryNFaces(patchI)
            faces = np.arange(startFace, startFace+nFaces)
            nodes = []
            for face in faces:
                nodes.extend(faceData[face])
            
            boundaries[patchName]['faces']=faces
            boundaries[patchName]['nodes']=nodes

        return boundaries

    def readCellInfo(self):
        """
        Read the boundary file information for this case and store in the
        OFData dictionary.

        Parameters
        ----------
        owner, neighbour : array, dimension NFaces*1
            owner and neighbour of a face in constant/polyMesh/owner and neighbour
        """

        owner = np.zeros(self.nLocalFaces, 'intc')
        neighbour = np.zeros(self.nLocalInternalFaces, 'intc')

        # internal face owner
        for faceI in range(self.nLocalInternalFaces):
            owner[faceI] = self.ofMesh.getLocalFaceOwner(faceI)
        
        # boundary face owner
        counterI = 0
        for patchI in range(self.nLocalBoundaryPatches):
            nFaces = self.ofMesh.getLocalBoundaryNFaces(patchI)
            for faceI in range(nFaces):
                totalFaceI = self.nLocalInternalFaces + counterI
                owner[totalFaceI] = self.ofMesh.getLocalBoundaryFaceOwner(patchI,faceI)
                counterI+=1
        
        for faceI in range(self.nLocalInternalFaces):
            neighbour[faceI] = self.ofMesh.getLocalFaceNeighbour(faceI)
    
        return owner, neighbour
    
    def writeVolumeMeshPoints(self,points):
        """
        Write the most recent points to a file
        NOTE: this will write mesh to the solution 
        folder, instead of constant/polyMesh/points
        """

        points = points.reshape((int(len(points)/3), 3))

        for pointI, _ in enumerate(points):
            for compI in range(3):
                value = points[pointI][compI]
                self.ofMesh.setMeshPointCoord(pointI, compI, value)
        
        self.ofMesh.updateMesh()
        self.ofMesh.writeMesh()

        # NOTE: the above writeMesh() call will write mesh to 0/polyMesh/points
        # we need to mv the points to constant/polyMesh/points
        fileOrig = self.fileNames['pointsFile0']
        fileNew = self.fileNames['pointsFile']
        fileOrigDir = self.fileNames['pointsFile0Dir']
        try:
            shutil.move(fileOrig, fileNew)
        except:
            raise Error('Can not move %s to %s'%(fileOrig, fileNew))
        
        try:
            shutil.rmtree(fileOrigDir) 
        except:
            raise Error('Can not remove %s'%(fileOrigDir))

        self.comm.Barrier()

    
    def checkMeshCompression(self):
    
        # check if mesh files are properly compressed. This can happen if writecompress is on, but the mesh files
        # have not been compressed. In this case, we will manually compress all the mesh files (except for boundary),
        # and delete the uncompressed ones.
        if self.writeCompression():
            if self.comm.rank==0:
                print("Checking if we need to compress the mesh files")
                meshFileDir='constant/polyMesh'
                for file1 in os.listdir(meshFileDir):
                    if (not '.gz' in file1) and (not file1 == 'boundary'):
                        compFileName = os.path.join(meshFileDir,file1)
                        if os.path.isfile(compFileName):
                            with open(compFileName, 'rb') as f_in, gzip.open(compFileName+'.gz', 'wb') as f_out:
                                shutil.copyfileobj(f_in,f_out)
                            os.remove(compFileName)
        self.comm.Barrier()

    def writeCompression(self):
        # check whether to use compressed file names 
        writecompress=False
        workingDir = os.getcwd()
        controlDictPath = os.path.join(workingDir,'system/controlDict')
        fControl=open(controlDictPath,'r')
        lines=fControl.readlines()
        fControl.close()
        for line in lines:
            if 'writeCompression' in line:
                cols=line.split()
                if 'on' in cols[1]:
                    writecompress=True
        return writecompress
