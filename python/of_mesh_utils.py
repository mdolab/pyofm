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
import re
import shutil
import numpy as np
import math
import time
import subprocess
import gzip
from mpi4py import MPI
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

def writeCompression():
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

def checkMeshCompression():

    comm = MPI.COMM_WORLD

    # check if mesh files are properly compressed. This can happen if writecompress is on, but the mesh files
    # have not been compressed. In this case, we will manually compress all the mesh files (except for boundary),
    # and delete the uncompressed ones.
    if writeCompression():
        if comm.rank==0:
            print("Checking if we need to compress the mesh files")
            meshFileDir='constant/polyMesh'
            for file1 in os.listdir(meshFileDir):
                if (not '.gz' in file1) and (not file1 == 'boundary'):
                    compFileName = os.path.join(meshFileDir,file1)
                    if os.path.isfile(compFileName):
                        with open(compFileName, 'rb') as f_in, gzip.open(compFileName+'.gz', 'wb') as f_out:
                            shutil.copyfileobj(f_in,f_out)
                        os.remove(compFileName)
    comm.Barrier()

def getFileNames(caseDir, comm=None):
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
    if comm is None:
        comm = MPI.COMM_WORLD

    checkMeshCompression()

    postfix=''
    if writeCompression():
        postfix='.gz'
    
    if comm.size > 1:
        fileNames['refPointsFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/points_orig%s'%(comm.rank,postfix))
        fileNames['pointsFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/points%s'%(comm.rank,postfix))
        fileNames['boundaryFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/boundary'%comm.rank)
        fileNames['faceFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/faces%s'%(comm.rank,postfix))
        fileNames['ownerFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/owner%s'%(comm.rank,postfix))
        fileNames['neighbourFile'] = os.path.join(caseDir, 'processor%d/constant/polyMesh/neighbour%s'%(comm.rank,postfix))
        fileNames['varBaseDir'] = os.path.join(caseDir,'processor%d'%comm.rank)
    else:
        fileNames['refPointsFile'] = os.path.join(caseDir, 'constant/polyMesh/points_orig%s'%postfix)
        fileNames['pointsFile'] = os.path.join(caseDir, 'constant/polyMesh/points%s'%postfix)
        fileNames['boundaryFile'] = os.path.join(caseDir, 'constant/polyMesh/boundary')
        fileNames['faceFile'] = os.path.join(caseDir, 'constant/polyMesh/faces%s'%postfix)
        fileNames['ownerFile'] = os.path.join(caseDir, 'constant/polyMesh/owner%s'%postfix)
        fileNames['neighbourFile'] = os.path.join(caseDir, 'constant/polyMesh/neighbour%s'%postfix)
        fileNames['varBaseDir'] = caseDir

    return fileNames
    
# -----------------------------------
# SnappyHexMesh generation routines
# -----------------------------------

def updateVGSnappyHexMeshEdgeFile(vgType,vgList,comm=None):
    '''
    Update the edges files (*.eMesh) for VGs
    vgType: 1-Rectangular 2-Cylindrical
    vgList: list of VGs containing all the VG geometry parameters,
    i.e., x, y, z, dx, dy, dz, r
    '''
    
    if comm==None:
        comm=MPI.COMM_WORLD

    # loop over the components and generate the lists to write
    nVG = len(vgList)
    listVarNames = ['name','VGx','VGy','VGz','VGdx','VGdy','VGdz','VGr']
    vgVar = OrderedDict()

    for var in listVarNames:
        vgVar[var] = []

    for vg in vgList:
        vgVar['name'].append(vg)
        vgVar['VGx'].append(vgList[vg]['x'])
        vgVar['VGy'].append(vgList[vg]['y'])
        vgVar['VGz'].append(vgList[vg]['z'])
        if (vgType == 1):
            vgVar['VGdx'].append(vgList[vg]['dx'])
            vgVar['VGdy'].append(vgList[vg]['dy'])
        elif (vgType == 2):
            vgVar['VGr'].append(vgList[vg]['r'])
        else:
            raise Error("vgType not supported! Avaiable options are: 1-rectangular 2-cylindrical")
        vgVar['VGdz'].append(vgList[vg]['dz'])
        
    # calculate the edge segments and write them to vgs.eMesh
    px=[]
    py=[]
    pz=[]
    edgeEnd1=[]
    edgeEnd2=[]
    if vgType ==1: # rectangular vg
        for i in range(nVG):
            z1=vgVar['VGz'][i]-vgVar['VGdz'][i]/2.0
            z2=vgVar['VGz'][i]+vgVar['VGdz'][i]/2.0
            # indices for the two ends of all the rectangular VG edges
            rectIndex1=np.array([0,1,2,3,4,5,6,7,0,1,2,3])
            rectIndex2=np.array([1,2,3,0,5,6,7,4,4,5,6,7])
            for z in [z1,z2]:
                # sw
                x = vgVar['VGx'][i]-vgVar['VGdx'][i]/2.0
                y = vgVar['VGy'][i]-vgVar['VGdy'][i]/2.0
                px.append(x)
                py.append(y)
                pz.append(z)
                # se
                x = vgVar['VGx'][i]+vgVar['VGdx'][i]/2.0
                px.append(x)
                py.append(y)
                pz.append(z)
                # ne
                y = vgVar['VGy'][i]+vgVar['VGdy'][i]/2.0
                px.append(x)
                py.append(y)
                pz.append(z)
                # nw
                x = vgVar['VGx'][i]-vgVar['VGdx'][i]/2.0
                px.append(x)
                py.append(y)
                pz.append(z)
            endgeEnd1.append(int((i+1)*rectIndex1))
            endgeEnd2.append(int((i+1)*rectIndex2))

    elif vgType ==2: # cylindrical vg
        pCount = 0
        for i in range(nVG):
            z1=vgVar['VGz'][i]-vgVar['VGdz'][i]/2.0
            z2=vgVar['VGz'][i]+vgVar['VGdz'][i]/2.0
            for z in [z1,z2]:
                segCounter=1
                segMax=50
                for deg in np.linspace(0.0,math.radians(360.0),segMax):
                    x=vgVar['VGx'][i]+vgVar['VGr'][i]*math.cos(deg)
                    y=vgVar['VGy'][i]+vgVar['VGr'][i]*math.sin(deg)
                    px.append(x)
                    py.append(y)
                    pz.append(z)
                    edgeEnd1.append(pCount)
                    pCount+=1
                    if segCounter ==segMax:
                        edgeEnd2.append(pCount-segMax+1)
                    else:
                        edgeEnd2.append(pCount)
                    segCounter+=1
    
    if comm.rank==0:
        # write eMesh to file          
        workingDirectory = os.getcwd()
        fileName = os.path.join(workingDirectory,'constant/triSurface/vgs.eMesh')
        f = open(fileName,'w')
        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       featureEdgeMesh;\n')
        f.write('    location    "constant/triSurface";\n')
        f.write('    object      vgs.eMesh;\n')
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        
        nPoint = len(px)
        f.write('// points:\n')
        f.write('%d\n'%nPoint)
        f.write('(\n')
        for i in range(len(px)):
            f.write('(%f %f %f)\n'%(px[i],py[i],pz[i]))
        f.write(')\n')
        
        nEdge = len(edgeEnd1)
        f.write('// edges:\n')
        f.write('%d\n'%nEdge)
        f.write('(\n')
        for i in range(len(edgeEnd1)):
            f.write('(%d %d)\n'%(edgeEnd1[i],edgeEnd2[i]))
        f.write(')\n')
    comm.Barrier() 
    
    return
    

def updateVGSnappyHexMeshDict(vgType,vgList,comm=None):
    '''
    Update the VG geometry parameters in snappyHexMeshDict
    vgType: 1-Rectangular 2-Cylindrical
    vgList: list of VGs containing all the VG geometry parameters,
    i.e., x, y, z, dx, dy, dz, r
    '''
    
    if comm==None:
        comm=MPI.COMM_WORLD

    # loop over the components and generate the lists to write
    nVG = len(vgList)
    listVarNames = ['name','VGx','VGy','VGz','VGdx','VGdy','VGdz','VGr']
    vgVar = OrderedDict()

    for var in listVarNames:
        vgVar[var] = []

    for vg in vgList:
        vgVar['name'].append(vg)
        vgVar['VGx'].append(vgList[vg]['x'])
        vgVar['VGy'].append(vgList[vg]['y'])
        vgVar['VGz'].append(vgList[vg]['z'])
        if (vgType == 1):
            vgVar['VGdx'].append(vgList[vg]['dx'])
            vgVar['VGdy'].append(vgList[vg]['dy'])
        elif (vgType == 2):
            vgVar['VGr'].append(vgList[vg]['r'])
        else:
            raise Error("vgType not supported! Avaiable options are: 1-rectangular 2-cylindrical")
        vgVar['VGdz'].append(vgList[vg]['dz'])
        
    # write vg parameters to dict
    if comm.rank == 0:
    
        workingDirectory = os.getcwd()
        fileName = os.path.join(workingDirectory,'system/snappyHexMeshDict')
        f = open(fileName,'r')
        lines1 = f.readlines()
        f.close()
        
        # first remove all the previous vg geometry parameters 
        f = open(fileName,'w')
        geometryBlock = False
        removeVG = False
        for line in lines1:
        
            if 'geometry' in line:
                geometryBlock=True
                openBracket = 0
                closeBracket = 0
                f.write(line)
                continue

            if geometryBlock==True:
            
                if '{' in line: 
                    openBracket += 1
                    
                if '}' in line: 
                    closeBracket += 1
                    
                if openBracket == closeBracket:
                    geometryBlock=False
            
                if 'vg' in line:
                    removeVG = True
                    continue
                    
                if removeVG == True and ('}' not in line):
                    continue
                elif removeVG == True and ('}' in line):
                    removeVG = False
                    continue
                else:
                    f.write(line)
            else:
                f.write(line)
                
        f.close()
        
        # now add new vg geometry parameters
        f = open(fileName,'r')
        lines2 = f.readlines()
        f.close()
        
        f = open(fileName,'w')
        geometryBlock = False
        addVG = False
        for line in lines2:
 
            if 'geometry' in line:
                openBracket = 0
                closeBracket = 0
                geometryBlock = True
                f.write(line)
                continue

            if geometryBlock == True:

                if '{' in line: 
                    openBracket += 1
                    
                if '}' in line: 
                    closeBracket += 1
                    
                if openBracket == closeBracket:
                    addVG = True
                    geometryBlock = False
                    
                if addVG == True:
                    for i in range(nVG):
                        if vgType ==1:
                            x1 = vgVar['VGx'][i]-vgVar['VGdx'][i]/2.0
                            y1 = vgVar['VGy'][i]-vgVar['VGdy'][i]/2.0
                            z1 = vgVar['VGz'][i]-vgVar['VGdz'][i]/2.0
                            x2 = vgVar['VGx'][i]+vgVar['VGdx'][i]/2.0
                            y2 = vgVar['VGy'][i]+vgVar['VGdy'][i]/2.0
                            z2 = vgVar['VGz'][i]+vgVar['VGdz'][i]/2.0
                            f.write('        %s\n'%vgVar['name'][i])
                            f.write('        {\n')
                            f.write('            type searchableBox;\n')
                            f.write('            min (%f %f %f);\n'%(x1,y1,z1))
                            f.write('            max (%f %f %f);\n'%(x2,y2,z2))
                            f.write('        }\n')
                            if i == nVG-1:
                                addVG = False
                                f.write('    }\n')  
                        elif vgType ==2:   
                            x1 = vgVar['VGx'][i]
                            y1 = vgVar['VGy'][i]
                            z1 = vgVar['VGz'][i]-vgVar['VGdz'][i]/2.0
                            x2 = x1
                            y2 = y1
                            z2 = vgVar['VGz'][i]+vgVar['VGdz'][i]/2.0
                            radius = vgVar['VGr'][i]
                            f.write('        %s\n'%vgVar['name'][i])
                            f.write('        {\n')
                            f.write('            type searchableCylinder;\n')
                            f.write('            point1 (%f %f %f);\n'%(x1,y1,z1))
                            f.write('            point2 (%f %f %f);\n'%(x2,y2,z2))
                            f.write('            radius %f;\n'%radius)
                            f.write('        }\n')
                            if i == nVG-1:
                                addVG = False
                                f.write('    }\n')  

                else:
                    f.write(line)
   
            else:
                f.write(line)

        f.close()
    
    comm.Barrier() 
    
    return
    
def genSnappyHexMesh(mpiSpawnRun=True,comm=None):
    '''
    Generate snappyHexMesh
    '''
    
    if comm==None:
        comm=MPI.COMM_WORLD
    
    # we need to first delete the previous mesh and run blockMesh
    if comm.rank==0:
    
        print("Delete the previous mesh")
        # delete all processor folders
        for i in range(comm.size):
            if os.path.exists('processor%d'%i):
                try:
                    shutil.rmtree("processor%d"%i)
                except:
                    raise Error('pyOptFOAM: Unable to remove processor%d directory'%i)
        
        cwd = os.getcwd()
        blockMeshDictPath1 = os.path.join(cwd,'constant/polyMesh/blockMeshDict')
        blockMeshDictPath2 = os.path.join(cwd,'constant/polyMesh.orig/blockMeshDict')
        polyMeshPath1 = os.path.join(cwd,'constant/polyMesh/')
        polyMeshPath2 = os.path.join(cwd,'constant/polyMesh.orig/')

        # backup and remove polyMesh
        if os.path.exists(polyMeshPath2):
            shutil.rmtree(polyMeshPath2)
        shutil.move(polyMeshPath1,polyMeshPath2)
        os.mkdir(polyMeshPath1)
        shutil.copy(blockMeshDictPath2,blockMeshDictPath1)
        f=open('blockMeshLog','w')
        subprocess.call("blockMesh",stdout=f,stderr=subprocess.STDOUT, shell=False)
        f.close()
        
        if comm.size>1:
            # decompose domains
            f = open('decomposeParLog','w')
            try:
                subprocess.call("decomposePar",stdout=f,stderr=subprocess.STDOUT,shell=False)
            except:
                raise Error('pyOptFOAM: status %d: Unable to run decomposePar'%status)
            f.close()   
        
    logName='snappyHexMeshLog'
    
    if comm.size>1 and mpiSpawnRun==True:
    
        # Setup the indicator file to tell the python layer that snappyHexMesh
        # is finished running
        finishFile = 'snappyHexMeshFinished.txt' 
        # Remove the indicator file if it exists
        if os.path.isfile(finishFile):
            if comm.rank==0:
                try:
                    os.remove(finishFile)
                except:
                    raise Error('pyOptFOAM: Unable to remove %s'%finishFile)
                    sys.exit(0)
                # end
            # end
        # end
    
        # write the bash script to run the solver
        scriptName = 'runSnappyHexMesh.sh'
        
        writeParallelSnappyHexMeshScript(logName,scriptName)
        if comm.rank==0:
            print("Running snappyHexMesh.")
            
        # Run snappyHexMesh through a bash script
        child = comm.Spawn('sh', [scriptName],comm.size,
                                MPI.INFO_NULL, root=0)

        # Now wait for snappyHexMesh to finish. The bash script will create
        # the empty finishFile when it completes
        counter = 0
        checkTime = 1.0
        while not os.path.exists(finishFile):
            time.sleep(checkTime)
            counter+=1
        # end
        # Wait for all of the processors to get here
        comm.Barrier()
        
        if comm.rank==0:
            print("snappyHexMesh Finished!")
            
        miscSetupSnappyHexMesh()

    elif comm.size>1 and not mpiSpawnRun==False:
    
        runFile  = 'runSnappyHexMesh'
        finishFile = 'jobFinished'
        
        if comm.rank==0:
            # Remove the finish file if it exists
            if os.path.isfile(finishFile):
                try:
                    os.remove(finishFile)
                except:
                    raise Error('pyOptFOAM: Unable to remove %s'%finishFile)
        
            # Remove the log file if it exists
            if os.path.isfile(logFileName):
                try:
                    os.remove(logFileName)
                except:
                    raise Error('pyOptFOAM: Unable to remove %s'%logFileName)
        
            # touch the run file
            fTouch=open(runFile,'w')
            fTouch.close()
            
        comm.Barrier()
                                
        # check if the job finishes
        checkTime = 1.0
        while not os.path.isfile(finishFile):
            time.sleep(checkTime)
        comm.Barrier()
        
        if comm.rank==0:
            print("snappyHexMesh Finished!")
        
        miscSetupSnappyHexMesh()
        
    else: # serial cases
    
        print("Running snappyHexMesh.")
        f=open(logName,'w')
        try:
            subprocess.call(["snappyHexMesh","-overwrite"],stdout=f,stderr=subprocess.STDOUT, shell=False)
        except:
            raise Error('pyOptFOAM: Unable to run snappyHexMesh')
            exit(0)
        f.close()
        print("snappyHexMesh Finished!")
        
        # renumber the mesh to reduce bandwidth   
        print("Renumbering Mesh")  
        f=open('renumberMeshLog','w')
        subprocess.call(["renumberMesh","-overwrite"],stdout=f,stderr=subprocess.STDOUT, shell=False)
        f.close()
                    
    # snappyHexMesh generated
    return

def miscSetupSnappyHexMesh(comm=None):
    '''
    Misc setup for parallel snappyHexMesh, including reconstruct and renumber the mesh
    '''

    if comm==None:
        comm=MPI.COMM_WORLD
        
    if comm.rank==0:
        
        # reconstruct the mesh
        print("Reconstructing Mesh")
        f=open('reconstructParMeshLog','w')
        subprocess.call(["reconstructParMesh","-latestTime"],stdout=f,stderr=subprocess.STDOUT, shell=False)
        f.close()
        
        # move the reconstructed mesh to constant/polyMesh
        cwd = os.getcwd()
        blockMeshDictPath1 = os.path.join(cwd,'constant/polyMesh/blockMeshDict')
        blockMeshDictPath2 = os.path.join(cwd,'constant/polyMesh.orig/blockMeshDict')
        constantPath = os.path.join(cwd,'constant/')
        polyMeshPath = os.path.join(cwd,'constant/polyMesh/')
        newMeshPath = os.path.join(cwd,'3/polyMesh/') # NOTE: we assume the new mesh is in this folder
        shutil.rmtree(polyMeshPath)
        shutil.move(newMeshPath,constantPath)
        shutil.move(blockMeshDictPath2,blockMeshDictPath1)
        # delete processor*
        for i in range(comm.size):
            if os.path.exists('processor%d'%i):
                try:
                    shutil.rmtree("processor%d"%i)
                except:
                    raise Error('pyOptFOAM: Unable to remove processor%d directory'%i)
                    
        # renumber the mesh to reduce bandwidth   
        print("Renumbering Mesh")  
        f=open('renumberMeshLog','w')
        subprocess.call(["renumberMesh","-overwrite"],stdout=f,stderr=subprocess.STDOUT, shell=False)
        f.close()
        
        # clean up
        shutil.rmtree('constant/polyMesh.orig')
        shutil.rmtree('3')
        
    comm.Barrier()

    return
    
def writeParallelSnappyHexMeshScript(logFileName,scriptName,comm=None):

    if comm==None:
        comm=MPI.COMM_WORLD
        
    if comm.rank==0:
        f = open(scriptName,'w')
        
        f.write('#!/bin/bash \n')
        f.write('snappyHexMesh -parallel > %s\n'%logFileName)
        f.write('touch snappyHexMeshFinished.txt\n')
        f.write('# End of the script file\n')
        
        f.close()
        
    comm.Barrier()
    
    return

def isInside2DTri(P,T):
    '''
    Check if a point is inside a 2D triangle
 
    Parameters
    ----------
    P : array 1x2
        point coordinate
    T : array 3x2
        triangle corner coordinates

    Example
    -------
    P=np.array([0.5,0.6])
    T=np.array([[0.,0.],[1.,0.],[0.,1.]])
    print isInside2DTri(P,T)
    '''

    u=T[1]-T[0]
    v=T[2]-T[0]
    # cross product for area
    a=u[0]*v[1]-u[1]*v[0]
    a=abs(a)

    l=P-T[0]
    m=P-T[1]
    n=P-T[2]

    b1=l[0]*m[1]-l[1]*m[0]
    b2=l[0]*n[1]-l[1]*n[0]
    b3=m[0]*n[1]-m[1]*n[0]

    b=abs(b1)+abs(b2)+abs(b3)

    if abs(a-b)>1e-12:
        return 0
    else:
        return 1

def isInsideMesh(pointCoords,meshPtX,meshPtY,meshPtZ):
    '''
    Check if a point is inside a closed, 3D triangulated surface mesh
    Here we will use a ray intersection algorithm. We shoot a ray in the x direction
    If this ray intersect with the mesh in odd times, this point is inside the mesh
    otherwise, it is outside
    NOTE: to avoid checking every triangle surface of a mesh which will be slow,
    we can pass in a "parial" mesh where its y and z are a little larger than 
    the maxY and maxZ of the FFD box. This is because we will alway shoot a ray 
    in the +x direction, so this "partial" mesh can be considered a "closed" one
 
    Parameters
    ----------
    pointCoords : array 1x3
        point coordinate
    meshPtX : array nx3
        X coordinates for the surface mesh, the 3 columns are for the 3 corners

    Example
    -------
    P=np.array([0.1,0.5,0.6])
    meshPtX=np.array([[0.,0.1,0.2],[0.2,0.3,0.4]])
    isInside2DTri(P,meshPtX,meshPtY,meshPtZ)
    '''

    P=np.array([pointCoords[1],pointCoords[2]],dtype='d')
    nInterSec=0
    interSectedAll=[]
    for i in range(len(meshPtX)):
        T=np.array([[meshPtY[i][0],meshPtZ[i][0]],
                    [meshPtY[i][1],meshPtZ[i][1]],
                    [meshPtY[i][2],meshPtZ[i][2]]],dtype='d')
        isIn=isInside2DTri(P,T)
        if isIn:
            # also need to check if the x coordinate of this T is larger than P
            maxX=-1000000.0
            for k in range(3):
                if (meshPtX[i][k]-maxX)>0:
                    maxX=meshPtX[i][k]
            if (maxX-pointCoords[0])>0:
                nInterSec+=1
                interSec=np.array([[meshPtX[i][0],meshPtY[i][0],meshPtZ[i][0]],
                                   [meshPtX[i][1],meshPtY[i][1],meshPtZ[i][1]],
                                   [meshPtX[i][2],meshPtY[i][2],meshPtZ[i][2]]],dtype='d')
                print("Intersectd triangle:\n ",interSec)
                interSectedAll.append(interSec)

    # NOTE: in some cases, the ray can shoot right on the edge of a triangle surface
    # So this will end up getting 2 intersections, but it should still count as 1
    # Here we need to make sure all the intersected triangle surfces we collected
    # are not connected to each other. This can be done by checking if they share any commonn points
    for i in range(len(interSectedAll)):
        for j in range(len(interSectedAll)):
            if j>i:
                edgeInterSecFound=0
                for k in range(3):
                    for l in range(3):
                        TDiff=interSectedAll[i][k]-interSectedAll[j][l]
                        if abs(np.dot(TDiff,TDiff))<1e-12:
                            edgeInterSecFound=1
                if edgeInterSecFound==1:
                    print("ray shooting on the edge of a triangle!")
                    nInterSec-=1

    if nInterSec%2==0:
        return False
    else:
        return True

def readSTL(stlFileName,scaleFactor=1.0):
    """
    Read and return the coordinates and face normal from a STL file
 
    Parameters
    ----------
    stlFileName : str
        name of the stl file to convert
    scaleFactor : float
        scale factor for the stl file
    Output:
    ptX, ptY, ptZ: coordinates
    fNormalX,fNormalY,fNormalZ: face normal
    patchName: names of the patches in the stl file
    patchFaceN: number of faces for each patch
    """
    
    print ("Reading stl from: %s"%stlFileName)
    
    fIn = open(stlFileName,'r')
    lines = fIn.readlines()
    fIn.close()
    
    # get the total patch number
    patchN=0
    for line in lines:
        if ('solid' in line) and ('endsolid' not in line):
            patchN += 1
    
    # read face normal, patch names, and count face number for each patch
    fNormalX=[]
    fNormalY=[]
    fNormalZ=[]
    # store the boundary patchName and their face number, this will be used
    # in the OpenFOAM boundary file 
    patchName = []
    patchFaceN = [0]*patchN
    faceN=0
    patchI=0
    for line in lines:
    
        if ('solid' in line) and ('endsolid' not in line):
            col = line.split()
            patchName.append(col[1])
            faceN=0
        if 'endsolid' in line:
            patchFaceN[patchI]=faceN
            patchI+=1
            
        if 'facet normal' in line:
            col = line.split()
            fNormalX.append(col[2])
            fNormalY.append(col[3])
            fNormalZ.append(col[4])
            faceN += 1

    # read points
    ptX=[]
    ptY=[]
    ptZ=[]
    readLines = False
    for line in lines:
    
        if 'outer loop' in line:
            readLines = True
            idx=0
            continue
    
        if readLines == False:
            continue
        if idx<=2:
            col = line.split()
            ptX.append(col[1])
            ptY.append(col[2])
            ptZ.append(col[3])
            idx +=1
        else:
            readLines = False

    print ("Reading stl from: %s: completed!"%stlFileName)
    
    return ptX,ptY,ptZ,fNormalX,fNormalY,fNormalZ,patchName,patchFaceN
    

def stlToFoamBoundaryMesh(stlFileName,scaleFactor=1.0):
    """
    Convert a stl file to OpenFOAM surface mesh. This will generate a surafce mesh in constant/polyMesh.
    Note: only the points, faces, owner, and neighbour files will be generated. When viewing the surface
    mesh in paraview, only select the surface mesh (uncheck the volume mesh).
 
    Parameters
    ----------
    stlFileName : str
        name of the stl file to convert
    scaleFactor : float
        scale factor for the stl file
    """
    
    print ("Converting %s to OpenFOAM surface mesh"%stlFileName)
    
    ptX,ptY,ptZ,fNormalX,fNormalY,fNormalZ,patchName,patchFaceN = readSTL(stlFileName,scaleFactor)
    
    ############### write points
    fPoints = open('constant/polyMesh/points','w')
    # write the file header
    fPoints.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    fPoints.write('| =========                 |                                                 |\n')
    fPoints.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    fPoints.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    fPoints.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    fPoints.write('|    \\\\/     M anipulation  |                                                 |\n')
    fPoints.write('\*---------------------------------------------------------------------------*/\n')
    fPoints.write('FoamFile\n')
    fPoints.write('{\n')
    fPoints.write('    version     2.0;\n')
    fPoints.write('    format      ascii;\n')
    fPoints.write('    class       vectorField;\n')
    fPoints.write('    location    "constant/polyMesh";\n')
    fPoints.write('    object      points;\n')
    fPoints.write('}\n')
    fPoints.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    fPoints.write('\n')
    
    fPoints.write('%d\n'%len(ptX))
    fPoints.write('(\n')
    for i in range(len(ptX)):
        fPoints.write('(%f %f %f)\n'%(scaleFactor*float(ptX[i]),scaleFactor*float(ptY[i]),scaleFactor*float(ptZ[i])))
    fPoints.write(')\n')
    fPoints.close()
    
    ################ write faces
    fFaces = open('constant/polyMesh/faces','w')
    # write the file header
    fFaces.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    fFaces.write('| =========                 |                                                 |\n')
    fFaces.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    fFaces.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    fFaces.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    fFaces.write('|    \\\\/     M anipulation  |                                                 |\n')
    fFaces.write('\*---------------------------------------------------------------------------*/\n')
    fFaces.write('FoamFile\n')
    fFaces.write('{\n')
    fFaces.write('    version     2.0;\n')
    fFaces.write('    format      ascii;\n')
    fFaces.write('    class       faceList;\n')
    fFaces.write('    location    "constant/polyMesh";\n')
    fFaces.write('    object      faces;\n')
    fFaces.write('}\n')
    fFaces.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    fFaces.write('\n')
    
    fFaces.write('%d\n'%len(fNormalX))
    fFaces.write('(\n')
    for i in range(len(fNormalX)):
        fFaces.write('3(%d %d %d)\n'%(i*3,i*3+1,i*3+2))
    fFaces.write(')\n')
    fFaces.close()

    ################ write face normal
    fFaces = open('constant/polyMesh/normals','w')
    # write the file header
    fFaces.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    fFaces.write('| =========                 |                                                 |\n')
    fFaces.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    fFaces.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    fFaces.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    fFaces.write('|    \\\\/     M anipulation  |                                                 |\n')
    fFaces.write('\*---------------------------------------------------------------------------*/\n')
    fFaces.write('FoamFile\n')
    fFaces.write('{\n')
    fFaces.write('    version     2.0;\n')
    fFaces.write('    format      ascii;\n')
    fFaces.write('    class       faceList;\n')
    fFaces.write('    location    "constant/polyMesh";\n')
    fFaces.write('    object      normals;\n')
    fFaces.write('}\n')
    fFaces.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    fFaces.write('\n')
    
    fFaces.write('%d\n'%len(fNormalX))
    fFaces.write('(\n')
    for i in range(len(fNormalX)):
        fFaces.write('%g %g %g\n'%(float(fNormalX[i]),float(fNormalY[i]),float(fNormalZ[i])))
    fFaces.write(')\n')
    fFaces.close()

    ################ write owner
    # note we don't actually need owner information for the surface mesh, so we simply assign zeros here
    fOwner = open('constant/polyMesh/owner','w')
    # write the file header
    fOwner.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    fOwner.write('| =========                 |                                                 |\n')
    fOwner.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    fOwner.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    fOwner.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    fOwner.write('|    \\\\/     M anipulation  |                                                 |\n')
    fOwner.write('\*---------------------------------------------------------------------------*/\n')
    fOwner.write('FoamFile\n')
    fOwner.write('{\n')
    fOwner.write('    version     2.0;\n')
    fOwner.write('    format      ascii;\n')
    fOwner.write('    class       labelList;\n')
    fOwner.write('    location    "constant/polyMesh";\n')
    fOwner.write('    object      owner;\n')
    fOwner.write('}\n')
    fOwner.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    fOwner.write('\n')
    
    fOwner.write('%d\n'%len(fNormalX))
    fOwner.write('(\n')
    for i in range(len(fNormalX)):
        fOwner.write('0\n')
    fOwner.write(')\n')        
    fOwner.close()
    
    
    ################ write neighbour
    # note we don't actually need neighbour information for the surface mesh, so we simply assign zeros here
    fNeighbour = open('constant/polyMesh/neighbour','w')
    # write the file header
    fNeighbour.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    fNeighbour.write('| =========                 |                                                 |\n')
    fNeighbour.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    fNeighbour.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    fNeighbour.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    fNeighbour.write('|    \\\\/     M anipulation  |                                                 |\n')
    fNeighbour.write('\*---------------------------------------------------------------------------*/\n')
    fNeighbour.write('FoamFile\n')
    fNeighbour.write('{\n')
    fNeighbour.write('    version     2.0;\n')
    fNeighbour.write('    format      ascii;\n')
    fNeighbour.write('    class       labelList;\n')
    fNeighbour.write('    location    "constant/polyMesh";\n')
    fNeighbour.write('    object      neighbour;\n')
    fNeighbour.write('}\n')
    fNeighbour.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    fNeighbour.write('\n')

    fNeighbour.write('%d\n'%len(fNormalX))
    fNeighbour.write('(\n')
    for i in range(len(fNormalX)):
        fNeighbour.write('0\n')
    fNeighbour.write(')\n')       
    fNeighbour.close()
    
    
    ################ write boundary
    fBoundary = open('constant/polyMesh/boundary','w')
    # write the file header
    fBoundary.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    fBoundary.write('| =========                 |                                                 |\n')
    fBoundary.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    fBoundary.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    fBoundary.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    fBoundary.write('|    \\\\/     M anipulation  |                                                 |\n')
    fBoundary.write('\*---------------------------------------------------------------------------*/\n')
    fBoundary.write('FoamFile\n')
    fBoundary.write('{\n')
    fBoundary.write('    version     2.0;\n')
    fBoundary.write('    format      ascii;\n')
    fBoundary.write('    class       polyBoundaryMesh;\n')
    fBoundary.write('    location    "constant/polyMesh";\n')
    fBoundary.write('    object      boundary;\n')
    fBoundary.write('}\n')
    fBoundary.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    fBoundary.write('\n')
    
    fBoundary.write('%d\n'%len(patchName))
    fBoundary.write('(\n')
    startFace = 0
    for i in range(len(patchName)):

        fBoundary.write('    %s\n'%patchName[i])
        fBoundary.write('    {\n')
        fBoundary.write('        type       wall;\n')
        fBoundary.write('        nFaces     %d;\n'%patchFaceN[i])
        fBoundary.write('        startFace  %d;\n'%startFace)
        fBoundary.write('    }\n')
        
        startFace += patchFaceN[i]
        
    fBoundary.write(')\n')
    
    fBoundary.close()
    
    return

def reflectDVFFD(FFDFileNameIn,FFDFileNameOut,axis='y'):
    '''
    Do a reflection of the current DV FFDs wrt the specified axis. It will do something like this:
    
    Before reflection:

    |---|---|---|     | y axis
    | 1 | 2 | 3 |     |
    |---|---|---|     |-------- x axis
    
    After reflection wrt the y axis:

    |---|---|---|     | y axis
    | 1 | 2 | 3 |     | 
    |---|---|---|     |--------- x axis
    | 4 | 5 | 6 |
    |---|---|---|        
    
    Note: the numbers denote the plot3d block number. This assumes you initially have 3 blocks. After reflection,
    three new blocks are added. We basically flip the sign of the y coordinates for the new blocks. So the size 
    of the blocks will remain same, but the y data ordering will be reflected. Also note that we will have 
    duplicated FFD points after the reflection.
    
    Parameters
    ----------
    FFDFileNameIn : str
        Location of the plot3d FFD file for reflection
    FFDFileNameOut : str
        Location of the reflected plot3d FFD file
    axis: str
        Reflect the FFD points wrt to this axis
    '''
    
    if axis=='x':
        iAxis = 1
    elif axis=='y':
        iAxis = 2
    elif axis=='z':
        iAxis = 3
    else:
        raise Error("reflection axis not valid")
    
    f=open(FFDFileNameIn,'r')
    lines = f.readlines()
    f.close()
    
    # write the old lines to the new FFD file
    fOut = open(FFDFileNameOut,'w')
    xCounter = 0
    for line in lines:
        if xCounter == 0:
            fOut.write(str(int(line)*2)+'\n')
        elif xCounter == 1:
            fOut.write(line.strip()+'  '+line)
        else:
            fOut.write(line)
        xCounter +=1

    # get the total number of blocks
    nBlockInit = int(lines[0].split()[0])
    
    # write the reflection data to the new FFDs            
    xCounter = 0
    for line in lines:
        if xCounter < 2:
            pass
        elif (xCounter-1-iAxis)%3 ==0 :
            cols = line.split()
            for col in cols:
                fOut.write(str(-float(col))+'  ')
            fOut.write('\n')
        else:
            fOut.write(line)

        xCounter+=1
    fOut.close()
    
    return
    
def fixFaceIndexFormat(fileNames):
    '''
    Call this in the readFaceInfo function if the face indices in constant/polyMesh/faces 
    take multiple lines, since it will mess up the readFaceInfo function. 
    Normal face format is this:
    3(1 2 3)
    But sometimes it may become this:
    3
    (
    1
    2
    3
    )
    '''   
    
    facesOrig = fileNames['faceFile']

    if writeCompression():
        fIn = gzip.open(facesOrig, 'rb')
    else:
        fIn = open(facesOrig, 'r')

    lines = fIn.readlines()
    linesAll = b''.join(lines)
    fIn.close()
    
    # search the face file and get the data block
    faceDataBlock = re.search(r'\s*[0-9]{1,100}\s*\n*\((.*)\)\s*\n*',linesAll.decode(),re.DOTALL)
    faceDataBlockList=faceDataBlock.group(1).split()
    faceDataBlockSplitLines = faceDataBlock.group(1).splitlines()

    # check whether we need to fix the inconsistent format
    faceDataFormat = re.compile(r'\s*[0-9]{1,100}\s*\(.*\)\s*\n*')
    needFix = 0
    for i in faceDataBlockSplitLines:
        res = faceDataFormat.match(i)
        if (not res) and (not i == '\n') and (not len(i)==0):
            needFix =1
            break

    #  inconsistent format detected, do the fix
    if needFix:
    
        print('Fixing the inconsistent face index format. ProcI: %d... '%MPI.COMM_WORLD.rank)
        
        # backup the original face file
        facesCopy = fileNames['faceFile']+'_bk' 
        if not os.path.exists(facesCopy):
            try:
                print('Copying the orginal faces to faces_bk...')
                shutil.copyfile(facesOrig,facesCopy)
            except:
                raise Error('pyOptFOAM: Unable to copy %s to %s.'%(facesOrig,facesCopy))
                sys.exit(0)

        # fix the format
        newFaceDataBlock = []
        for i in faceDataBlockList:
            if ')' in i:
                newFaceDataBlock.append(i+'\n')
            else:
                newFaceDataBlock.append(i+' ')
        
        # now write the new file using the fixed format
        if writeCompression():
            fOut = gzip.open(facesOrig, 'wb')
        else:
            fOut = open(facesOrig, 'w')
        
        # the line that has the number of faces
        nFaceLine = re.compile(r'\s*([0-9]{1,100})\s*\n')
        
        dataStart = 0
        for line in lines:
        
            # check if we can start writing the new data
            res = nFaceLine.match(line.decode())
            if res and dataStart ==0:
                dataStart =1
                fOut.write(line)
                fOut.write('(\n')
            
            if dataStart:
                for i in newFaceDataBlock:
                    fOut.write(i)
                break
            else:
                fOut.write(line)
        fOut.write(')\n')

        fOut.close()
        
    MPI.COMM_WORLD.Barrier()
    return

        
# -----------------------------
# Reading routines
# -----------------------------

def readVolumeMeshPoints(fileNames):
    '''
    Return a numpy array of the mesh points
    '''

    # Open the points file for reading
    if writeCompression():
        f = gzip.open(fileNames['pointsFile'], 'rb')
    else:
        f = open(fileNames['pointsFile'], 'r')
        
    # Read Regular Header
    foamHeader, N, oneLineData = _parseFoamHeader(f)

    try:
        fmt = foamHeader['format'].lower()
    except:
        fmt = 'ascii' # assume ascii if format is missing.

    x = np.zeros((N, 3),dtype='d')
    pointLine = re.compile(r'\(\s*(\S*)\s*(\S*)\s*(\S*)\s*\)\s*\n')

    if fmt == 'binary':
        raise Error('Binary Reading is not yet supported.')
    else:
        k = 0
        j = 0
        for line in f:
            if j>=N:
                break
            res = pointLine.match(line.decode())
            if res:
                k += 1
                for idim in range(3):
                    x[j, idim] = float(line[res.start(idim+1):res.end(idim+1)])
            j+=1

        if k != N:
            raise Error('Error reading grid coordinates. Expected %d'
                        ' coordinates but found %d'%(N, k))
    f.close()

    return x

def readFaceInfo(fileNames):
    """
    Read the face info for this case.

    """
    
    # check and fix the face index format
    fixFaceIndexFormat(fileNames)

    if writeCompression():
        f = gzip.open(fileNames['faceFile'], 'rb')
    else:
        f = open(fileNames['faceFile'], 'r')

    # Read Regular Header
    foamHeader, N, oneLineData = _parseFoamHeader(f)

    try:
        fmt = foamHeader['format'].lower()
    except:
        fmt = 'ascii' # assume ascii if format is missing.

    if fmt == 'binary':
        raise Error('Binary Reading is not yet supported.')
    else:
        faces = [[] for k in range(N)]
        counter = 0
        #for j in range(N):                
        for line in f:
            line = line.replace(b'(', b' ')
            line = line.replace(b')', b' ')
            aux = line.split()
            nNode = int(aux[0])
            for k in range(1, nNode+1):
                faces[counter].append(int(aux[k]))
            counter +=1
            if counter>=N:
                break


    f.close()

    return faces

def readBoundaryInfo(fileNames,faceData):
    '''
    read the boundary file information for this case and store in a dict.
    '''

    # Open the boundary file
    f = open(fileNames['boundaryFile'], 'r')

    # Read Regular Header
    foamHeader, N, oneLineData = _parseFoamHeader(f)

    # We don't actually need to know the number of blocks...just
    # read them all:
    boundaries = {}
    keyword = re.compile(r'\s*([a-zA-Z0-9]{1,100})\s*\n')
    for line in f:
        try:
            lineD=line.decode()
        except:
            lineD=line
        res = keyword.match(lineD)
        if res:
            boundaryName = line[res.start(1):res.end(1)]
            blkData = _readBlock(f)

            # Now we can just read out the info we need:
            startFace = int(blkData['startFace'])
            nFaces = int(blkData['nFaces'])
            faces = np.arange(startFace, startFace+nFaces)
            nodes = []
            for face in faces:
                nodes.extend(faceData[face])
            boundaries[boundaryName] = {
                'faces':faces,
                'nodes':nodes,
                'type':blkData['type']}
    f.close()

    return   boundaries

def readCellInfo(fileNames):
    """Read the boundary file information for this case and store in the
    OFData dictionary."""

    # ------------- Read the owners file ---------------
    if writeCompression():
        f = gzip.open(fileNames['ownerFile'], 'rb')
    else:
        f = open(fileNames['ownerFile'], 'r')

    # Read Regular Header
    foamHeader, N, oneLineData = _parseFoamHeader(f)
    try:
        fmt = foamHeader['format'].lower()
    except:
        fmt = 'ascii' # assume ascii if format is missing.

    owner = np.zeros(N, 'intc')

    if fmt == 'binary':
        raise Error('Binary Reading is not yet supported.')
    else:    
        counter = 0
        for line in f:
            try:
                lineD=line.decode()
            except:
                lineD=line
            if ')' in lineD:
                break
            vals =  np.fromstring(line, dtype='int',sep=" ")
            #print vals
            for i in range(len(vals)):
                owner[counter]=vals[i]
                counter+=1
        # owner = np.fromfile(f, dtype='int', count=2, sep=" ")
        # print 'ownerfromfile',owner,np.fromfile(f, dtype='int', count=2, sep=" ")
        #owner = owner.astype('intc')

    f.close()

    # ------------- Read the neighbour file ---------------
    if writeCompression():
        f = gzip.open(fileNames['neighbourFile'], 'rb')
    else:
        f = open(fileNames['neighbourFile'], 'r')

    # Read Regular Header
    foamHeader, N, oneLineData = _parseFoamHeader(f)

    try:
        fmt = foamHeader['format'].lower()
    except:
        fmt = 'ascii' # assume ascii if format is missing.

    neighbour = np.zeros(N, 'intc')
    if fmt == 'binary':
        raise Error('Binary Reading is not yet supported.')
    else:
        counter = 0
        for line in f:
            try:
                lineD=line.decode()
            except:
                lineD=line
            if ')' in lineD:
                break
            vals =  np.fromstring(line, dtype='int',sep=" ")
            #print vals
            for i in range(len(vals)):
                neighbour[counter]=vals[i]
                counter+=1
        # neighbour = np.fromfile(f, dtype='int', count=N, sep=' ')
        # neighbour = owner.astype('intc')

    return owner, neighbour

def _parseFoamHeader(f, nDim=1):
    """Generic function to read the openfoam file header up to the point
    where it determines the number of subsequent 'stuff' to read

    Parameters
    ----------
    f : file handle
        file to be parsed

    Returns
    -------
    foamDict : dictionary
         Dictionary of the data contained in the header
    i : int 
         The next line to start at
    N : int
         The number of entries to read
    """
    keyword = re.compile(r'\s*[a-zA-Z]{1,100}\s*\n')

    blockOpen = False
    foamDict = {}

    for line in f:
        try:
            lineD=line.decode()
        except:
            lineD=line
        res = keyword.match(lineD)
        if res:
            headerName = line[res.start():res.end()]
            foamDict[headerName] = _readBlock(f,nDim)
            break

    # Now we need to match the number followed by an open bracket:
    # Note that numberHead and numberHeaderOneLine used to have the
    # 'ur' prefix. This is not compatible with Python3, but it appears
    # it isn't necessary anyway. 
    numberHeader = re.compile(r'\s*(\d{1,100})\s*\n')
    numberHeaderOneLine = re.compile(r'\s*(\d{1,100})\(')
    data = re.compile(r'\s*([a-zA-Z]*)\s*(.*);\s*')
    field =re.compile(r'\s*([a-zA-Z]+)\s+([a-zA-Z\s\<\>]+)\s*')
    nufield = re.compile(r'\s*([a-zA-Z]+)\s+([a-zA-Z]+)\s+([a-zA-Z\s\<\>]+)\s*')
    ufield =re.compile(r'\s*([a-zA-Z]+)\s+([a-zA-Z]+)\s+([0-9\.eE\-]+)\s*')
    ufield3 =re.compile(r'\s*([a-zA-Z]+)\s+([a-zA-Z]+)\s+\(\s*([0-9\.eE\-]+)\s+([0-9\.eE\-]+)\s+([0-9\.eE\-]+)\s*\)\s*')
    oneLineData = False
    foamDict['uniformValue']=None
    exitOnNext = False

    for line in f:
        try:
            lineD=line.decode()
        except:
            lineD=line

        if exitOnNext:
            return foamDict,  N, oneLineData
        res = numberHeader.match(lineD)
        resOneLine = numberHeaderOneLine.search(lineD)
        dataTest = data.match(lineD)
        fieldTest = nufield.match(lineD)
        uFieldTest = ufield.match(lineD)
        uFieldTest3 = ufield3.match(lineD)

        if uFieldTest:
            foamDict[uFieldTest.group(1)]=uFieldTest.group(2)+' '+uFieldTest.group(3)
            foamDict['fieldType'] = uFieldTest.group(2)
            foamDict['uniformValue']=[uFieldTest.group(3)]
            return foamDict, 0, oneLineData
        elif uFieldTest3: 
            foamDict[uFieldTest3.group(1)]=uFieldTest3.group(2)+' '+uFieldTest3.group(3)+' '+uFieldTest3.group(4)+' '+uFieldTest3.group(5)
            foamDict['fieldType'] = uFieldTest3.group(2)
            foamDict['uniformValue']=[uFieldTest3.group(3),uFieldTest3.group(4),uFieldTest3.group(5)]
            return foamDict, 0, oneLineData
        elif fieldTest:                
            foamDict[fieldTest.group(1)] = fieldTest.group(2)+' '+fieldTest.group(3)
            foamDict['fieldType'] = fieldTest.group(2)
        elif dataTest:
            foamDict[dataTest.group(1)] = dataTest.group(2)

        if res:
            # we have found what we are looking for. If there is a bracket in 
            # this line return right away. else, return after one more line read
            if resOneLine:
                N = int(line[resOneLine.start(0):resOneLine.end(0)-1])
                oneLineData=True
                return foamDict,  N, oneLineData
            else:
                N = int(line[res.start(0):res.end(0)])
                exitOnNext=True
    raise Error("Could not find the starting data in openFoam file")

def _readBlock(f, nDim=1):
    """
    Generic code to read an openFoam type block structure
    """

    openBracket = re.compile(r'\s*\{\s*\n')
    closeBracket = re.compile(r'\s*\}\s*\n')
    data = re.compile(r'\s*([a-zA-Z]*)\s*(.*);\s*\n')

    blockOpen = False
    blk = {}
    for line in f:
        try:
            lineD=line.decode()
        except:
            lineD=line

        if not blockOpen:
            res = openBracket.match(lineD)
            if res:
                blockOpen = True

        else:
            res = data.match(lineD)
            if res:
                if 'nonuniform' in res.group(2):
                    field = _readField(f,nDim)
                else:
                    field = res.group(2)
                blk[res.group(1)] = field
            if closeBracket.match(lineD):
                break

    return blk


def _readField(f, nDim):
    """
    Generic code to read an openFoam type field data structure
    """

    openBracket = re.compile(r'\s*\(\s*')
    closeBracket = re.compile(r'\s*\)\s*')
    data = re.compile(r'\s*([a-zA-Z]*)\s*(.*);\s*\n')
    dataNSC = re.compile(r'\s*([a-zA-Z]*)\s*(.*)\s*\n')
    numberHeader = re.compile(r'\s*(\d{1,100})\s*')

    blockOpen = False
    sizeFound = False

    field = {}

    for line in f:
        if not sizeFound:
            # figure out the size of this field block
            sizeTest = numberHeader.search(line.decode())

            if sizeTest:
                field['size'] = int(sizeTest.group(1))
                sizeFound=True
                # now also check for a block open in this line
                openTest = openBracket.search(line.decode())
                if openTest:
                    blockOpen=True
                    field['value']=[]
                    # now if size is 0 also check for close bracket
                    if field['size']==0:
                        closeTest = closeBracket.search(line.decode())
                        if closeTest:
                            blockOpen=False
                            return field

        elif not blockOpen:
            openTest = openBracket.match(line.decode())
            if openTest:
                blockOpen=True
                field['value']=[]

        else:
            closeTest = closeBracket.search(line.decode())
            if closeTest:
                blockOpen=False
                return field
            else:
                if nDim==1:
                    varLine = re.compile(r'\s*(\S*)\s*')
                elif nDim==2:
                    varLine = re.compile(r'\s*\((\S*)\s*(\S*)\)\s*')
                elif nDim == 3:
                    varLine = re.compile(r'\s*\((\S*)\s*(\S*)\s*(\S*)\)\s*')
                else:
                    raise Error("nDim >3 not yet supported.")

                res = varLine.match(line.decode())
                if res:
                    field['value'].append(res.group(1))

def readFoamVarFile(fileNames, varName, timeDir,nCells, nDim):
    '''
    Generic function to read a FOAM variable file at a given time.
    '''

    # Open the var file for reading
    varDir = os.path.join(fileNames['varBaseDir'],timeDir)
    fileName = os.path.join(varDir, varName)

    if writeCompression():
        f = gzip.open(fileName+'.gz', 'rb')
    else:
        f = open(fileName, 'r')

    var = {}

    # Read Regular Header
    foamHeader, N, oneLineData = _parseFoamHeader(f,nDim)

    var['header']=foamHeader
    try:
        fmt = foamHeader['format'].lower() 
    except:
        fmt = 'ascii' # assume ascii if format is missing.

    if foamHeader['fieldType'].rstrip() == 'uniform':
        nVals = nCells
        # convert all vars to non-uniform for now... 
        # NOTE: comment out the following for now, we don't want to write
        # nonuniform field
        #if nDim ==1:
        #    foamHeader['internalField'] = 'nonuniform List<scalar> \n'
        #elif nDim==3:
        #    foamHeader['internalField'] = 'nonuniform List<vector> \n'
    else:
        nVals = N
        if not(nCells == N):
            raise Error("nCells doesn't match the var length")

    var['values'] = np.zeros((nVals, nDim),dtype='d')

    keyword = re.compile(r'\s*([a-zA-Z]{1,100})\s*\n')

    if fmt == 'binary':
        raise Error('Binary Reading is not yet supported.')
    else:
        k = 0

        if foamHeader['uniformValue'] != None:
            # this data is all specified with a single value
            for j in range(nVals):
                for idim in range(nDim):
                    var['values'][j, idim] = float(foamHeader['uniformValue'][idim])

        elif oneLineData:

            # create the regular expression for the single line
            dataPack = re.compile(r'\(.*\)\;')
            # read the line from the file
            line = f.readline()

            # Now parse the data
            dataLine = dataPack.search(line.decode())

            data = line[dataLine.start(0)+1:dataLine.end(0)-2]

            if nDim==1:
                varLine = re.compile(r'\s*(\S*)\s*')
            elif nDim==2:
                varLine = re.compile(r'\s*\((\S*)\s*(\S*)\)\s*')
            elif nDim == 3:
                varLine = re.compile(r'\s*\((\S*)\s*(\S*)\s*(\S*)\)\s*')
            else:
                raise Error("nDim >3 not yet supported.")

            for j in range(N):
                res = varLine.match(data)

                if res:
                    k += 1
                    for idim in range(nDim):
                        var['values'][j, idim] = float(
                            data[res.start(idim+1):res.end(idim+1)])

                    data = data[res.end(idim+1)+1:]

            if k != N:
                raise Error('Error reading variable. Expected %d'
                            ' values but found %d'%(N, k))

            #i = i + 1

        else:

            if nDim==1:
                varLine = re.compile(r'\s*(\S*)\s*\n')
            elif nDim==2:
                varLine = re.compile(r'\s*\((\S*)\s*(\S*)\s*\n')
            elif nDim == 3:
                varLine = re.compile(r'\s*\((\S*)\s*(\S*)\s*(\S*)\)\s*\n')
            else:
                raise Error("nDim >3 not yet supported.")

            #for j in range(N):
            j = 0
            for line in f:
                res = varLine.match(line.decode())

                if res:
                    k += 1
                    for idim in range(nDim):
                        var['values'][j, idim] = float(
                            line[res.start(idim+1):res.end(idim+1)])
                j+=1
                if j>=N:
                    break

            if k != N:
                raise Error('Error reading variable. Expected %d'
                            ' values but found %d'%(N, k))


    for line in f:

        res = keyword.match(line.decode())
        if res:
            boundaryName = line[res.start(1):res.end(1)]
            var['boundaryName']=boundaryName

            blkData = _readBoundaryBlock(f,nDim)
            var[boundaryName]=blkData

    f.close()
    return var

def _readBoundaryBlock(f,nDim=0):
    """
    Code to read an openFoam boundary block structure in vars
    """
    #keyword = re.compile(r'\s*([a-zA-Z]{1,100})\s*\n')  
    keyword = re.compile(r'\s*([a-zA-Z][a-zA-Z0-9]{1,100})\s*\n')
    openBracket = re.compile(r'\s*\{\s*\n')
    closeBracket = re.compile(r'\s*\}\s*\n')
    data = re.compile(r'\s*([a-zA-Z]*)\s*(.*);\s*\n')
   # dataNSC = re.compile(r'\s*([a-zA-Z]*)\s*(.*)\s*\n') # NSC part is not working well, need to fix this


    blockOpen = False
    blk = OrderedDict()#{}
    for line in f:
        if not blockOpen:
            res = openBracket.match(line.decode())
            if res:
                blockOpen = True
        else:
            keywd = keyword.match(line.decode())
            res = data.match(line.decode())
           # resNSC = dataNSC.match(line.decode())

            if keywd:
                # this is the start of another block, call recursively
                blkData = _readBoundaryBlock(f,nDim)
                blk[keywd.group(1)] = blkData
            elif res:
                if 'nonuniform' in res.group(2):
                    #field = _readField(f,nDim)
                    # NOTE: _readField will return None and stop for 
                    # nonuniform 0() case in parallel
                    # so changet it to field = res.group(2)   
                    field = res.group(2)   
                else:
                    field = res.group(2)                
                blk[res.group(1)] = field#res.group(2)
            #elif resNSC:
                # this is the start of a set that is multiple lines long
                #if 'nonuniform' in resNSC.group(2):
                #    field = _readField(f,nDim)
                #else:
                #    field = resNSC.group(2)

                #blk[resNSC.group(1)] = field#res.group(2)


            if closeBracket.match(line.decode()):
                break

    return blk

# -------------------------------------------------------
# Writing routines
# -------------------------------------------------------

def _writeOpenFOAMVolumePoints(fileNames,nodes):
    '''
    Write the most recent points to a file
    '''
    fileName = fileNames['pointsFile']
    if writeCompression():
        f = gzip.open(fileName, 'wb')
    else:
        f = open(fileName, 'w')

    # write the file header
    f.write(b'/*--------------------------------*- C++ -*----------------------------------*\ \n')
    f.write(b'| =========                 |                                                 |\n')
    f.write(b'| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    f.write(b'|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    f.write(b'|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    f.write(b'|    \\\\/     M anipulation  |                                                 |\n')
    f.write(b'\*---------------------------------------------------------------------------*/\n')
    f.write(b'FoamFile\n')
    f.write(b'{\n')
    f.write(b'    version     2.0;\n')
    f.write(b'    format      ascii;\n')
    f.write(b'    class       vectorField;\n')
    f.write(b'    location    "constant/polyMesh";\n')
    f.write(b'    object      points;\n')
    f.write(b'}\n')
    f.write(b'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    f.write(b'\n')
    f.write(b'\n')

    nodes = nodes.reshape((int(len(nodes)/3), 3))
    nPoints = len(nodes)
    f.write(b'%d\n'% nPoints)
    f.write(b'(\n')
    for i in range(nPoints):
        if abs(nodes[i, 0])<1e-16:
            nodes[i, 0]=0.0
        if abs(nodes[i, 1])<1e-16:
            nodes[i, 1]=0.0
        if abs(nodes[i, 2])<1e-16:
            nodes[i, 2]=0.0
        f.write(b'(%20.15e %20.15e %20.15e)\n'%(nodes[i, 0], nodes[i, 1], nodes[i, 2]))
    f.write(b')\n\n\n')
    f.write(b'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

    return
    
def writeFoamVarFile(var, timeDir):
    '''
    Generic function to write a FOAM variable file at a given time.
    '''
    
    comm = MPI.COMM_WORLD
        
    # Open the var file for reading
    workingDirectory = os.getcwd()
    if comm.size>1:
        baseDir = os.path.join(workingDirectory,'processor%d'%comm.rank)
        varDir = os.path.join(baseDir,timeDir)
    else:
        varDir = os.path.join(workingDirectory,timeDir)
    # end
  
    varName = var['header']['FoamFile\n']['object']
    if(len(var['values'].shape)==1):
        nDim = 1
    else:
        nDim =  var['values'].shape[1]
    fileName = os.path.join(varDir, varName)
    #print('inwrite',fileName,self.comm.rank)
    if writeCompression():
        f = gzip.open(fileName, 'wb')
    else:
        f = open(fileName, 'w')

    dataType = var['header']['FoamFile\n']['class']#'volScalarField'
  
    # write the file header
    f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    f.write('| =========                 |                                                 |\n')
    f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
    f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    f.write('|    \\\\/     M anipulation  |                                                 |\n')
    f.write('\*---------------------------------------------------------------------------*/\n')
    f.write('FoamFile\n')
    f.write('{\n')
    f.write('    version     2.0;\n')
    f.write('    format      ascii;\n')
    f.write('    class       %s;\n'%dataType)
    f.write('    location    "%s";\n'%timeDir)
    f.write('    object      %s;\n'%varName)
    f.write('}\n')
    f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    f.write('\n')
    f.write('dimensions      %s;\n'%(var['header']['dimensions']))
    f.write('\n')
    if nDim ==1:
        v1= float(var['header']['uniformValue'][0])
        f.write('internalField uniform %f;\n'%(v1))
    else:
        v1= float(var['header']['uniformValue'][0])
        v2= float(var['header']['uniformValue'][1])
        v3= float(var['header']['uniformValue'][2])
        f.write('internalField uniform (%f %f %f);\n'%(v1,v2,v3))
    f.write('\n')
    
    #print('intField',var['header']['internalField'])
    
    # NOTE: comment this out since we don't want to write a list now
    #nVar = len(var['values'])
    #f.write('%d\n'% nVar)
    #f.write('(\n')
    #for i in range(nVar):
    #    if nDim==1:
    #        if(len(var['values'].shape)==1):
    #            f.write('%20.16f\n'%var['values'][i])
    #        else:
    #            f.write('%20.16f\n'%var['values'][i,0])
    #    else:
    #        f.write('(')
    #        for j in range(nDim):
    #            f.write('%20.16f '%var['values'][i,j])
    #          
    #        f.write('\n')
    #f.write(')\n;\n\n')
    
    # now write boundary
    bName = var['boundaryName']
    f.write('%s\n{\n'%bName)
    #print 'var',bName,var[bName]
    for key in var[bName]:#['blkNames']:
    
        f.write('    %s\n    {\n'%key)
        for data in var[bName][key]:
            if type(var[bName][key][data])==dict:
                if var[bName][key][data]['size']==0:
                    f.write('     %s   nonuniform 0();\n'%(data))
                else:
                    f.write('     %s   nonuniform List<%s>\n%s\n(\n'%(data,'scalar',var[bName][key][data]['size']))
                    for i in range(var[bName][key][data]['size']):
                        f.write('  %s\n'%(var[bName][key][data]['value'][i]))
                    f.write(')\n')
            elif not (data == 'blkNames' or data=='uniformValue' or data=='fieldType'):
                f.write('        %s    %s;\n'%(data,var[bName][key][data]))
        f.write('    }\n')
    f.write('}\n')
          
    f.write('\n\n')
    f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
    f.close()
    
    return

    def _writeInitialDir(self):
        '''
        write out the initial directory
        '''
        self._writePFile()
        self._writeUFile()
        self._writeAlphaFile()
        self._writenutFile()
        self._writenuTildaFile()

    def _writePFile(self):
        
         # Open the var file for reading
        workingDirectory = os.getcwd()
        timeDir='0b'
        varDir = os.path.join(workingDirectory,timeDir)
                
        varName = 'p'
        fileName = os.path.join(varDir, varName)

        #print('inwrite',fileName,self.comm.rank)
        if writeCompression():
            f = gzip.open(fileName, 'wb')
        else:
            f = open(fileName, 'w')

        dataType = 'volScalarField'
        
        # write the file header
        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       %s;\n'%dataType)
        f.write('    location    "%s";\n'%timeDir)
        f.write('    object      %s;\n'%varName)
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 2 -2 0 0 0 0];\n')
        f.write('\n')
        f.write('internalField uniform 0')

        f.write(';\n\n')

        f.write('boundaryField\n')
        f.write('{\n')
        #print('ofboundaries',self.OFData['boundaries'].keys())
        for key in self.OFData['boundaries']:
            f.write('   %s\n     {\n'%key)
            print(key,self.OFData['boundaries'][key]['type'])
            if(self.OFData['boundaries'][key]['type']=='wall'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='patch'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='inlet'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='outlet'):
                f.write('        type           fixedValue;\n')
                f.write('        value          uniform 0;\n')
            f.write('     }\n')
        f.write('}\n')


        f.write('\n\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

        f.close()

    def _writeUFile(self):
        
         # Open the var file for reading
        workingDirectory = os.getcwd()
        timeDir='0b'
        varDir = os.path.join(workingDirectory,timeDir)

        initialU = [26.8224,0,0]

        varName = 'U'
        fileName = os.path.join(varDir, varName)

        #print('inwrite',fileName,self.comm.rank)
        if writeCompression():
            f = gzip.open(fileName, 'wb')
        else:
            f = open(fileName, 'w')

        dataType = 'volVectorField'
        
        # write the file header
        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       %s;\n'%dataType)
        f.write('    location    "%s";\n'%timeDir)
        f.write('    object      %s;\n'%varName)
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 1 -1 0 0 0 0];\n')
        f.write('\n')
        f.write('internalField uniform (%f %f %f)'%(initialU[0],initialU[1],initialU[2]))

        f.write(';\n\n')

        f.write('boundaryField\n')
        f.write('{\n')
        #print('ofboundaries',self.OFData['boundaries'].keys())
        for key in self.OFData['boundaries']:
            f.write('   %s\n     {\n'%key)
            print(key,self.OFData['boundaries'][key]['type'])
            if(self.OFData['boundaries'][key]['type']=='wall'):
                f.write('        type           fixedValue;\n')
                f.write('        value          uniform (0 0 0);\n')
            elif(self.OFData['boundaries'][key]['type']=='patch'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='inlet'):
                f.write('        type           fixedValue;\n')
                f.write('        value          uniform (%f %f %f);\n'%(initialU[0],initialU[1],initialU[2]))
            elif(self.OFData['boundaries'][key]['type']=='outlet'):
                f.write('        type           zeroGradient;\n')
            f.write('     }\n')
        f.write('}\n')


        f.write('\n\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

        f.close()

    def _writeAlphaFile(self):
        
         # Open the var file for reading
        workingDirectory = os.getcwd()
        timeDir='0b'
        varDir = os.path.join(workingDirectory,timeDir)
                
        varName = 'alpha'
        fileName = os.path.join(varDir, varName)

        #print('inwrite',fileName,self.comm.rank)
        if writeCompression():
            f = gzip.open(fileName, 'wb')
        else:
            f = open(fileName, 'w')

        dataType = 'volScalarField'
        
        # write the file header
        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       %s;\n'%dataType)
        f.write('    location    "%s";\n'%timeDir)
        f.write('    object      %s;\n'%varName)
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 0 -1 0 0 0 0];\n')
        f.write('\n')
        f.write('internalField uniform 0')

        f.write(';\n\n')

        f.write('boundaryField\n')
        f.write('{\n')
        #print('ofboundaries',self.OFData['boundaries'].keys())
        for key in self.OFData['boundaries']:
            f.write('   %s\n     {\n'%key)
            print(key,self.OFData['boundaries'][key]['type'])
            if(self.OFData['boundaries'][key]['type']=='wall'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='patch'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='inlet'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='outlet'):
                f.write('        type           fixedValue;\n')
                f.write('        value          uniform 0;\n')
            f.write('     }\n')
        f.write('}\n')


        f.write('\n\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

        f.close()

    def _writenutFile(self):
        
         # Open the var file for reading
        workingDirectory = os.getcwd()
        timeDir='0b'
        varDir = os.path.join(workingDirectory,timeDir)
                
        initialNut =0.14
        varName = 'nut'
        fileName = os.path.join(varDir, varName)

        #print('inwrite',fileName,self.comm.rank)
        if writeCompression():
            f = gzip.open(fileName, 'wb')
        else:
            f = open(fileName, 'w')

        dataType = 'volScalarField'
        
        # write the file header
        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       %s;\n'%dataType)
        f.write('    location    "%s";\n'%timeDir)
        f.write('    object      %s;\n'%varName)
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 2 -1 0 0 0 0];\n')
        f.write('\n')
        f.write('internalField uniform %f'%initialNut)

        f.write(';\n\n')

        f.write('boundaryField\n')
        f.write('{\n')
        #print('ofboundaries',self.OFData['boundaries'].keys())
        for key in self.OFData['boundaries']:
            f.write('   %s\n     {\n'%key)
            print(key,self.OFData['boundaries'][key]['type'])
            if(self.OFData['boundaries'][key]['type']=='wall'):
                f.write('       type            nutUSpaldingWallFunction;\n')
                f.write('       value           uniform 0;\n')
            elif(self.OFData['boundaries'][key]['type']=='patch'):
                f.write('        type           zeroGradient;\n')
            elif(self.OFData['boundaries'][key]['type']=='inlet'):
                f.write('        type           fixedValue;\n')
                f.write('        value          uniform %f;\n'%initialNut)
            elif(self.OFData['boundaries'][key]['type']=='outlet'):
                f.write('        type           zeroGradient;\n')
            f.write('     }\n')
        f.write('}\n')


        f.write('\n\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

        f.close()

    def _writenuTildaFile(self):
        
         # Open the var file for reading
        workingDirectory = os.getcwd()
        timeDir='0b'
        varDir = os.path.join(workingDirectory,timeDir)
                
        initialNuTilda =0.14
        varName = 'nuTilda'
        fileName = os.path.join(varDir, varName)

        #print('inwrite',fileName,self.comm.rank)
        if writeCompression():
            f = gzip.open(fileName, 'wb')
        else:
            f = open(fileName, 'w')

        dataType = 'volScalarField'
        
        # write the file header
        f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\\    /   O peration     | Version:  v1812                                 |\n')
        f.write('|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\\/     M anipulation  |                                                 |\n')
        f.write('\*---------------------------------------------------------------------------*/\n')
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('    version     2.0;\n')
        f.write('    format      ascii;\n')
        f.write('    class       %s;\n'%dataType)
        f.write('    location    "%s";\n'%timeDir)
        f.write('    object      %s;\n'%varName)
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        f.write('dimensions      [0 2 -1 0 0 0 0];\n')
        f.write('\n')
        f.write('internalField uniform %f'%initialNuTilda)

        f.write(';\n\n')

        f.write('boundaryField\n')
        f.write('{\n')
        #print('ofboundaries',self.OFData['boundaries'].keys())
        for key in self.OFData['boundaries']:
            f.write('   %s\n     {\n'%key)
            print(key,self.OFData['boundaries'][key]['type'])
            if(self.OFData['boundaries'][key]['type']=='wall'):
                f.write('       type            fixedValue;\n')
                f.write('       value           uniform 0;\n')
            elif(self.OFData['boundaries'][key]['type']=='patch'):
                f.write('        type           zeroGradient;\n')
                # f.write('       type            fixedValue;\n')
                # f.write('       value           uniform 0;\n')
            elif(self.OFData['boundaries'][key]['type']=='inlet'):
                f.write('        type           fixedValue;\n')
                f.write('        value          uniform %f;\n'%initialNuTilda)
            elif(self.OFData['boundaries'][key]['type']=='outlet'):
                f.write('        type           zeroGradient;\n')
            f.write('     }\n')
        f.write('}\n')


        f.write('\n\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

        f.close()
