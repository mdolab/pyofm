
############################################################
# DO NOT USE THIS SCRIPT AS A REFERENCE FOR HOW TO USE pyOFM
# THIS SCRIPT USES PRIVATE INTERNAL FUNCTIONALITY THAT IS
# SUBJECT TO CHANGE!!
############################################################

# ======================================================================
#         Imports
# ======================================================================
import sys,os,copy
import numpy
from mdo_regression_helper import *
from idwarp import USMesh
from mpi4py import MPI

# First thing we will do is define a complete set of default options
# that will be reused as we do differnt tests.  These are the default
# options as July 15, 2015.

defOpts = {
    'gridFile':None,
    'aExp': 3.0,
    'bExp': 5.0,
    'LdefFact':1.0,
    'alpha':0.25,
    'errTol':0.0005,
    'evalMode':'fast',
    'symmTol':1e-6,
    'useRotations':True,
    'bucketSize':8,
    'fileType':None

}

def printHeader(testName):
    if MPI.COMM_WORLD.rank == 0:
        print('+' + '-'*78 + '+')
        print('| Test Name: ' + '%-66s'%testName + '|')
        print('+' + '-'*78 + '+')


def test1():
    # Test the Ahmed body openfoam mesh
    sys.stdout.flush()
    #change directory to the correct test case
    os.chdir('./input/ahmedBodyMesh/')

    file_name = os.getcwd()

    meshOptions = copy.deepcopy(defOpts)

    meshOptions.update(
        {'gridFile':file_name,
         'fileType':'OpenFOAM',
         'symmetryPlanes':[[[0,0,0], [0,1,0]]],
     }
    )
    # Create warping object
    mesh = USMesh(options=meshOptions)

    # Extract Surface Coordinates
    coords0 = mesh.getSurfaceCoordinates()

    vCoords = mesh.getCommonGrid()
    val = MPI.COMM_WORLD.reduce(numpy.sum(vCoords.flatten()),op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of vCoords Inital:')
        reg_write(val,1e-8,1e-8)

    new_coords = coords0.copy()
    # Do a stretch:
    for i in range(len(coords0)):
        length = coords0[i,2]
        new_coords[i,0] += .05*length

    # Reset the newly computed surface coordiantes
    mesh.setSurfaceCoordinates(new_coords)
    mesh.warpMesh()

    vCoords = mesh.getWarpGrid()
    val = MPI.COMM_WORLD.reduce(numpy.sum(vCoords.flatten()),op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of vCoords Warped:')
        reg_write(val,1e-8,1e-8)

    # Create a dXv vector to do test the mesh warping with:
    dXv_warp = numpy.linspace(0,1.0, mesh.warp.griddata.warpmeshdof)
    if MPI.COMM_WORLD.rank == 0:
        print('Computing Warp Deriv')
    mesh.warpDeriv(dXv_warp,solverVec=False)
    dXs = mesh.getdXs()

    val = MPI.COMM_WORLD.reduce(numpy.sum(dXs.flatten()),op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of dxs:')
        reg_write(val,1e-8,1e-8)

    if MPI.COMM_WORLD.rank == 0:
        print('Verifying Warp Deriv')
    mesh.verifyWarpDeriv(dXv_warp,solverVec=False,dofStart=0,dofEnd=10,h=1e-9)

    #change back to the original directory
    os.chdir('../../')

def test2():
    # Test some basic pyOFM functions uing the Ahmed body openfoam mesh
    sys.stdout.flush()
    #change directory to the correct test case
    os.chdir('./input/ahmedBodyMesh/')

    from pyofm import PYOFM

    ofm = PYOFM(MPI.COMM_WORLD)

    # test points
    points = ofm.readVolumeMeshPoints()
    val = MPI.COMM_WORLD.reduce(numpy.sum(points.flatten()),op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of points initial:')
        reg_write(val,1e-8,1e-8)

    # test faces
    faces = ofm.readFaceInfo()
    fSum = 0
    for face in faces:
        for point in face:
            fSum += point
    val = MPI.COMM_WORLD.reduce(fSum,op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of faces:')
        reg_write(val,1e-8,1e-8)
    
    # test boundaries
    bcSum = 0.0
    boundaries = ofm.readBoundaryInfo(faces)
    for key in list(boundaries.keys()):
        boundary = boundaries[key]
        nodes = boundary['nodes']
        faces = boundary['faces']
        for node in nodes:
            bcSum += node
        for face in faces:
            bcSum += face
    val = MPI.COMM_WORLD.reduce(bcSum,op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of boundary nodes and faces:')
        reg_write(val,1e-8,1e-8)

    # test owners and neighbours
    owners, neighbours = ofm.readCellInfo()
    onSum = 0.0
    for owner in owners:
        onSum += owner
    for neighbour in neighbours:
        onSum += neighbour
    val = MPI.COMM_WORLD.reduce(onSum,op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of owners and neighbours:')
        reg_write(val,1e-8,1e-8)
    
    # test write and read mesh points
    points1D = points.flatten()
    points1D *= 10.0 # scale the point
    ofm.writeVolumeMeshPoints(points1D)
    pointsNew = ofm.readVolumeMeshPoints()
    val = MPI.COMM_WORLD.reduce(numpy.sum(pointsNew.flatten()),op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print('Sum of points new:')
        reg_write(val,1e-8,1e-8)
    
    # reset the points 
    points1D /= 10.0 # scale the point back
    ofm.writeVolumeMeshPoints(points1D)

    #change back to the original directory
    os.chdir('../../')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Run all tests
        test1()
        test2()
    else:
        # Run individual ones
        if 'test1' in sys.argv:
            test1()
        if 'test2' in sys.argv:
            test2()
