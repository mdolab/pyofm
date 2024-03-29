
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
from petsc4py import PETSc
from mdo_regression_helper import *
if 'complex' in sys.argv:
    from idwarp import USMesh_C as USMesh
else:
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

h = 1e-40

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
    mesh = USMesh(options=meshOptions, debug=True)

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

    if not 'complex' in sys.argv:
       
        if MPI.COMM_WORLD.rank == 0:
            print('Computing Warp Deriv')
        mesh.warpDeriv(dXv_warp,solverVec=False)
        dXs = mesh.getdXs()
            
        val = MPI.COMM_WORLD.reduce(numpy.sum(dXs.flatten()),op=MPI.SUM)
        if MPI.COMM_WORLD.rank == 0:
            print('Sum of dxs:')
            reg_write(val,1e-8,1e-8)
    else:
               
        # add a complex perturbation on all surface nodes simultaneously:
        for i in range(len(coords0)):
            new_coords[i,:] += h*1j

        # Reset the newly computed surface coordiantes
        mesh.setSurfaceCoordinates(new_coords)
        mesh.warpMesh()  

        vCoords = mesh.getWarpGrid()
        deriv = numpy.imag(vCoords)/h
        deriv = numpy.dot(dXv_warp,deriv)
        val = MPI.COMM_WORLD.reduce(numpy.sum(deriv),op=MPI.SUM)

        if MPI.COMM_WORLD.rank == 0:
            print('Sum of dxs:')
            reg_write(val,1e-8,1e-8)
        

    del mesh
    #os.system('rm -fr *.cgns *.dat')
    #change back to the original directory
    os.chdir('../../')

if __name__ == '__main__':
    if len(sys.argv) == 1 or(len(sys.argv) == 2 and 'complex' in sys.argv):
        # testall
        test1()

    else:
        # Run individual ones
        if 'test1' in sys.argv:
            test1()
      
