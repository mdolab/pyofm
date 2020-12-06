/*---------------------------------------------------------------------------*\

    pyOFM  : Python interface for OpenFOAM mesh
    Version : v1.2

\*---------------------------------------------------------------------------*/
#include "OFMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
OFMesh::OFMesh(
    char* argsAll)
    : meshPtr_(nullptr),
      runTimePtr_(nullptr)
{
    argsAll_ = argsAll;
}

OFMesh::~OFMesh()
{
}

void OFMesh::readMesh()
{
// read the mesh and compute some sizes
#include "setArgs.H"
#include "setRootCasePython.H"

    Info << "Reading the OpenFOAM mesh.." << endl;

    runTimePtr_.reset(
        new Time(
            Time::controlDictName,
            args));

    Time& runTime = runTimePtr_();

    word regionName = fvMesh::defaultRegion;
    meshPtr_.reset(
        new fvMesh(
            IOobject(
                regionName,
                runTime.timeName(),
                runTime,
                IOobject::MUST_READ)));

    fvMesh& mesh = meshPtr_();

    nLocalPoints_ = mesh.nPoints();

    nLocalCells_ = mesh.nCells();

    nLocalFaces_ = mesh.nFaces();

    nLocalInternalFaces_ = mesh.nInternalFaces();

    // initialize pointField and assign its values based on the initial mesh
    pointField_.setSize(nLocalPoints_);

    forAll(pointField_, pointI)
    {
        for (label compI = 0; compI < 3; compI++)
        {
            pointField_[pointI][compI] = mesh.points()[pointI][compI];
        }
    }

    nLocalBoundaryPatches_ = 0;
    forAll(mesh.boundaryMesh(), patchI)
    {
        nLocalBoundaryPatches_++;
    }

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
