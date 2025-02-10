/*---------------------------------------------------------------------------*\

    pyOFM  : Python interface for OpenFOAM mesh
    Version : v1.2

    Description:
        Functions to read OpenFOAM mesh information

\*---------------------------------------------------------------------------*/

#ifndef OFMesh_H
#define OFMesh_H

#include "fvCFD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class OFMesh Declaration
\*---------------------------------------------------------------------------*/

class OFMesh
{

private:
    /// Disallow default bitwise copy construct
    OFMesh(const OFMesh&);

    /// Disallow default bitwise assignment
    void operator=(const OFMesh&);

    /// all the arguments
    char* argsAll_;

    autoPtr<fvMesh> meshPtr_;

    autoPtr<Time> runTimePtr_;

    autoPtr<argList> argsPtr_;

    label nLocalPoints_;

    label nLocalCells_;

    label nLocalFaces_;

    label nLocalInternalFaces_;

    label nLocalBoundaryPatches_;

    pointField pointField_;

public:
    /// Constructors
    OFMesh(char* argsAll);

    /// Destructor
    virtual ~OFMesh();

    void readMesh();

    double getMeshPointCoord(
        const label pointI,
        const label compI) const
    {
#if defined(CODI_ADR) || defined(CODI_ADF)
        return meshPtr_().points()[pointI][compI].getValue();
#else
        return meshPtr_().points()[pointI][compI];
#endif
    }

    void setMeshPointCoord(
        const label pointI,
        const label compI,
        const double value)
    {
        pointField_[pointI][compI] = value;
    }

    label getMeshFacePointIndex(
        const label faceI,
        const label pointI) const
    {
        const label& facePointSize = meshPtr_().faces()[faceI].size();
        if (pointI > facePointSize - 1)
        {
            FatalErrorIn("getMeshFacePointIndex") << "Face " << faceI << " has " << facePointSize << " points."
                                                  << " while the require pointI is " << pointI << abort(FatalError);
        }
        return meshPtr_().faces()[faceI][pointI];
    }

    label getNLocalPoints() const
    {
        return nLocalPoints_;
    }

    label getNLocalCells() const
    {
        return nLocalCells_;
    }

    label getNLocalFaces() const
    {
        return nLocalFaces_;
    }

    label getNLocalInternalFaces() const
    {
        return nLocalInternalFaces_;
    }

    label getNFacePoints(const label faceI) const
    {
        return meshPtr_().faces()[faceI].size();
    }

    void writeMesh() const
    {
        meshPtr_().write();
    }

    void updateMesh()
    {
        meshPtr_().movePoints(pointField_);
    }

    label getNLocalBoundaryPatches() const
    {
        return nLocalBoundaryPatches_;
    }

    word getLocalBoundaryName(const label patchI) const
    {
        return meshPtr_().boundaryMesh()[patchI].name();
    }

    word getLocalBoundaryType(const label patchI) const
    {
        return meshPtr_().boundaryMesh()[patchI].type();
    }

    label getLocalBoundaryStartFace(const label patchI) const
    {
        return meshPtr_().boundaryMesh()[patchI].start();
    }

    label getLocalBoundaryNFaces(const label patchI) const
    {
        return meshPtr_().boundaryMesh()[patchI].size();
    }

    label getLocalFaceOwner(const label faceI) const
    {
        return meshPtr_().owner()[faceI];
    }

    label getLocalFaceNeighbour(const label faceI) const
    {
        return meshPtr_().neighbour()[faceI];
    }

    label getLocalBoundaryFaceOwner(
        const label patchI,
        const label faceI) const
    {
        const UList<label>& pFaceCells = meshPtr_().boundaryMesh()[patchI].faceCells();
        return pFaceCells[faceI];
    }

    void readField(
        const word fieldName,
        const word fieldType,
        const word timeName,
        double* field)
    {
        if (fieldType == "volScalarField")
        {
            volScalarField state(
                IOobject(
                    fieldName,
                    timeName,
                    meshPtr_(),
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false),
                meshPtr_());

            forAll(state, cellI)
            {
#if defined(CODI_ADR) || defined(CODI_ADF)
                field[cellI] = state[cellI].getValue();
#else
                field[cellI] = state[cellI];
#endif
            }
        }
        else if (fieldType == "volVectorField")
        {
            volVectorField state(
                IOobject(
                    fieldName,
                    timeName,
                    meshPtr_(),
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE,
                    false),
                meshPtr_());

            label counterI = 0;
            forAll(state, cellI)
            {
                for (label i = 0; i < 3; i++)
                {
#if defined(CODI_ADR) || defined(CODI_ADF)
                field[counterI] = state[cellI][i].getValue();
#else
                field[counterI] = state[cellI][i];
#endif
                    counterI++;
                }
            }
        }
        else
        {
            FatalErrorIn("readField") << "fieldType " << fieldType
                                      << " not supported! Options are volScalariField or volVectorField" << abort(FatalError);
        }
    }

    void writeField(
        const word fieldName,
        const word fieldType,
        const double* field)
    {
        if (fieldType == "volScalarField")
        {
            volScalarField state(
                IOobject(
                    fieldName,
                    "0",
                    meshPtr_(),
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false),
                meshPtr_(),
                dimensionedScalar(fieldName, dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
                "zeroGradient");

            forAll(state, cellI)
            {
                state[cellI] = field[cellI];
            }
            state.correctBoundaryConditions();
            state.write();
        }
        else if (fieldType == "volVectorField")
        {
            volVectorField state(
                IOobject(
                    fieldName,
                    "0",
                    meshPtr_(),
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false),
                meshPtr_(),
                dimensionedVector(fieldName, dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::zero),
                "zeroGradient");

            label counterI = 0;
            forAll(state, cellI)
            {
                for (label i = 0; i < 3; i++)
                {
                    state[cellI][i] = field[counterI];
                    counterI++;
                }
            }
            state.correctBoundaryConditions();
            state.write();
        }
        else
        {
            FatalErrorIn("readField") << "fieldType " << fieldType
                                      << " not supported! Options are volScalariField or volVectorField" << abort(FatalError);
        }
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
