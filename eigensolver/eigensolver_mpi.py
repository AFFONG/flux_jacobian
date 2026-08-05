"""
Global flux Jacobian stability analysis -- single solve, MPI-parallel.

Run with:
    mpiexec -n 4 python solve_single.py

Every process runs this same script; PETSc's COMM_WORLD groups them
into one distributed solve. Print statements are guarded so only
rank 0 writes output, avoiding duplicated prints from every process.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'   # avoid thread oversubscription
os.environ['MKL_NUM_THREADS'] = '1'   # alongside MPI ranks -- each rank
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # should use 1 thread, not many

import time
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.getRank()
nprocs = comm.getSize()


def log(msg):
    """Only rank 0 prints -- otherwise every process would print the
    same line nprocs times."""
    if rank == 0:
        print(msg)


# =====================================================================
# 1. Read the Jacobian -- every rank reads the full file, then keeps
#    only the rows PETSc assigns it (simplest correct approach; fine
#    at this matrix size, since scipy sparse storage is compact even
#    if read redundantly per rank)
# =====================================================================
def read_jacobian(path):
    J = sp.load_npz(path)
    J = J.tocsr().astype(PETSc.ScalarType)
    log(f"Loaded Jacobian: shape={J.shape}, nnz={J.nnz}, "
        f"density={J.nnz / (J.shape[0]*J.shape[1]):.2e}")
    return J


def scipy_csr_to_petsc_parallel(J_csr, comm):
    """Distributes the matrix across MPI ranks: PETSc decides the
    row partitioning, and each rank inserts only its own local rows."""
    n = J_csr.shape[0]
    A = PETSc.Mat().createAIJ(size=(n, n), comm=comm)
    A.setUp()

    Istart, Iend = A.getOwnershipRange()
    for i in range(Istart, Iend):
        row = J_csr.getrow(i)
        cols = row.indices
        vals = row.data
        if len(cols) > 0:
            A.setValues(i, cols, vals)

    A.assemble()
    return A


# =====================================================================
# 2. Solve with shift-invert (identical to your serial version --
#    PETSc/SLEPc objects are already comm-aware once A is distributed)
# =====================================================================
def solve_shift_invert(J, sigma, nev=10, ncv=None, tol=1e-10, max_it=2000):
    n = J.getSize()[0]
    if ncv is None:
        ncv = min(n, max(3 * nev, 30))

    E = SLEPc.EPS().create(comm=comm)
    E.setOperators(J)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setDimensions(nev=nev, ncv=ncv)
    E.setTolerances(tol=tol, max_it=max_it)

    st = E.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(sigma)

    ksp = st.getKSP()
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    try:
        pc.setFactorSolverType('mumps')   # MUMPS does the actual
        solver_used = 'mumps'              # parallel factorization
    except PETSc.Error:
        pc.setFactorSolverType('petsc')
        solver_used = 'petsc (built-in, no mumps found)'

    E.setTarget(sigma)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setFromOptions()

    log(f"Solving: sigma={sigma}, nev={nev}, ncv={ncv}, "
        f"factorization={solver_used}, nprocs={nprocs}")
    E.solve()
    return E


# =====================================================================
# 3. Report + save -- only rank 0 does file I/O and printing, since
#    every rank has access to the same converged results via SLEPc's
#    internal MPI communication (getEigenpair etc. return the full
#    vector gathered appropriately)
# =====================================================================

def gather_vec_to_rank0(vec, comm):
    scatter, vec_full = PETSc.Scatter.toZero(vec)
    scatter.scatter(vec, vec_full, False, PETSc.Scatter.Mode.FORWARD)
    if comm.getRank() == 0:
        return vec_full.getArray().copy()
    return None


def report_and_save(E, J, nev, sigma, out_path, residual_tol=1e-6):
    nconv = E.getConverged()
    log(f"\nConverged eigenpairs: {nconv} / {nev} requested")

    vr, vi = J.createVecs()
    eigenvalues = np.zeros(nconv, dtype=complex)
    residuals = np.zeros(nconv)

    N = J.getSize()[0]
    eigenvectors = np.zeros((nconv, N), dtype=complex) if rank == 0 else None

    for i in range(nconv):
        val = E.getEigenpair(i, vr, vi)
        eigenvalues[i] = val
        residuals[i] = E.computeError(i)

        # every rank must call these -- they're collective operations
        vr_full = gather_vec_to_rank0(vr, comm) # gather vectors from all ranks to rank0
        vi_full = gather_vec_to_rank0(vi, comm) # gather vectors from all ranks to rank0

        # only rank 0 actually has non-None data to use
        if rank == 0:
            eigenvectors[i, :] = vr_full + 1j * vi_full

    if rank == 0:
        for i in range(nconv):
            lam = eigenvalues[i]
            err = residuals[i]
            status = ("UNRELIABLE" if err > residual_tol
                       else "UNSTABLE" if lam.real > 0
                       else "MARGINAL" if abs(lam.real) < 1e-3
                       else "stable")
            print(f"{i:>3} {lam.real:>14.6f} {lam.imag:>14.6f} "
                  f"{err:>12.2e}  {status}")

        np.savez(out_path,
                 eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                 residuals=residuals, sigma=np.array([sigma]),
                 nev=nev, nprocs=nprocs)
        print(f"\nSaved to {out_path}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    Mesh, Re, Mach = 21228, 60, 0.2
    data_dir = "/home/ahf25/git/flux_jacobian/data/flux_jacobian_assembly_v4"
    JACOBIAN_PATH = f"{data_dir}/jacobian_cylinder_{Mesh}_Re{Re}_M{Mach}_fd.npz"

    f = 9.505
    SIGMA = 0.0 + f * 2 * np.pi * 1j
    nev, ncv = 10, 300

    J_csr = read_jacobian(JACOBIAN_PATH)
    J = scipy_csr_to_petsc_parallel(J_csr, comm)

    t0 = time.perf_counter()
    E = solve_shift_invert(J, sigma=SIGMA, nev=nev, ncv=ncv)
    runtime = time.perf_counter() - t0
    log(f"Run time: {runtime:.2f}s ({nprocs} MPI ranks)")


    out_dir = "/home/ahf25/git/flux_jacobian/data/eigendata"
    out_path = f"{out_dir}/eigendata_{Mesh}_Re{Re}_M{Mach}_nev{nev}_ncv{ncv}_np{nprocs}.npz"
    report_and_save(E, J, nev, SIGMA, out_path)