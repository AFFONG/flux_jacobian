"""
Global flux Jacobian stability analysis -- single solve, MPI-parallel.
No MUMPS OOC (standard in-core MUMPS factorization).

Run with:
    mpirun -np 10 python solve_single.py --case-name OAT --mach 0.73 --aoa 35 ...

Every process runs this same script; PETSc's COMM_WORLD groups them
into one distributed solve. Print statements are guarded so only
rank 0 writes output, avoiding duplicated prints from every process.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'   # avoid thread oversubscription
os.environ['MKL_NUM_THREADS'] = '1'   # alongside MPI ranks -- each rank
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # should use 1 thread, not many

import time
import argparse
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
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
# 1. Read the Jacobian and distribute (in pieces) it across ranks
# Only save the full jacobian matrix in rank = 0
# every ranks hold it local slices
# =====================================================================

def read_and_distribute_jacobian(path, comm):
    mpi_comm = comm.tompi4py()
    rank = mpi_comm.Get_rank()

    if rank == 0:
        J = sp.load_npz(path).tocsr().astype(PETSc.ScalarType)
        n = J.shape[0]
        print(f"Loaded Jacobian: shape={J.shape}, nnz={J.nnz}")
    else:
        J = None
        n = None
    n = mpi_comm.bcast(n, root=0)

    A = PETSc.Mat().createAIJ(size=(n, n), comm=comm)
    A.setUp()
    Istart, Iend = A.getOwnershipRange()

    # gather every rank's (Istart, Iend) on rank 0
    all_ranges = mpi_comm.gather((Istart, Iend), root=0)

    if rank == 0:
        for dest, (r_start, r_end) in enumerate(all_ranges):
            block = J[r_start:r_end]
            if dest == 0:
                local_indptr = block.indptr.copy()
                local_indices = block.indices.copy()
                local_data = block.data.copy()
            else:
                mpi_comm.send((block.indptr, block.indices, block.data),
                               dest=dest, tag=100)
        del J  # free the full matrix on rank 0 once distributed
    else:
        local_indptr, local_indices, local_data = mpi_comm.recv(source=0, tag=100)

    A.setValuesCSR(local_indptr, local_indices, local_data)
    A.assemble()
    return A


# =====================================================================
# 2. Solve with shift-invert (standard in-core MUMPS, no OOC)
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
        solver_used = 'mumps'              # parallel factorization (in-core)
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
# 3. Report + save -- only rank 0 does file I/O and printing
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
        vr_full = gather_vec_to_rank0(vr, comm)
        vi_full = gather_vec_to_rank0(vi, comm)

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
# CLI args
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="MPI shift-invert eigensolver (no MUMPS OOC)")
    p.add_argument("--jacobian",   type=str, required=True, help="Path to Jacobian .npz")
    p.add_argument("--out-dir",    type=str, required=True, help="Output directory")

    p.add_argument("--case-name",  type=str,   default="cylinder",
                   help="Case identifier for output filename")
    p.add_argument("--mesh",       type=int,   default=None)
    p.add_argument("--re",         type=float, default=None)
    p.add_argument("--mach",       type=float, default=None)
    p.add_argument("--aoa",        type=float, default=None)
    p.add_argument("--mesh-ver",   type=int,   default=None)

    p.add_argument("--sigma-real", type=float, default=0.0)
    p.add_argument("--sigma-imag", type=float, required=True,
                   help="Im(sigma) in rad/s -- e.g. 2*pi*f for a frequency estimate f (Hz)")
    p.add_argument("--nev",        type=int,   default=10)
    p.add_argument("--ncv",        type=int,   default=None)
    p.add_argument("--tol",        type=float, default=1e-10)
    p.add_argument("--max-it",     type=int,   default=2000)
    return p.parse_args()


def build_output_tag(args):
    """Build the case-identifying portion of the output filename,
    using whichever identifiers were actually provided."""
    parts = [args.case_name]
    if args.mesh is not None:
        parts.append(str(args.mesh))
    if args.re is not None:
        parts.append(f"Re{int(args.re)}")
    if args.mach is not None:
        parts.append(f"M{args.mach}")
    if args.aoa is not None:
        parts.append(f"A{args.aoa}")
    return "_".join(parts)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    args = parse_args()
    sigma = args.sigma_real + args.sigma_imag * 1j
    nev, ncv = args.nev, args.ncv

    log(f"Case: {args.case_name}  mesh={args.mesh}  re={args.re}  "
        f"mach={args.mach}  aoa={args.aoa}")
    log(f"Jacobian path: {args.jacobian}")

    J = read_and_distribute_jacobian(args.jacobian, comm)

    t0 = time.perf_counter()
    E = solve_shift_invert(J, sigma=sigma, nev=nev, ncv=ncv)
    runtime = time.perf_counter() - t0
    log(f"Run time: {runtime:.2f}s ({nprocs} MPI ranks)")

    ncv_used = E.getDimensions()[1]
    tag = build_output_tag(args)
    fname = f"eigendata_{tag}_nev{nev}_ncv{ncv_used}_np{nprocs}.npz"
    out_path = os.path.join(args.out_dir, fname)

    os.makedirs(args.out_dir, exist_ok=True)
    report_and_save(E, J, nev, sigma, out_path)

# =====================================================================
# how to run
# =====================================================================
# conda activate petsc-complex
# cd /home/ahf25/git/flux_jacobian/eigensolver
#
# tmux new -s eigensolve
#
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
#
# ── cylinder case ────────────────────────────────────────────────────
# nohup mpirun -np 20 python -u eigensolver_mpi_v3.py \
#     --jacobian   "/home/ahf25/git/flux_jacobian/data/flux_jacobian_assembly_v4/v3_mesh/jacobian_cylinder_765960_Re60_M0.2_fd.npz" \
#     --out-dir    "/home/ahf25/git/flux_jacobian/data/eigendata/v3_mesh" \
#     --case-name  cylinder \
#     --mesh       765960 \
#     --re         60 \
#     --mach       0.2 \
#     --sigma-real 0.0 \
#     --sigma-imag 59.72 \
#     --nev        10 \
#     --ncv        300 \
#     > solve_output_cylinder.log 2>&1 &
#
# ── OAT15 case ───────────────────────────────────────────────────────
# nohup mpirun -np 38 python -u eigensolver_mpi_v3.py \
#     --jacobian   "/home/ahf25/git/flux_jacobian/data/flux_jacobian_assembly_v5/jacobian_OAT15_M0.73_A35_fd.npz" \
#     --out-dir    "/home/ahf25/git/flux_jacobian/data/eigendata/OAT15" \
#     --case-name  OAT \
#     --mach       0.73 \
#     --aoa        3.5 \
#     --sigma-real 0.0 \
#     --sigma-imag 440 \
#     --nev        5 \
#     --ncv        30 \
#     > solve_output_OAT15.log 2>&1 &
# #
# check status
# ps aux | grep solve_single.py
# kill <PID>

# check RAM usage
# watch -n 2 free -h