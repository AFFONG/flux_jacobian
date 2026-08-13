# =============================================================================
# GLOBAL STABILITY ANALYSIS — SHIFT-INVERT EIGENSOLVER
#
# Pipeline:
#   1. Read sparse Jacobian from .npz (CSR) format
#   2. Solve eigenproblem via SLEPc shift-invert (Krylov-Schur)
#   3. Report and save converged eigenpairs
#
# Requires: petsc-complex environment (PETSc/SLEPc complex scalar build)
#
# MUMPS out-of-core (OOC):
#   Enabled by default for large meshes where the LU factorisation
#   exceeds available RAM. MUMPS spills factor blocks to disk and
#   swaps them in/out during the solve.
#   → Point ooc_dir to a fast disk with sufficient free space (SSD preferred)
#   → Tune icntl_23 = total_RAM_MB / np  (max RAM per MPI process)
#
# Tunable parameters:
#   sigma      : complex shift  — place near expected physical eigenvalue
#   nev        : number of eigenvalues requested
#   ncv        : Krylov subspace size  (default: max(3*nev, 30))
#   tol        : eigensolver convergence tolerance  (default 1e-10)
#   ooc_dir    : directory for MUMPS temporary OOC files
#   icntl_14   : MUMPS extra working memory headroom in %  (default 80)
#   icntl_23   : MUMPS max RAM per MPI process in MB
#
# Usage:
#   mpirun -np 10 python eigensolver.py
# =============================================================================

import os
import time
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from petsc4py import PETSc
from slepc4py import SLEPc


# =============================================================================
# 1. Read the Jacobian
# =============================================================================

def read_jacobian(path):
    """Load sparse Jacobian from scipy .npz and cast to PETSc scalar type."""
    J = sp.load_npz(path)
    J = J.tocsr().astype(PETSc.ScalarType)
    print(f"Loaded Jacobian: shape={J.shape}, nnz={J.nnz}, "
          f"density={J.nnz / (J.shape[0]*J.shape[1]):.2e}")
    return J


def scipy_csr_to_petsc(J_csr):
    Mat = PETSc.Mat().createAIJ(size=J_csr.shape,
                                 csr=(J_csr.indptr,
                                      J_csr.indices,
                                      J_csr.data))
    Mat.assemble()
    return Mat


# =============================================================================
# 2. Solve with shift-invert + MUMPS OOC
# =============================================================================

def solve_shift_invert(J, sigma, nev=10, ncv=None, tol=1e-10, max_it=2000,
                       ooc_dir="/scratch/mumps_ooc",
                       icntl_14=80,
                       icntl_23=12000):
    """
    Solve eigenproblem via Krylov-Schur shift-invert with MUMPS OOC.

    Parameters
    ----------
    sigma     : complex shift
    nev       : number of eigenvalues requested
    ncv       : Krylov subspace size (default: max(3*nev, 30))
    tol       : convergence tolerance
    max_it    : maximum number of iterations
    ooc_dir   : directory for MUMPS OOC temporary files
    icntl_14  : MUMPS extra working memory headroom (%)
    icntl_23  : MUMPS max RAM per MPI process (MB)
                set to total_RAM_MB / np
    """
    n = J.getSize()[0]
    if ncv is None:
        ncv = min(n, max(3 * nev, 30))

    # ── MUMPS out-of-core settings ────────────────────────────────────────────
    os.makedirs(ooc_dir, exist_ok=True)
    os.environ['MUMPS_OOC_TMPDIR'] = ooc_dir
    opts = PETSc.Options()
    opts['mat_mumps_icntl_22'] = 1          # enable OOC
    opts['mat_mumps_icntl_14'] = icntl_14   # extra working memory headroom (%)
    opts['mat_mumps_icntl_23'] = icntl_23   # max RAM per process (MB)

    # ── Eigensolver setup ─────────────────────────────────────────────────────
    E = SLEPc.EPS().create()
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
        pc.setFactorSolverType('mumps')
        solver_used = 'mumps (OOC enabled)'
    except PETSc.Error:
        pc.setFactorSolverType('petsc')
        solver_used = 'petsc built-in (no MUMPS found)'

    E.setTarget(sigma)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setFromOptions()

    print(f"Solving: sigma={sigma}, nev={nev}, ncv={ncv}, "
          f"solver={solver_used}")
    print(f"MUMPS OOC dir : {ooc_dir}")
    print(f"icntl_14={icntl_14}%  icntl_23={icntl_23} MB/process")
    E.solve()
    return E


# =============================================================================
# 3. Extract and report results
# =============================================================================

def report_results(E, J, nev, residual_tol=1e-6):
    nconv         = E.getConverged()
    nev_requested = E.getDimensions()[0]
    print(f"\nConverged eigenpairs: {nconv} / {nev_requested} requested")
    if nconv < nev_requested:
        print("  WARNING: fewer eigenpairs converged than requested.")

    vr, vi  = J.createVecs()
    results = []
    for i in range(nconv):
        val = E.getEigenpair(i, vr, vi)
        err = E.computeError(i)
        results.append({
            'eigenvalue': val,
            'residual':   err,
            'vec_real':   vr.getArray().copy(),
            'vec_imag':   vi.getArray().copy(),
        })

    print(f"\n{'#':>3} {'Re(λ)':>14} {'Im(λ)':>14} {'residual':>12}  status")
    print("-" * 66)
    for i, r in enumerate(results[:nev]):
        lam = r['eigenvalue']
        err = r['residual']
        if err > residual_tol:
            status = "UNRELIABLE"
        elif lam.real > 0:
            status = "UNSTABLE"
        elif abs(lam.real) < 1e-3:
            status = "MARGINAL"
        else:
            status = "stable"
        print(f"{i:>3} {lam.real:>14.6f} {lam.imag:>14.6f} "
              f"{err:>12.2e}  {status}")
    return results


def print_eigenvalues(results, residual_tol=1e-6):
    print(f"\n{len(results)} converged eigenvalues:")
    print(f"{'#':>3} {'Re(λ)':>14} {'Im(λ)':>14} {'residual':>12}  status")
    print("-" * 66)
    for i, r in enumerate(results):
        lam = r['eigenvalue']
        err = r['residual']
        if err > residual_tol:     status = "UNRELIABLE"
        elif lam.real > 0:         status = "UNSTABLE"
        elif abs(lam.real) < 1e-3: status = "MARGINAL"
        else:                      status = "stable"
        print(f"{i:>3} {lam.real:>14.6f} {lam.imag:>14.6f} "
              f"{err:>12.2e}  {status}")


# =============================================================================
# 4. Save eigenpairs
# =============================================================================

def save_eigendata(E, J, sigma, nev, ncv, out_path):
    """Save eigenvalues, eigenvectors, residuals and runtime to .npz."""
    nconv = E.getConverged()
    N     = J.getSize()[0]
    vr, vi = J.createVecs()

    eigenvalues  = np.zeros(nconv, dtype=complex)
    eigenvectors = np.zeros((nconv, N), dtype=complex)
    residuals    = np.zeros(nconv)

    for i in range(nconv):
        val               = E.getEigenpair(i, vr, vi)
        eigenvalues[i]    = val
        residuals[i]      = E.computeError(i)
        eigenvectors[i,:] = vr.getArray() + 1j * vi.getArray()

    np.savez(out_path,
             eigenvalues  = eigenvalues,
             eigenvectors = eigenvectors,
             residuals    = residuals,
             sigma        = np.array([sigma]),
             nev          = nev,
             ncv          = ncv)
    print(f"Saved {nconv} eigenpairs → {out_path}")


# =============================================================================
# 5. Plot eigenspectrum
# =============================================================================

def plot_eigenspectrum(results, residual_tol=1e-6,
                       save_path='eigenspectrum.png'):
    eigs      = np.array([r['eigenvalue'] for r in results])
    residuals = np.array([r['residual']   for r in results])
    reliable  = residuals <= residual_tol

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axvline(0, color='black', lw=1.0)
    ax.axvspan(eigs.real.min() - 0.1, 0, color='tab:blue', alpha=0.06)
    ax.axvspan(0, eigs.real.max() + 0.1, color='tab:red',  alpha=0.06)
    ax.scatter(eigs.real[ reliable],  eigs.imag[ reliable],
               c='tab:blue', s=45, edgecolor='white', lw=0.5,
               label='converged', zorder=3)
    if (~reliable).any():
        ax.scatter(eigs.real[~reliable], eigs.imag[~reliable],
                   c='gray', s=45, marker='x',
                   label='residual above tol', zorder=3)
    idx_lead = np.argmax(eigs.real)
    ax.annotate(f"λ={eigs[idx_lead].real:.4f}+{eigs[idx_lead].imag:.4f}j",
                xy=(eigs[idx_lead].real, eigs[idx_lead].imag),
                xytext=(10, 10), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Re(λ)  (growth rate)')
    ax.set_ylabel('Im(λ)  (frequency rad/s)')
    ax.set_title('Eigenspectrum')
    ax.legend(fontsize=8);  ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


# =============================================================================
# 6. Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Shift-invert eigensolver with MUMPS OOC")
    p.add_argument("--jacobian",   type=str,   required=True,       help="Path to Jacobian .npz")
    p.add_argument("--out-dir",    type=str,   default="./data",    help="Output directory")
    p.add_argument("--sigma-real", type=float, default=0.0,         help="Re(sigma)")
    p.add_argument("--sigma-imag", type=float, default=59.72,       help="Im(sigma)")
    p.add_argument("--nev",        type=int,   default=10,          help="Number of eigenvalues")
    p.add_argument("--ncv",        type=int,   default=None,        help="Krylov subspace size")
    p.add_argument("--tol",        type=float, default=1e-10,       help="Convergence tolerance")
    p.add_argument("--max-it",     type=int,   default=2000,        help="Max iterations")
    p.add_argument("--ooc-dir",    type=str,   default="/scratch/mumps_ooc",
                   help="MUMPS OOC temp directory (point to large disk)")
    p.add_argument("--icntl-14",   type=int,   default=80,          help="MUMPS working memory headroom %%")
    p.add_argument("--icntl-23",   type=int,   default=12000,       help="MUMPS max RAM per process (MB)")
    p.add_argument("--mesh",       type=int,   default=0)
    p.add_argument("--re",         type=float, default=60.0)
    p.add_argument("--mach",       type=float, default=0.2)
    return p.parse_args()


def main():
    args  = parse_args()
    sigma = args.sigma_real + 1j * args.sigma_imag
    nev   = args.nev
    ncv   = args.ncv

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Shift-invert eigensolver")
    print(f"  Jacobian : {args.jacobian}")
    print(f"  sigma    : {sigma}")
    print(f"  nev={nev}  ncv={ncv if ncv else 'auto'}  tol={args.tol}")
    print(f"  OOC dir  : {args.ooc_dir}")
    print(f"  icntl_14 : {args.icntl_14}%")
    print(f"  icntl_23 : {args.icntl_23} MB/process")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    J_csr = read_jacobian(args.jacobian)
    J     = scipy_csr_to_petsc(J_csr)

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    E       = solve_shift_invert(J, sigma,
                                  nev      = nev,
                                  ncv      = ncv,
                                  tol      = args.tol,
                                  max_it   = args.max_it,
                                  ooc_dir  = args.ooc_dir,
                                  icntl_14 = args.icntl_14,
                                  icntl_23 = args.icntl_23)
    runtime = time.perf_counter() - t0
    print(f"\nRuntime: {runtime:.1f}s")

    # ── Report ────────────────────────────────────────────────────────────────
    results = report_results(E, J, nev)
    ncv_used = E.getDimensions()[1]

    # ── Save ──────────────────────────────────────────────────────────────────
    tag      = f"_nev{nev}_ncv{ncv_used}_sigR{args.sigma_real:+.0f}"
    out_path = os.path.join(args.out_dir,
                             f"eigendata_{args.mesh}_Re{int(args.re)}"
                             f"_M{args.mach}{tag}.npz")
    save_eigendata(E, J, sigma, nev, ncv_used, out_path)

    # ── Save runtime into the same file ───────────────────────────────────────
    d = dict(np.load(out_path, allow_pickle=True))
    d['runtime_s'] = runtime
    np.savez(out_path, **d)

    plot_eigenspectrum(results,
                       save_path=out_path.replace('.npz', '_spectrum.png'))


if __name__ == "__main__":
    main()

# nohup mpirun -np 10 python -u eigensolver_v2.py \
#     --jacobian  "/home/ahf25/git/flux_jacobian/data/jacobian_1500000.npz" \
#     --out-dir   "/home/ahf25/git/flux_jacobian/data/eigendata" \
#     --sigma-real 0.0 \
#     --sigma-imag 59.72 \
#     --nev        10 \
#     --ncv        30 \
#     --ooc-dir   "/scratch/mumps_ooc" \
#     --icntl-14   80 \
#     --icntl-23   12000 \
#     --mesh       1500000 \
#     --re         60 \
#     --mach       0.2 \
#     > eigensolver.log 2>&1 &