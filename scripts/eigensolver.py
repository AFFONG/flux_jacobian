#!/usr/bin/env python3
"""
eigensolver.py
===============
Standalone CLI version of your eigensolver_v2 notebook (SLEPc/PETSc
shift-invert stability solver). The numerical routines below are copied
as-is from that notebook -- only the hardcoded case parameters at the
bottom have been replaced with an argparse CLI, and interactive plotting
(plt.show()) has been made optional/non-blocking so this runs headless
over SSH/tmux/cron.

Requires the complex-scalar PETSc/SLEPc build (petsc-complex env), since
sigma is a general complex number here -- same requirement as your
original notebook.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Basic run, shift at 0 + i*2*pi*9.505 (i.e. targeting a 9.505 Hz mode):

    python3 eigensolver.py \
        --jacobian /path/to/data/jacobian_cylinder_21228_Re60_M0.2_fd.npz \
        --freq-hz 9.505 \
        --nev 10 --ncv 300 \
        --out /path/to/data/eigendata_21228_Re60_M0.2_nev10_ncv300.npz

Equivalent, giving sigma directly instead of a frequency:

    python3 eigensolver.py \
        --jacobian jacobian.npz \
        --sigma-real 0.0 --sigma-imag 59.7154 \
        --nev 10 --ncv 300 \
        --out eigendata.npz

If --out is omitted, it is derived from the input filename, e.g.
"jacobian_cylinder_21228_Re60_M0.2_fd.npz" -> in the same directory,
"eigendata_cylinder_21228_Re60_M0.2_fd_nev10_ncv300.npz".

For parallel factorization (e.g. MUMPS across ranks), launch under MPI:

    mpirun -n 4 python3 eigensolver.py --jacobian ... --out ...

Run `python3 eigensolver.py --help` for the full flag list.

--------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------
By default this script writes exactly one artifact: the eigendata .npz
(eigenvalues, eigenvectors, residuals, sigma, nev, ncv used) at --out.
Pass --plot-spectrum to additionally save a PNG of the eigenspectrum next
to --out (off by default to keep the pipeline artifact-minimal).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Headless-safe: never try to open a display / block on plt.show().
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from petsc4py import PETSc
from slepc4py import SLEPc


# =============================================================================
# 1. Read the Jacobian
# =============================================================================

def read_jacobian(path):
    """Load a sparse Jacobian saved via scipy.sparse.save_npz.
    Returns a CSR matrix cast to PETSc's scalar type (complex128
    in this environment)."""
    J = sp.load_npz(path)
    J = J.tocsr().astype(PETSc.ScalarType)
    print(f"Loaded Jacobian: shape={J.shape}, nnz={J.nnz}, "
          f"density={J.nnz / (J.shape[0] * J.shape[1]):.2e}")
    return J


def scipy_csr_to_petsc(J_csr):
    Mat = PETSc.Mat().createAIJ(size=J_csr.shape,
                                 csr=(J_csr.indptr, J_csr.indices, J_csr.data))
    Mat.assemble()
    return Mat


# =============================================================================
# 2. Solve with shift-invert
# =============================================================================

def solve_shift_invert(J, sigma, nev=10, ncv=None, tol=1e-10, max_it=2000):
    """
    sigma  : complex shift -- place this near where you expect the
             largest-growth-rate / marginal eigenvalues to sit
             (e.g. 0+0j, or 0 + 1j*omega_guess if you have a frequency
             estimate from a prior run or physical intuition).
    nev    : number of eigenvalues requested.
    ncv    : Krylov subspace size. Larger = more robust convergence,
             especially for clustered eigenvalues near a Hopf crossing,
             at the cost of more memory/compute per iteration. A common
             rule of thumb is ncv >= 2*nev, with more headroom (3-4x)
             if you expect closely spaced eigenvalues. Default here
             picks max(3*nev, 30), capped at the matrix dimension.
    """
    n = J.getSize()[0]
    if ncv is None:
        ncv = min(n, max(3 * nev, 30))

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
        solver_used = 'mumps'
    except PETSc.Error:
        pc.setFactorSolverType('petsc')
        solver_used = 'petsc (built-in, no mumps found)'

    E.setTarget(sigma)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setFromOptions()

    print(f"Solving: sigma={sigma}, nev={nev}, ncv={ncv}, "
          f"factorization={solver_used}")
    E.solve()
    return E


# =============================================================================
# 3. Extract and report results
# =============================================================================

def report_results(E, J, nev, residual_tol=1e-6):
    nconv = E.getConverged()
    nev_requested = E.getDimensions()[0]

    print(f"\nConverged eigenpairs: {nconv} / {nev_requested} requested")
    if nconv < nev_requested:
        print("  WARNING: fewer eigenpairs converged than requested.")
        print("  Consider: increasing ncv, increasing max_it, or checking")
        print("  whether sigma is placed sensibly relative to the spectrum.")

    vr, vi = J.createVecs()
    results = []
    for i in range(nev):
        val = E.getEigenpair(i, vr, vi)
        err = E.computeError(i)
        results.append({
            'eigenvalue': val,
            'residual': err,
            'vec_real': vr.getArray().copy(),
            'vec_imag': vi.getArray().copy(),
        })

    top_results = results[:nev]

    print(f"\n{'#':>3} {'Re(lambda)':>14} {'Im(lambda)':>14} {'residual':>12}  status")
    print("-" * 66)
    for i, r in enumerate(top_results):
        lam = r['eigenvalue']
        err = r['residual']
        if err > residual_tol:
            status = "UNRELIABLE (residual above tol)"
        elif lam.real > 0:
            status = "UNSTABLE"
        elif abs(lam.real) < 1e-3:
            status = "MARGINAL (near Re=0 -- check carefully)"
        else:
            status = "stable"
        print(f"{i:>3} {lam.real:>14.6f} {lam.imag:>14.6f} {err:>12.2e}  {status}")

    return top_results


def print_eigenvalues(results, residual_tol=1e-6):
    """Print ALL converged eigenvalues (not just top nev)."""
    print(f"\n{len(results)} eigenvalues:")
    print(f"{'#':>3} {'Re(lambda)':>14} {'Im(lambda)':>14} {'residual':>12}  status")
    print("-" * 66)
    for i, r in enumerate(results):
        lam = r['eigenvalue']
        err = r['residual']
        if err > residual_tol:
            status = "UNRELIABLE (residual above tol)"
        elif lam.real > 0:
            status = "UNSTABLE"
        elif abs(lam.real) < 1e-3:
            status = "MARGINAL (near Re=0)"
        else:
            status = "stable"
        print(f"{i:>3} {lam.real:>14.6f} {lam.imag:>14.6f} {err:>12.2e}  {status}")


def plot_eigenspectrum(results, residual_tol=1e-6, save_path='eigenspectrum.png'):
    """Save (not show -- headless) the eigenspectrum on the complex plane."""
    eigs = np.array([r['eigenvalue'] for r in results])
    residuals = np.array([r['residual'] for r in results])
    reliable = residuals <= residual_tol

    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.axvspan(min(eigs.real.min(), -0.1) - 0.05, 0, color='tab:blue', alpha=0.06)
    ax.axvspan(0, max(eigs.real.max(), 0.1) + 0.05, color='tab:red', alpha=0.06)
    ax.axvline(0, color='black', lw=1.0)

    ax.scatter(eigs.real[reliable], eigs.imag[reliable],
               c='tab:blue', s=45, edgecolor='white', linewidth=0.5,
               label='converged (residual OK)', zorder=3)
    if (~reliable).any():
        ax.scatter(eigs.real[~reliable], eigs.imag[~reliable],
                   c='gray', s=45, marker='x',
                   label='residual above tolerance -- do not trust', zorder=3)

    idx_lead = np.argmax(eigs.real)
    ax.annotate(f'lambda = {eigs[idx_lead].real:.4f} + {eigs[idx_lead].imag:.4f}j',
                xy=(eigs[idx_lead].real, eigs[idx_lead].imag),
                xytext=(10, 10), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Re(lambda)  (growth rate)')
    ax.set_ylabel('Im(lambda)  (frequency, rad/s or non-dim)')
    ax.set_title('Eigenspectrum of the global flux Jacobian')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_eigendata(E, J, sigma, out_path='eigendata.npz'):
    """
    Saves:
      eigenvalues : complex array, shape (nconv,)
      eigenvectors: complex array, shape (nconv, N) -- each row is one
                    eigenvector, full length N (matches Jacobian dimension)
      residuals   : real array, shape (nconv,)
      sigma       : the complex shift used for this solve
      nev, ncv    : solver settings used, for reproducibility
    """
    nconv = E.getConverged()
    N = J.getSize()[0]
    nev_requested, ncv_used, _ = E.getDimensions()

    vr, vi = J.createVecs()
    eigenvalues = np.zeros(nconv, dtype=complex)
    eigenvectors = np.zeros((nconv, N), dtype=complex)
    residuals = np.zeros(nconv)

    for i in range(nconv):
        val = E.getEigenpair(i, vr, vi)
        eigenvalues[i] = val
        residuals[i] = E.computeError(i)
        eigenvectors[i, :] = vr.getArray() + 1j * vi.getArray()

    np.savez(out_path,
             eigenvalues=eigenvalues,
             eigenvectors=eigenvectors,
             residuals=residuals,
             sigma=np.array([sigma]),
             nev=nev_requested,
             ncv=ncv_used)

    print(f"Saved {nconv} eigenpairs to {out_path}")
    print(f"  eigenvalues.shape  = {eigenvalues.shape}")
    print(f"  eigenvectors.shape = {eigenvectors.shape}  (row i = eigenvector for eigenvalues[i])")


def print_interpretation_guide():
    print("""
--------------------------------------------------------------------
How to interpret this output:

1. Re(lambda) > 0  -> that mode grows in time -> globally unstable.
   Re(lambda) < 0  -> decays -> stable.
   Re(lambda) ~ 0  -> marginal; this is the Hopf-relevant regime.

2. A genuine Hopf bifurcation shows up as a COMPLEX-CONJUGATE PAIR
   (nonzero Im(lambda), and you should see its conjugate elsewhere
   in the list or in a re-run with wider nev) with Re(lambda) crossing
   zero as you vary your control parameter (Reynolds number). A real
   eigenvalue crossing zero alone would indicate a different
   (steady/pitchfork) bifurcation, not Hopf.

3. Trust ONLY eigenpairs with residual comfortably below your solver
   tolerance (1e-6 to 1e-8 is typical). An eigenvalue with residual
   above tolerance is not converged -- don't draw physical conclusions
   from it, especially near Re(lambda)=0 where you need real precision.

4. If nconv < nev requested: SLEPc could not converge everything you
   asked for within max_it iterations at this ncv. Don't assume the
   unconverged ones don't exist -- widen ncv/max_it and re-run before
   concluding anything about the missing eigenvalues.

5. Sanity checks before trusting a Hopf conclusion:
   - Increase ncv and re-run: do the top eigenvalues change more than
     your tolerance? If yes, ncv was too small.
   - Nudge sigma slightly and re-run: do the same eigenvalues reappear?
     If eigenvalues disappear/appear with small sigma changes, you may
     be missing modes near the edge of what shift-invert "saw".
   - If you have a mesh-refinement study available, confirm the
     leading eigenvalue's real part doesn't move significantly under
     refinement -- a Hopf point that moves with mesh resolution isn't
     trustworthy yet.
--------------------------------------------------------------------
""")


# =============================================================================
# CLI entry point
# =============================================================================

def default_out_path(jacobian_path: Path, nev: int, ncv: int) -> Path:
    """
    Derive an eigendata output path from the jacobian filename, mirroring
    your existing data/eigendata naming convention, e.g.:
        jacobian_cylinder_21228_Re60_M0.2_fd.npz
            -> eigendata_cylinder_21228_Re60_M0.2_fd_nev10_ncv300.npz
    """
    stem = jacobian_path.stem
    if stem.startswith('jacobian_'):
        stem = stem[len('jacobian_'):]
    name = f"eigendata_{stem}_nev{nev}_ncv{ncv}.npz"
    return jacobian_path.parent / name


def parse_args():
    p = argparse.ArgumentParser(
        description="Solve for eigenvalues/eigenvectors of a global flux Jacobian near a target shift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--jacobian', required=True, type=Path,
                    help="Path to the Jacobian .npz (scipy CSR, from assemble_jacobian.py).")

    sigma_group = p.add_mutually_exclusive_group()
    sigma_group.add_argument('--sigma-imag', type=float, default=None,
                              help="Imaginary part of the shift sigma (rad/s). Mutually exclusive with --freq-hz.")
    sigma_group.add_argument('--freq-hz', type=float, default=None,
                              help="Target frequency in Hz; sets sigma_imag = 2*pi*freq_hz. "
                                   "Mutually exclusive with --sigma-imag.")
    p.add_argument('--sigma-real', type=float, default=0.0,
                    help="Real part of the shift sigma (growth rate guess, default 0.0).")

    p.add_argument('--nev', type=int, default=10,
                    help="Number of eigenvalues to request (default 10).")
    p.add_argument('--ncv', type=int, default=None,
                    help="Krylov subspace size (default: max(3*nev, 30), capped at matrix size).")
    p.add_argument('--tol', type=float, default=1e-10,
                    help="SLEPc convergence tolerance (default 1e-10).")
    p.add_argument('--max-it', type=int, default=2000,
                    help="Max SLEPc iterations (default 2000).")
    p.add_argument('--residual-tol', type=float, default=1e-6,
                    help="Residual threshold for the console 'reliable' flag (default 1e-6).")

    p.add_argument('--out', type=Path, default=None,
                    help="Output path for eigendata .npz. If omitted, derived from --jacobian, --nev, --ncv.")
    p.add_argument('--plot-spectrum', action='store_true',
                    help="Also save a PNG of the eigenspectrum next to --out (off by default).")
    p.add_argument('--quiet', action='store_true',
                    help="Suppress the printed interpretation guide.")

    args = p.parse_args()
    if args.sigma_imag is not None:
        sigma_imag = args.sigma_imag
    elif args.freq_hz is not None:
        sigma_imag = 2 * np.pi * args.freq_hz
    else:
        sigma_imag = 0.0
    args.sigma = args.sigma_real + sigma_imag * 1j
    return args


def main():
    args = parse_args()

    out_path = args.out or default_out_path(args.jacobian, args.nev,
                                             args.ncv if args.ncv is not None else max(3 * args.nev, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    J_csr = read_jacobian(args.jacobian)
    J = scipy_csr_to_petsc(J_csr)

    t0 = time.perf_counter()
    E = solve_shift_invert(J, sigma=args.sigma, nev=args.nev, ncv=args.ncv,
                            tol=args.tol, max_it=args.max_it)
    t = time.perf_counter() - t0

    top_results = report_results(E, J, args.nev, residual_tol=args.residual_tol)
    print(f"Run time: {t:.2f}s")

    save_eigendata(E, J, sigma=args.sigma, out_path=str(out_path))

    if args.plot_spectrum:
        png_path = out_path.with_suffix('.png')
        plot_eigenspectrum(top_results, residual_tol=args.residual_tol, save_path=str(png_path))

    if not args.quiet:
        print_interpretation_guide()


if __name__ == '__main__':
    sys.exit(main())
