# flux_jacobian

Flux-Jacobian eigenvalue analysis for 2D CFD cases (cylinder wakes, OAT15 airfoil) —
a stability-analysis pipeline that assembles a global flux Jacobian, then finds its eigenvalues/eigenmodes (SLEPc/PETSc shift-invert) to study flow instabilities.

## Pipeline

```
flux_jacobians/vN  --[assemble Jacobian]-->  jacobian_*.npz
jacobian_*.npz     --[eigensolver/vN]-->     eigendata_*.npz
eigendata_*.npz    --[notebooks/eigen_post.ipynb]-->  results/figures/
```

Raw CFD case inputs and the intermediate `.npz` data for assembled global flux jacobians and eigenvalues are not in the repo.

## Directory structure (tracked files only)

```
notebooks/          Active analysis notebooks: mesh/BC prep, CFD post-processing, eigen post-processing
flux_jacobians/     Flux-Jacobian assembly notebooks/scripts, versioned v1-v5
eigensolver/        Eigenvalue solver notebooks/scripts, versioned v1-v3
results/figures/    Output plots (SVG/PNG)
archive/notebooks/  Superseded notebook(s) kept for reference
```

### `notebooks/`
The active, working notebooks for the pipeline:
- `mesh_convert.ipynb` — mesh conversion
- `boundary_list.ipynb` — boundary node list generation
- `eigen_solver.ipynb` — eigenvalue solver (notebook front-end)
- `eigen_post.ipynb` — eigenvalue post-processing
- `plot_flux_jacobian_v2.ipynb` — Jacobian plotting
- `plot_sr.ipynb` — spectral radius plotting
- `verifications_flux_jacobian_v4.ipynb` — verification checks
- `cfd_post/` — CFD post-processing: `cfd_post.ipynb`, `plots.ipynb`, `plot_residuals.ipynb`,
  `unsteady.ipynb`, `unsteady_cl_plot.ipynb`, `unsteady_animation_v2.ipynb`/`_v3.ipynb`
- `testing/` — 1D/2D test notebooks (`1D_test.ipynb`, `2D_test.ipynb`) and associated `.mtx` test matrices

### `flux_jacobians/`
Notebooks/scripts that assemble the global flux Jacobian (inviscid/viscous/boundary contributions),
grouped by pipeline version:
- `v1/` — original assembly + boundary/viscous/inviscid Jacobian notebooks (analytic and FD variants)
- `v2/`, `v3/` — FD assembly + boundary/viscous notebooks, refactored per version
- `v4/` — FD assembly notebook plus `assemble_jacobian_v4.py`
- `v5/` — current version: FD assembly, inviscid Jacobian notebook, `assemble_jacobian_v5.py`

### `eigensolver/`
Eigenvalue solver notebooks/scripts (SLEPc/PETSc shift-invert), versioned:
- `v1/` — `eigensolver.ipynb`, `eigensolver_mpi.py`
- `v2/` — `eigensolver_v2.ipynb`, `eigensolver_v2.py`
- `v3/` — `eigensolver_mpi_v3.py`
