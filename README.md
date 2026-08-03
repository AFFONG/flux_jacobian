# flux_jacobian

Flux-Jacobian eigenvalue analysis for 2D cylinder flow cases (CFD stability analysis pipeline).

## Directory structure

```
cases/            Raw CFD solver run directories (mesh, solution, restart files) — one per case
notebooks/        Active analysis notebooks (pipeline code)
flux_jacobians/    Flux-Jacobian assembly notebooks, versioned v1/v2/v3 (code, not data)
data/              Generated data artifacts (Jacobian matrices, eigendata) produced by the notebooks
results/figures/   Output plots (SVG/PNG)
archive/           Superseded notebook versions and old data no longer in active use
```

### `cases/`
One folder per CFD run, named `cylinder_<mesh_nodes>_Re<Reynolds>_M<Mach>` (mesh node count omitted
where the case predates that convention, e.g. `cylinder_Re40`, `cylinder_Re50`). Contains raw solver
input/output files (`cylinder.*`, `.rst`, `.sol`, `.plt`, logs, etc.).

### `notebooks/`
The active, working notebooks for the pipeline:
- `mesh_convert.ipynb` — mesh conversion
- `boundary_list.ipynb` — boundary node list generation
- `cfd_post.ipynb`, `u_profile.ipynb` — CFD post-processing
- `eigen_solver.ipynb` — eigenvalue solver
- `eigen_post.ipynb` — eigenvalue post-processing
- `plot_flux_jacobian.ipynb` — Jacobian plotting
- `verifications_flux_jacobian.ipynb` — verification checks
- `testing/` — 1D/2D test notebooks (`1D_test.ipynb`, `2D_test.ipynb`) and associated `.mtx` test matrices

Each of these is the latest version of its notebook; earlier iterations live in `archive/notebooks/`
and `archive/testing/`.

### `flux_jacobians/`
Notebooks that assemble the global flux Jacobian (inviscid/viscous/boundary contributions), grouped by
pipeline version (`v1`, `v2`, `v3`). This is **code**, distinct from `data/flux_jacobian_assembly_*/`
below, which holds the data those notebooks produce.

### `data/`
Generated artifacts, not raw CFD output:
- `flux_jacobian_assembly_v1.2/`, `v2/`, `v3/` — assembled Jacobian matrices (`.npz`) from each pipeline
  version (`v1`, the earliest/largest run, has been moved to `archive/data/`)
- `eigendata/` — eigenvalue/eigenmode solver output (`.npz`)

### `results/figures/`
Rendered plots: flux-Jacobian sensitivity SVGs (`Re60_M0.2_*.svg`) and the `u_profile_combined.png`
velocity profile figure.

### `archive/`
Superseded material kept for reference rather than deleted:
- `archive/data/flux_jacobian_assembly_v1/` — the original (5GB) assembly run, superseded by v1.2/v2/v3
- `archive/notebooks/` — prior versions of the root notebooks (`*_v1.ipynb`)
- `archive/testing/` — prior versions of the 1D/2D test notebooks (`*_v1.ipynb` through `*_v3.ipynb`)

## Known issue

Several active notebooks still contain hardcoded relative paths to the pre-reorg locations
(e.g. `2d_cylinder_*`, `flux_jacobian_assembly_v2/`) and need updating to the new `cases/` and `data/`
paths above:
- `notebooks/eigen_post.ipynb`
- `notebooks/plot_flux_jacobian.ipynb`
- `notebooks/verifications_flux_jacobian.ipynb`
- `notebooks/boundary_list.ipynb`
- `notebooks/u_profile.ipynb`
