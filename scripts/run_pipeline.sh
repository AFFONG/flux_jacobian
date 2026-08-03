#!/usr/bin/env bash
#
# run_pipeline.sh
# ================
# Chains assemble_jacobian.py -> eigensolver.py for one CFD case, and logs
# both steps to a timestamped log file. This is a template: edit the
# "USER CONFIG" block below per case/run (or copy this file per case),
# then run:
#
#     bash run_pipeline.sh
#
# The only artifacts written are the two .npz files (jacobian + eigendata)
# plus a .log file next to them -- nothing else.
#
# Requires: the conda environment named below must have both toolchains
# installed -- pyau3d/numpy/scipy/pandas for the assembly step, and
# petsc4py/slepc4py (complex-scalar build) for the eigensolver step.
# MAKE SURE I SET UP petsc4py/slepc4py (complex-scalar build) , scipy in my env

set -euo pipefail

# --------------------------------------------------------------------------
# USER CONFIG -- edit per case
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="flux-jacobian"                                # <-- set to your actual conda env name

CASE_DIR="/path/to/cases/cylinder_21228_Re60_M0.2"      # raw CFD case, outside this repo
BC_CONFIG="$SCRIPT_DIR/example_bc_config.json"           # copy + edit per case
OUT_DIR="/path/to/data/cylinder_21228_Re60_M0.2"         # where the two npz files land
TAG="cylinder_21228_Re60_M0.2"                           # used only to name the output files below

VISCOUS=true            # true -> pass --viscous to assemble_jacobian.py
EPS_VISC=1e-8
EPS_GHOST=1e-6

FREQ_HZ=9.505            # target shedding frequency guess (Hz) -> sigma_imag = 2*pi*FREQ_HZ
SIGMA_REAL=0.0
NEV=10
NCV=300

JACOBIAN_OUT="$OUT_DIR/jacobian_${TAG}_fd.npz"
EIGEN_OUT="$OUT_DIR/eigendata_${TAG}_nev${NEV}_ncv${NCV}.npz"

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# Activate conda inside a non-interactive shell. `conda activate` on its own
# won't work here without sourcing conda's shell hook first.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

{
    echo "=== conda env: $CONDA_ENV ($(which python3)) ==="
    echo "=== assemble_jacobian.py :: $(date) ==="
    VISCOUS_FLAG=()
    if [ "$VISCOUS" = true ]; then
        VISCOUS_FLAG=(--viscous)
    fi

    python3 "$SCRIPT_DIR/assemble_jacobian.py" \
        --case-dir "$CASE_DIR" \
        --bc-config "$BC_CONFIG" \
        --eps-visc "$EPS_VISC" \
        --eps-ghost "$EPS_GHOST" \
        "${VISCOUS_FLAG[@]}" \
        --out "$JACOBIAN_OUT"

    echo "=== eigensolver.py :: $(date) ==="
    python3 "$SCRIPT_DIR/eigensolver.py" \
        --jacobian "$JACOBIAN_OUT" \
        --freq-hz "$FREQ_HZ" \
        --sigma-real "$SIGMA_REAL" \
        --nev "$NEV" \
        --ncv "$NCV" \
        --out "$EIGEN_OUT"

    echo "=== done :: $(date) ==="
    echo "Jacobian: $JACOBIAN_OUT"
    echo "Eigendata: $EIGEN_OUT"
} 2>&1 | tee "$LOG_FILE"
