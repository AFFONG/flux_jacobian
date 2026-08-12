"""
=============================================================================
GLOBAL FLUX JACOBIAN ASSEMBLER — FINITE DIFFERENCE VERSION (v4)
Self-contained: all dependencies included.

Updates from v1:
  (1) viscous_flux_jacobians_fd   — relative eps scaling  (default eps_visc  = 1e-8)
  (2) ghost_state_jacobian_fd     — relative eps scaling  (default eps_ghost = 1e-6)
  (3) boundary_flux_jacobian_fd   — exact non-frozen ghost chain rule:
                                    J_bc = A+ + A- @ dUg/dUi  (riemann)
  (4) assemble_global_jacobian_fd — both eps values exposed in signature

Updates from v2:
  (5) slip_wall_flux_jacobian     — exact analytic BC Jacobian (replaces FD/Roe):
                                    J_bc = dF/dV|_W @ dV/dU
                                    F_W = [0, P*nx, P*ny, P*nz, 0]  (u_n = 0)
  (6) dV_dU                       — primitive/conservative Jacobian (5x5, 3D)
  (7) assemble_global_jacobian_fd — Area scaling applied to all face contributions
                                    (fixes factor-of-2 error at corner BC nodes)

Updates from v3:
  (8) boundary_flux_jacobian_fd   — no-slip wall: non-frozen ghost chain rule:
                                    J_bc = dFv/dUi + dFv/dUg @ dUg/dUi
                                    dUg/dUi = diag([1,-1,-1,-1,1])  (exact, no FD)
  (9) extract_cell_volumes        — two-pass volume extraction: col 5 for i-nodes,
                                    geometric dual-mesh fallback for j-only nodes
 (10) assemble_global_jacobian_fd — volume normalisation now internal:
                                    J_phys = diag(1/Vol) @ J_raw  -> [1/s]
                                    controlled by normalise_volume=True (default)

Usage:
    python assemble_jacobian_v4.py --mesh 765960 --re 60 --mach 0.2
    python assemble_jacobian_v4.py --mesh 765960 --re 60 --mach 0.2 --viscous
    python assemble_jacobian_v4.py --help
=============================================================================
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import diags, lil_matrix, save_npz

from pyau3d.utils import PltFileUtils, UnkFileUtils


# =============================================================================
# SECTION 1 — Inviscid flux Jacobian building blocks
# =============================================================================

def compute_dFdU(U, normal, gamma=1.4):
    """Inviscid flux Jacobian dF/dU at state U in direction normal."""
    rho = U[0]
    rho_u, rho_v, rho_w, rho_E = U[1], U[2], U[3], U[4]
    nx, ny, nz = normal[0], normal[1], normal[2]
    u = rho_u / rho;  v = rho_v / rho;  w = rho_w / rho
    E = rho_E / rho
    phi = 0.5 * (gamma - 1.0) * (u**2 + v**2 + w**2)
    V   = nx*u + ny*v + nz*w
    a1  = gamma*E - phi
    a2  = gamma - 1.0
    a3  = gamma - 2.0
    A   = np.zeros((5, 5))
    A[0, 1] = nx;  A[0, 2] = ny;  A[0, 3] = nz
    A[1, 0] = nx*phi - u*V
    A[1, 1] = V - a3*nx*u;  A[1, 2] = ny*u - a2*nx*v;  A[1, 3] = nz*u - a2*nx*w;  A[1, 4] = a2*nx
    A[2, 0] = ny*phi - v*V
    A[2, 1] = nx*v - a2*ny*u;  A[2, 2] = V - a3*ny*v;  A[2, 3] = nz*v - a2*ny*w;  A[2, 4] = a2*ny
    A[3, 0] = nz*phi - w*V
    A[3, 1] = nx*w - a2*nz*u;  A[3, 2] = ny*w - a2*nz*v;  A[3, 3] = V - a3*nz*w;  A[3, 4] = a2*nz
    A[4, 0] = V * (phi - a1)
    A[4, 1] = a1*nx - a2*u*V;  A[4, 2] = a1*ny - a2*v*V
    A[4, 3] = a1*nz - a2*w*V;  A[4, 4] = gamma*V
    return A


def absolute_normal_jacobian(U, n, gamma=1.4):
    """Absolute-value normal Jacobian |A_n|."""
    U  = np.asarray(U, dtype=float)
    n  = np.asarray(n, dtype=float);  n = n / np.linalg.norm(n)
    rho = U[0];  vel = U[1:4] / rho;  E = U[4] / rho
    q2  = np.dot(vel, vel)
    p   = (gamma - 1.0) * rho * (E - 0.5*q2)
    c   = np.sqrt(gamma * p / rho)
    H   = E + p / rho
    qn  = np.dot(vel, n)
    M2  = q2 / c**2;  Mn = qn / c;  g1 = gamma - 1.0
    # |qn| contribution
    mid = np.zeros((5, 5))
    mid[0, 0]     =  1.0 - 0.5*g1*M2
    mid[0, 1:4]   =  (g1/c**2)*vel;  mid[0, 4] = -(g1/c**2)
    mid[1:4, 0]   = -0.5*g1*M2*vel + qn*n
    mid[1:4, 1:4] =  (g1/c**2)*np.outer(vel, vel) + np.eye(3) - np.outer(n, n)
    mid[1:4, 4]   = -(g1/c**2)*vel
    mid[4, 0]     =  qn**2 - 0.5*q2*(1.0 + 0.5*g1*M2)
    mid[4, 1:4]   =  (1.0 + 0.5*g1*M2)*vel - qn*n;  mid[4, 4] = -0.5*g1*M2
    # |qn - c| contribution
    l1 = np.empty(5)
    l1[0] = 0.25*g1*M2 + 0.5*Mn
    l1[1:4] = -(g1/(2*c**2))*vel - n/(2*c);  l1[4] = g1/(2*c**2)
    r1 = np.empty(5)
    r1[0] = 1.0;  r1[1:4] = vel - c*n;  r1[4] = H - qn*c
    A1 = np.outer(r1, l1)
    # |qn + c| contribution
    l3 = np.empty(5)
    l3[0] = 0.25*g1*M2 - 0.5*Mn
    l3[1:4] = -(g1/(2*c**2))*vel + n/(2*c);  l3[4] = g1/(2*c**2)
    r3 = np.empty(5)
    r3[0] = 1.0;  r3[1:4] = vel + c*n;  r3[4] = H + qn*c
    A3 = np.outer(r3, l3)
    return abs(qn - c)*A1 + abs(qn)*mid + abs(qn + c)*A3


def roe_average(UL, UR, gamma=1.4):
    """Roe-averaged conservative state."""
    UL = np.asarray(UL, dtype=float);  UR = np.asarray(UR, dtype=float)
    rhoL, rhoR = UL[0], UR[0]
    velL = UL[1:4]/rhoL;  velR = UR[1:4]/rhoR
    EL = UL[4]/rhoL;      ER  = UR[4]/rhoR
    pL = (gamma-1.0)*rhoL*(EL - 0.5*np.dot(velL, velL))
    pR = (gamma-1.0)*rhoR*(ER - 0.5*np.dot(velR, velR))
    HL = EL + pL/rhoL;  HR = ER + pR/rhoR
    wL = np.sqrt(rhoL);  wR = np.sqrt(rhoR);  ws = wL + wR
    rho_roe = wL * wR
    vel_roe = (wL*velL + wR*velR) / ws
    H_roe   = (wL*HL   + wR*HR)   / ws
    q2_roe  = np.dot(vel_roe, vel_roe)
    E_roe   = (H_roe + (gamma-1.0)*0.5*q2_roe) / gamma
    U_roe   = np.empty(5)
    U_roe[0] = rho_roe;  U_roe[1:4] = rho_roe*vel_roe;  U_roe[4] = rho_roe*E_roe
    return U_roe


def inviscid_flux_jacobians(U_i, U_j, n, gamma=1.4):
    """
    Roe-split inviscid flux Jacobian blocks.
    Returns (J_L, J_R) = (A+, A-) at the Roe-averaged state.
    """
    U_roe     = roe_average(U_i, U_j, gamma)
    abs_A_roe = absolute_normal_jacobian(U_roe, n, gamma=gamma)
    J_L = 0.5 * (compute_dFdU(U_i, n, gamma) + abs_A_roe)   # A+
    J_R = 0.5 * (compute_dFdU(U_j, n, gamma) - abs_A_roe)   # A-
    return J_L, J_R


# =============================================================================
# SECTION 2 — Viscous flux and Jacobian
# =============================================================================

def cons_to_prim(U, gamma, R_gas):
    """Conservative -> primitive variables."""
    rho = U[0]
    u, v, w = U[1]/rho, U[2]/rho, U[3]/rho
    E = U[4] / rho
    p = (gamma-1.0) * rho * (E - 0.5*(u*u + v*v + w*w))
    T = p / (rho * R_gas)
    return rho, u, v, w, p, T


def Fnv_numeric(U_i, U_j, n, ds,
                gamma=1.4, R_gas=287.0, Pr=0.72,
                mu0=1.716e-5, T0=273.15, Suth_C=110.4):
    """Numeric normal viscous flux F_n^v(U_i, U_j, n, ds)."""
    nx, ny, nz = n
    rho_L, u_L, v_L, w_L, p_L, T_L = cons_to_prim(U_i, gamma, R_gas)
    rho_R, u_R, v_R, w_R, p_R, T_R = cons_to_prim(U_j, gamma, R_gas)
    u_avg = (u_L+u_R)/2;  v_avg = (v_L+v_R)/2
    w_avg = (w_L+w_R)/2;  T_avg = (T_L+T_R)/2
    mu_avg = mu0 * (T0+Suth_C) / (T_avg+Suth_C) * (T_avg/T0)**1.5
    dudn = (u_R-u_L)/ds;  dvdn = (v_R-v_L)/ds
    dwdn = (w_R-w_L)/ds;  dTdn = (T_R-T_L)/ds
    div    = dudn*nx + dvdn*ny + dwdn*nz
    tau_xx = mu_avg*(2*dudn*nx - (2/3)*div)
    tau_yy = mu_avg*(2*dvdn*ny - (2/3)*div)
    tau_zz = mu_avg*(2*dwdn*nz - (2/3)*div)
    tau_xy = mu_avg*(dudn*ny + dvdn*nx)
    tau_xz = mu_avg*(dudn*nz + dwdn*nx)
    tau_yz = mu_avg*(dvdn*nz + dwdn*ny)
    tau_nx = tau_xx*nx + tau_xy*ny + tau_xz*nz
    tau_ny = tau_xy*nx + tau_yy*ny + tau_yz*nz
    tau_nz = tau_xz*nx + tau_yz*ny + tau_zz*nz
    tau_nn = tau_nx*u_avg + tau_ny*v_avg + tau_nz*w_avg
    kappa  = gamma * mu_avg / (Pr*(gamma-1.0))
    q_n    = -kappa * dTdn
    return np.array([0., -tau_nx, -tau_ny, -tau_nz, -tau_nn + q_n])


def viscous_flux_jacobians_fd(U_i, U_j, n, ds, eps,
                               gamma=1.4, R_gas=287.0, Pr=0.72,
                               mu0=1.716e-5, T0=273.15, Suth_C=110.4,
                               eps_abs=1e-14):
    """
    Forward FD viscous flux Jacobian with relative eps scaling.
    h_k = eps * max(|U[k]|, eps_abs)
    """
    U_i = np.asarray(U_i, dtype=float)
    U_j = np.asarray(U_j, dtype=float)
    F_base = Fnv_numeric(U_i, U_j, n, ds, gamma, R_gas, Pr, mu0, T0, Suth_C)
    J_i_fd = np.zeros((5, 5))
    J_j_fd = np.zeros((5, 5))
    for k in range(5):
        h_i = eps * max(abs(U_i[k]), eps_abs)
        h_j = eps * max(abs(U_j[k]), eps_abs)
        U_i_fwd = U_i.copy();  U_i_fwd[k] += h_i
        J_i_fd[:, k] = (Fnv_numeric(U_i_fwd, U_j, n, ds, gamma, R_gas, Pr, mu0, T0, Suth_C)
                         - F_base) / h_i
        U_j_fwd = U_j.copy();  U_j_fwd[k] += h_j
        J_j_fd[:, k] = (Fnv_numeric(U_i, U_j_fwd, n, ds, gamma, R_gas, Pr, mu0, T0, Suth_C)
                         - F_base) / h_j
    return J_i_fd, J_j_fd


# =============================================================================
# SECTION 3 — Ghost state and its Jacobian
# =============================================================================

def riemann_invariant_bc(U_int, n, gamma, u_b, v_b, w_b, T_b, P_b, R_gas=287.0):
    """Ghost cell state via Riemann invariant BC (4-regime selection)."""
    U_int = np.asarray(U_int, dtype=float)
    n     = np.asarray(n,     dtype=float);  n = n / np.linalg.norm(n)
    cv    = R_gas / (gamma - 1.0)
    rho_i = U_int[0];  vel_i = U_int[1:4]/rho_i;  E_i = U_int[4]/rho_i
    Vn_i  = np.dot(vel_i, n);  Vt_i = vel_i - Vn_i*n
    p_i   = (gamma-1.0) * rho_i * (E_i - 0.5*np.dot(vel_i, vel_i))
    T_i   = p_i / (rho_i*R_gas);  c_i = np.sqrt(gamma*R_gas*T_i)
    rho_b = P_b / (R_gas*T_b);  vel_b = np.array([u_b, v_b, w_b])
    Vn_b  = np.dot(vel_b, n);  Vt_b = vel_b - Vn_b*n
    c_b   = np.sqrt(gamma*R_gas*T_b)
    fac   = 2.0 / (gamma - 1.0)
    Rp_int = Vn_i + fac*c_i
    Rm_b   = Vn_b - fac*c_b
    Mn_i   = Vn_i / c_i
    if Mn_i <= -1.0:
        Vn_wall = Vn_b;  c_wall = c_b;  Vt_wall = Vt_b
        s_wall  = P_b / (rho_b**gamma)
    elif Mn_i >= 1.0:
        Vn_wall = Vn_i;  c_wall = c_i;  Vt_wall = Vt_i
        s_wall  = p_i / (rho_i**gamma)
    elif Vn_i < 0.0:
        Vn_wall = 0.5*(Rp_int + Rm_b);  c_wall = 0.25*(gamma-1.0)*(Rp_int - Rm_b)
        Vt_wall = Vt_b;  s_wall = P_b / (rho_b**gamma)
    else:
        Vn_wall = 0.5*(Rp_int + Rm_b);  c_wall = 0.25*(gamma-1.0)*(Rp_int - Rm_b)
        Vt_wall = Vt_i;  s_wall = p_i / (rho_i**gamma)
    if c_wall <= 0.0:
        raise ValueError(f"Non-physical c_wall={c_wall:.4f}")
    rho_wall = (c_wall**2 / (gamma*s_wall))**(1.0/(gamma-1.0))
    p_wall   = s_wall * rho_wall**gamma
    T_wall   = p_wall / (rho_wall*R_gas)
    vel_wall = Vn_wall*n + Vt_wall
    E_wall   = cv*T_wall + 0.5*np.dot(vel_wall, vel_wall)
    U_wall_c = np.array([rho_wall, rho_wall*vel_wall[0],
                          rho_wall*vel_wall[1], rho_wall*vel_wall[2],
                          rho_wall*E_wall])
    return 2.0*U_wall_c - U_int


def ghost_state(U, bc_type, n, gamma=1.4, R_gas=287.0,
                u_b=0.0, v_b=0.0, w_b=0.0, T_b=300.0, P_b=101325.0):
    """Ghost cell state for slip / noslip / riemann BCs."""
    U = np.asarray(U, dtype=float)
    n = np.asarray(n, dtype=float);  n = n / np.linalg.norm(n)
    rho = U[0];  vel = U[1:4]/rho;  rho_E = U[4]
    if bc_type == 'slip':
        vn = np.dot(vel, n)
        return np.array([rho, *(rho*(vel - 2.0*vn*n)), rho_E])
    elif bc_type == 'noslip':
        return np.array([rho, *(-rho*vel), rho_E])
    elif bc_type == 'riemann':
        return riemann_invariant_bc(U, n, gamma, u_b, v_b, w_b, T_b, P_b, R_gas)
    else:
        raise ValueError(f"Unknown bc_type '{bc_type}'.")


def ghost_state_jacobian_fd(U_i, bc_type, n,
                             gamma=1.4, R_gas=287.0,
                             u_b=0.0, v_b=0.0, w_b=0.0,
                             T_b=288.15, P_b=101325.0,
                             eps=1e-6, eps_abs=1e-14):
    """
    dU_ghost/dU_i (5x5) via forward FD with relative eps scaling.
    h_k = eps * max(|U_i[k]|, eps_abs)
    """
    U_i    = np.asarray(U_i, dtype=float)
    n      = np.asarray(n,   dtype=float);  n = n / np.linalg.norm(n)
    Ug_base = ghost_state(U_i, bc_type, n, gamma, R_gas, u_b, v_b, w_b, T_b, P_b)
    dUg     = np.zeros((5, 5))
    for k in range(5):
        h  = eps * max(abs(U_i[k]), eps_abs)
        ej = np.zeros(5);  ej[k] = h
        Ug_fwd    = ghost_state(U_i + ej, bc_type, n, gamma, R_gas, u_b, v_b, w_b, T_b, P_b)
        dUg[:, k] = (Ug_fwd - Ug_base) / h
    return dUg


def dV_dU(rho, u, v, w, gamma):
    """
    Jacobian of primitive V=[rho,u,v,w,p] w.r.t. conservative U=[rho,rhou,rhov,rhow,rhoE]
    5x5 matrix (3D).
    """
    g1 = gamma - 1.0
    q2 = u**2 + v**2 + w**2
    return np.array([
        [ 1.0,      0.0,     0.0,     0.0,    0.0],
        [-u/rho,  1.0/rho,   0.0,     0.0,    0.0],
        [-v/rho,    0.0,   1.0/rho,   0.0,    0.0],
        [-w/rho,    0.0,     0.0,   1.0/rho,  0.0],
        [0.5*q2*g1, -u*g1, -v*g1,  -w*g1,    g1 ],
    ])


def slip_wall_flux_jacobian(U_i, n, gamma=1.4):
    """
    Exact slip-wall boundary flux Jacobian dF_bc/dU_i (5x5).
    F_W = [0, P*nx, P*ny, P*nz, 0]  (u_n = 0)
    Chain rule: dF/dU = dF/dV|_W @ dV/dU
    """
    U_i = np.asarray(U_i, dtype=float)
    n   = np.asarray(n,   dtype=float);  n = n / np.linalg.norm(n)
    nx, ny, nz = n
    rho = U_i[0]
    u, v, w = U_i[1]/rho, U_i[2]/rho, U_i[3]/rho
    dFdV_W       = np.zeros((5, 5))
    dFdV_W[1, 4] = nx
    dFdV_W[2, 4] = ny
    dFdV_W[3, 4] = nz
    return dFdV_W @ dV_dU(rho, u, v, w, gamma)


# =============================================================================
# SECTION 4 — Boundary flux Jacobian (non-frozen ghost)
# =============================================================================

def boundary_flux_jacobian_fd(U_i, bc_type, n, ds,
                               eps_visc, eps_ghost,
                               gamma=1.4, R_gas=287.0,
                               Pr=0.72, mu0=1.716e-5,
                               T0=273.15, Suth_C=110.4,
                               u_b=0.0, v_b=0.0, w_b=0.0,
                               T_b=300.0, P_b=101325.0):
    """
    Boundary flux Jacobian dF_bc/dU_i (5x5).
    slip    — exact analytic:  J_bc = dF/dV|_W @ dV/dU
    riemann — non-frozen ghost chain rule:  J_bc = A+ + A- @ dUg/dUi
    noslip  — non-frozen ghost viscous:  J_bc = dFv/dUi + dFv/dUg @ dUg/dUi
    """
    U_i = np.asarray(U_i, dtype=float)
    n   = np.asarray(n,   dtype=float);  n = n / np.linalg.norm(n)
    if bc_type == 'slip':
        return slip_wall_flux_jacobian(U_i, n, gamma)
    elif bc_type == 'riemann':
        U_g = ghost_state(U_i, 'riemann', n, gamma, R_gas, u_b, v_b, w_b, T_b, P_b)
        J_L, J_R = inviscid_flux_jacobians(U_i, U_g, n, gamma)
        dUg_dUi  = ghost_state_jacobian_fd(
            U_i, 'riemann', n,
            gamma=gamma, R_gas=R_gas,
            u_b=u_b, v_b=v_b, w_b=w_b,
            T_b=T_b, P_b=P_b,
            eps=eps_ghost)
        return J_L + J_R @ dUg_dUi
    elif bc_type == 'noslip':
        U_g = ghost_state(U_i, 'noslip', n)
        J_i_visc, J_g_visc = viscous_flux_jacobians_fd(
            U_i, U_g, n, max(ds, 1e-14),
            eps=eps_visc, gamma=gamma, R_gas=R_gas,
            Pr=Pr, mu0=mu0, T0=T0, Suth_C=Suth_C)
        dUg_dUi = np.diag([1.0, -1.0, -1.0, -1.0, 1.0])   # exact, no FD
        return J_i_visc + J_g_visc @ dUg_dUi
    else:
        raise ValueError(f"Unknown bc_type '{bc_type}'.")


# =============================================================================
# SECTION 5 — Volume extraction and global assembler
# =============================================================================

def extract_cell_volumes(fortfile, coord, N):
    """
    Pass 1: fill from col 5 wherever node appears as i.
    Pass 2: geometric dual-mesh fallback for j-only nodes.
    """
    cell_vol = np.zeros(N)
    for row in fortfile:
        i = int(row[0]) - 1
        cell_vol[i] = row[5]
    geo_vol = np.zeros(N)
    for row in fortfile:
        i    = int(row[0]) - 1
        j    = int(row[1]) - 1
        A_ij = np.asarray(row[2:5], dtype=float)
        Area = np.linalg.norm(A_ij)
        if Area < 1e-14:
            continue
        n_ij = A_ij / Area
        ds   = max(abs(np.dot(coord[j] - coord[i], n_ij)), 1e-14)
        slab = 0.5 * Area * ds
        geo_vol[i] += slab
        geo_vol[j] += slab
    missing_mask           = (cell_vol == 0)
    cell_vol[missing_mask] = geo_vol[missing_mask]
    still_missing = np.where(cell_vol == 0)[0]
    if len(still_missing):
        raise ValueError(f"Volume still zero for nodes: {still_missing}")
    print(f"  Volumes from col 5 (direct)      : {(~missing_mask).sum()}")
    print(f"  Volumes from geometry (fallback)  : {missing_mask.sum()}")
    return cell_vol


def assemble_global_jacobian_fd(fortfile, U_list, coord, boundary_list,
                                 gamma=1.4, R_gas=287.0,
                                 viscous=False,
                                 Pr=0.72, mu0=1.716e-5,
                                 T0=273.15, Suth_C=110.4,
                                 eps_visc=1e-8,
                                 eps_ghost=1e-6,
                                 include_bc=True,
                                 normalise_volume=True,
                                 cell_vol=None):
    N       = U_list.shape[0]
    J       = lil_matrix((5*N, 5*N))
    visc_kw = dict(gamma=gamma, R_gas=R_gas, Pr=Pr, mu0=mu0, T0=T0, Suth_C=Suth_C)

    # ── Interior faces ─────────────────────────────────────────────────────────
    for row in fortfile:
        i    = int(row[0]) - 1;  j = int(row[1]) - 1
        A_ij = np.asarray(row[2:5], dtype=float)
        Area = np.linalg.norm(A_ij)
        if Area < 1e-14:
            continue
        n_ij = A_ij / Area
        J_L, J_R = inviscid_flux_jacobians(U_list[i], U_list[j], n_ij, gamma)
        if viscous:
            ds      = max(abs(float(np.dot(coord[j] - coord[i], n_ij))), 1e-14)
            Jv_L, Jv_R = viscous_flux_jacobians_fd(
                U_list[i], U_list[j], n_ij, ds, eps=eps_visc, **visc_kw)
            J_L = J_L + Jv_L;  J_R = J_R + Jv_R
        ri = slice(5*i, 5*i+5);  ci = slice(5*i, 5*i+5)
        rj = slice(5*j, 5*j+5);  cj = slice(5*j, 5*j+5)
        J[ri, ci] += J_L * Area
        J[ri, cj] += J_R * Area
        J[rj, ci] -= J_L * Area
        J[rj, cj] -= J_R * Area

    # ── Boundary faces ─────────────────────────────────────────────────────────
    if include_bc:
        bc_map = {0: 'riemann', 1: 'slip', 2: 'noslip'}
        for row in boundary_list:
            i    = int(row[0]) - 1
            A_bc = np.asarray(row[1:4], dtype=float)
            Area = np.linalg.norm(A_bc)
            ds   = float(row[4])
            if Area < 1e-14:
                continue
            n_bc    = A_bc / Area
            bc_type = bc_map[int(row[5])]
            u_b, v_b, w_b = float(row[6]), float(row[7]), float(row[8])
            T_b, P_b      = float(row[9]), float(row[10])
            J_bc = boundary_flux_jacobian_fd(
                U_list[i], bc_type, n_bc, ds,
                eps_visc=eps_visc, eps_ghost=eps_ghost,
                u_b=u_b, v_b=v_b, w_b=w_b, T_b=T_b, P_b=P_b,
                **visc_kw)
            ri = slice(5*i, 5*i+5)
            J[ri, ri] += J_bc * Area

    J_csr = J.tocsr()

    # ── Volume normalisation ────────────────────────────────────────────────────
    if normalise_volume:
        if cell_vol is None:
            cell_vol = extract_cell_volumes(fortfile, coord, N)
        D_inv = np.repeat(1.0 / cell_vol, 5)
        J_csr = diags(D_inv) @ J_csr    # J_phys in [1/s]

    return J_csr


# =============================================================================
# SECTION 6 — Mesh/BL helper functions
# =============================================================================

def area_normals(coord, ifac3=None, ifac4=None):
    """Area-weighted normal vectors (Ax, Ay, Az) accumulated from surface faces."""
    coord = np.asarray(coord, dtype=float)
    anor  = np.zeros_like(coord)
    if ifac3 is not None and len(ifac3) > 0:
        ifac3 = np.asarray(ifac3)
        v0, v1, v2 = coord[ifac3[:, 0]], coord[ifac3[:, 1]], coord[ifac3[:, 2]]
        face_anor = 0.5 * np.cross(v1 - v0, v2 - v0)
        contrib   = -face_anor / 3.0
        for k in range(3):
            np.add.at(anor, ifac3[:, k], contrib)
    if ifac4 is not None and len(ifac4) > 0:
        ifac4 = np.asarray(ifac4)
        v0, v1, v2, v3 = (coord[ifac4[:, 0]], coord[ifac4[:, 1]],
                           coord[ifac4[:, 2]], coord[ifac4[:, 3]])
        n1 = 0.5 * np.cross(v1 - v0, v2 - v0)
        n2 = 0.5 * np.cross(v2 - v0, v3 - v0)
        face_anor = n1 + n2
        contrib   = -face_anor / 4.0
        for k in range(4):
            np.add.at(anor, ifac4[:, k], contrib)
    return anor


def boundary_geometry(pltfile, coord, fortfile, flag):
    """Wall-normal distance ds for each boundary node on surface flag."""
    surface_nodes, tri_connect, quad_connect = pltfile.extract_surface_real(flag=flag)
    surface_coord        = coord[surface_nodes]
    surface_area_normals = area_normals(surface_coord, ifac3=tri_connect, ifac4=quad_connect)
    n_hat = surface_area_normals / np.linalg.norm(surface_area_normals, axis=1, keepdims=True)
    local_index = -np.ones(coord.shape[0], dtype=int)
    local_index[surface_nodes] = np.arange(len(surface_nodes))
    surface_mask = np.zeros(coord.shape[0], dtype=bool)
    surface_mask[surface_nodes] = True
    ds_lists = {}
    for row in fortfile:
        a = int(row[0]) - 1;  b = int(row[1]) - 1
        a_surf = surface_mask[a];  b_surf = surface_mask[b]
        if a_surf and not b_surf:
            node, neigh = a, b
        elif b_surf and not a_surf:
            node, neigh = b, a
        else:
            continue
        normal   = n_hat[local_index[node]]
        ds_proj  = abs(np.dot(coord[neigh] - coord[node], normal))
        ds_lists.setdefault(node, []).append(ds_proj)
    node_ids = np.array(sorted(ds_lists.keys()))
    ds       = np.array([np.mean(ds_lists[n]) for n in node_ids])
    return np.column_stack([node_ids, surface_area_normals, ds])


def build_boundary_list(pltfile, coord, fortfile, flags_bc_map):
    """
    boundary_list : (B, 12) ndarray
        col 0    : node i (1-based)
        col 1-3  : [Ax, Ay, Az]
        col 4    : ds
        col 5    : bc_type  0=riemann | 1=slip | 2=noslip
        col 6-8  : u_b, v_b, w_b
        col 9    : T_b
        col 10   : P_b
        col 11   : flag
    """
    rows = []
    for flag, bc in flags_bc_map.items():
        geom = boundary_geometry(pltfile, coord, fortfile, flag)
        n    = geom.shape[0]
        bc_cols = np.tile([
            float(bc['bc_type']), float(bc['u_b']), float(bc['v_b']),
            float(bc['w_b']),     float(bc['T_b']), float(bc['P_b']),
        ], (n, 1))
        node_1based = geom[:, [0]] + 1.0
        flag_col    = np.full((n, 1), flag, dtype=float)
        rows.append(np.hstack([node_1based, geom[:, 1:5], bc_cols, flag_col]))
    return np.vstack(rows)


def specific_energy(rho, p, ux, uy, uz, gamma=1.4):
    return p / ((gamma - 1.0) * rho) + 0.5 * (ux**2 + uy**2 + uz**2)


def conservative_variables(rst, gamma):
    """Primitive rst object -> conservative variable array (N, 5)."""
    E = specific_energy(rst.rho, rst.p, rst.ux, rst.uy, rst.uz, gamma)
    return np.column_stack((
        rst.rho,
        rst.rho * rst.ux,
        rst.rho * rst.uy,
        rst.rho * rst.uz,
        rst.rho * E,
    ))


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble global flux Jacobian (v4) and save to .npz")
    p.add_argument("--mesh",      type=int,   default=765960,  help="Number of mesh nodes")
    p.add_argument("--re",        type=float, default=60.0,    help="Reynolds number")
    p.add_argument("--mach",      type=float, default=0.2,     help="Mach number")
    p.add_argument("--mesh-ver",  type=int,   default=3,       help="Mesh version")
    p.add_argument("--viscous",   action="store_true",         help="Include viscous fluxes")
    p.add_argument("--no-bc",     action="store_true",         help="Exclude boundary conditions")
    p.add_argument("--eps-visc",  type=float, default=1e-8,    help="FD step for viscous flux")
    p.add_argument("--eps-ghost", type=float, default=1e-6,    help="FD step for ghost state")
    p.add_argument("--case-dir",  type=str,
                   default="/home/ahf25/CFD_2d_cylinder_all/Steady/Ma{mach}/v{ver}_mesh/2d_cylinder_{mesh}_Re{re}",
                   help="Case directory template")
    p.add_argument("--data-dir",  type=str,
                   default="../../data/flux_jacobian_assembly_v4",
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    Mesh     = args.mesh
    Re       = args.re
    Mach     = args.mach
    mesh_ver = args.mesh_ver
    gamma    = 1.4
    R_gas    = 287.0

    case_dir = args.case_dir.format(
        mach=Mach, ver=mesh_ver, mesh=Mesh, re=int(Re))
    data_dir = os.path.join(args.data_dir, f"v{mesh_ver}_mesh")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Global Flux Jacobian Assembler v4")
    print(f"  Mesh={Mesh}  Re={Re}  Ma={Mach}  viscous={args.viscous}")
    print(f"  eps_visc={args.eps_visc}  eps_ghost={args.eps_ghost}")
    print(f"  Case dir : {case_dir}")
    print(f"  Output   : {data_dir}")
    print("=" * 60)

    # ── Load files ─────────────────────────────────────────────────────────────
    print("\n[1/4] Reading mesh and solution files...")
    pltfile = PltFileUtils(f"{case_dir}/cylinder.plt")
    rstfile = UnkFileUtils(f"{case_dir}/cylinder.rst", extend=False)
    fortfile = pd.read_csv(f"{case_dir}/fort.864",
                            sep=r'\s+', header=None).to_numpy()
    rstfile._primitive()
    U_list = conservative_variables(rstfile, gamma)
    coord  = pltfile.coord
    print(f"  Nodes: {U_list.shape[0]}  Faces: {fortfile.shape[0]}")

    # ── Boundary conditions ─────────────────────────────────────────────────────
    print("\n[2/4] Building boundary list...")
    u_in = 68.0525;  T_in = 288.15;  P_in = 1.32702
    flags_bc_map = {
        1: {'bc_type': 0, 'u_b': u_in,  'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
        2: {'bc_type': 0, 'u_b': u_in,  'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
        3: {'bc_type': 1, 'u_b': 0.0,   'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
        4: {'bc_type': 1, 'u_b': 0.0,   'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
        5: {'bc_type': 1, 'u_b': 0.0,   'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
        6: {'bc_type': 1, 'u_b': 0.0,   'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
        7: {'bc_type': 2, 'u_b': 0.0,   'v_b': 0.0, 'w_b': 0.0, 'T_b': T_in, 'P_b': P_in},
    }
    boundary_list = build_boundary_list(pltfile, coord, fortfile, flags_bc_map)
    print(f"  Boundary nodes: {boundary_list.shape[0]}")

    # ── Assemble ────────────────────────────────────────────────────────────────
    print("\n[3/4] Assembling Jacobian...")
    t0 = time.perf_counter()
    J  = assemble_global_jacobian_fd(
            fortfile, U_list, coord, boundary_list,
            viscous          = args.viscous,
            eps_visc         = args.eps_visc,
            eps_ghost        = args.eps_ghost,
            include_bc       = not args.no_bc,
            normalise_volume = True)
    runtime = time.perf_counter() - t0
    print(f"  Done.  shape={J.shape}  nnz={J.nnz}  runtime={runtime:.1f}s")

    # ── Save ────────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving...")
    out_path = os.path.join(
        data_dir,
        f"jacobian_cylinder_{Mesh}_Re{int(Re)}_M{Mach}_fd.npz")
    save_npz(out_path, J)
    print(f"  Saved -> {out_path}")
    print(f"\nTotal runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()

# how to run 
#　conda activate /home/ahf25/anaconda3/envs/pyau3d_env
# nohup python -u assemble_jacobian_v4.py \
#     --mesh     1515828 \
#     --re       60 \
#     --mach     0.2 \
#     --mesh-ver 3 \
#     --viscous \
#     --eps-visc  1e-8 \
#     --eps-ghost 1e-6 \
#     --case-dir  "/home/ahf25/CFD_2d_cylinder_all/Steady/Ma0.2/v3_mesh/2d_cylinder_1515828_Re60" \
#     --data-dir  "/home/ahf25/git/flux_jacobian/data/flux_jacobian_assembly_v4" \
#     > assemble_1515828.log 2>&1 &