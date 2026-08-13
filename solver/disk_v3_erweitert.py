import numpy as np
import matplotlib.pyplot as plt
GM = 1.0
r0 = 1.0
Sigma0 = 1.0
p_slope = 1.0
h0 = 0.05
A_gap = 0.7
r_gap = 1.0
w_gap = 0.21
alpha = 0.01
St = 0.01
eps0 = 0.01
delta_t = 0.8
r_min, r_max = (0.3, 5.0)
N = 400
t_end_orbits = 20.0
CFL = 0.4
T_orb = 2.0 * np.pi / np.sqrt(GM / r0 ** 3)
t_end = t_end_orbits * T_orb
SIGMA_FLOOR = 1e-12
SIG_G_EQ = None
L_G_EQ = None
TOMINAGA_MOM = True
DAMP_FRAC_IN = 0.1
DAMP_FRAC_OUT = 0.1
DAMP_TAU_FRAC = 0.1

def rebuild_grid(N_new):
    global N, dr, r, r_face, cs, v_K, Om_K, H, nu, D_d, cd2
    N = N_new
    log_step = np.log(r_max / r_min) / (N - 2)
    r_face = np.empty(N + 1)
    r_face[1:N] = np.geomspace(r_min, r_max, N - 1)
    r_face[0] = r_face[1] / np.exp(log_step)
    r_face[N] = r_face[N - 1] * np.exp(log_step)
    r = np.sqrt(r_face[:-1] * r_face[1:])
    dr = r_face[1:] - r_face[:-1]
    cs = h0 * np.sqrt(GM / r0) * (r / r0) ** (-0.25)
    v_K = np.sqrt(GM / r)
    Om_K = np.sqrt(GM / r ** 3)
    H = cs / Om_K
    nu = alpha * cs ** 2 / Om_K
    D_d = delta_t * nu / (1.0 + St ** 2)
    cd2 = D_d * Om_K
rebuild_grid(N)

def initial_state(nsh_drift_on=True):
    gap = 1.0 - A_gap * np.exp(-0.5 * ((r - r_gap) / w_gap) ** 2)
    sig = Sigma0 * (r / r0) ** (-p_slope) * np.maximum(gap, 0.001)
    sig[0] = sig[1]
    sig[-1] = sig[-2]
    Pi = cs ** 2 * sig
    dPi = np.zeros_like(sig)
    dPi[1:-1] = (Pi[2:] - Pi[:-2]) / (r[2:] - r[:-2])
    u_phi = np.sqrt(np.maximum(v_K ** 2 + r * dPi / np.maximum(sig, SIGMA_FLOOR), 0.0))
    u_r = np.zeros_like(r)
    sig_d = eps0 * sig
    if nsh_drift_on:
        eta = -(r / (2.0 * np.maximum(Pi, 1e-30))) * (H / r) ** 2 * dPi
        v_r_d = (u_r - 2.0 * eta * v_K * St) / (1.0 + St ** 2)
        v_phi_d = (u_phi + (u_r + eta * v_K) * St) / (1.0 + St ** 2)
    else:
        v_r_d = np.zeros_like(r)
        v_phi_d = v_K.copy()
    mom_r = sig * u_r
    L = sig * r * u_phi
    mom_r_d = sig_d * v_r_d
    L_d = sig_d * r * v_phi_d
    return (sig, mom_r, L, sig_d, mom_r_d, L_d)

def apply_outflow_bc(sig, mom_r, L):
    sig[0] = sig[1]
    sig[-1] = sig[-2]
    L[0] = L[1]
    L[-1] = L[-2]
    u_in = mom_r[1] / max(sig[1], SIGMA_FLOOR)
    u_out = mom_r[-2] / max(sig[-2], SIGMA_FLOOR)
    mom_r[0] = min(u_in, 0.0) * sig[0]
    mom_r[-1] = max(u_out, 0.0) * sig[-1]

def cyl_donor_cell_flux(q, u_face):
    n = len(q)
    F = np.zeros(n + 1)
    F[1:-1] = np.where(u_face[1:-1] >= 0.0, q[:-1] * u_face[1:-1], q[1:] * u_face[1:-1])
    F[0] = q[0] * u_face[0]
    F[-1] = q[-1] * u_face[-1]
    return F * r_face

def vanleer_slope(q):
    n = len(q)
    s = np.zeros(n)
    dq_l = q[1:-1] - q[:-2]
    dq_r = q[2:] - q[1:-1]
    prod = dq_l * dq_r
    sum_ = dq_l + dq_r
    s[1:-1] = np.where(prod > 0.0, 2.0 * prod / np.where(sum_ != 0.0, sum_, 1.0), 0.0)
    return s

def minmod_slope(q):
    n = len(q)
    s = np.zeros(n)
    dq_l = q[1:-1] - q[:-2]
    dq_r = q[2:] - q[1:-1]
    s[1:-1] = np.where(dq_l * dq_r > 0.0, np.where(np.abs(dq_l) < np.abs(dq_r), dq_l, dq_r), 0.0)
    return s

def cyl_muscl_bouchut_flux(q, v, dt, r_face_arr):
    n = len(q)
    s_q = vanleer_slope(q)
    cfl_L = v[:-1] * dt / dr[:-1]
    cfl_R = v[1:] * dt / dr[1:]
    qL = q[:-1] + 0.5 * (1.0 - cfl_L) * s_q[:-1]
    qR = q[1:] - 0.5 * (1.0 + cfl_R) * s_q[1:]
    vL = v[:-1]
    vR = v[1:]
    F = np.zeros(n + 1)
    F[1:-1] = r_face_arr[1:-1] * (qL * np.maximum(vL, 0.0) + qR * np.minimum(vR, 0.0))
    F[0] = r_face_arr[0] * q[0] * v[0]
    F[-1] = r_face_arr[-1] * q[-1] * v[-1]
    return F

def cyl_muscl_rusanov_dust_flux(sig_d, mom_r_d, L_d, v_r, cd, dt, r_face_arr):
    n = len(sig_d)
    s_sig = vanleer_slope(sig_d)
    s_mr = vanleer_slope(mom_r_d)
    s_L = vanleer_slope(L_d)
    cfl_L = v_r[:-1] * dt / dr[:-1]
    cfl_R = v_r[1:] * dt / dr[1:]
    sig_L = sig_d[:-1] + 0.5 * (1.0 - cfl_L) * s_sig[:-1]
    sig_R = sig_d[1:] - 0.5 * (1.0 + cfl_R) * s_sig[1:]
    mr_L = mom_r_d[:-1] + 0.5 * (1.0 - cfl_L) * s_mr[:-1]
    mr_R = mom_r_d[1:] - 0.5 * (1.0 + cfl_R) * s_mr[1:]
    Ld_L = L_d[:-1] + 0.5 * (1.0 - cfl_L) * s_L[:-1]
    Ld_R = L_d[1:] - 0.5 * (1.0 + cfl_R) * s_L[1:]
    sig_L = np.maximum(sig_L, SIGMA_FLOOR)
    sig_R = np.maximum(sig_R, SIGMA_FLOOR)
    vL = mr_L / sig_L
    vR = mr_R / sig_R
    cdL = cd[:-1]
    cdR = cd[1:]
    PiL = cdL ** 2 * sig_L
    PiR = cdR ** 2 * sig_R
    Fsig_L = sig_L * vL
    Fsig_R = sig_R * vR
    Fmr_L = mr_L * vL + PiL
    Fmr_R = mr_R * vR + PiR
    FL_L = Ld_L * vL
    FL_R = Ld_R * vR
    alpha = np.maximum(np.abs(vL) + cdL, np.abs(vR) + cdR)
    F_sig_int = 0.5 * (Fsig_L + Fsig_R) - 0.5 * alpha * (sig_R - sig_L)
    F_mr_int = 0.5 * (Fmr_L + Fmr_R) - 0.5 * alpha * (mr_R - mr_L)
    F_L_int = 0.5 * (FL_L + FL_R) - 0.5 * alpha * (Ld_R - Ld_L)
    F_sig = np.zeros(n + 1)
    F_mr = np.zeros(n + 1)
    F_L = np.zeros(n + 1)
    F_sig[1:-1] = r_face_arr[1:-1] * F_sig_int
    F_mr[1:-1] = r_face_arr[1:-1] * F_mr_int
    F_L[1:-1] = r_face_arr[1:-1] * F_L_int
    Pi0 = cd[0] ** 2 * sig_d[0]
    PiN = cd[-1] ** 2 * sig_d[-1]
    F_sig[0] = r_face_arr[0] * sig_d[0] * v_r[0]
    F_sig[-1] = r_face_arr[-1] * sig_d[-1] * v_r[-1]
    F_mr[0] = r_face_arr[0] * (mom_r_d[0] * v_r[0] + Pi0)
    F_mr[-1] = r_face_arr[-1] * (mom_r_d[-1] * v_r[-1] + PiN)
    F_L[0] = r_face_arr[0] * L_d[0] * v_r[0]
    F_L[-1] = r_face_arr[-1] * L_d[-1] * v_r[-1]
    return (F_sig, F_mr, F_L)

def cyl_isothermal_dust_flux(sig_d, mom_r_d, L_d, v_r, cd, r_face_arr):
    n = len(sig_d)
    sig_L = sig_d[:-1]
    sig_R = sig_d[1:]
    mr_L = mom_r_d[:-1]
    mr_R = mom_r_d[1:]
    Ld_L = L_d[:-1]
    Ld_R = L_d[1:]
    vL = v_r[:-1]
    vR = v_r[1:]
    cdL = cd[:-1]
    cdR = cd[1:]
    PiL = cdL ** 2 * sig_L
    PiR = cdR ** 2 * sig_R
    Fsig_L = sig_L * vL
    Fsig_R = sig_R * vR
    Fmr_L = mr_L * vL + PiL
    Fmr_R = mr_R * vR + PiR
    FL_L = Ld_L * vL
    FL_R = Ld_R * vR
    alpha = np.maximum(np.abs(vL) + cdL, np.abs(vR) + cdR)
    F_sig_int = 0.5 * (Fsig_L + Fsig_R) - 0.5 * alpha * (sig_R - sig_L)
    F_mr_int = 0.5 * (Fmr_L + Fmr_R) - 0.5 * alpha * (mr_R - mr_L)
    F_L_int = 0.5 * (FL_L + FL_R) - 0.5 * alpha * (Ld_R - Ld_L)
    F_sig = np.zeros(n + 1)
    F_mr = np.zeros(n + 1)
    F_L = np.zeros(n + 1)
    F_sig[1:-1] = r_face_arr[1:-1] * F_sig_int
    F_mr[1:-1] = r_face_arr[1:-1] * F_mr_int
    F_L[1:-1] = r_face_arr[1:-1] * F_L_int
    Pi0 = cd[0] ** 2 * sig_d[0]
    PiN = cd[-1] ** 2 * sig_d[-1]
    F_sig[0] = r_face_arr[0] * sig_d[0] * v_r[0]
    F_sig[-1] = r_face_arr[-1] * sig_d[-1] * v_r[-1]
    F_mr[0] = r_face_arr[0] * (mom_r_d[0] * v_r[0] + Pi0)
    F_mr[-1] = r_face_arr[-1] * (mom_r_d[-1] * v_r[-1] + PiN)
    F_L[0] = r_face_arr[0] * L_d[0] * v_r[0]
    F_L[-1] = r_face_arr[-1] * L_d[-1] * v_r[-1]
    return (F_sig, F_mr, F_L)

def cyl_muscl_isothermal_gas_flux(sig, mom_r, L, u_r, cs_arr, dt, r_face_arr, sig_eq=None, L_eq=None):
    n = len(sig)
    s_sig = minmod_slope(sig)
    s_mr = minmod_slope(mom_r)
    s_L = minmod_slope(L)
    cfl_L = u_r[:-1] * dt / dr[:-1]
    cfl_R = u_r[1:] * dt / dr[1:]
    sig_L = sig[:-1] + 0.5 * (1.0 - cfl_L) * s_sig[:-1]
    sig_R = sig[1:] - 0.5 * (1.0 + cfl_R) * s_sig[1:]
    mr_L = mom_r[:-1] + 0.5 * (1.0 - cfl_L) * s_mr[:-1]
    mr_R = mom_r[1:] - 0.5 * (1.0 + cfl_R) * s_mr[1:]
    L_L = L[:-1] + 0.5 * (1.0 - cfl_L) * s_L[:-1]
    L_R = L[1:] - 0.5 * (1.0 + cfl_R) * s_L[1:]
    sig_L = np.maximum(sig_L, SIGMA_FLOOR)
    sig_R = np.maximum(sig_R, SIGMA_FLOOR)
    uL = mr_L / sig_L
    uR = mr_R / sig_R
    csL = cs_arr[:-1]
    csR = cs_arr[1:]
    PiL = csL ** 2 * sig_L
    PiR = csR ** 2 * sig_R
    Fsig_L = sig_L * uL
    Fsig_R = sig_R * uR
    Fmr_L = mr_L * uL + PiL
    Fmr_R = mr_R * uR + PiR
    FL_L = L_L * uL
    FL_R = L_R * uR
    alpha = np.maximum(np.abs(uL) + csL, np.abs(uR) + csR)
    alpha_eq = np.maximum(csL, csR)
    if sig_eq is not None:
        s_sigeq = minmod_slope(sig_eq)
        s_Leq = minmod_slope(L_eq)
        sigeq_L = sig_eq[:-1] + 0.5 * s_sigeq[:-1]
        sigeq_R = sig_eq[1:] - 0.5 * s_sigeq[1:]
        Leq_L = L_eq[:-1] + 0.5 * s_Leq[:-1]
        Leq_R = L_eq[1:] - 0.5 * s_Leq[1:]
        corr_sig = 0.5 * alpha_eq * (sigeq_R - sigeq_L)
        corr_L = 0.5 * alpha_eq * (Leq_R - Leq_L)
    else:
        corr_sig = 0.0
        corr_L = 0.0
    F_sig_int = 0.5 * (Fsig_L + Fsig_R) - 0.5 * alpha * (sig_R - sig_L) + corr_sig
    F_mr_int = 0.5 * (Fmr_L + Fmr_R) - 0.5 * alpha * (mr_R - mr_L)
    F_L_int = 0.5 * (FL_L + FL_R) - 0.5 * alpha * (L_R - L_L) + corr_L
    F_sig = np.zeros(n + 1)
    F_mr = np.zeros(n + 1)
    F_L = np.zeros(n + 1)
    F_sig[1:-1] = r_face_arr[1:-1] * F_sig_int
    F_mr[1:-1] = r_face_arr[1:-1] * F_mr_int
    F_L[1:-1] = r_face_arr[1:-1] * F_L_int
    Pi0 = cs_arr[0] ** 2 * sig[0]
    PiN = cs_arr[-1] ** 2 * sig[-1]
    F_sig[0] = r_face_arr[0] * sig[0] * u_r[0]
    F_sig[-1] = r_face_arr[-1] * sig[-1] * u_r[-1]
    F_mr[0] = r_face_arr[0] * (mom_r[0] * u_r[0] + Pi0)
    F_mr[-1] = r_face_arr[-1] * (mom_r[-1] * u_r[-1] + PiN)
    F_L[0] = r_face_arr[0] * L[0] * u_r[0]
    F_L[-1] = r_face_arr[-1] * L[-1] * u_r[-1]
    return (F_sig, F_mr, F_L)

def gas_pressure_grad_wb(sig):
    n = len(sig)
    Pi = cs ** 2 * sig
    s = minmod_slope(sig)
    sig_L = np.maximum(sig[:-1] + 0.5 * s[:-1], SIGMA_FLOOR)
    sig_R = np.maximum(sig[1:] - 0.5 * s[1:], SIGMA_FLOOR)
    PiL = cs[:-1] ** 2 * sig_L
    PiR = cs[1:] ** 2 * sig_R
    F = np.zeros(n + 1)
    F[1:-1] = r_face[1:-1] * 0.5 * (PiL + PiR)
    F[0] = r_face[0] * Pi[0]
    F[-1] = r_face[-1] * Pi[-1]
    dPi = np.zeros_like(sig)
    interior = slice(1, n - 1)
    dPi[interior] = (F[2:-1] - F[1:-2]) / (r[interior] * dr[interior]) - Pi[interior] / r[interior]
    return dPi

def cyl_pressureless_riemann_flux(q, v, r_face_arr):
    n = len(q)
    F = np.zeros(n + 1)
    F[1:-1] = r_face_arr[1:-1] * (q[:-1] * np.maximum(v[:-1], 0.0) + q[1:] * np.minimum(v[1:], 0.0))
    F[0] = r_face_arr[0] * q[0] * v[0]
    F[-1] = r_face_arr[-1] * q[-1] * v[-1]
    return F

def cyl_muscl_flux(q, u_face, dt):
    n = len(q)
    s = minmod_slope(q)
    cfl_L = u_face[1:-1] * dt / dr[:-1]
    cfl_R = u_face[1:-1] * dt / dr[1:]
    qL = q[:-1] + 0.5 * (1.0 - cfl_L) * s[:-1]
    qR = q[1:] - 0.5 * (1.0 + cfl_R) * s[1:]
    F = np.zeros(n + 1)
    F[1:-1] = np.where(u_face[1:-1] >= 0.0, qL, qR) * u_face[1:-1] * r_face[1:-1]
    F[0] = q[0] * u_face[0] * r_face[0]
    F[-1] = q[-1] * u_face[-1] * r_face[-1]
    return F

def face_velocity(u_cell):
    n = len(u_cell)
    uf = np.zeros(n + 1)
    uf[1:-1] = 0.5 * (u_cell[:-1] + u_cell[1:])
    uf[1] = min(u_cell[1], 0.0)
    uf[-2] = max(u_cell[-2], 0.0)
    uf[0] = uf[1]
    uf[-1] = uf[-2]
    return uf

def epstein_drag_update(sig, mom_r, L, sig_d, mom_r_d, L_d, dt):
    if St <= 0.0:
        return (mom_r, L, mom_r_d, L_d)
    u_r_n = mom_r / sig
    u_phi_n = L / (sig * r)
    v_r_n = mom_r_d / sig_d
    v_phi_n = L_d / (sig_d * r)
    tau_s = St / Om_K
    eps = sig_d / sig
    fac = -np.expm1(-dt * (1.0 + eps) / tau_s)
    f_g = eps / (1.0 + eps)
    f_d = 1.0 / (1.0 + eps)
    drel_r = v_r_n - u_r_n
    drel_phi = v_phi_n - u_phi_n
    u_r_n += f_g * drel_r * fac
    v_r_n -= f_d * drel_r * fac
    u_phi_n += f_g * drel_phi * fac
    v_phi_n -= f_d * drel_phi * fac
    return (sig * u_r_n, sig * r * u_phi_n, sig_d * v_r_n, sig_d * r * v_phi_n)

def hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt):
    n = len(sig)
    u_r = mom_r / np.maximum(sig, SIGMA_FLOOR)
    v_r = mom_r_d / np.maximum(sig_d, SIGMA_FLOOR)
    u_face = face_velocity(u_r)
    F_sig = cyl_muscl_flux(sig, u_face, dt)
    F_mr = cyl_muscl_flux(mom_r, u_face, dt)
    F_L = cyl_muscl_flux(L, u_face, dt)
    safe = sig > 10.0 * SIGMA_FLOOR
    Omega = np.where(safe, L / (np.maximum(sig, SIGMA_FLOOR) * r * r), Om_K)
    nu_face = 0.5 * (nu[:-1] + nu[1:])
    sig_face = 0.5 * (sig[:-1] + sig[1:])
    dOmega = (Omega[1:] - Omega[:-1]) / (r[1:] - r[:-1])
    rf = r_face[1:-1]
    F_visc = -nu_face * sig_face * rf ** 3 * dOmega
    F_visc[0] = 0.0
    F_visc[-1] = 0.0
    F_L[1:-1] += F_visc
    cd = np.sqrt(np.maximum(cd2, 0.0))
    F_sig_d, F_mr_d, F_L_d = cyl_isothermal_dust_flux(sig_d, mom_r_d, L_d, v_r, cd, r_face)
    eps_face_diff = (sig_d[1:] / np.maximum(sig[1:], SIGMA_FLOOR) - sig_d[:-1] / np.maximum(sig[:-1], SIGMA_FLOOR)) / (r[1:] - r[:-1])
    D_face = 0.5 * (D_d[:-1] + D_d[1:])
    sg_face_diff = 0.5 * (sig[:-1] + sig[1:])
    F_diff = -D_face * sg_face_diff * rf * eps_face_diff
    F_diff[0] = 0.0
    F_diff[-1] = 0.0
    rvphi_cell = L_d / np.maximum(sig_d, SIGMA_FLOOR)
    upwind_L = F_diff >= 0.0
    vr_up = np.where(upwind_L, v_r[:-1], v_r[1:])
    rvphi_up = np.where(upwind_L, rvphi_cell[:-1], rvphi_cell[1:])
    F_sig_d[1:-1] += F_diff
    if TOMINAGA_MOM:
        F_mr_d[1:-1] += F_diff * vr_up
        F_L_d[1:-1] += F_diff * rvphi_up
    interior = slice(1, n - 1)
    geom = dt / (r[interior] * dr[interior])
    sig_new = sig.copy()
    mr_new = mom_r.copy()
    L_new = L.copy()
    sig_d_new = sig_d.copy()
    mr_d_new = mom_r_d.copy()
    L_d_new = L_d.copy()
    sig_new[interior] -= geom * (F_sig[2:-1] - F_sig[1:-2])
    mr_new[interior] -= geom * (F_mr[2:-1] - F_mr[1:-2])
    L_new[interior] -= geom * (F_L[2:-1] - F_L[1:-2])
    sig_d_new[interior] -= geom * (F_sig_d[2:-1] - F_sig_d[1:-2])
    mr_d_new[interior] -= geom * (F_mr_d[2:-1] - F_mr_d[1:-2])
    L_d_new[interior] -= geom * (F_L_d[2:-1] - F_L_d[1:-2])
    floor_g = sig_new < 10.0 * SIGMA_FLOOR
    floor_d = sig_d_new < 10.0 * SIGMA_FLOOR
    sig_new = np.maximum(sig_new, SIGMA_FLOOR)
    sig_d_new = np.maximum(sig_d_new, SIGMA_FLOOR)
    L_new[floor_g] = sig_new[floor_g] * r[floor_g] * v_K[floor_g]
    mr_new[floor_g] = 0.0
    L_d_new[floor_d] = sig_d_new[floor_d] * r[floor_d] * v_K[floor_d]
    mr_d_new[floor_d] = 0.0
    u_phi_new = L_new / (sig_new * r)
    v_phi_new = L_d_new / (sig_d_new * r)
    dL_g = L_new - sig_new * r * v_K
    dL_d = L_d_new - sig_d_new * r * v_K
    Pi = cs ** 2 * sig_new
    dPi = np.zeros_like(sig_new)
    dPi[1:-1] = (Pi[2:] - Pi[:-2]) / (r[2:] - r[:-2])
    S_r_g = dL_g * (u_phi_new + v_K) / r ** 2 - dPi
    mr_new[interior] += dt * S_r_g[interior]
    Pi_d = cd2 * sig_d_new
    S_r_d = dL_d * (v_phi_new + v_K) / r ** 2 + Pi_d / r
    mr_d_new[interior] += dt * S_r_d[interior]
    mr_new, L_new, mr_d_new, L_d_new = epstein_drag_update(sig_new, mr_new, L_new, sig_d_new, mr_d_new, L_d_new, dt)
    apply_outflow_bc(sig_new, mr_new, L_new)
    apply_outflow_bc(sig_d_new, mr_d_new, L_d_new)
    return (sig_new, mr_new, L_new, sig_d_new, mr_d_new, L_d_new)

def build_damping_rate():
    domain_w = r_max - r_min
    r_damp_in = r_min + DAMP_FRAC_IN * domain_w
    r_damp_out = r_max - DAMP_FRAC_OUT * domain_w
    R = np.zeros_like(r)
    mask_in = r < r_damp_in
    if mask_in.any():
        R[mask_in] = ((r_damp_in - r[mask_in]) / (r_damp_in - r_min)) ** 3
    mask_out = r > r_damp_out
    if mask_out.any():
        R[mask_out] = ((r[mask_out] - r_damp_out) / (r_max - r_damp_out)) ** 3
    tau_local = DAMP_TAU_FRAC * 2.0 * np.pi / Om_K
    return R / tau_local

def apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d, target, rate, dt):
    sig_t, mom_r_t, L_t, sig_d_t, mom_r_d_t, L_d_t = target
    fac = np.minimum(dt * rate, 1.0)
    sig -= fac * (sig - sig_t)
    mom_r -= fac * (mom_r - mom_r_t)
    L -= fac * (L - L_t)
    sig_d -= fac * (sig_d - sig_d_t)
    mom_r_d -= fac * (mom_r_d - mom_r_d_t)
    L_d -= fac * (L_d - L_d_t)
    return (sig, mom_r, L, sig_d, mom_r_d, L_d)

def compute_dt(sig, mom_r, sig_d, mom_r_d):
    u_r = mom_r[1:-1] / np.maximum(sig[1:-1], SIGMA_FLOOR)
    v_r = mom_r_d[1:-1] / np.maximum(sig_d[1:-1], SIGMA_FLOOR)
    cd = np.sqrt(cd2[1:-1])
    speed = np.maximum(np.abs(u_r) + cs[1:-1], np.abs(v_r) + cd)
    dr_int = dr[1:-1]
    dt_adv = CFL * float(np.min(dr_int / np.maximum(speed, 1e-30)))
    dt_visc = 0.5 * float(np.min(dr_int ** 2 / np.maximum(nu[1:-1], 1e-30)))
    dt_diff = 0.5 * float(np.min(dr_int ** 2 / np.maximum(D_d[1:-1], 1e-30)))
    return min(dt_adv, dt_visc, dt_diff)

def run(nsh_drift_on=True, damping_on=True):
    sig, mom_r, L, sig_d, mom_r_d, L_d = initial_state(nsh_drift_on=nsh_drift_on)
    apply_outflow_bc(sig, mom_r, L)
    apply_outflow_bc(sig_d, mom_r_d, L_d)
    target = (sig.copy(), mom_r.copy(), L.copy(), sig_d.copy(), mom_r_d.copy(), L_d.copy())
    damp_rate = build_damping_rate() if damping_on else np.zeros_like(r)
    n_snap = 5
    snap_dt = t_end / n_snap
    snaps = [(0.0, sig.copy(), L.copy(), sig_d.copy(), L_d.copy())]
    t_next = snap_dt
    t, k = (0.0, 0)
    while t < t_end:
        dt = min(compute_dt(sig, mom_r, sig_d, mom_r_d), t_end - t)
        sig, mom_r, L, sig_d, mom_r_d, L_d = hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt)
        if damping_on:
            sig, mom_r, L, sig_d, mom_r_d, L_d = apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d, target, damp_rate, dt)
        t += dt
        k += 1
        if t >= t_next or t >= t_end - 1e-12:
            snaps.append((t, sig.copy(), L.copy(), sig_d.copy(), L_d.copy()))
            t_next += snap_dt
        if k % 500 == 0:
            print(f'  step={k:6d}  t={t / T_orb:6.2f} Orbits  dt={dt:.4e}')
    return snaps

def saw(a):
    a = a[1:-1]
    d2 = np.abs(a[2:] - 2.0 * a[1:-1] + a[:-2])
    sc = np.abs(a[2:]) + 2.0 * np.abs(a[1:-1]) + np.abs(a[:-2]) + 1e-30
    return float((d2 / sc).max())

def print_diag(snaps):
    print(f"\n{'t/T_orb':>10s}  {'min(Σ_g)':>10s}  {'max(Σ_g)':>10s}  {'min(ε)':>10s}  {'max(ε)':>10s}  {'sawΣ_g':>8s}  {'sawΣ_d':>8s}")
    for ts, sg, _, sd, _ in snaps:
        s = sg[1:-1]
        eps = sd[1:-1] / np.maximum(sg[1:-1], SIGMA_FLOOR)
        print(f'{ts / T_orb:10.2f}  {s.min():10.3e}  {s.max():10.3e}  {eps.min():10.3e}  {eps.max():10.3e}  {saw(sg):8.2e}  {saw(sd):8.2e}')

def plot_snaps(snaps, title, pdfname):
    r_plot = r[1:-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps)))
    for (ts, sg, L_, sd, L_d_), c in zip(snaps, colors):
        label = f'{ts / T_orb:.1f} Orbits'
        axes[0].plot(r_plot, sg[1:-1], color=c, lw=1.5, label=label)
        axes[1].plot(r_plot, sd[1:-1], color=c, lw=1.5)
        axes[2].plot(r_plot, sd[1:-1] / np.maximum(sg[1:-1], SIGMA_FLOOR), color=c, lw=1.5)
    axes[2].axhline(eps0, ls='--', color='gray', lw=0.8, label='ε₀')
    for ax, ttl, ylbl in zip(axes, ['Gas Σ_g', 'Staub Σ_d', 'ε = Σ_d/Σ_g'], ['Σ_g', 'Σ_d', 'ε']):
        ax.set_xlabel('r')
        ax.set_ylabel(ylbl)
        ax.set_title(ttl)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(pdfname, bbox_inches='tight')
    print(f'Gespeichert: {pdfname}')

def simulate(A_gap_val, St_val, nsh_drift_on, label, pdfname, damping_on=True, N_val=400, cd_factor=1.0):
    global A_gap, St, cd2
    A_gap = A_gap_val
    St = St_val
    rebuild_grid(N_val)
    cd2 = cd2 * cd_factor ** 2
    print(f"\n══ {label} ══  N={N_val}, A_gap={A_gap_val}, St={St_val}, NSH-IC={('an' if nsh_drift_on else 'aus')}, Damping={('an' if damping_on else 'aus')}, cd_factor={cd_factor}")
    snaps = run(nsh_drift_on=nsh_drift_on, damping_on=damping_on)
    print_diag(snaps)
    plot_snaps(snaps, label, pdfname)
    return snaps
if __name__ == '__main__':
    import os as _os
    _os.makedirs('output', exist_ok=True)
    simulate(A_gap_val=0.7, St_val=0.01, nsh_drift_on=True, label='Erweitert: Tominaga-Diffusion (P4) + N=400 (P6)', pdfname='output/disk_v3_erweitert.pdf', damping_on=True, N_val=400, cd_factor=0.0)
    plt.show()
