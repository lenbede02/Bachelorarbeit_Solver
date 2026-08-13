import numpy as np
import matplotlib.pyplot as plt
GM       = 1.0
r0       = 1.0
Sigma0   = 1.0
p_slope  = 1.0
h0       = 0.05
A_gap    = 0.0
r_gap    = 1.0
w_gap    = 0.21
q_planet = 4.5e-4
r_planet = 1.0
T_RAMP_ORBITS = 10.0
C_torque      = 0.4478
alpha    = 0.01
St       = 0.1
eps0     = 0.01
delta_t  = 0.8
r_min, r_max = 0.3, 5.0
N            = 4000
t_end_orbits = 100.0
CFL          = 0.4
T_orb = 2.0 * np.pi / np.sqrt(GM / r0**3)
t_end = t_end_orbits * T_orb
SIGMA_FLOOR = 1e-12
TOMINAGA_MOM = True
DUST_BACKREACTION = True
DAMP_FRAC_IN  = 0.10
DAMP_FRAC_OUT = 0.10
DAMP_TAU_FRAC = 0.1
def planet_specific_torque(r_arr):
    if q_planet <= 0.0:
        return np.zeros_like(r_arr)
    cs_p   = h0 * np.sqrt(GM / r0) * (r_planet / r0)**(-0.25)
    Om_p   = np.sqrt(GM / r_planet**3)
    H_p    = cs_p / Om_p
    Dc     = 1.3 * H_p
    dist   = np.maximum(np.abs(r_arr - r_planet), 1e-30)
    Lam    = np.sign(r_arr - r_planet) * C_torque * 0.5 * q_planet**2 \
             * (GM / r_arr) * (r_arr / dist)**4
    Lam    = np.where(dist < Dc, 0.0, Lam)
    return Lam

dr = None
r = None
r_face = None
cs = None
v_K = None
Om_K = None
H = None
nu = None
D_d = None
cd2 = None
torque_p = None

def rebuild_grid(N_new):
    global N, dr, r, r_face, cs, v_K, Om_K, H, nu, D_d, cd2, torque_p
    N        = N_new
    log_step = np.log(r_max / r_min) / (N - 2) 
    r_face   = np.empty(N + 1)
    r_face[1:N]  = np.geomspace(r_min, r_max, N - 1) 
    r_face[0]    = r_face[1]    / np.exp(log_step) 
    r_face[N]    = r_face[N-1]  * np.exp(log_step)
    r            = np.sqrt(r_face[:-1] * r_face[1:]) 
    dr           = r_face[1:] - r_face[:-1]
    cs   = h0 * np.sqrt(GM / r0) * (r / r0)**(-0.25)
    v_K  = np.sqrt(GM / r)
    Om_K = np.sqrt(GM / r**3)
    H    = cs / Om_K
    nu   = alpha * cs**2 / Om_K
    D_d  = delta_t * nu/ (1.0 + St**2)
    cd2  = D_d * Om_K
    torque_p = planet_specific_torque(r)
rebuild_grid(N)
def initial_state(nsh_drift_on=True, stationary_visc=False):
    gap = 1.0 - A_gap * np.exp(-0.5 * ((r - r_gap) / w_gap)**2)
    sig = Sigma0 * (r / r0)**(-p_slope) * np.maximum(gap, 0)
    sig[0]  = sig[1]
    sig[-1] = sig[-2]
    Pi  = cs**2 * sig
    dPi = np.zeros_like(sig)
    dPi[1:-1] = (Pi[2:] - Pi[:-2]) / (r[2:] - r[:-2])
    u_phi = np.sqrt(np.maximum(v_K**2 + r * dPi / np.maximum(sig, SIGMA_FLOOR), 0.0))
    u_r   = np.zeros_like(r)
    if stationary_visc:
        u_r = -1.5 * nu / r
    sig_d = eps0 * sig
    if nsh_drift_on:
        eta     = -(r / (2.0 * np.maximum(Pi, 1e-30))) * (H / r)**2 * dPi
        v_r_d   = (u_r   - 2.0 * eta * v_K * St) / (1.0 + St**2)
        v_phi_d = (u_phi + (u_r + eta * v_K) * St) / (1.0 + St**2)
    else:
        v_r_d   = np.zeros_like(r)
        v_phi_d = v_K.copy()
    mom_r   = sig * u_r
    L       = sig * r * u_phi
    mom_r_d = sig_d * v_r_d
    L_d     = sig_d * r * v_phi_d
    return sig, mom_r, L, sig_d, mom_r_d, L_d
def apply_outflow_bc(sig, mom_r, L):
    sig[0]  = sig[1];   sig[-1] = sig[-2]
    L[0]    = L[1];     L[-1]   = L[-2]
    u_in    = mom_r[1]  / max(sig[1],  SIGMA_FLOOR)
    u_out   = mom_r[-2] / max(sig[-2], SIGMA_FLOOR)
    mom_r[0]  = min(u_in,  0.0) * sig[0]
    mom_r[-1] = max(u_out, 0.0) * sig[-1]
def minmod_slope(q): #übernommen aus den Übungen der Vorlesung Fundamentals in Simulation Methods. Mit Kees abgesprochen
    n = len(q)
    s = np.zeros(n)
    dq_l = q[1:-1] - q[:-2]
    dq_r = q[2:]   - q[1:-1]
    s[1:-1] = np.where(
        dq_l * dq_r > 0.0,
        np.where(np.abs(dq_l) < np.abs(dq_r), dq_l, dq_r),
        0.0)
    return s
def cyl_isothermal_dust_flux(sig_d, mom_r_d, L_d, v_r, cd, r_face_arr):
    n = len(sig_d)
    sig_L = sig_d[:-1];   sig_R = sig_d[1:]
    mr_L  = mom_r_d[:-1]; mr_R  = mom_r_d[1:]
    Ld_L  = L_d[:-1];     Ld_R  = L_d[1:]
    vL    = v_r[:-1];     vR    = v_r[1:]
    cdL   = cd[:-1];      cdR   = cd[1:]
    PiL   = cdL**2 * sig_L
    PiR   = cdR**2 * sig_R
    Fsig_L = sig_L * vL;             Fsig_R = sig_R * vR
    Fmr_L  = mr_L * vL + PiL;        Fmr_R  = mr_R * vR + PiR
    FL_L   = Ld_L * vL;              FL_R   = Ld_R * vR
    alpha  = np.maximum(np.abs(vL) + cdL, np.abs(vR) + cdR)
    F_sig_int = 0.5 * (Fsig_L + Fsig_R) - 0.5 * alpha * (sig_R - sig_L)
    F_mr_int  = 0.5 * (Fmr_L  + Fmr_R)  - 0.5 * alpha * (mr_R  - mr_L)
    F_L_int   = 0.5 * (FL_L   + FL_R)   - 0.5 * alpha * (Ld_R  - Ld_L)
    F_sig = np.zeros(n + 1)
    F_mr  = np.zeros(n + 1)
    F_L   = np.zeros(n + 1)
    F_sig[1:-1] = r_face_arr[1:-1] * F_sig_int
    F_mr [1:-1] = r_face_arr[1:-1] * F_mr_int
    F_L  [1:-1] = r_face_arr[1:-1] * F_L_int
    Pi0 = cd[0]**2  * sig_d[0]
    PiN = cd[-1]**2 * sig_d[-1]
    F_sig[0]  = r_face_arr[0]  * sig_d[0]  * v_r[0]
    F_sig[-1] = r_face_arr[-1] * sig_d[-1] * v_r[-1]
    F_mr [0]  = r_face_arr[0]  * (mom_r_d[0]  * v_r[0]  + Pi0)
    F_mr [-1] = r_face_arr[-1] * (mom_r_d[-1] * v_r[-1] + PiN)
    F_L  [0]  = r_face_arr[0]  * L_d[0]  * v_r[0]
    F_L  [-1] = r_face_arr[-1] * L_d[-1] * v_r[-1]
    return F_sig, F_mr, F_L
def cyl_muscl_flux(q, u_face, dt):
    n = len(q)
    s = minmod_slope(q)
    cfl_L = u_face[1:-1] * dt / dr[:-1]
    cfl_R = u_face[1:-1] * dt / dr[1:]
    qL  = q[:-1] + 0.5 * (1.0 - cfl_L) * s[:-1]
    qR  = q[1:]  - 0.5 * (1.0 + cfl_R) * s[1:]
    F = np.zeros(n + 1)
    F[1:-1] = np.where(u_face[1:-1] >= 0.0, qL, qR) * u_face[1:-1] * r_face[1:-1]
    F[0]  = q[0]  * u_face[0]  * r_face[0]
    F[-1] = q[-1] * u_face[-1] * r_face[-1]
    return F
def face_velocity(u_cell):
    n  = len(u_cell)
    uf = np.zeros(n + 1)
    uf[1:-1] = 0.5 * (u_cell[:-1] + u_cell[1:])
    uf[1]    = min(u_cell[1],  0.0)
    uf[-2]   = max(u_cell[-2], 0.0)
    uf[0]    = uf[1]
    uf[-1]   = uf[-2]
    return uf
def epstein_drag_update(sig, mom_r, L, sig_d, mom_r_d, L_d, dt):
    if St <= 0.0:
        return mom_r, L, mom_r_d, L_d
    u_r_n   = mom_r   / sig
    u_phi_n = L       / (sig   * r)
    v_r_n   = mom_r_d / sig_d
    v_phi_n = L_d     / (sig_d * r)
    tau_s = St / Om_K
    eps   = sig_d / sig
    if DUST_BACKREACTION:
        rate = (1.0 + eps) / tau_s
        f_g  = eps / (1.0 + eps)
        f_d  = 1.0 / (1.0 + eps)
    else:
        rate = 1.0 / tau_s
        f_g  = 0.0
        f_d  = 1.0
    fac   = -np.expm1(-dt * rate)
    drel_r   = v_r_n   - u_r_n
    drel_phi = v_phi_n - u_phi_n
    u_r_n   += f_g * drel_r   * fac
    v_r_n   -= f_d * drel_r   * fac
    u_phi_n += f_g * drel_phi * fac
    v_phi_n -= f_d * drel_phi * fac
    return (sig   * u_r_n,   sig   * r * u_phi_n,
            sig_d * v_r_n,   sig_d * r * v_phi_n)
def hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt, t_now=None):
    n = len(sig)
    u_r = mom_r   / np.maximum(sig,   SIGMA_FLOOR)
    v_r = mom_r_d / np.maximum(sig_d, SIGMA_FLOOR)
    u_face = face_velocity(u_r)
    F_sig = cyl_muscl_flux(sig,   u_face, dt)
    F_mr  = cyl_muscl_flux(mom_r, u_face, dt)
    F_L   = cyl_muscl_flux(L,     u_face, dt)
    safe   = sig > 10.0 * SIGMA_FLOOR
    Omega  = np.where(safe, L / (np.maximum(sig, SIGMA_FLOOR) * r * r), Om_K)
    nu_face  = 0.5 * (nu[:-1]  + nu[1:])
    sig_face = 0.5 * (sig[:-1] + sig[1:])
    dOmega   = (Omega[1:] - Omega[:-1]) / (r[1:] - r[:-1])
    rf       = r_face[1:-1]
    F_visc   = -nu_face * sig_face * rf**3 * dOmega
    F_visc[0]  = 0.0
    F_visc[-1] = 0.0
    F_L[1:-1] += F_visc
    cd = np.sqrt(np.maximum(cd2, 0.0))
    F_sig_d, F_mr_d, F_L_d = cyl_isothermal_dust_flux(
        sig_d, mom_r_d, L_d, v_r, cd, r_face)
    eps_face_diff = (sig_d[1:] / np.maximum(sig[1:], SIGMA_FLOOR) -
                     sig_d[:-1] / np.maximum(sig[:-1], SIGMA_FLOOR)) / (r[1:] - r[:-1])
    D_face        = 0.5 * (D_d[:-1]  + D_d[1:])
    sg_face_diff  = 0.5 * (sig[:-1] + sig[1:])
    F_diff        = -D_face * sg_face_diff * rf * eps_face_diff
    F_diff[0]  = 0.0
    F_diff[-1] = 0.0
    rvphi_cell = L_d / np.maximum(sig_d, SIGMA_FLOOR)
    upwind_L   = F_diff >= 0.0
    vr_up      = np.where(upwind_L, v_r[:-1],        v_r[1:])
    rvphi_up   = np.where(upwind_L, rvphi_cell[:-1], rvphi_cell[1:])
    F_sig_d[1:-1] += F_diff
    if TOMINAGA_MOM:
        F_mr_d [1:-1] += F_diff * vr_up
        F_L_d  [1:-1] += F_diff * rvphi_up
    interior = slice(1, n - 1)
    geom     = dt / (r[interior] * dr[interior])
    sig_new   = sig.copy();    mr_new   = mom_r.copy();   L_new   = L.copy()
    sig_d_new = sig_d.copy();  mr_d_new = mom_r_d.copy(); L_d_new = L_d.copy()
    sig_new  [interior] -= geom * (F_sig [2:-1] - F_sig [1:-2])
    mr_new   [interior] -= geom * (F_mr  [2:-1] - F_mr  [1:-2])
    L_new    [interior] -= geom * (F_L   [2:-1] - F_L   [1:-2])
    sig_d_new[interior] -= geom * (F_sig_d[2:-1] - F_sig_d[1:-2])
    mr_d_new [interior] -= geom * (F_mr_d [2:-1] - F_mr_d [1:-2])
    L_d_new  [interior] -= geom * (F_L_d  [2:-1] - F_L_d  [1:-2])
    floor_g = sig_new   < 10.0 * SIGMA_FLOOR
    floor_d = sig_d_new < 10.0 * SIGMA_FLOOR
    sig_new   = np.maximum(sig_new,   SIGMA_FLOOR)
    sig_d_new = np.maximum(sig_d_new, SIGMA_FLOOR)
    L_new   [floor_g] = sig_new  [floor_g] * r[floor_g] * v_K[floor_g]
    mr_new  [floor_g] = 0.0
    L_d_new [floor_d] = sig_d_new[floor_d] * r[floor_d] * v_K[floor_d]
    mr_d_new[floor_d] = 0.0
    if t_now is not None and T_RAMP_ORBITS > 0.0:
        ramp = min(t_now / (T_RAMP_ORBITS * T_orb), 1.0)**2
    else:
        ramp = 1.0
    L_new  [interior] += dt * ramp * sig_new  [interior] * torque_p[interior]
    u_phi_new = L_new   / (sig_new   * r)
    v_phi_new = L_d_new / (sig_d_new * r)
    dL_g  = L_new   - sig_new   * r * v_K
    dL_d  = L_d_new - sig_d_new * r * v_K
    Pi  = cs**2 * sig_new
    dPi = np.zeros_like(sig_new)
    dPi[1:-1] = (Pi[2:] - Pi[:-2]) / (r[2:] - r[:-2])
    S_r_g = dL_g * (u_phi_new + v_K) / r**2 - dPi
    mr_new[interior] += dt * S_r_g[interior]
    Pi_d  = cd2 * sig_d_new
    S_r_d = dL_d * (v_phi_new + v_K) / r**2 + Pi_d / r
    mr_d_new[interior] += dt * S_r_d[interior]
    mr_new, L_new, mr_d_new, L_d_new = epstein_drag_update(
        sig_new, mr_new, L_new, sig_d_new, mr_d_new, L_d_new, dt)
    apply_outflow_bc(sig_new,   mr_new,   L_new)
    apply_outflow_bc(sig_d_new, mr_d_new, L_d_new)
    return sig_new, mr_new, L_new, sig_d_new, mr_d_new, L_d_new
def build_damping_rate():
    domain_w   = r_max - r_min
    r_damp_in  = r_min + DAMP_FRAC_IN  * domain_w
    r_damp_out = r_max - DAMP_FRAC_OUT * domain_w
    R = np.zeros_like(r)
    mask_in = r < r_damp_in
    if mask_in.any():
        R[mask_in] = ((r_damp_in - r[mask_in]) / (r_damp_in - r_min))**3
    mask_out = r > r_damp_out
    if mask_out.any():
        R[mask_out] = ((r[mask_out] - r_damp_out) / (r_max - r_damp_out))**3
    tau_local = DAMP_TAU_FRAC * 2.0 * np.pi / Om_K
    return R / tau_local
def apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d,
                   target, rate, dt):
    sig_t, mom_r_t, L_t, sig_d_t, mom_r_d_t, L_d_t = target
    fac = np.minimum(dt * rate, 1.0)
    sig     -= fac * (sig     - sig_t)
    mom_r   -= fac * (mom_r   - mom_r_t)
    L       -= fac * (L       - L_t)
    sig_d   -= fac * (sig_d   - sig_d_t)
    mom_r_d -= fac * (mom_r_d - mom_r_d_t)
    L_d     -= fac * (L_d     - L_d_t)
    return sig, mom_r, L, sig_d, mom_r_d, L_d
def compute_dt(sig, mom_r, sig_d, mom_r_d):
    u_r    = mom_r[1:-1]   / np.maximum(sig[1:-1],   SIGMA_FLOOR)
    v_r    = mom_r_d[1:-1] / np.maximum(sig_d[1:-1], SIGMA_FLOOR)
    cd     = np.sqrt(cd2[1:-1])
    speed  = np.maximum(np.abs(u_r) + cs[1:-1], np.abs(v_r) + cd)
    dr_int  = dr[1:-1]
    dt_adv  = CFL * float(np.min(dr_int / np.maximum(speed, 1e-30)))
    dt_visc = 0.5 * float(np.min(dr_int**2 / np.maximum(nu[1:-1],  1e-30)))
    dt_diff = 0.5 * float(np.min(dr_int**2 / np.maximum(D_d[1:-1], 1e-30)))
    return min(dt_adv, dt_visc, dt_diff)
def run(nsh_drift_on=True, damping_on=True, stationary_visc=False, n_snap=5):
    sig, mom_r, L, sig_d, mom_r_d, L_d = initial_state(
        nsh_drift_on=nsh_drift_on, stationary_visc=stationary_visc)
    apply_outflow_bc(sig,   mom_r,   L)
    apply_outflow_bc(sig_d, mom_r_d, L_d)
    target = (sig.copy(), mom_r.copy(), L.copy(),
              sig_d.copy(), mom_r_d.copy(), L_d.copy())
    damp_rate = build_damping_rate() if damping_on else np.zeros_like(r)
    snap_dt  = t_end / n_snap
    snaps    = [(0.0, sig.copy(), L.copy(), sig_d.copy(), L_d.copy())]
    t_next   = snap_dt
    t, k = 0.0, 0
    while t < t_end:
        dt = min(compute_dt(sig, mom_r, sig_d, mom_r_d), t_end - t)
        sig, mom_r, L, sig_d, mom_r_d, L_d = hydro_step(
            sig, mom_r, L, sig_d, mom_r_d, L_d, dt, t_now=t)
        if damping_on:
            sig, mom_r, L, sig_d, mom_r_d, L_d = apply_damping(
                sig, mom_r, L, sig_d, mom_r_d, L_d,
                target, damp_rate, dt)
        t += dt; k += 1
        if t >= t_next or t >= t_end - 1e-12:
            snaps.append((t, sig.copy(), L.copy(), sig_d.copy(), L_d.copy()))
            t_next += snap_dt
        if k % 500 == 0:
            print(f"  step={k:6d}  t={t/T_orb:6.2f} Orbits  dt={dt:.4e}")
    return snaps
def saw(a):
    a  = a[1:-1]
    d2 = np.abs(a[2:] - 2.0 * a[1:-1] + a[:-2])
    sc = np.abs(a[2:]) + 2.0 * np.abs(a[1:-1]) + np.abs(a[:-2]) + 1e-30
    return float((d2 / sc).max())

def plot_snaps(snaps, title, pdfname):
    r_plot = r[1:-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps)))
    for (ts, sg, L_, sd, L_d_), c in zip(snaps, colors):
        label = f"{ts/T_orb:.1f} Orbits"
        axes[0].plot(r_plot, sg[1:-1],                    color=c, lw=1.5, label=label)
        axes[1].plot(r_plot, sd[1:-1],                    color=c, lw=1.5)
        axes[2].plot(r_plot, sd[1:-1] / np.maximum(sg[1:-1], SIGMA_FLOOR),
                     color=c, lw=1.5)
    axes[2].axhline(eps0, ls="--", color="gray", lw=0.8, label="ε₀")
    for ax, ttl, ylbl in zip(axes,
                              ["Gas Σ_g", "Staub Σ_d", "ε = Σ_d/Σ_g"],
                              ["Σ_g", "Σ_d", "ε"]):
        ax.set_xlabel("r"); ax.set_ylabel(ylbl); ax.set_title(ttl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if q_planet > 0.0:
            ax.axvline(r_planet, ls=":", color="crimson", lw=1.0)
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(pdfname, bbox_inches="tight")
    print(f"Gespeichert: {pdfname}")
def plot_snaps_interactive(snaps, title):
    from matplotlib.widgets import Slider
    r_plot = r[1:-1]
    times  = np.array([s[0] for s in snaps]) / T_orb
    def _eps(s):
        return s[3][1:-1] / np.maximum(s[1][1:-1], SIGMA_FLOOR)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.22, top=0.86)
    i0 = len(snaps) - 1
    lineG, = axes[0].plot(r_plot, snaps[i0][1][1:-1], color="C0", lw=1.6)
    lineD, = axes[1].plot(r_plot, snaps[i0][3][1:-1], color="C1", lw=1.6)
    lineE, = axes[2].plot(r_plot, _eps(snaps[i0]),    color="C2", lw=1.6)
    axes[2].axhline(eps0, ls="--", color="gray", lw=0.8, label="ε₀")
    def _ylim(getter):
        v = np.concatenate([getter(s) for s in snaps])
        v = v[np.isfinite(v) & (v > 0)]
        return (v.min() * 0.7, v.max() * 1.4) if v.size else (1e-3, 1.0)
    ylims = [_ylim(lambda s: s[1][1:-1]),
             _ylim(lambda s: s[3][1:-1]),
             _ylim(_eps)]
    for ax, ttl, ylbl, yl in zip(axes,
            ["Gas Σ_g", "Staub Σ_d", "ε = Σ_d/Σ_g"], ["Σ_g", "Σ_d", "ε"], ylims):
        ax.set_xlabel("r"); ax.set_ylabel(ylbl); ax.set_title(ttl)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*yl)
        ax.grid(True, which="both", alpha=0.3)
        if q_planet > 0.0:
            ax.axvline(r_planet, ls=":", color="crimson", lw=1.0)
    axes[2].legend(fontsize=8)
    sup = fig.suptitle(f"{title}   —   t = {times[i0]:.1f} Orbits")
    step  = (times[-1] - times[0]) / max(len(times) - 1, 1)
    ax_sl = fig.add_axes([0.15, 0.07, 0.70, 0.03])
    slider = Slider(ax_sl, "t [Orbits]", float(times[0]), float(times[-1]),
                    valinit=float(times[i0]), valstep=float(step))
    def _update(_val):
        i = int(np.argmin(np.abs(times - slider.val)))
        lineG.set_ydata(snaps[i][1][1:-1])
        lineD.set_ydata(snaps[i][3][1:-1])
        lineE.set_ydata(_eps(snaps[i]))
        sup.set_text(f"{title}   —   t = {times[i]:.1f} Orbits")
        fig.canvas.draw_idle()
    slider.on_changed(_update)
    fig._slider = slider
    return fig
def simulate(A_gap_val, St_val, nsh_drift_on, label, pdfname,
             damping_on=True, N_val=400, cd_factor=1.0,
             q_planet_val=None, r_planet_val=None,
             stationary_visc=False, t_orbits_val=None,
             interactive=True, n_snap=60):
    global A_gap, St, cd2, q_planet, r_planet, t_end_orbits, t_end
    A_gap = A_gap_val
    St    = St_val
    if q_planet_val is not None:
        q_planet = q_planet_val
    if r_planet_val is not None:
        r_planet = r_planet_val
    if t_orbits_val is not None:
        t_end_orbits = t_orbits_val
        t_end        = t_end_orbits * T_orb
    rebuild_grid(N_val)
    cd2   = cd2 * (cd_factor ** 2)
    print(f"\n {label}  N={N_val}, A_gap={A_gap_val}, St={St_val}, "
          f"q_planet={q_planet:g}, r_planet={r_planet:g}, t={t_end_orbits:g} Orb, "
          f"NSH-IC={'an' if nsh_drift_on else 'aus'}, "
          f"Damping={'an' if damping_on else 'aus'}, cd_factor={cd_factor}")
    snaps = run(nsh_drift_on=nsh_drift_on, damping_on=damping_on,
                stationary_visc=stationary_visc,
                n_snap=(n_snap if interactive else 5))
    
    if interactive:
        plot_snaps_interactive(snaps, label)
    else:
        plot_snaps(snaps, label, pdfname)
    return snaps
if __name__ == "__main__":
    import os as _os
    _os.makedirs("output", exist_ok=True)
    simulate(A_gap_val=0.0, St_val=0.1, nsh_drift_on=False,
             label="Lindblad-Planetenlücke, q=4.5e-4, N=400",
             pdfname="output/disk_v3_erweitert_linblad.pdf",
             damping_on=True, N_val=400, cd_factor=1.0,
             q_planet_val=4.5e-4, r_planet_val=1.0,
             stationary_visc=True, t_orbits_val=200.0)
    plt.show()
