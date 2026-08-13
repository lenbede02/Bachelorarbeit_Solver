import os
import numpy as np
import matplotlib.pyplot as plt
import disk_v3_erweitert_linblad as d3
import disk_v3_erweitert as d3_base
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(_HERE, '..', 'thesis', 'figures', 'verifikation'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
import numval
numval.install('verifikation')
import progress
C_CAL = 0.4478

def configure(A_gap_val=0.0, St_val=0.01, alpha_val=0.01, N_val=200, cd_factor=1.0):
    d3.A_gap = A_gap_val
    d3.St = St_val
    d3.alpha = alpha_val
    d3.q_planet = 0.0
    d3.rebuild_grid(N_val)
    d3.cd2 = d3.cd2 * cd_factor ** 2

def run_full_snaps(t_end_orbits=20.0, n_snap=10, nsh_drift_on=True, damping_on=True, stationary_u_r=False, track_boundary_flux=False, label=''):
    t_end = t_end_orbits * d3.T_orb
    (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.initial_state(nsh_drift_on=nsh_drift_on)
    if stationary_u_r:
        u_r_stat = -1.5 * d3.nu / d3.r
        mom_r[:] = sig * u_r_stat
    d3.apply_outflow_bc(sig, mom_r, L)
    d3.apply_outflow_bc(sig_d, mom_r_d, L_d)
    target = (sig.copy(), mom_r.copy(), L.copy(), sig_d.copy(), mom_r_d.copy(), L_d.copy())
    damp_rate = d3.build_damping_rate() if damping_on else np.zeros_like(d3.r)
    snap_dt = t_end / n_snap
    snaps = [_snap(0.0, sig, mom_r, L, sig_d, mom_r_d, L_d)]
    t_next = snap_dt
    L_flux_in_acc = 0.0
    L_flux_out_acc = 0.0
    flux_log = [(0.0, 0.0, 0.0)]
    bar = progress.Bar(label) if label else None
    (t, k) = (0.0, 0)
    while t < t_end:
        dt = min(d3.compute_dt(sig, mom_r, sig_d, mom_r_d), t_end - t)
        if track_boundary_flux:
            u_r_ia = mom_r[1] / max(sig[1], d3.SIGMA_FLOOR)
            u_r_ob = mom_r[-2] / max(sig[-2], d3.SIGMA_FLOOR)
            v_r_ia = mom_r_d[1] / max(sig_d[1], d3.SIGMA_FLOOR)
            v_r_ob = mom_r_d[-2] / max(sig_d[-2], d3.SIGMA_FLOOR)
            L_flux_in = (L[1] * min(u_r_ia, 0.0) + L_d[1] * min(v_r_ia, 0.0)) * 2 * np.pi * d3.r_face[1]
            L_flux_out = (L[-2] * max(u_r_ob, 0.0) + L_d[-2] * max(v_r_ob, 0.0)) * 2 * np.pi * d3.r_face[-2]
            L_flux_in_acc += L_flux_in * dt
            L_flux_out_acc += L_flux_out * dt
        (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt)
        if damping_on:
            (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d, target, damp_rate, dt)
        t += dt
        k += 1
        if t >= t_next or t >= t_end - 1e-12:
            snaps.append(_snap(t, sig, mom_r, L, sig_d, mom_r_d, L_d))
            flux_log.append((t, L_flux_in_acc, L_flux_out_acc))
            t_next += snap_dt
        if bar is not None:
            bar.update(t / t_end)
        elif k % 500 == 0:
            print(f'  step={k:6d}  t={t / d3.T_orb:6.2f} Orbits  dt={dt:.4e}')
    if bar is not None:
        bar.done()
    if track_boundary_flux:
        return (snaps, flux_log)
    return snaps

def _snap(t, sig, mom_r, L, sig_d, mom_r_d, L_d):
    return {'t': t, 'sig': sig.copy(), 'mom_r': mom_r.copy(), 'L': L.copy(), 'sig_d': sig_d.copy(), 'mom_r_d': mom_r_d.copy(), 'L_d': L_d.copy()}

def V1_lbp_stationary():
    print('\n══ V1: Lynden-Bell & Pringle stationäre Akkretion ══')
    configure(A_gap_val=0.0, St_val=0.01, alpha_val=0.01, N_val=400, cd_factor=1.0)
    snaps = run_full_snaps(t_end_orbits=60.0, n_snap=12, damping_on=True, stationary_u_r=True)
    snaps_settled = [s for s in snaps if s['t'] / d3.T_orb >= 40.0 - 1e-09]
    r_active = d3.r[1:-1]
    sig_ic = snaps[0]['sig'][1:-1]
    domain_w = d3.r_max - d3.r_min
    r_in = d3.r_min + d3.DAMP_FRAC_IN * domain_w
    r_out = d3.r_max - d3.DAMP_FRAC_OUT * domain_w
    bulk = (r_active > r_in) & (r_active < r_out)
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps_settled)))
    for (s, c) in zip(snaps_settled, colors):
        ratio = s['sig'][1:-1] / sig_ic
        axes[0].plot(r_active, ratio, color=c, lw=1.3, label=f"t = {s['t'] / d3.T_orb:.1f} Orb")
    s_end = snaps_settled[-1]
    u_r_e = s_end['mom_r'][1:-1] / np.maximum(s_end['sig'][1:-1], d3.SIGMA_FLOOR)
    Mdot_e = -2.0 * np.pi * r_active * s_end['sig'][1:-1] * u_r_e
    axes[1].plot(r_active, Mdot_e, color='C0', lw=1.4, label=f"t = {s_end['t'] / d3.T_orb:.0f} Orb")
    axes[0].axhline(1.0, ls='--', color='gray', lw=0.6)
    axes[0].axvspan(d3.r_min, r_in, alpha=0.15, color='gray')
    axes[0].axvspan(r_out, d3.r_max, alpha=0.15, color='gray')
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\\Sigma_g(r,t)\\,/\\,\\Sigma_g(r,0)$')
    axes[0].set_title('V1a: Stationarität von Σ_g (eingeschwungen, t = 40–60 Orb)')
    axes[0].set_xscale('log')
    axes[0].set_ylim(0.95, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    nu_r0 = d3.alpha * (d3.h0 * np.sqrt(d3.GM / d3.r0)) ** 2 / np.sqrt(d3.GM / d3.r0 ** 3)
    Mdot_analyt = 3.0 * np.pi * nu_r0 * d3.Sigma0
    axes[1].axhline(Mdot_analyt, ls='--', color='red', lw=1.0, label=f'Ṁ = 3πν(r₀)Σ₀ = {Mdot_analyt:.2e}')
    axes[1].axvspan(d3.r_min, r_in, alpha=0.15, color='gray')
    axes[1].axvspan(r_out, d3.r_max, alpha=0.15, color='gray')
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('$\\dot M = -2\\pi\\,r\\,\\Sigma_g\\,u_r$')
    axes[1].set_title('V1b: Massenfluss-Konstanz (t = 60 Orb)')
    axes[1].set_xscale('log')
    axes[1].set_ylim(0.0, 3.0 * Mdot_analyt)
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('V1: LBP-Stationarität nach Anlauf-Transient — Damping-Zonen grau hinterlegt')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V1_lbp_stationary.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    final_dev = float(np.abs(snaps[-1]['sig'][1:-1][bulk] / sig_ic[bulk] - 1.0).max())
    nu_r0 = d3.alpha * (d3.h0 * np.sqrt(d3.GM / d3.r0)) ** 2 / np.sqrt(d3.GM / d3.r0 ** 3)
    Mdot_analyt = 3.0 * np.pi * nu_r0 * d3.Sigma0
    u_r_final = snaps[-1]['mom_r'][1:-1] / np.maximum(snaps[-1]['sig'][1:-1], d3.SIGMA_FLOOR)
    Mdot_final = -2.0 * np.pi * r_active * snaps[-1]['sig'][1:-1] * u_r_final
    Mdot_bulk = Mdot_final[bulk]
    Mdot_rel_err = float(np.abs(Mdot_bulk - Mdot_analyt).max() / abs(Mdot_analyt))
    print(f'V1a: max |Σ_g(t_end)/Σ_g(0) − 1| im Bulk     = {final_dev:.3e}')
    print(f'V1b: max |Ṁ_num − Ṁ_analyt|/|Ṁ_analyt| Bulk  = {Mdot_rel_err:.3e}')
    print(f'     (Ṁ_analyt = 3π·ν(r_0)·Σ_0 = {Mdot_analyt:.3e})')
    return (final_dev, Mdot_rel_err)

def V2_nsh_drift():
    print('\n══ V2: NSH-Driftlösung ══')
    (damp_in_save, damp_out_save) = (d3.DAMP_FRAC_IN, d3.DAMP_FRAC_OUT)
    (d3.DAMP_FRAC_IN, d3.DAMP_FRAC_OUT) = (0.15, 0.15)
    configure(A_gap_val=0.0, St_val=0.1, alpha_val=0.01, N_val=400, cd_factor=1.0)
    snaps = run_full_snaps(t_end_orbits=5.0, n_snap=5, damping_on=True)
    sig_ic_full = snaps[0]['sig']
    Pi_full = d3.cs ** 2 * sig_ic_full
    dPi_full = np.zeros_like(sig_ic_full)
    dPi_full[1:-1] = (Pi_full[2:] - Pi_full[:-2]) / (d3.r[2:] - d3.r[:-2])
    r_active = d3.r[1:-1]
    Pi = Pi_full[1:-1]
    dPi = dPi_full[1:-1]
    H_a = d3.H[1:-1]
    v_K_a = d3.v_K[1:-1]
    eta = -(r_active / (2.0 * np.maximum(Pi, 1e-30))) * (H_a / r_active) ** 2 * dPi
    v_r_NSH = -2.0 * eta * v_K_a * d3.St / (1.0 + d3.St ** 2)
    domain_w = d3.r_max - d3.r_min
    r_in = d3.r_min + d3.DAMP_FRAC_IN * domain_w
    r_out = d3.r_max - d3.DAMP_FRAC_OUT * domain_w
    bulk = (r_active > r_in) & (r_active < r_out)
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps)))
    for (s, c) in zip(snaps, colors):
        v_r_d = s['mom_r_d'][1:-1] / np.maximum(s['sig_d'][1:-1], d3.SIGMA_FLOOR)
        axes[0].plot(r_active, v_r_d, color=c, lw=1.3, label=f"t = {s['t'] / d3.T_orb:.1f} Orb")
        rel_err = (v_r_d - v_r_NSH) / np.maximum(np.abs(v_r_NSH), 1e-30)
        axes[1].plot(r_active, rel_err, color=c, lw=1.3)
    axes[0].plot(r_active, v_r_NSH, '--', color='red', lw=1.8, label='v_r,NSH (analytisch)')
    v_r_d_fin = snaps[-1]['mom_r_d'][1:-1] / np.maximum(snaps[-1]['sig_d'][1:-1], d3.SIGMA_FLOOR)
    r_bulk = r_active[bulk]
    vrd_bulk = v_r_d_fin[bulk]
    vnsh_bulk = v_r_NSH[bulk]
    nbin = 14
    edges = np.geomspace(r_bulk[0], r_bulk[-1], nbin + 1)
    bidx = np.clip(np.digitize(r_bulk, edges) - 1, 0, nbin - 1)
    have = [b for b in range(nbin) if np.any(bidx == b)]
    r_bin = np.array([r_bulk[bidx == b].mean() for b in have])
    vrd_bin = np.array([vrd_bulk[bidx == b].mean() for b in have])
    vnsh_bin = np.array([vnsh_bulk[bidx == b].mean() for b in have])
    relerr_bin = (vrd_bin - vnsh_bin) / np.maximum(np.abs(vnsh_bin), 1e-30)
    axes[0].plot(r_bin, vrd_bin, 'ko-', lw=1.5, ms=4, label='numerisch (radial gemittelt)')
    axes[1].plot(r_bin, relerr_bin, 'ko-', lw=1.6, ms=4, zorder=5, label='radial gemittelt (Drift-Signal)')
    axes[0].axvspan(d3.r_min, r_in, alpha=0.15, color='gray')
    axes[0].axvspan(r_out, d3.r_max, alpha=0.15, color='gray')
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$v_{r,d}$')
    axes[0].set_title('V2a: Staubdrift numerisch vs. analytisch')
    axes[0].set_xscale('log')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(0.0, ls='--', color='gray', lw=0.6)
    axes[1].axvspan(d3.r_min, r_in, alpha=0.15, color='gray')
    axes[1].axvspan(r_out, d3.r_max, alpha=0.15, color='gray')
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('$(v_{r,d}-v_{r,\\mathrm{NSH}})\\,/\\,|v_{r,\\mathrm{NSH}}|$')
    axes[1].set_title('V2b: Relativer Fehler (dünn: pro Zelle, dick: gemittelt)')
    axes[1].set_xscale('log')
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('V2: NSH-Drift — St=0.1, α=1e-2, A_gap=0, 5 Orbits (feine Gitter-Oszillationen = druckfreier Staub, mitteln sich heraus)')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V2_nsh_drift.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    v_r_d_final = snaps[-1]['mom_r_d'][1:-1] / np.maximum(snaps[-1]['sig_d'][1:-1], d3.SIGMA_FLOOR)
    rel_err_final = (v_r_d_final - v_r_NSH) / np.maximum(np.abs(v_r_NSH), 1e-30)
    rel_err_max = float(np.abs(rel_err_final[bulk]).max()) if bulk.any() else float('nan')
    rel_err_rms = float(np.sqrt(np.mean(rel_err_final[bulk] ** 2))) if bulk.any() else float('nan')
    rel_err_mean = float(np.abs(np.mean(rel_err_final[bulk]))) if bulk.any() else float('nan')
    rel_err_binned_rms = float(np.sqrt(np.mean(relerr_bin ** 2)))
    print(f'V2 Bulk-Statistik (außerhalb Damping-Zonen):')
    print(f'  |⟨Δv_r,d⟩|/|v_NSH| (Mittel)  = {rel_err_mean:.3e}   ← Drift-Übereinstimmung im Mittel')
    print(f'  RMS radial gemittelt        = {rel_err_binned_rms:.3e}   ← Gitter-Oszill. herausgemittelt')
    print(f'  RMS pro Zelle               = {rel_err_rms:.3e}   (inkl. druckfreier-Staub-Oszillationen)')
    print(f'  max pro Zelle               = {rel_err_max:.3e}   (Oszillations-Spitze, dokumentiert)')
    (d3.DAMP_FRAC_IN, d3.DAMP_FRAC_OUT) = (damp_in_save, damp_out_save)
    return (rel_err_max, rel_err_binned_rms, rel_err_mean)

def V3_angular_momentum():
    print('\n══ V3: Drehimpulserhaltung mit Rand-Bilanz ══')
    configure(A_gap_val=0.7, St_val=0.01, alpha_val=0.01, N_val=400, cd_factor=1.0)
    (snaps, flux_log) = run_full_snaps(t_end_orbits=20.0, n_snap=40, damping_on=False, track_boundary_flux=True)
    times = np.array([s['t'] for s in snaps])
    L_int = np.array([np.sum((s['L'][1:-1] + s['L_d'][1:-1]) * 2.0 * np.pi * d3.r[1:-1] * d3.dr[1:-1]) for s in snaps])
    flux_arr = np.array(flux_log)
    L_balance = L_int + flux_arr[:, 2] - flux_arr[:, 1]
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(times / d3.T_orb, L_int / L_int[0], 'b-', lw=1.4, label='L_aktiv(t) / L(0)')
    axes[0].plot(times / d3.T_orb, L_balance / L_int[0], 'g-', lw=1.4, label='(L_aktiv + L_Rand) / L(0)')
    axes[0].axhline(1.0, ls='--', color='gray', lw=0.6)
    axes[0].set_xlabel('t / T_orb')
    axes[0].set_ylabel('normierter Drehimpuls')
    axes[0].set_title('V3a: Erhaltung mit Randfluss-Korrektur')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times / d3.T_orb, (L_balance - L_int[0]) / L_int[0], 'r-', lw=1.4)
    axes[1].axhline(0.0, ls='--', color='gray', lw=0.6)
    axes[1].set_xlabel('t / T_orb')
    axes[1].set_ylabel('$\\Delta L_\\mathrm{tot}\\,/\\,L(0)$')
    axes[1].set_title('V3b: Numerischer Erhaltungsfehler (bilanziert)')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('V3: Drehimpulserhaltung — Damping AUS, Outflow + Rand-Bilanz')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V3_angular_momentum.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    drift_raw = float(np.abs(L_int / L_int[0] - 1.0).max())
    drift_balance = float(np.abs((L_balance - L_int[0]) / L_int[0]).max())
    print(f'V3a: max |L_aktiv/L(0) − 1|             = {drift_raw:.3e}  (inkl. Randverlust)')
    print(f'V3b: max |(L_aktiv + L_Rand − L(0))/L(0)| = {drift_balance:.3e}  (= reiner Numerikfehler)')
    return drift_balance

def gaussian_bump(r, r0_bump, sigma_bump, amplitude):
    return amplitude * np.exp(-0.5 * ((r - r0_bump) / sigma_bump) ** 2)

def fit_gaussian_moments(r, f, mask=None):
    if mask is None:
        mask = np.ones_like(f, dtype=bool)
    w = np.maximum(f[mask], 0.0)
    if w.sum() < 1e-30:
        return (float('nan'), float('nan'))
    mu = float(np.sum(r[mask] * w) / w.sum())
    var = float(np.sum((r[mask] - mu) ** 2 * w) / w.sum())
    return (mu, float(np.sqrt(max(var, 0.0))))

def _ic_with_dust_bump(r0_bump, sigma_bump, amplitude_eps, nsh_drift_on=True):
    (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.initial_state(nsh_drift_on=nsh_drift_on)
    v_r_d = mom_r_d / np.maximum(sig_d, d3.SIGMA_FLOOR)
    v_phi_d = L_d / (np.maximum(sig_d, d3.SIGMA_FLOOR) * d3.r)
    bump = gaussian_bump(d3.r, r0_bump, sigma_bump, amplitude_eps * sig)
    sig_d = sig_d + bump
    mom_r_d = sig_d * v_r_d
    L_d = sig_d * d3.r * v_phi_d
    d3.apply_outflow_bc(sig, mom_r, L)
    d3.apply_outflow_bc(sig_d, mom_r_d, L_d)
    return (sig, mom_r, L, sig_d, mom_r_d, L_d)

def V4_pure_advection():
    print('\n══ V4: Pure Advection (Bump-Tracking) ══')
    configure(A_gap_val=0.0, St_val=0.1, alpha_val=0.01, N_val=600, cd_factor=1.0)
    (r0_bump, sigma_bump) = (2.5, 0.15)

    def _run_with_ic(sig, mom_r, L, sig_d, mom_r_d, L_d, t_end, n_snap=6):
        target = (sig.copy(), mom_r.copy(), L.copy(), sig_d.copy(), mom_r_d.copy(), L_d.copy())
        dr = d3.build_damping_rate()
        out = [_snap(0.0, sig, mom_r, L, sig_d, mom_r_d, L_d)]
        snap_dt = t_end / n_snap
        (t, t_next) = (0.0, snap_dt)
        while t < t_end:
            dt = min(d3.compute_dt(sig, mom_r, sig_d, mom_r_d), t_end - t)
            (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt)
            (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d, target, dr, dt)
            t += dt
            if t >= t_next or t >= t_end - 1e-12:
                out.append(_snap(t, sig, mom_r, L, sig_d, mom_r_d, L_d))
                t_next += snap_dt
        return out
    t_end = 50.0 * d3.T_orb
    (sig0, mom_r0, L0, sig_d0, mom_r_d0, L_d0) = d3.initial_state(nsh_drift_on=True)
    d3.apply_outflow_bc(sig0, mom_r0, L0)
    d3.apply_outflow_bc(sig_d0, mom_r_d0, L_d0)
    snaps_ref = _run_with_ic(sig0, mom_r0, L0, sig_d0, mom_r_d0, L_d0, t_end)
    (sigB, mom_rB, LB, sig_dB, mom_r_dB, L_dB) = _ic_with_dust_bump(r0_bump, sigma_bump, amplitude_eps=0.02, nsh_drift_on=True)
    snaps = _run_with_ic(sigB, mom_rB, LB, sig_dB, mom_r_dB, L_dB, t_end)
    Pi0 = d3.cs ** 2 * snaps[0]['sig']
    dPi = np.zeros_like(Pi0)
    dPi[1:-1] = (Pi0[2:] - Pi0[:-2]) / (d3.r[2:] - d3.r[:-2])
    eta_arr = -(d3.r / (2 * np.maximum(Pi0, 1e-30))) * (d3.H / d3.r) ** 2 * dPi
    idx_r0 = int(np.argmin(np.abs(d3.r - r0_bump)))
    v_NSH = -2.0 * eta_arr[idx_r0] * d3.v_K[idx_r0] * d3.St / (1.0 + d3.St ** 2)
    (mu_log, sig_log) = ([], [])
    box_half = 4.0 * sigma_bump
    for (sB, sR) in zip(snaps, snaps_ref):
        mu_expected = r0_bump if not mu_log else mu_log[-1]
        box = (d3.r > mu_expected - box_half) & (d3.r < mu_expected + box_half)
        excess = sB['sig_d'][box] - sR['sig_d'][box]
        (mu, sg) = fit_gaussian_moments(d3.r[box], excess)
        mu_log.append(mu)
        sig_log.append(sg)
    times = np.array([s['t'] for s in snaps])
    (mu_log, sig_log) = (np.array(mu_log), np.array(sig_log))
    mu_analyt = r0_bump + v_NSH * times
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps)))
    for (s, sR, c) in zip(snaps, snaps_ref, colors):
        axes[0].plot(d3.r, s['sig_d'] - sR['sig_d'], color=c, lw=1.2, label=f"t={s['t'] / d3.T_orb:.1f} Orb")
    axes[0].set_xlim(r0_bump - 5 * sigma_bump, r0_bump + 5 * sigma_bump)
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\\Sigma_d - \\Sigma_{d,\\mathrm{ref}}$')
    axes[0].set_title('V4a: Bump-Excess (Σ_d − Reference)')
    axes[0].axhline(0.0, ls=':', color='gray', lw=0.6)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times / d3.T_orb, mu_log, 'bo-', lw=1.2, label='numerisch ⟨r⟩')
    axes[1].plot(times / d3.T_orb, mu_analyt, 'r--', lw=1.4, label='analyt. r₀+v_NSH·t')
    axes[1].set_xlabel('t / T_orb')
    axes[1].set_ylabel('Bump-Schwerpunkt')
    axes[1].set_title(f'V4b: v_NSH(r₀) = {v_NSH:.3e}')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('V4: Advection+Drift — A=0, St=0.1, α=1e-2, 50 Orbits')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V4_pure_advection.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    pos_err = float(np.abs(mu_log[-1] - mu_analyt[-1]) / abs(mu_analyt[-1] - r0_bump + 1e-30))
    width_drift = float((sig_log[-1] - sig_log[0]) / sig_log[0])
    print(f'V4a Position-Fehler bei t_end:  |Δμ|/|Δr_analyt| = {pos_err:.3e}')
    print(f'V4b Breiten-Drift (physik-dominiert, vgl. V5): Δσ/σ₀ = {width_drift:+.3e}')
    return (pos_err, width_drift)

def V5_pure_diffusion(N_val=800, save_pdf=True):
    print(f'\n══ V5: Pure Diffusion (Bump-Verbreiterung), N={N_val} ══')
    configure(A_gap_val=0.0, St_val=0.001, alpha_val=0.01, N_val=N_val, cd_factor=1.0)
    (r0_bump, sigma_bump) = (2.5, 0.2)
    (sig, mom_r, L, sig_d, mom_r_d, L_d) = _ic_with_dust_bump(r0_bump, sigma_bump, amplitude_eps=0.1, nsh_drift_on=False)
    target = (sig.copy(), mom_r.copy(), L.copy(), sig_d.copy(), mom_r_d.copy(), L_d.copy())
    damp_rate = d3.build_damping_rate()
    t_end = 20.0 * d3.T_orb
    snap_dt = t_end / 10
    snaps = [_snap(0.0, sig, mom_r, L, sig_d, mom_r_d, L_d)]
    (t, t_next) = (0.0, snap_dt)
    bar = progress.Bar(f'V5  N={N_val}')
    while t < t_end:
        dt = min(d3.compute_dt(sig, mom_r, sig_d, mom_r_d), t_end - t)
        (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt)
        (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d, target, damp_rate, dt)
        t += dt
        bar.update(t / t_end)
        if t >= t_next or t >= t_end - 1e-12:
            snaps.append(_snap(t, sig, mom_r, L, sig_d, mom_r_d, L_d))
            t_next += snap_dt
    bar.done()
    box = (d3.r > r0_bump - 5 * sigma_bump) & (d3.r < r0_bump + 5 * sigma_bump)
    sig0_bg = snaps[0]['sig_d'] - gaussian_bump(d3.r, r0_bump, sigma_bump, 0.1 * snaps[0]['sig'])
    (times, sigmas) = ([], [])
    for s in snaps:
        excess = s['sig_d'] - sig0_bg
        (_, sg) = fit_gaussian_moments(d3.r, excess, mask=box)
        times.append(s['t'])
        sigmas.append(sg)
    (times, sigmas) = (np.array(times), np.array(sigmas))
    var = sigmas ** 2
    idx_r0 = int(np.argmin(np.abs(d3.r - r0_bump)))
    D0 = float(d3.D_d[idx_r0])
    var_analyt = var[0] + 2.0 * D0 * times
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps)))
    for (s, c) in zip(snaps, colors):
        axes[0].plot(d3.r, s['sig_d'], color=c, lw=1.2, label=f"t={s['t'] / d3.T_orb:.1f} Orb")
    axes[0].set_xlim(r0_bump - 6 * sigma_bump, r0_bump + 6 * sigma_bump)
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\\Sigma_d$')
    axes[0].set_title('V5a: Σ_d-Bump-Verbreiterung')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times / d3.T_orb, var, 'bo-', lw=1.2, label='numerisch σ²(t)')
    axes[1].plot(times / d3.T_orb, var_analyt, 'r--', lw=1.4, label=f'σ²(0) + 2·D_d·t   (D_d={D0:.2e})')
    axes[1].set_xlabel('t / T_orb')
    axes[1].set_ylabel('$\\sigma^2$')
    axes[1].set_title('V5b: Varianz-Wachstum')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f'V5: Pure Diffusion — A=0, St=1e-3 (Drift ≈ 0), α=1e-2, N={N_val}')
    plt.tight_layout()
    if save_pdf:
        fname = os.path.join(OUTPUT_DIR, 'V5_pure_diffusion.pdf')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Gespeichert: {fname}')
    plt.close()
    (slope_num, _) = np.polyfit(times, var, 1)
    slope_err = float(abs(slope_num - 2.0 * D0) / (2.0 * D0))
    print(f'V5  d(σ²)/dt numerisch = {slope_num:.3e},  2·D_d = {2 * D0:.3e}')
    print(f'    rel. Fehler        = {slope_err:.3e}')
    return slope_err

def V5_convergence(Ns=(400, 800, 1600)):
    print('\n══ V5b: Diffusions-Konvergenz (Slope-Fehler vs. N) ══')
    Ns = list(Ns)
    errs = []
    for (i, N) in enumerate(Ns):
        print(f'  → V5-Lauf {i + 1}/{len(Ns)}  N={N}')
        errs.append(V5_pure_diffusion(N_val=N, save_pdf=False))
    errs = np.array(errs, float)
    N_arr = np.array(Ns, float)
    (slope, _) = np.polyfit(np.log(N_arr), np.log(np.maximum(errs, 1e-30)), 1)
    (fig, ax) = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.loglog(N_arr, errs, 'bo-', lw=1.4, label=f'|Δslope|/(2·D_d)  (Rate ≈ {-slope:.2f})')
    ax.loglog(N_arr, errs[0] * (N_arr[0] / N_arr) ** 1, 'k:', lw=0.8, label='1. Ordnung (∝ 1/N)')
    ax.set_xlabel('N')
    ax.set_ylabel('rel. Slope-Fehler')
    ax.set_title('V5b: Diffusions-Konvergenz — numerischer Diffusionsboden → 0')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V5b_diffusion_convergence.pdf')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    for (N, e) in zip(Ns, errs):
        print(f'    N={N:5d}  |Δslope|/(2D_d) = {e:.3e}')
    numval.save_rows('verifikation', 'V5b_diffusion_convergence', ['N', 'slope_fehler_rel'], list(zip(Ns, errs)))
    return dict(Ns=Ns, errs=errs.tolist())

def V6_steady_state_trap(N_val=800, save_pdf=True, return_detail=False):
    print(f'\n══ V6: Steady-State Dullemond-Trap, N={N_val} ══')
    configure(A_gap_val=0.7, St_val=0.1, alpha_val=0.001, N_val=N_val, cd_factor=1.0)
    snaps = run_full_snaps(t_end_orbits=150.0, n_snap=8, damping_on=True, label=f'V6  N={N_val}')
    s_fin = snaps[-1]
    eps = s_fin['sig_d'][1:-1] / np.maximum(s_fin['sig'][1:-1], d3.SIGMA_FLOOR)
    r_act = d3.r[1:-1]
    Pi = d3.cs[1:-1] ** 2 * s_fin['sig'][1:-1]
    box_outer = (r_act > 1.0) & (r_act < 1.0 + 4 * d3.w_gap)
    if box_outer.any():
        r_peak = float(r_act[box_outer][int(np.argmax(Pi[box_outer]))])
    else:
        r_peak = 1.0 + d3.w_gap
    eps_bg = float(np.median(eps))
    eps_excess = eps - eps_bg
    ipk = int(np.argmax(eps_excess))
    r_pk_num = float(r_act[ipk])
    half = 0.5 * eps_excess[ipk]
    r_half_in = r_pk_num
    for k in range(ipk, 0, -1):
        if eps_excess[k] <= half:
            f = (half - eps_excess[k]) / (eps_excess[k + 1] - eps_excess[k] + 1e-30)
            r_half_in = r_act[k] + f * (r_act[k + 1] - r_act[k])
            break
    hwhm_in = max(r_pk_num - r_half_in, 1e-30)
    sig_num = hwhm_in / np.sqrt(2.0 * np.log(2.0))
    Pi_full = d3.cs ** 2 * s_fin['sig']
    dPi_full = np.zeros_like(Pi_full)
    dPi_full[1:-1] = (Pi_full[2:] - Pi_full[:-2]) / (d3.r[2:] - d3.r[:-2])
    eta_full = -(d3.r / (2 * np.maximum(Pi_full, 1e-30))) * (d3.H / d3.r) ** 2 * dPi_full
    v_drift = -2.0 * eta_full * d3.v_K * d3.St / (1.0 + d3.St ** 2)
    dv_dr = np.zeros_like(v_drift)
    dv_dr[1:-1] = (v_drift[2:] - v_drift[:-2]) / (d3.r[2:] - d3.r[:-2])
    idx_peak = int(np.argmin(np.abs(d3.r - r_peak)))
    v_prime = float(dv_dr[idx_peak])
    D_peak = float(d3.D_d[idx_peak])
    sig_analyt = float(np.sqrt(D_peak / max(abs(v_prime), 1e-30)))
    v_r_d_pk = float(s_fin['mom_r_d'][idx_peak] / max(s_fin['sig_d'][idx_peak], d3.SIGMA_FLOOR))
    cd_pk = float(np.sqrt(max(float(d3.cd2[idx_peak]), 0.0)))
    D_num = 0.5 * (abs(v_r_d_pk) + cd_pk) * float(d3.dr[idx_peak])
    Dnum_ratio = D_num / max(D_peak, 1e-30)
    eps_analyt = eps_bg + eps_excess.max() * np.exp(-0.5 * ((r_act - r_peak) / sig_analyt) ** 2)
    (fig, ax) = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(r_act, eps, 'b-', lw=1.4, label=f"numerisch (t={s_fin['t'] / d3.T_orb:.0f} Orb)")
    ax.plot(r_act, eps_analyt, 'r--', lw=1.4, label=f'Dullemond-Gauß: σ_trap = {sig_analyt:.3e}')
    ax.plot([r_half_in, r_pk_num], [eps_bg + half, eps_bg + half], 'g-', lw=2.5, label=f'innere HWHM → σ_num = {sig_num:.3e}')
    ax.axvline(r_pk_num, ls=':', color='gray', lw=0.8)
    ax.axhline(eps_bg, ls=':', color='gray', lw=0.6)
    ax.set_xlabel('r')
    ax.set_ylabel('$\\varepsilon = \\Sigma_d/\\Sigma_g$')
    ax.set_title(f'V6: Steady-State-Trap — σ_num (innere HWHM) = {sig_num:.3e},  σ_analyt = {sig_analyt:.3e},  Verh. = {sig_num / sig_analyt:.2f}')
    ax.set_xlim(r_peak - 3 * d3.w_gap, r_peak + 3 * d3.w_gap)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_pdf:
        fname = os.path.join(OUTPUT_DIR, 'V6_steady_state_trap.pdf')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Gespeichert: {fname}')
    plt.close()
    sig_err = float(abs(sig_num - sig_analyt) / max(sig_analyt, 1e-30))
    print(f'V6  σ_trap numerisch (innere HWHM) = {sig_num:.3e}')
    print(f'    σ_trap analyt. (lokal-linear)  = {sig_analyt:.3e}')
    print(f'    Verhältnis num/analyt           = {sig_num / sig_analyt:.2f}  (Soll ~1)')
    print(f'    numer. Diffusionsboden D_num/D_d = {Dnum_ratio:.3e}  (≪1 ⇒ Fallenbreite physikalisch gesetzt)')
    print(f'    Hinweis: die innere ε-Flanke folgt der Dullemond-Drift-Diffusions-')
    print(f'    Balance; der äußere Schwanz ist das (asymmetrische) Druckgradient-')
    print(f'    Profil und gehört nicht zur lokalen Gauß-Breite.')
    if return_detail:
        return dict(sig_err=sig_err, sig_num=sig_num, sig_analyt=sig_analyt, D_num=D_num, D_peak=D_peak, Dnum_ratio=Dnum_ratio)
    return sig_err

def V6_convergence(Ns=(400, 800, 1600)):
    print('\n══ V6b: Trap-Breiten-Konvergenz (σ_num, D_num/D_d vs. N) ══')
    Ns = list(Ns)
    (sig_num, sig_an, ratio) = ([], [], [])
    for (i, N) in enumerate(Ns):
        print(f'  → V6-Lauf {i + 1}/{len(Ns)}  N={N}')
        d = V6_steady_state_trap(N_val=N, save_pdf=False, return_detail=True)
        sig_num.append(d['sig_num'])
        sig_an.append(d['sig_analyt'])
        ratio.append(d['Dnum_ratio'])
    N_arr = np.array(Ns, float)
    (sig_num, sig_an, ratio) = map(lambda a: np.array(a, float), (sig_num, sig_an, ratio))
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(N_arr, sig_num, 'bo-', lw=1.4, label='$\\sigma_\\mathrm{num}$ (innere HWHM)')
    axes[0].plot(N_arr, sig_an, 'r--', lw=1.4, label="$\\sigma_\\mathrm{analyt}=\\sqrt{D_d/|v'|}$")
    axes[0].set_xscale('log')
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('$\\sigma_\\mathrm{trap}$')
    axes[0].set_title('V6b: Fallenbreite konvergiert gegen Dullemond')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, which='both')
    axes[1].loglog(N_arr, ratio, 'gs-', lw=1.4, label='$D_\\mathrm{num}/D_d$')
    axes[1].loglog(N_arr, ratio[0] * (N_arr[0] / N_arr) ** 1, 'k:', lw=0.8, label='∝ 1/N')
    axes[1].set_xlabel('N')
    axes[1].set_ylabel('$D_\\mathrm{num}/D_d$')
    axes[1].set_title('V6b: numerischer Diffusionsboden → 0')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V6b_trap_convergence.pdf')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    for (N, sn, sa, rr) in zip(Ns, sig_num, sig_an, ratio):
        print(f'    N={N:5d}  σ_num={sn:.3e}  σ_analyt={sa:.3e}  num/an={sn / max(sa, 1e-30):.2f}  D_num/D_d={rr:.2e}')
    numval.save_rows('verifikation', 'V6b_trap_convergence', ['N', 'sigma_num', 'sigma_analyt', 'verh_num_an', 'Dnum_Dd'], [(N, sn, sa, sn / max(sa, 1e-30), rr) for (N, sn, sa, rr) in zip(Ns, sig_num, sig_an, ratio)])
    return dict(Ns=Ns, sig_num=sig_num.tolist(), sig_analyt=sig_an.tolist(), Dnum_ratio=ratio.tolist())

def V7_convergence():
    print('\n══ V7: Resolution-Konvergenz ══')
    Ns = [100, 200, 400, 800, 1600, 3200, 6400]
    profiles_d = {}
    profiles_g = {}
    for (i, N) in enumerate(Ns):
        configure(A_gap_val=0.7, St_val=0.1, alpha_val=0.01, N_val=N, cd_factor=1.0)
        snaps = run_full_snaps(t_end_orbits=10.0, n_snap=1, damping_on=True, label=f'V7  N={N} [{i + 1}/{len(Ns)}]')
        r_act = d3.r[1:-1].copy()
        profiles_d[N] = (r_act, snaps[-1]['sig_d'][1:-1].copy())
        profiles_g[N] = (r_act, snaps[-1]['sig'][1:-1].copy())
    (r_ref, sd_ref) = profiles_d[Ns[-1]]
    (r_ref_g, sg_ref) = profiles_g[Ns[-1]]
    (err_d, err_g) = ([], [])
    for N in Ns[:-1]:
        (r_n, sd_n) = profiles_d[N]
        sd_interp = np.interp(r_ref, r_n, sd_n)
        L1_d = float(np.sum(np.abs(sd_interp - sd_ref) * np.gradient(r_ref)) / np.sum(np.abs(sd_ref) * np.gradient(r_ref)))
        err_d.append(L1_d)
        (r_n, sg_n) = profiles_g[N]
        sg_interp = np.interp(r_ref_g, r_n, sg_n)
        L1_g = float(np.sum(np.abs(sg_interp - sg_ref) * np.gradient(r_ref_g)) / np.sum(np.abs(sg_ref) * np.gradient(r_ref_g)))
        err_g.append(L1_g)
    (err_d, err_g) = (np.array(err_d), np.array(err_g))
    N_arr = np.array(Ns[:-1], dtype=float)
    (slope_d, _) = np.polyfit(np.log(N_arr), np.log(err_d), 1)
    (slope_g, _) = np.polyfit(np.log(N_arr), np.log(err_g), 1)
    (fig, ax) = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.loglog(N_arr, err_d, 'bo-', lw=1.3, label=f'Σ_d  (Rate ≈ {-slope_d:.2f}, erwartet 1)')
    ax.loglog(N_arr, err_g, 'rs-', lw=1.3, label=f'Σ_g  (Rate ≈ {-slope_g:.2f}, erwartet ~1, zeitlimitiert)')
    ax.loglog(N_arr, err_d[0] * (N_arr[0] / N_arr) ** 1, 'b:', lw=0.8, alpha=0.6)
    ax.loglog(N_arr, err_g[0] * (N_arr[0] / N_arr) ** 2, 'r:', lw=0.8, alpha=0.6)
    ax.set_xlabel('N')
    ax.set_ylabel(f'$L^1$-Fehler vs. Referenz $N={Ns[-1]}$')
    ax.set_title('V7: Resolution-Konvergenz (Gas: MUSCL+minmod, Staub: Rusanov)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V7_convergence.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    print(f'V7  Konvergenzrate Σ_d = {-slope_d:.3f}  (Erwartung ~1)')
    print(f'    Konvergenzrate Σ_g = {-slope_g:.3f}  (Erwartung ~1, zeitlimitiert; vgl. V10)')
    return (-slope_d, -slope_g)
_PS_DEFAULTS = dict(A_gap=0.7, w_gap=0.2, h0=0.05, alpha=0.01, St=0.1, N=600, t_orbits=200.0, cd_factor=1.0)

def _ps_configure(A_gap, w_gap, h0, alpha, St, N, cd_factor):
    d3.A_gap = A_gap
    d3.w_gap = w_gap
    d3.h0 = h0
    d3.alpha = alpha
    d3.St = St
    d3.q_planet = 0.0
    d3.rebuild_grid(N)
    d3.cd2 = d3.cd2 * cd_factor ** 2

def _ps_run_one(label, **params):
    p = dict(_PS_DEFAULTS)
    p.update(params)
    _ps_configure(p['A_gap'], p['w_gap'], p['h0'], p['alpha'], p['St'], p['N'], p['cd_factor'])
    t_end = p['t_orbits'] * d3.T_orb
    (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.initial_state(nsh_drift_on=True)
    d3.apply_outflow_bc(sig, mom_r, L)
    d3.apply_outflow_bc(sig_d, mom_r_d, L_d)
    target = (sig.copy(), mom_r.copy(), L.copy(), sig_d.copy(), mom_r_d.copy(), L_d.copy())
    damp_rate = d3.build_damping_rate()
    n_diag = 50
    diag_dt = t_end / n_diag
    times = [0.0]
    max_eps = [float((sig_d[1:-1] / np.maximum(sig[1:-1], d3.SIGMA_FLOOR)).max())]
    r_active = d3.r[1:-1]
    r_gap_val = 1.0
    w = p['w_gap']
    trap_mask = (r_active >= r_gap_val - w) & (r_active <= r_gap_val + 2.0 * w)
    M_trap = [float(np.sum(sig_d[1:-1][trap_mask] * 2 * np.pi * r_active[trap_mask] * d3.dr[1:-1][trap_mask]))]
    snaps = [(0.0, sig.copy(), sig_d.copy())]
    t_next = diag_dt
    t = 0.0
    while t < t_end:
        dt = min(d3.compute_dt(sig, mom_r, sig_d, mom_r_d), t_end - t)
        (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.hydro_step(sig, mom_r, L, sig_d, mom_r_d, L_d, dt)
        (sig, mom_r, L, sig_d, mom_r_d, L_d) = d3.apply_damping(sig, mom_r, L, sig_d, mom_r_d, L_d, target, damp_rate, dt)
        t += dt
        if t >= t_next or t >= t_end - 1e-12:
            eps = sig_d[1:-1] / np.maximum(sig[1:-1], d3.SIGMA_FLOOR)
            times.append(t)
            max_eps.append(float(eps.max()))
            M_trap.append(float(np.sum(sig_d[1:-1][trap_mask] * 2 * np.pi * r_active[trap_mask] * d3.dr[1:-1][trap_mask])))
            snaps.append((t, sig.copy(), sig_d.copy()))
            t_next += diag_dt
    return dict(label=label, params=p, times=np.array(times), max_eps=np.array(max_eps), M_trap=np.array(M_trap), snaps=snaps, r_active=r_active.copy())

def mrn_weights(St_list, q=3.5):
    St_arr = np.asarray(St_list, dtype=float)
    w = St_arr ** (4.0 - q)
    return w / w.sum()

def _ps_run_multipop(St_list, q=3.5, label='multipop', **params):
    w = mrn_weights(St_list, q=q)
    runs = []
    for (St_i, w_i) in zip(St_list, w):
        p_i = dict(params)
        p_i['St'] = St_i
        res = _ps_run_one(f'{label}/St={St_i:g}', **p_i)
        res['weight'] = float(w_i)
        runs.append(res)
    r_act = runs[0]['r_active']
    n_snap = len(runs[0]['snaps'])
    combined_snaps = []
    for j in range(n_snap):
        t_j = runs[0]['snaps'][j][0]
        sig_g_j = runs[0]['snaps'][j][1]
        sig_d_tot = sum((r['weight'] * r['snaps'][j][2] for r in runs))
        combined_snaps.append((t_j, sig_g_j, sig_d_tot))
    return dict(label=label, St_list=list(St_list), weights=w.tolist(), runs=runs, combined_snaps=combined_snaps, times=runs[0]['times'], r_active=r_act.copy(), params=runs[0]['params'])

def V8_multipop_consistency():
    print('\n══ V8: Multi-Pop-Wrapper-Determinismus (kein Physiktest) ══')
    St_list = [0.01, 0.1]
    base = dict(A_gap=0.7, w_gap=0.2, h0=0.05, alpha=0.01, N=150, t_orbits=5.0, cd_factor=1.0)
    mp = _ps_run_multipop(St_list, q=3.5, label='V8-MP', **base)
    singles = []
    for St_i in St_list:
        params = dict(base)
        params['St'] = St_i
        singles.append(_ps_run_one(f'V8-single-St={St_i}', **params))
    print(f'\nV8 — Diff zwischen Multi-Pop-Sublauf und Single-Pop-Lauf:')
    max_diffs = []
    for (i, St_i) in enumerate(St_list):
        sd_mp = mp['runs'][i]['snaps'][-1][2]
        sd_sg = singles[i]['snaps'][-1][2]
        diff = float(np.max(np.abs(sd_mp - sd_sg)))
        rel = diff / float(np.max(np.abs(sd_sg)) + 1e-30)
        max_diffs.append(rel)
        print(f'  St = {St_i:g}:  max |ΔΣ_d| = {diff:.3e}  (rel = {rel:.3e})')
    (fig, ax) = plt.subplots(1, 1, figsize=(8, 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(St_list)))
    for (i, (St_i, c)) in enumerate(zip(St_list, colors)):
        r_act = mp['runs'][i]['r_active']
        sd_mp = mp['runs'][i]['snaps'][-1][2][1:-1]
        sd_sg = singles[i]['snaps'][-1][2][1:-1]
        ax.plot(r_act, sd_mp, color=c, lw=1.6, label=f'MP, St={St_i}')
        ax.plot(r_act, sd_sg, color=c, lw=0.9, ls='--', label=f'Single, St={St_i}')
    ax.set_xlabel('r')
    ax.set_ylabel('$\\Sigma_d$')
    ax.set_title('V8: Wrapper-Determinismus — Multi-Pop-Sublauf (durchgez.) vs. Single-Pop (gestr.)', fontsize=10)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V8_multipop_consistency.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    return float(max(max_diffs))

def V9_dustybox():
    print('\n══ V9: DUSTYBOX (analytischer Drag-Kern) ══')
    configure(A_gap_val=0.0, St_val=0.1, alpha_val=0.01, N_val=200, cd_factor=1.0)
    eps0 = 0.5
    sig = np.ones_like(d3.r)
    sig_d = eps0 * sig
    u_r = np.zeros_like(d3.r)
    u_phi = d3.v_K.copy()
    (dvr, dvphi) = (0.05, 0.03)
    mom_r = sig * u_r
    L = sig * d3.r * u_phi
    mom_r_d = sig_d * (u_r + dvr)
    L_d = sig_d * d3.r * (u_phi + dvphi)
    tau_s = d3.St / d3.Om_K
    idx = d3.N // 2
    p_r0 = (mom_r + mom_r_d).copy()
    L_tot0 = (L + L_d).copy()
    dt = 0.05 * float(tau_s.min())
    ts = [0.0]
    num_r = [dvr]
    num_phi = [dvphi]
    pr_err = [0.0]
    L_err = [0.0]
    for k in range(200):
        (mom_r, L, mom_r_d, L_d) = d3.epstein_drag_update(sig, mom_r, L, sig_d, mom_r_d, L_d, dt)
        ts.append((k + 1) * dt)
        num_r.append(float((mom_r_d / sig_d - mom_r / sig)[idx]))
        num_phi.append(float((L_d / (sig_d * d3.r) - L / (sig * d3.r))[idx]))
        pr_err.append(float(np.abs(mom_r + mom_r_d - p_r0).max()))
        L_err.append(float(np.abs(L + L_d - L_tot0).max()))
    ts = np.array(ts)
    ana_r = dvr * np.exp(-(1 + eps0) * ts / tau_s[idx])
    ana_phi = dvphi * np.exp(-(1 + eps0) * ts / tau_s[idx])
    err_r = float(np.abs(np.array(num_r) - ana_r).max() / dvr)
    err_phi = float(np.abs(np.array(num_phi) - ana_phi).max() / dvphi)
    mom_cons = max(max(pr_err), max(L_err))
    mr2 = sig * u_r
    L2 = sig * d3.r * u_phi
    mrd2 = sig_d * (u_r + dvr)
    Ld2 = sig_d * d3.r * (u_phi + dvphi)
    (mr2, L2, mrd2, Ld2) = d3.epstein_drag_update(sig, mr2, L2, sig_d, mrd2, Ld2, 50.0 * float(tau_s.max()))
    drel_stiff = float(np.abs(mrd2 / sig_d - mr2 / sig).max())
    (fig, ax) = plt.subplots(1, 1, figsize=(7.5, 5))
    x = ts / tau_s[idx]
    ax.semilogy(x, np.abs(num_r), 'bo', ms=3, label='num |Δv_r|')
    ax.semilogy(x, np.abs(ana_r), 'b-', lw=1.0, label='$\\exp(-(1+\\varepsilon)t/\\tau_s)$')
    ax.semilogy(x, np.abs(num_phi), 'rs', ms=3, label='num |Δv_φ|')
    ax.semilogy(x, np.abs(ana_phi), 'r-', lw=1.0)
    ax.set_xlabel('$t/\\tau_s$')
    ax.set_ylabel('$|\\Delta v|$')
    ax.set_title(f'V9: DUSTYBOX — err_r={err_r:.1e}, Impuls-Erh.={mom_cons:.1e}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V9_dustybox.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    print(f'V9  max rel. Fehler Δv_r vs exp  = {err_r:.2e}')
    print(f'    max rel. Fehler Δv_φ vs exp  = {err_phi:.2e}')
    print(f'    Impuls-Erhaltung (max|Δ|)    = {mom_cons:.2e}  (radial + Drehimpuls)')
    print(f'    steifer Limes |Δrel|(τ_s→0)  = {drel_stiff:.2e}  (Soll → 0)')
    return (err_r, mom_cons)

def V10_time_convergence():
    print('\n══ V10: Zeit-Konvergenz (CFL-Sweep, festes N) ══')
    (N_fix, t_orb) = (200, 5.0)
    (CFLs, CFL_ref) = ([0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05], 0.0125)
    CFL_save = d3.CFL

    def run_cfl(cfl):
        configure(A_gap_val=0.7, St_val=0.1, alpha_val=0.01, N_val=N_fix, cd_factor=1.0)
        d3.CFL = cfl
        (sig0, mr0, L0, sd0, mrd0, Ld0) = d3.initial_state(nsh_drift_on=True)
        d3.apply_outflow_bc(sig0, mr0, L0)
        d3.apply_outflow_bc(sd0, mrd0, Ld0)
        dt0 = d3.compute_dt(sig0, mr0, sd0, mrd0)
        snaps = run_full_snaps(t_end_orbits=t_orb, n_snap=1, damping_on=True, label=f'V10 CFL={cfl:g}')
        return (d3.r[1:-1].copy(), snaps[-1]['sig_d'][1:-1].copy(), float(dt0))
    (r_ref, sd_ref, _) = run_cfl(CFL_ref)
    (errs, dts) = ([], [])
    for cfl in CFLs:
        (r_n, sd_n, dt0) = run_cfl(cfl)
        L1 = float(np.sum(np.abs(sd_n - sd_ref) * np.gradient(r_ref)) / np.sum(np.abs(sd_ref) * np.gradient(r_ref)))
        errs.append(L1)
        dts.append(dt0)
    d3.CFL = CFL_save
    (errs, dts) = (np.array(errs), np.array(dts))
    valid = errs > 0
    slope = float(np.polyfit(np.log(dts[valid]), np.log(errs[valid]), 1)[0]) if valid.sum() >= 2 else float('nan')
    (fig, ax) = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.loglog(dts, errs, 'bo-', lw=1.3, label=f'L1(Σ_d) vs dt  (Rate ≈ {slope:.2f})')
    ax.loglog(dts, errs[-1] * (dts / dts[-1]) ** 1, 'k:', lw=0.8, label='1. Ordnung')
    ax.set_xlabel('dt')
    ax.set_ylabel('$L^1$-Fehler vs. kleinstes dt')
    ax.set_title('V10: Zeit-Konvergenz (festes N=200, CFL-Sweep)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V10_time_convergence.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    print(f"V10 dt-Werte (∝ CFL erwartet): {', '.join((f'{d:.2e}' for d in dts))}")
    print(f'    Zeit-Konvergenzrate = {slope:.3f}  (Erwartung ~1)')
    return slope

def V12_wellbalance_floor():
    print('\n══ V12: Well-Balancing (Langlauf) + Floor-Inaktivität ══')
    configure(A_gap_val=0.0, St_val=0.1, alpha_val=0.0, N_val=400, cd_factor=1.0)
    bk_save = d3.DUST_BACKREACTION
    d3.DUST_BACKREACTION = False
    try:
        snaps = run_full_snaps(t_end_orbits=50.0, n_snap=10, nsh_drift_on=False, damping_on=False, label='V12 Well-Balancing')
    finally:
        d3.DUST_BACKREACTION = bk_save
    r_act = d3.r[1:-1]
    domain_w = d3.r_max - d3.r_min
    r_in = d3.r_min + d3.DAMP_FRAC_IN * domain_w
    r_out = d3.r_max - d3.DAMP_FRAC_OUT * domain_w
    bulk = (r_act > r_in) & (r_act < r_out)
    sig0 = snaps[0]['sig'][1:-1]
    sig_end = snaps[-1]['sig'][1:-1]
    wb_drift = float(np.abs(sig_end[bulk] / np.maximum(sig0[bulk], d3.SIGMA_FLOOR) - 1.0).max())
    u_r_end = snaps[-1]['mom_r'][1:-1] / np.maximum(sig_end, d3.SIGMA_FLOOR)
    u_r_rel = float((np.abs(u_r_end) / d3.cs[1:-1])[bulk].max())
    saw_end = float(d3.saw(snaps[-1]['sig']))
    floor_g = float(sig_end[bulk].min() / d3.SIGMA_FLOOR)
    floor_d = float(snaps[-1]['sig_d'][1:-1][bulk].min() / d3.SIGMA_FLOOR)
    floor_margin = floor_g
    (sig0f, mr0, L0, sd0, mrd0, Ld0) = d3.initial_state(nsh_drift_on=False, stationary_visc=False)
    u_phi0 = L0 / (sig0f * d3.r)
    dL0 = L0 - sig0f * d3.r * d3.v_K
    Pi0 = d3.cs ** 2 * sig0f
    dPi0 = np.zeros_like(sig0f)
    dPi0[1:-1] = (Pi0[2:] - Pi0[:-2]) / (d3.r[2:] - d3.r[:-2])
    S0 = dL0 * (u_phi0 + d3.v_K) / d3.r ** 2 - dPi0
    scale = float(np.abs(dPi0[1:-1][bulk]).max())
    wb_res = float(np.abs(S0[1:-1][bulk]).max())
    wb_rel = wb_res / max(scale, 1e-300)
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snaps)))
    for (s, c) in zip(snaps, colors):
        axes[0].plot(r_act, s['sig'][1:-1] / np.maximum(sig0, d3.SIGMA_FLOOR), color=c, lw=1.1, label=f"t={s['t'] / d3.T_orb:.0f} Orb")
    axes[0].axhline(1.0, ls='--', color='gray', lw=0.6)
    axes[0].set_xscale('log')
    axes[0].set_ylim(1 - 1e-06, 1 + 1e-06)
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\\Sigma_g(r,t)/\\Sigma_g(r,0)$')
    axes[0].set_title(f'V12a: Well-Balancing — max. Drift {wb_drift:.1e}')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, which='both')
    axes[1].plot(r_act, np.abs(u_r_end) / d3.cs[1:-1], 'C0-', lw=1.3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('$|u_r|/c_s$')
    axes[1].set_title(f'V12b: spurioser Radialimpuls — max {u_r_rel:.1e}')
    axes[1].grid(True, alpha=0.3, which='both')
    fig.suptitle('V12: Well-Balancing-Langlauf (q=0, α=0, kein Damping/Rückreaktion)')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V12_wellbalance_floor.pdf')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    print(f'V12  Drift max|ΔΣ/Σ| (Bulk, 50 Orb, Damping AUS) = {wb_drift:.3e}  (randgetrieben, NICHT das Well-Balancing-Mass)')
    print(f'     spurioser Radialimpuls max|u_r|/c_s    = {u_r_rel:.3e}')
    print(f'     Gitterrauschen saw(Σ_g)                = {saw_end:.3e}')
    print(f'     Quellterm-Residuum max|S_r| (t=0)      = {wb_res:.3e}  → relativ {wb_rel:.2e}  (DAS ist das Well-Balancing)')
    print(f'     Floor-Marge Staub min(Σ_d)/FLOOR (Bulk)= {floor_d:.3e}')
    print(f'     Floor-Marge Gas   min(Σ_g)/FLOOR (Bulk)= {floor_margin:.3e}  (≫1 ⇒ Floor nie aktiv im ROI)')
    return (wb_drift, saw_end, floor_margin, u_r_rel, wb_res, wb_rel)

def _h_p(r_p=1.0):
    cs_p = d3.h0 * np.sqrt(d3.GM / d3.r0) * (r_p / d3.r0) ** (-0.25)
    Om_p = np.sqrt(d3.GM / r_p ** 3)
    return cs_p / Om_p / r_p

def _kanagawa(q, alpha, h_p):
    K = q ** 2 * h_p ** (-5) / alpha
    return (1.0 / (1.0 + 0.04 * K), K)

def _run(q, A_gap, t_orbits, N=1600, St_val=0.1, alpha_val=0.01, C_torque=C_CAL, cd_factor=1.0, n_track=120, label=''):
    d3.A_gap = A_gap
    d3.q_planet = q
    d3.r_planet = 1.0
    d3.St = St_val
    d3.alpha = alpha_val
    d3.C_torque = C_torque
    d3.rebuild_grid(N)
    d3.cd2 = d3.cd2 * cd_factor ** 2
    (sig, mr, L, sd, mrd, Ld) = d3.initial_state(nsh_drift_on=True, stationary_visc=True)
    d3.apply_outflow_bc(sig, mr, L)
    d3.apply_outflow_bc(sd, mrd, Ld)
    tgt = (sig.copy(), mr.copy(), L.copy(), sd.copy(), mrd.copy(), Ld.copy())
    dr = d3.build_damping_rate()
    idx = int(np.argmin(np.abs(d3.r - 1.0)))
    S0 = d3.Sigma0 * (1.0 / d3.r0) ** (-d3.p_slope)
    t_end = t_orbits * d3.T_orb
    track_dt = t_end / n_track
    times = [0.0]
    depth = [sig[idx] / S0]
    (t, t_next) = (0.0, track_dt)
    bar = progress.Bar(label) if label else None
    while t < t_end:
        dt = min(d3.compute_dt(sig, mr, sd, mrd), t_end - t)
        (sig, mr, L, sd, mrd, Ld) = d3.hydro_step(sig, mr, L, sd, mrd, Ld, dt, t_now=t)
        (sig, mr, L, sd, mrd, Ld) = d3.apply_damping(sig, mr, L, sd, mrd, Ld, tgt, dr, dt)
        t += dt
        if bar is not None:
            bar.update(t / t_end)
        if t >= t_next or t >= t_end - 1e-12:
            times.append(t)
            depth.append(sig[idx] / S0)
            t_next += track_dt
    if bar is not None:
        bar.done()
    snap = {'sig': sig.copy(), 'sig_d': sd.copy(), 'r': d3.r.copy()}
    return (np.array(times), np.array(depth), snap, S0)

def V0_equivalence(N=200, t_orbits=15.0):
    print('\n══ V0: Äquivalenz Lindblad(q_planet=0) ≡ Basis-Solver ══')
    h0_ref = d3.h0

    def short(mod, q0=None):
        mod.A_gap = 0.7
        mod.St = 0.1
        mod.alpha = 0.01
        mod.h0 = h0_ref
        if q0 is not None:
            mod.q_planet = q0
        mod.rebuild_grid(N)
        (s, mrr, l, sdd, mrd, ld) = mod.initial_state(nsh_drift_on=True)
        mod.apply_outflow_bc(s, mrr, l)
        mod.apply_outflow_bc(sdd, mrd, ld)
        tgt = (s.copy(), mrr.copy(), l.copy(), sdd.copy(), mrd.copy(), ld.copy())
        drr = mod.build_damping_rate()
        te = t_orbits * mod.T_orb
        t = 0.0
        while t < te:
            dt = min(mod.compute_dt(s, mrr, sdd, mrd), te - t)
            (s, mrr, l, sdd, mrd, ld) = mod.hydro_step(s, mrr, l, sdd, mrd, ld, dt)
            (s, mrr, l, sdd, mrd, ld) = mod.apply_damping(s, mrr, l, sdd, mrd, ld, tgt, drr, dt)
            t += dt
        return (s, mrr, l, sdd, mrd, ld)
    b = short(d3_base)
    g = short(d3, q0=0.0)
    names = ['Σ_g', 'mom_r,g', 'L_g', 'Σ_d', 'mom_r,d', 'L_d']
    diffs = [float(np.max(np.abs(bi - gi))) for (bi, gi) in zip(b, g)]
    print('  max|Lindblad(q=0) − Basis| je Feld:')
    for (nm, dd) in zip(names, diffs):
        print(f'    {nm:<9s} {dd:.2e}')
    worst = max(diffs)
    print(f'  → max über alle Felder = {worst:.2e}   (Soll ≈ 0 ⇒ Kern identisch)')
    return worst

def V11_planet_gap():
    print('\n══ V11: Persistente Planetenlücke (Lindblad) vs. diffusives Zulaufen ══')
    q_ref = 0.00045
    alpha_val = 0.01
    h_p = _h_p()
    print(f'  Lauf A: Planet AN (q={q_ref:.1e}), glatter Start, 250 Orbits ...')
    (tA, dA, snapA, S0) = _run(q_ref, A_gap=0.0, t_orbits=250.0, label='V11 A: Planet AN')
    print('  Lauf B: Planet AUS (q=0), aufgeprägte Gauß-Lücke A=0.7, 250 Orbits ...')
    (tB, dB, snapB, _) = _run(0.0, A_gap=0.7, t_orbits=250.0, label='V11 B: Planet AUS')
    qs = [0.0002, 0.0003, q_ref, 0.0006]
    print(f'  Kanagawa-Sweep über q = {qs} (je 250 Orbits) ...')
    depth_ss = []
    for (j, q) in enumerate(qs):
        (tq, dq, _, _) = _run(q, A_gap=0.0, t_orbits=250.0, label=f'V11 Kanagawa q={q:.1e} [{j + 1}/{len(qs)}]')
        mask = tq > tq[-1] * 0.7
        depth_ss.append(float(np.mean(dq[mask])))
        print(f'    q = {q:.2e}  →  stationäre Tiefe = {depth_ss[-1]:.3f}')
    depth_ss = np.array(depth_ss)
    Ks = np.array([q ** 2 * h_p ** (-5) / alpha_val for q in qs])
    depth_kan = 1.0 / (1.0 + 0.04 * Ks)
    (fig, (ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 4.5))
    (dk_ref, K_ref) = _kanagawa(q_ref, alpha_val, h_p)
    ax1.plot(tA / d3.T_orb, dA, 'b-', lw=1.8, label=f'Planet AN (q={q_ref:.1e})')
    ax1.plot(tB / d3.T_orb, dB, 'r-', lw=1.8, label='Planet AUS, IC-Lücke (alt)')
    ax1.axhline(dk_ref, ls='--', color='b', lw=0.9, label=f'Kanagawa (K={K_ref:.0f}) = {dk_ref:.2f}')
    ax1.axhline(1.0, ls=':', color='gray', lw=0.8)
    ax1.set_xlabel('t / T_orb')
    ax1.set_ylabel('$\\Sigma_g(r_p)/\\Sigma_0$')
    ax1.set_title('V11a: Lückentiefe — persistent vs. zulaufend')
    ax1.set_ylim(0.0, 1.15)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.loglog(Ks, depth_ss, 'bo', ms=8, label='Modell (stationär)')
    Kfine = np.geomspace(Ks.min() * 0.6, Ks.max() * 1.7, 100)
    ax2.loglog(Kfine, 1.0 / (1.0 + 0.04 * Kfine), 'k--', lw=1.4, label='Kanagawa 2015')
    ax2.axvline(K_ref, ls=':', color='gray', lw=0.8, label='Kalibrierpunkt')
    ax2.set_xlabel('$K = q^2 (H/r_p)^{-5}/\\alpha$')
    ax2.set_ylabel('$\\Sigma_\\mathrm{min}/\\Sigma_0$')
    ax2.set_title('V11b: Gleichgewichtstiefe vs. Kanagawa')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    r_act = snapA['r'][1:-1]
    sgA = snapA['sig'][1:-1]
    sdA = snapA['sig_d'][1:-1]
    epsA = sdA / np.maximum(sgA, d3.SIGMA_FLOOR)
    ax3.plot(r_act, sgA / S0, 'b-', lw=1.6, label='$\\Sigma_g/\\Sigma_0$')
    ax3.plot(r_act, epsA / d3.eps0, 'g-', lw=1.6, label='$\\varepsilon/\\varepsilon_0$ (Staubfalle)')
    ax3.axvline(1.0, ls=':', color='crimson', lw=1.0, label='Planet $r_p$')
    ax3.axhline(1.0, ls=':', color='gray', lw=0.6)
    ax3.set_xlabel('r')
    ax3.set_ylabel('normiert')
    ax3.set_title(f'V11c: stationäre Struktur (t={tA[-1] / d3.T_orb:.0f} Orb)')
    ax3.set_xscale('log')
    ax3.set_xlim(0.5, 2.8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    fig.suptitle('V11: Planeten-Lindblad-Lücke — Persistenz, Kanagawa-Skalierung, Staubfalle')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'V11_planet_gap.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {fname}')
    mA = tA > tA[-1] * 0.7
    depth_A_ss = float(np.mean(dA[mA]))
    drift_A = float(np.polyfit(tA[mA] / d3.T_orb, dA[mA], 1)[0])
    closed_B = float(dB[-1] - dB[0])
    iref = qs.index(q_ref)
    rel_ref = abs(depth_ss[iref] - depth_kan[iref]) / depth_kan[iref]
    print(f'\nV11a Persistenz (Planet AN):')
    print(f'     stationäre Tiefe         = {depth_A_ss:.3f}   (Kanagawa {dk_ref:.3f})')
    print(f'     Drift d(Tiefe)/d(Orbit)  = {drift_A:+.2e}    (Soll ≈ 0 ⇒ Lücke STEHT)')
    print(f'V11  Zulaufen (Planet AUS):')
    print(f'     Δ-Tiefe über Lauf        = {closed_B:+.3f}    (>0 ⇒ Lücke läuft ZU)')
    print(f'V11b Kanagawa @ Referenz:     rel. Abw. = {rel_ref:.2f}')
    return (depth_A_ss, drift_A, closed_B, rel_ref)
if __name__ == '__main__':
    worst_v0 = V0_equivalence()
    (dev_v1_sig, dev_v1_mdot) = V1_lbp_stationary()
    (v2_max, v2_rms, v2_mean) = V2_nsh_drift()
    dev_v3 = V3_angular_momentum()
    (pos_v4, width_v4) = V4_pure_advection()
    slope_err_v5 = V5_pure_diffusion()
    sig_err_v6 = V6_steady_state_trap()
    (rate_d_v7, rate_g_v7) = V7_convergence()
    diff_v8 = V8_multipop_consistency()
    (err_r_v9, mom_cons_v9) = V9_dustybox()
    rate_t_v10 = V10_time_convergence()
    (wb_v12, saw_v12, floor_v12, ur_v12, wbres_v12, wbrel_v12) = V12_wellbalance_floor()
    (depth_v11, drift_v11, closed_v11, rel_v11) = V11_planet_gap()
    if os.environ.get('RIGOR') == '1':
        V5_convergence()
        V6_convergence()
    print('\n═══ Zusammenfassung Verifikation (V0–V12) ═══')
    print(f'V0  Äquivalenz Lindblad(q=0)≡Basis: max|Δ|               = {worst_v0:.2e}  (≈0)')
    print(f'V1a Stationarität Σ:        max |ΔΣ/Σ|              = {dev_v1_sig:.2e}')
    print(f'V1b Ṁ-Übereinstimmung:      max |ΔṀ/Ṁ_analyt|       = {dev_v1_mdot:.2e}')
    print(f'V2  NSH-Drift:              RMS |Δv_r,d|/|v_NSH|    = {v2_rms:.2e}')
    print(f'                             ⟨Δv_r,d⟩ /|v_NSH|       = {v2_mean:.2e}  (mittlerer Offset)')
    print(f'V3  Drehimpulserhaltung:    max |ΔL/L| (bilanziert) = {dev_v3:.2e}')
    print(f'V4  Pure Advection:         |Δμ|/|Δr_analyt|        = {pos_v4:.2e}')
    print(f'V5  Pure Diffusion:         |Δslope|/(2D_d)         = {slope_err_v5:.2e}')
    print(f'V6  Steady-State Trap:      |Δσ_trap|/σ_analyt      = {sig_err_v6:.2e}')
    print(f'V7  Konvergenz Σ_d (Staub): Rate                    = {rate_d_v7:.2f}  (Erw. ~1)')
    print(f'    Konvergenz Σ_g (Gas):   Rate                    = {rate_g_v7:.2f}  (Erw. ~1, zeitlimitiert)')
    print(f'V8  Wrapper-Determinismus:  max rel. ΔΣ_d           = {diff_v8:.2e}  (kein Physiktest)')
    print(f'V9  DUSTYBOX-Drag:          rel. Fehler Δv_r        = {err_r_v9:.2e}')
    print(f'                             Impuls-Erhaltung max|Δ| = {mom_cons_v9:.2e}')
    print(f'V10 Zeit-Konvergenz:        Rate                    = {rate_t_v10:.2f}  (Erw. ~1)')
    print(f'V12 Well-Balancing-Drift:   max|ΔΣ/Σ| (Bulk)        = {wb_v12:.2e}  (Soll ~1e-10)')
    print(f'    Floor-Marge:            min(Σ)/SIGMA_FLOOR      = {floor_v12:.2e}  (≫1 ⇒ Floor inaktiv)')
    print(f'V11a stationäre Lückentiefe (Planet):                  = {depth_v11:.3f}')
    print(f'V11a Tiefen-Drift (Persistenz):                        = {drift_v11:+.2e}  (≈0 ⇒ steht)')
    print(f'V11  Zulaufen ohne Planet (Kontrolle):                 = {closed_v11:+.3f}  (>0 ⇒ zu)')
    print(f'V11b Kanagawa-Abweichung @ Referenz:                   = {rel_v11:.2f}')
    numval.save_rows('verifikation', 'V0-V11_zusammenfassung', ['test', 'groesse', 'wert'], [('V0', 'max|Δ| Lindblad(q=0)≡Basis', worst_v0), ('V1a', 'max|ΔΣ/Σ|', dev_v1_sig), ('V1b', 'max|ΔṀ/Ṁ_analyt|', dev_v1_mdot), ('V2', 'RMS|Δv_r,d|/|v_NSH|', v2_rms), ('V2', '⟨Δv_r,d⟩/|v_NSH| (Offset)', v2_mean), ('V2', 'max|Δv_r,d|/|v_NSH|', v2_max), ('V3', 'max|ΔL/L| (bilanziert)', dev_v3), ('V4', '|Δμ|/|Δr_analyt| (Position)', pos_v4), ('V4', 'Δσ/σ0 (Breiten-Drift)', width_v4), ('V5', '|Δslope|/(2D_d)', slope_err_v5), ('V6', '|Δσ_trap|/σ_analyt', sig_err_v6), ('V7', 'Konvergenzrate Σ_d (Staub)', rate_d_v7), ('V7', 'Konvergenzrate Σ_g (Gas)', rate_g_v7), ('V8', 'max rel. ΔΣ_d (Multi-Pop-Wrapper, Determinismus)', diff_v8), ('V9', 'rel. Fehler Δv_r (DUSTYBOX)', err_r_v9), ('V9', 'Impuls-Erhaltung max|Δ|', mom_cons_v9), ('V10', 'Zeit-Konvergenzrate', rate_t_v10), ('V11a', 'stationäre Lückentiefe', depth_v11), ('V11a', 'Tiefen-Drift (Persistenz)', drift_v11), ('V11', 'Zulaufen ohne Planet', closed_v11), ('V11b', 'Kanagawa-Abweichung @ Referenz', rel_v11), ('V12', 'Well-Balancing-Drift max|ΔΣ/Σ|', wb_v12), ('V12', 'saw(Σ_g) Gitterrauschen', saw_v12), ('V12', 'Quellterm-Residuum max|S_r| (t=0)', wbres_v12), ('V12', 'dito relativ zum Druckterm', wbrel_v12), ('V12', 'spurioser Radialimpuls max|u_r|/c_s', ur_v12), ('V12', 'Floor-Marge Gas min(Σ_g)/FLOOR (Bulk)', floor_v12)])
