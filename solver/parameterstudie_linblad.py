import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import disk_v3_erweitert_linblad as d3
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(_HERE, '..', 'thesis', 'figures', 'parameterstudie'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
import numval
numval.install('parameterstudie')
import progress
SMOKE = os.environ.get('SMOKE', '0') == '1'
DEFAULTS = dict(A_gap=0.0, w_gap=0.2, h0=0.05, alpha=0.01, St=0.1, q_planet=0.00045, C_torque=0.4478, N=1600 if not SMOKE else 120, t_orbits=250.0 if not SMOKE else 25.0, cd_factor=1.0)

def stationary(times, y, frac=0.3):
    t = np.asarray(times, float)
    y = np.asarray(y, float)
    if t.size < 3:
        return (float(y[-1]), float('nan'))
    mask = t > t[-1] * (1.0 - frac)
    if mask.sum() < 2:
        mask[-2:] = True
    y_ss = float(np.mean(y[mask]))
    slope = float(np.polyfit(t[mask] / d3.T_orb, y[mask], 1)[0])
    return (y_ss, slope)

def window_drift_pct(times, y, frac=0.3):
    t = np.asarray(times, float)
    y = np.asarray(y, float)
    if t.size < 3:
        return float('nan')
    mask = t > t[-1] * (1.0 - frac)
    if mask.sum() < 2:
        mask[-2:] = True
    ym = float(np.mean(y[mask]))
    if abs(ym) < 1e-30:
        return float('nan')
    return 100.0 * float(y[mask][-1] - y[mask][0]) / ym

def _h_p(h0, r_p=1.0):
    cs_p = h0 * np.sqrt(d3.GM / d3.r0) * (r_p / d3.r0) ** (-0.25)
    Om_p = np.sqrt(d3.GM / r_p ** 3)
    return cs_p / Om_p / r_p

def kanagawa(q, alpha, h0):
    K = q ** 2 * _h_p(h0) ** (-5) / alpha
    return (K, 1.0 / (1.0 + 0.04 * K))

def configure(p):
    d3.A_gap = p['A_gap']
    d3.w_gap = p['w_gap']
    d3.h0 = p['h0']
    d3.alpha = p['alpha']
    d3.St = p['St']
    d3.q_planet = p['q_planet']
    d3.r_planet = 1.0
    d3.C_torque = p['C_torque']
    d3.rebuild_grid(p['N'])
    d3.cd2 = d3.cd2 * p['cd_factor'] ** 2

def run_one(label, **params):
    prog = params.pop('prog', '')
    p = dict(DEFAULTS)
    p.update(params)
    configure(p)
    t_end = p['t_orbits'] * d3.T_orb
    rp = d3.r_planet
    (sig, mr, L, sd, mrd, Ld) = d3.initial_state(nsh_drift_on=True, stationary_visc=True)
    d3.apply_outflow_bc(sig, mr, L)
    d3.apply_outflow_bc(sd, mrd, Ld)
    tgt = (sig.copy(), mr.copy(), L.copy(), sd.copy(), mrd.copy(), Ld.copy())
    dr = d3.build_damping_rate()
    r_act = d3.r[1:-1]
    S0 = d3.Sigma0 * (rp / d3.r0) ** (-d3.p_slope)
    idx = int(np.argmin(np.abs(d3.r - rp)))
    H_p = d3.h0 * np.sqrt(d3.GM / d3.r0) * (d3.r_planet / d3.r0) ** (-0.25) / np.sqrt(d3.GM / d3.r_planet ** 3)
    trap_mask = (r_act >= rp) & (r_act <= rp + 12.0 * H_p)
    outer = (r_act > rp) & (r_act < rp + 16.0 * H_p)

    def _diag(sg, sd_):
        eps = sd_[1:-1] / np.maximum(sg[1:-1], d3.SIGMA_FLOOR)
        M = float(np.sum(sd_[1:-1][trap_mask] * 2 * np.pi * r_act[trap_mask] * d3.dr[1:-1][trap_mask]))
        return (float(eps.max()), M, float(sg[idx] / S0))
    n_diag = 60 if not SMOKE else 12
    diag_dt = t_end / n_diag
    (e0, M0, g0) = _diag(sig, sd)
    times = [0.0]
    max_eps = [e0]
    M_trap = [M0]
    depth = [g0]
    snaps = [(0.0, sig.copy(), sd.copy())]
    (t, t_next) = (0.0, diag_dt)
    bar = progress.Bar(f'{prog} {label}'.strip())
    while t < t_end:
        dt = min(d3.compute_dt(sig, mr, sd, mrd), t_end - t)
        (sig, mr, L, sd, mrd, Ld) = d3.hydro_step(sig, mr, L, sd, mrd, Ld, dt, t_now=t)
        (sig, mr, L, sd, mrd, Ld) = d3.apply_damping(sig, mr, L, sd, mrd, Ld, tgt, dr, dt)
        t += dt
        bar.update(t / t_end)
        if t >= t_next or t >= t_end - 1e-12:
            (e, M, g) = _diag(sig, sd)
            times.append(t)
            max_eps.append(e)
            M_trap.append(M)
            depth.append(g)
            snaps.append((t, sig.copy(), sd.copy()))
            t_next += diag_dt
    bar.done()
    Pi_fin = d3.cs[1:-1] ** 2 * snaps[-1][1][1:-1]
    if outer.any() and Pi_fin[outer].size:
        r_pmax = float(r_act[outer][int(np.argmax(Pi_fin[outer]))])
    else:
        r_pmax = rp + _h_p(p['h0'])
    w_eff = max(r_pmax - rp, 0.001)
    res = dict(label=label, params=p, times=np.array(times), max_eps=np.array(max_eps), M_trap=np.array(M_trap), depth=np.array(depth), snaps=snaps, r_active=r_act.copy(), S0=S0, w_eff=w_eff, r_pmax=r_pmax)
    numval.save_rows('parameterstudie', f'timeseries_{numval._slug(label)}', ['t_orb', 'depth', 'max_eps_rel', 'M_trap_rel'], list(zip(res['times'] / d3.T_orb, res['depth'], res['max_eps'] / d3.eps0, res['M_trap'] / max(res['M_trap'][0], 1e-30))))
    return res

def tau_trap(res, factor=2.0):
    idx = np.where(res['max_eps'] > factor * d3.eps0)[0]
    return float(res['times'][idx[0]] / d3.T_orb) if len(idx) else np.nan

def peclet(res):
    p = res['params']
    return p['St'] / (d3.delta_t * p['alpha']) * (res['w_eff'] / 1.0)

def sweep(param, values, fixed=None, t_orbits_fn=None):
    print(f'\n══ Sweep {param} ∈ {list(values)} ══')
    results = []
    for (i, v) in enumerate(values):
        params = dict(fixed or {})
        params[param] = v
        if t_orbits_fn is not None:
            params['t_orbits'] = t_orbits_fn(v)
        print(f"  → {param} = {v:g}  (t={params.get('t_orbits', DEFAULTS['t_orbits']):.0f} Orb)")
        results.append(run_one(f'{param}={v:g}', prog=f'[{param}-Sweep {i + 1}/{len(values)}]', **params))
    return results

def plot_sweep(results, sweep_name, pdfname):
    (fig, axes) = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
    for (res, c) in zip(results, colors):
        t_orb = res['times'] / d3.T_orb
        axes[0, 0].plot(t_orb, res['max_eps'] / d3.eps0, color=c, lw=1.5, label=res['label'])
        axes[0, 1].plot(t_orb, res['M_trap'] / max(res['M_trap'][0], 1e-30), color=c, lw=1.5)
        sg = res['snaps'][-1][1][1:-1]
        sdf = res['snaps'][-1][2][1:-1]
        axes[1, 0].plot(res['r_active'], sg / res['S0'], color=c, lw=1.5)
        axes[1, 1].plot(res['r_active'], sdf / np.maximum(sg, d3.SIGMA_FLOOR) / d3.eps0, color=c, lw=1.5)
    axes[0, 0].set_xlabel('t / T_orb')
    axes[0, 0].set_ylabel('$\\max_r\\,\\varepsilon/\\varepsilon_0$')
    axes[0, 0].axhline(1, ls='--', color='gray', lw=0.6)
    axes[0, 0].set_title('Staubanreicherung über Zeit')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('t / T_orb')
    axes[0, 1].set_ylabel('$M_\\mathrm{trap}(t)/M_\\mathrm{trap}(0)$')
    axes[0, 1].axhline(1, ls='--', color='gray', lw=0.6)
    axes[0, 1].set_title('Gefangene Staubmasse')
    axes[0, 1].grid(True, alpha=0.3)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xscale('log')
        ax.set_xlim(0.4, 3.5)
        ax.axvline(1.0, ls=':', color='crimson', lw=1)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlabel('r')
    axes[1, 0].axhline(1, ls=':', color='gray', lw=0.6)
    axes[1, 0].set_ylabel('$\\Sigma_g/\\Sigma_0$')
    axes[1, 0].set_title('finale Gas-Lücke + Pile-up')
    axes[1, 1].axhline(1, ls=':', color='gray', lw=0.6)
    axes[1, 1].set_ylabel('$\\varepsilon/\\varepsilon_0$')
    axes[1, 1].set_title('finale Staubfalle')
    fig.suptitle(f'Lindblad-Planetenlücke — Sweep über {sweep_name}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, pdfname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {path}')

def plot_reference_evolution(res, pdfname='linblad_reference_evolution.pdf'):
    snaps = res['snaps']
    r_act = res['r_active']
    n = len(snaps)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, n))
    (fig, axes) = plt.subplots(1, 3, figsize=(16, 4.5))
    (_, sig0, _) = snaps[0]
    axes[0].plot(r_act, sig0[1:-1] / res['S0'], 'k--', lw=1.0, alpha=0.6, label='$\\Sigma_g(r,0)$')
    step = max(1, n // 6)
    for (i, ((t, sig, sd_), c)) in enumerate(zip(snaps, colors)):
        lab = f't={t / d3.T_orb:.0f} Orb' if i % step == 0 else None
        axes[0].plot(r_act, sig[1:-1] / res['S0'], color=c, lw=1.1, label=lab)
        axes[1].plot(r_act, sd_[1:-1] / np.maximum(sig[1:-1], d3.SIGMA_FLOOR) / d3.eps0, color=c, lw=1.1)
    axes[0].set_xscale('log')
    axes[0].set_xlim(0.4, 3.5)
    axes[0].axhline(1, ls=':', color='gray', lw=0.6)
    axes[0].axvline(1.0, ls=':', color='crimson', lw=1)
    axes[0].set_xlabel('r')
    axes[0].set_ylabel('$\\Sigma_g/\\Sigma_0$')
    axes[0].set_title('Gas-Lücke (Verlauf)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, which='both')
    axes[1].set_xscale('log')
    axes[1].set_xlim(0.4, 3.5)
    axes[1].axhline(1, ls=':', color='gray', lw=0.6)
    axes[1].axvline(1.0, ls=':', color='crimson', lw=1)
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('$\\varepsilon/\\varepsilon_0$')
    axes[1].set_title('Staubfalle (Verlauf)')
    axes[1].grid(True, alpha=0.3, which='both')
    t_orb = res['times'] / d3.T_orb
    axes[2].plot(t_orb, res['max_eps'] / d3.eps0, 'b-', lw=1.5, label='$\\max_r\\varepsilon/\\varepsilon_0$')
    axes[2].plot(t_orb, res['M_trap'] / max(res['M_trap'][0], 1e-30), 'g-', lw=1.5, label='$M_\\mathrm{trap}/M_0$')
    axes[2].plot(t_orb, res['depth'], 'r-', lw=1.5, label='$\\Sigma_g(r_p)/\\Sigma_0$ (Tiefe)')
    axes[2].axhline(1, ls='--', color='gray', lw=0.6)
    axes[2].set_xlabel('t / T_orb')
    axes[2].set_title('Zeit-Diagnostik')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    (K, kd) = kanagawa(res['params']['q_planet'], res['params']['alpha'], res['params']['h0'])
    fig.suptitle(f"Referenz — q={res['params']['q_planet']:.1e} (K={K:.0f}, Kanagawa-Tiefe {kd:.2f}), St={res['params']['St']:.0e}, α={res['params']['alpha']:.0e}, Pe={peclet(res):.2f}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, pdfname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {path}')

def print_summary_table(results, sweep_name):
    print(f'\n──────── Zusammenfassung: {sweep_name} ────────')
    print(f"{'Lauf':<16s} {'Tiefe':>7s} {'dTiefe/Orb':>11s} {'K':>6s} {'maxε/ε0':>8s} {'ε-Drift%':>9s} {'M/M0':>6s} {'M-Drift%':>9s} {'τ_trap':>7s} {'Pe':>8s}")
    rows = []
    for res in results:
        p = res['params']
        (K, _) = kanagawa(p['q_planet'], p['alpha'], p['h0'])
        tt = tau_trap(res)
        ts = f'{tt:.1f}' if not np.isnan(tt) else '—'
        (depth_ss, depth_slope) = stationary(res['times'], res['depth'])
        (eps_ss, eps_slope) = stationary(res['times'], res['max_eps'])
        (M_ss, M_slope) = stationary(res['times'], res['M_trap'])
        M0 = max(res['M_trap'][0], 1e-30)
        eps_dr = window_drift_pct(res['times'], res['max_eps'])
        M_dr = window_drift_pct(res['times'], res['M_trap'])
        print(f"{res['label']:<16s} {depth_ss:7.3f} {depth_slope:11.2e} {K:6.0f} {eps_ss / d3.eps0:8.2f} {eps_dr:8.1f}% {M_ss / M0:6.2f} {M_dr:8.1f}% {ts:>7s} {peclet(res):8.2e}")
        rows.append((res['label'], p['q_planet'], p['St'], p['alpha'], p['h0'], p['t_orbits'], depth_ss, depth_slope, K, eps_ss / d3.eps0, eps_slope / d3.eps0, eps_dr, M_ss / M0, M_slope / M0, M_dr, tt, res['w_eff'], peclet(res)))
    numval.save_rows('parameterstudie', f'summary_{sweep_name}', ['lauf', 'q_planet', 'St', 'alpha', 'h0', 't_orbits', 'tiefe', 'tiefe_slope_pro_orb', 'K', 'maxeps_over_eps0', 'eps_slope_pro_orb', 'eps_drift_pct_fenster', 'M_over_M0', 'M_slope_pro_orb', 'M_drift_pct_fenster', 'tau_trap_orb', 'w_eff', 'Pe'], rows)

def plot_q_threshold(results, pdfname='linblad_q_threshold.pdf', eps_factor=2.0):
    q = np.array([r['params']['q_planet'] for r in results])
    K = np.array([kanagawa(r['params']['q_planet'], r['params']['alpha'], r['params']['h0'])[0] for r in results])
    eps = np.array([stationary(r['times'], r['max_eps'])[0] for r in results])
    M = np.array([stationary(r['times'], r['M_trap'])[0] / max(r['M_trap'][0], 1e-30) for r in results])
    thr = eps_factor * d3.eps0
    idx = np.where(eps > thr)[0]
    q_crit = float(q[idx[0]]) if len(idx) else np.nan
    (fig, axes) = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(q, eps / d3.eps0, 'o-', color='darkblue', lw=1.4)
    axes[0].axhline(eps_factor, ls=':', color='gray', lw=0.8, label=f'ε_thr={eps_factor:g}·ε₀')
    if not np.isnan(q_crit):
        axes[0].axvline(q_crit, ls='--', color='red', lw=1.0, label=f'q_crit≈{q_crit:.1e}')
    axes[0].set_xlabel('q_planet')
    axes[0].set_ylabel('$\\max_r\\varepsilon/\\varepsilon_0$ (t_end)')
    axes[0].set_title('Trapping-Schwelle vs. Planetenmasse')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(K, M, 's-', color='darkgreen', lw=1.4)
    axes[1].axhline(1.0, ls=':', color='gray', lw=0.8)
    axes[1].set_xlabel('Kanagawa K')
    axes[1].set_ylabel('$M_\\mathrm{trap}/M_\\mathrm{trap}(0)$')
    axes[1].set_title('Trap-Massenaufbau vs. K')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"q-Sweep — q_crit (ε>{eps_factor:g}ε₀) ≈ {('—' if np.isnan(q_crit) else f'{q_crit:.2e}')}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, pdfname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {path}')
    print(f"  q_crit (ε > {eps_factor:g}·ε₀) ≈ {('—' if np.isnan(q_crit) else f'{q_crit:.3e}')}")
    numval.save_rows('parameterstudie', 'q_threshold', ['q_planet', 'K', 'maxeps_over_eps0', 'M_over_M0'], list(zip(q, K, eps / d3.eps0, M)))
    numval.save_scalars('parameterstudie', 'q_threshold_qcrit', {'eps_factor': eps_factor, 'q_crit': q_crit})

def run_2d_map(x_name, x_values, y_name, y_values, **fixed):
    (n_y, n_x) = (len(y_values), len(x_values))
    eps_grid = np.zeros((n_y, n_x))
    M_grid = np.zeros((n_y, n_x))
    drift_grid = np.zeros((n_y, n_x))
    eps_drift_grid = np.zeros((n_y, n_x))
    print(f'\n══ 2D-Map {x_name}×{y_name}  ({n_x}×{n_y}={n_x * n_y} Läufe) ══')
    for (j, yv) in enumerate(y_values):
        for (i, xv) in enumerate(x_values):
            params = dict(fixed)
            params[x_name] = xv
            params[y_name] = yv
            print(f'  [{j * n_x + i + 1:>3d}/{n_x * n_y}] {x_name}={xv:g}, {y_name}={yv:g}')
            r = run_one(f'{x_name}={xv:g},{y_name}={yv:g}', prog=f'[Map {x_name}×{y_name} {j * n_x + i + 1}/{n_x * n_y}]', **params)
            eps_grid[j, i] = stationary(r['times'], r['max_eps'])[0]
            M_grid[j, i] = stationary(r['times'], r['M_trap'])[0] / max(r['M_trap'][0], 1e-30)
            drift_grid[j, i] = stationary(r['times'], r['depth'])[1]
            eps_drift_grid[j, i] = window_drift_pct(r['times'], r['max_eps'])
    (X, Y) = np.meshgrid(x_values, y_values)
    numval.save_map('parameterstudie', f'map_{x_name}_{y_name}', X, Y, eps=eps_grid, eps_over_eps0=eps_grid / d3.eps0, M_over_M0=M_grid, drift=drift_grid, eps_drift_pct=eps_drift_grid)
    return dict(x_name=x_name, y_name=y_name, x_values=np.asarray(x_values, float), y_values=np.asarray(y_values, float), X=X, Y=Y, eps=eps_grid, M=M_grid, drift=drift_grid, eps_drift=eps_drift_grid, t_orbits=fixed.get('t_orbits'), fixed=fixed)

def plot_2d_map(m, pdfname, log_x=True, log_y=False, fit_overlay=None, blind_mask=None, blind_label=None):
    (fig, ax) = plt.subplots(1, 1, figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(m['X'], m['Y'], np.log10(np.maximum(m['eps'] / d3.eps0, 0.001)), shading='auto', cmap='viridis')
    if blind_mask is not None and blind_mask.any():
        ax.contourf(m['X'], m['Y'], blind_mask.astype(float), levels=[0.5, 1.5], colors='none', hatches=['///'])
        ax.contour(m['X'], m['Y'], blind_mask.astype(float), levels=[0.5], colors='white', linewidths=1.0, linestyles='--')
        if blind_label:
            ax.plot([], [], ls='none', marker='s', mfc='none', mec='white', label=blind_label)
    fig.colorbar(pcm, ax=ax).set_label('$\\log_{10}(\\max_r\\varepsilon/\\varepsilon_0)\\,(t_\\mathrm{end})$')
    levels = [1.5 * d3.eps0, 2.0 * d3.eps0, 5.0 * d3.eps0, 10.0 * d3.eps0]
    try:
        cs = ax.contour(m['X'], m['Y'], m['eps'], levels=levels, colors=['cyan', 'white', 'orange', 'red'], linewidths=1.2)
        ax.clabel(cs, fmt={l: f'{l / d3.eps0:g}ε₀' for l in levels}, fontsize=8)
    except Exception:
        pass
    if fit_overlay is not None:
        ax.plot(fit_overlay['x'], fit_overlay['y'], 'k-', lw=1.6, label=fit_overlay.get('label', 'Fit'))
    if fit_overlay is not None or (blind_mask is not None and blind_label):
        ax.legend(fontsize=8, loc='best')
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlim(m['x_values'].min(), m['x_values'].max())
    ax.set_ylim(m['y_values'].min(), m['y_values'].max())
    ax.set_xlabel(m['x_name'])
    ax.set_ylabel(m['y_name'])
    fixedstr = ', '.join((f'{k}={v}' for (k, v) in m['fixed'].items() if k not in (m['x_name'], m['y_name'], 'N', 't_orbits', 'A_gap', 'w_gap')))
    ax.set_title(f'max ε  |  {fixedstr}')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, pdfname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {path}')

def extract_q_crit(map_Stq, eps0=0.01, candidates=(2.0, 1.5, 1.2, 1.1), min_valid=3):
    assert map_Stq['x_name'] == 'St' and map_Stq['y_name'] == 'q_planet'
    St = map_Stq['x_values']
    qv = map_Stq['y_values']

    def at(f):
        thr = f * eps0
        qc = np.full(len(St), np.nan)
        cens = np.zeros(len(St), bool)
        for i in range(len(St)):
            col = map_Stq['eps'][:, i]
            idx = np.where(col > thr)[0]
            if len(idx):
                k = idx[0]
                if k == 0:
                    qc[i] = qv[0]
                    cens[i] = True
                else:
                    qc[i] = qv[k - 1] + (thr - col[k - 1]) * (qv[k] - qv[k - 1]) / (col[k] - col[k - 1])
        return (qc, cens)
    best = None
    for f in candidates:
        (qc, cens) = at(f)
        nv = int(np.sum(np.isfinite(qc) & ~cens))
        if nv >= min_valid:
            return (St, qc, float(f), cens)
        if best is None or nv > best[3]:
            best = (qc, cens, f, nv)
    return (St, best[0], float(best[2]), best[1])

def fit_q_crit(St, q_crit, censored=None):
    m = np.isfinite(q_crit) & (q_crit > 0)
    if censored is not None:
        m = m & ~np.asarray(censored, bool)
    if m.sum() < 3:
        return None
    (slope, icpt) = np.polyfit(np.log(St[m]), np.log(q_crit[m]), 1)
    return dict(C=float(np.exp(icpt)), beta=float(slope), n_fit=int(m.sum()), n_censored=0 if censored is None else int(np.sum(censored)), formula=f'q_crit ≈ {np.exp(icpt):.2e}·St^{slope:.2f}')

def mrn_weights(St_list, q=3.5):
    w = np.asarray(St_list, float) ** (4.0 - q)
    return w / w.sum()

def run_multipop(St_list, q=3.5, label='MRN', **params):
    w = mrn_weights(St_list, q=q)
    print(f"\n══ Multi-Pop MRN q={q}: St={St_list}, w=[{', '.join((f'{x:.3f}' for x in w))}] ══")
    runs = []
    for (St_i, w_i) in zip(St_list, w):
        p = dict(params)
        p['St'] = St_i
        print(f'  → St={St_i:g} (w={w_i:.3f})')
        r = run_one(f'{label}/St={St_i:g}', **p)
        r['weight'] = float(w_i)
        runs.append(r)
    n_snap = len(runs[0]['snaps'])
    comb = []
    for j in range(n_snap):
        tj = runs[0]['snaps'][j][0]
        sgj = runs[0]['snaps'][j][1]
        sdt = sum((r['weight'] * r['snaps'][j][2] for r in runs))
        comb.append((tj, sgj, sdt))
    max_eps = np.array([float((s[2][1:-1] / np.maximum(s[1][1:-1], d3.SIGMA_FLOOR)).max()) for s in comb])
    return dict(label=label, St_list=list(St_list), weights=w.tolist(), runs=runs, combined_snaps=comb, times=runs[0]['times'], max_eps=max_eps, r_active=runs[0]['r_active'], S0=runs[0]['S0'])

def plot_multipop(mp, pdfname='linblad_multipop_MRN.pdf'):
    (fig, axes) = plt.subplots(2, 2, figsize=(13, 9))
    r_act = mp['r_active']
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(mp['runs'])))
    for (r, c) in zip(mp['runs'], colors):
        (_, sgf, sdf) = r['snaps'][-1]
        axes[0, 0].plot(r_act, sdf[1:-1], color=c, lw=1.3, label=f"St={r['params']['St']:g}, w={r['weight']:.2f}")
        axes[1, 0].plot(r['times'] / d3.T_orb, r['max_eps'] / d3.eps0, color=c, lw=1.3)
    (_, sgt, sdt) = mp['combined_snaps'][-1]
    axes[0, 1].plot(r_act, sdt[1:-1], 'k-', lw=1.6, label='Σ_d total (MRN)')
    axes[0, 1].plot(r_act, d3.eps0 * sgt[1:-1], 'b--', lw=0.8, label='ε₀·Σ_g (Start)')
    eps_tot = sdt[1:-1] / np.maximum(sgt[1:-1], d3.SIGMA_FLOOR)
    axes[1, 1].plot(r_act, eps_tot / d3.eps0, 'k-', lw=1.5)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xscale('log')
        ax.set_xlim(0.4, 3.5)
        ax.axvline(1.0, ls=':', color='crimson', lw=1)
        ax.set_xlabel('r')
    axes[0, 0].set_ylabel('$\\Sigma_{d,i}$')
    axes[0, 0].set_title('Einzelpopulationen Σ_d(t_end)')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_ylabel('$\\Sigma_{d,\\mathrm{tot}}$')
    axes[0, 1].set_title('Kombiniert (MRN-gewichtet)')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel('t / T_orb')
    axes[1, 0].set_ylabel('$\\max_r\\varepsilon_i/\\varepsilon_0$')
    axes[1, 0].set_title('max ε je Population')
    axes[1, 0].axhline(1, ls='--', color='gray', lw=0.6)
    axes[1, 1].axhline(1, ls='--', color='gray', lw=0.6)
    axes[1, 1].set_ylabel('$\\varepsilon_\\mathrm{tot}/\\varepsilon_0$')
    axes[1, 1].set_title('Total-ε(t_end)')
    fig.suptitle(f"Multi-Pop (MRN q=3.5) an Planetenlücke — St ∈ {mp['St_list']}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, pdfname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gespeichert: {path}')

def robustness_table():
    print(f'\n══ Numerische Robustheit (Default-Setup) ══')
    rows = []
    Ns = [300, 600, 1200] if not SMOKE else [80, 120]
    for N in Ns:
        rows.append((f'N = {N}', run_one(f'rob-N{N}', N=N)))
    (r_min0, r_max0) = (d3.r_min, d3.r_max)
    for (rmn, rmx, name) in [(0.2, 6.0, 'Domain weit'), (0.4, 4.0, 'Domain eng')]:
        (d3.r_min, d3.r_max) = (rmn, rmx)
        rows.append((name, run_one(f'rob-{name}')))
    (d3.r_min, d3.r_max) = (r_min0, r_max0)
    for cd in [0.0, 0.5, 1.0, 2.0] if not SMOKE else [0.0, 1.0]:
        rows.append((f'cd_factor={cd}', run_one(f'rob-cd{cd}', cd_factor=cd)))
    print(f"\n{'Variante':<20s} {'maxε(t_end)':>12s} {'M/M0':>8s} {'Tiefe':>7s} {'Δ% vs N600':>11s}")
    eps_ref = None
    dump = []
    for (name, res) in rows:
        e = res['max_eps'][-1]
        if name == 'N = 600':
            eps_ref = e
        d = (e - eps_ref) / eps_ref * 100.0 if eps_ref else 0.0
        print(f"{name:<20s} {e:>12.3e} {res['M_trap'][-1] / max(res['M_trap'][0], 1e-30):>8.2f} {res['depth'][-1]:>7.3f} {d:>+11.1f}")
        dump.append((name, e, res['M_trap'][-1] / max(res['M_trap'][0], 1e-30), res['depth'][-1], d))
    numval.save_rows('parameterstudie', 'robustheit', ['variante', 'maxeps_tend', 'M_over_M0', 'tiefe', 'delta_pct_vs_N600'], dump)
    return rows
if __name__ == '__main__':
    print(f"{'=' * 60}\nPARAMETERSTUDIE LINDBLAD-PLANETENLÜCKE  (SMOKE={SMOKE})\n{'=' * 60}")
    print('\n══ Referenz-Modell (Default) ══')
    res_ref = run_one('reference', **DEFAULTS)
    plot_reference_evolution(res_ref)
    print_summary_table([res_ref], 'Referenz')
    q_vals = np.geomspace(0.00015, 0.0009, 10) if not SMOKE else [0.0003, 0.0006]
    res_q = sweep('q_planet', q_vals)
    plot_sweep(res_q, 'q_planet (Planetenmasse)', 'linblad_sweep_q.pdf')
    plot_q_threshold(res_q)
    print_summary_table(res_q, 'q_planet')
    St_vals = np.geomspace(0.02, 1.0, 10) if not SMOKE else [0.1, 0.3]
    res_St = sweep('St', St_vals)
    plot_sweep(res_St, 'St (Stokes-Zahl)', 'linblad_sweep_St.pdf')
    print_summary_table(res_St, 'St')
    al_vals = np.geomspace(0.003, 0.03, 8) if not SMOKE else [0.01, 0.03]
    t_alpha = lambda a: float(np.clip(DEFAULTS['t_orbits'] * (0.01 / a) ** 0.7, 150, 600)) if not SMOKE else 25.0
    res_al = sweep('alpha', al_vals, t_orbits_fn=t_alpha)
    plot_sweep(res_al, 'α (Viskosität)', 'linblad_sweep_alpha.pdf')
    print_summary_table(res_al, 'alpha')
    h_vals = np.round(np.linspace(0.04, 0.11, 8), 3) if not SMOKE else [0.05, 0.07]
    res_h = sweep('h0', h_vals)
    plot_sweep(res_h, 'h_0 (Aspektverhältnis)', 'linblad_sweep_h0.pdf')
    print_summary_table(res_h, 'h0')
    mp_St = [0.03, 0.1, 0.3, 1.0] if not SMOKE else [0.1, 0.3]
    mp = run_multipop(mp_St, q=3.5, **DEFAULTS)
    plot_multipop(mp)
    NMAP = 12 if not SMOKE else 2
    N_MAP = 1600 if not SMOKE else 80
    T_MAP = 250.0 if not SMOKE else 25.0
    map_fixed = {k: v for (k, v) in DEFAULTS.items() if k not in ('St', 'q_planet', 't_orbits', 'N')}
    map_fixed['t_orbits'] = T_MAP
    map_fixed['N'] = N_MAP
    Stm = np.geomspace(0.02, 1.0, NMAP)
    qm = np.geomspace(3e-05, 0.0009, NMAP)
    map_Stq = run_2d_map('St', Stm, 'q_planet', qm, **map_fixed)
    plot_2d_map(map_Stq, 'linblad_map_St_q.pdf', log_x=True, log_y=True)
    (St_a, q_crit, used_f, cens) = extract_q_crit(map_Stq)
    print(f'\n──── q_crit aus St×q-Map (Schwelle ε>{used_f:g}ε₀) ────')
    for (s, qc, c) in zip(St_a, q_crit, cens):
        tag = '  ≤ (zensiert: unterste q-Zeile)' if c else ''
        print(f"  St={s:.2e} → q_crit={('—' if np.isnan(qc) else f'{qc:.2e}')}{tag}")
    n_free = int(np.sum(np.isfinite(q_crit) & ~cens))
    print(f'  frei aufgelöst: {n_free} von {len(St_a)}; zensiert: {int(cens.sum())}')
    if cens.any():
        print('  ACHTUNG: zensierte Punkte sind obere Schranken. Für einen belastbaren Fit muss die q-Achse nach unten verlängert werden.')
    numval.save_rows('parameterstudie', 'q_crit_von_St', ['St', 'q_crit', 'eps_factor', 'zensiert'], [(s, qc, used_f, int(c)) for (s, qc, c) in zip(St_a, q_crit, cens)])
    fit = fit_q_crit(St_a, q_crit, censored=cens)
    if fit is not None:
        print(f"  Fit über {fit['n_fit']} freie Punkte ({fit['n_censored']} zensierte ausgeschlossen): {fit['formula']}")
        Sf = np.geomspace(St_a.min(), St_a.max(), 50)
        plot_2d_map(map_Stq, 'linblad_map_St_q_fit.pdf', log_x=True, log_y=True, fit_overlay=dict(x=Sf, y=fit['C'] * Sf ** fit['beta'], label=fit['formula']))
    else:
        print('  KEIN Fit: zu wenige frei aufgelöste Punkte. q-Achse nach unten verlängern und erneut auswerten.')
    map_fixed2 = {k: v for (k, v) in DEFAULTS.items() if k not in ('St', 'alpha', 't_orbits', 'N')}
    map_fixed2['t_orbits'] = T_MAP
    map_fixed2['N'] = N_MAP
    al_m = np.geomspace(0.001, 0.03, NMAP)
    map_Sta = run_2d_map('St', Stm, 'alpha', al_m, **map_fixed2)
    plot_2d_map(map_Sta, 'linblad_map_St_alpha.pdf', log_x=True, log_y=True)
    robustness_table()
    print('\n═══ Parameterstudie (Lindblad) abgeschlossen ═══')
