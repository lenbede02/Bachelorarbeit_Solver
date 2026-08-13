import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parameterstudie_linblad as ps
import disk_v3_erweitert_linblad as d3
import numval
OUT = os.path.join(ps.OUTPUT_DIR, 'druckprofil_falle.pdf')

def make_figure(r, sig_g_rel, Pi, dPi, eps, *, rp, r_pmax, r_eps, r_zero, delta, dr_loc, H_p):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.6), sharex=True)
    ax = axes[0]
    ax.plot(r, sig_g_rel, color='C0', lw=1.7)
    ax.set_ylabel('$\\Sigma_g/\\Sigma_0$')
    ax.set_title('Selbstkonsistente Planetenlücke: Gas, Druck, Staubfalle')
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(r, Pi, color='C2', lw=1.7, label='$\\Pi=c_s^2\\Sigma_g$')
    ax.set_ylabel('$\\Pi$', color='C2')
    ax.tick_params(axis='y', labelcolor='C2')
    ax.grid(alpha=0.3)
    axb = ax.twinx()
    axb.plot(r, dPi, color='C3', lw=1.3, ls='--', label='$\\partial_r\\Pi$')
    axb.axhline(0.0, color='C3', lw=0.8, ls=':', alpha=0.8)
    near = (r > rp) & (r < rp + 8.0 * H_p)
    if near.any():
        m = float(np.max(np.abs(dPi[near])))
        axb.set_ylim(-2.2 * m, 2.2 * m)
    if np.isfinite(r_zero):
        axb.plot([r_zero], [0.0], 'o', color='C3', ms=7, markeredgecolor='k', markeredgewidth=0.5, zorder=6)
    axb.set_ylabel('$\\partial_r\\Pi$', color='C3')
    axb.tick_params(axis='y', labelcolor='C3')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper left')
    ax = axes[2]
    ax.plot(r, eps, color='C4', lw=1.7)
    ax.plot([r_eps], [float(np.max(eps[(r > rp) & (r < rp + 16.0 * H_p)]))], 'v', color='C4', ms=10, markeredgecolor='k', markeredgewidth=0.5, zorder=5)
    ax.set_ylabel('$\\varepsilon/\\varepsilon_0$')
    ax.set_xlabel('$r$')
    ax.grid(alpha=0.3)
    ax.text(0.025, 0.6, f'$r_\\varepsilon - r_{{\\Pi,\\max}} = {delta:+.4f}$\n$= {delta / dr_loc:+.2f}\\,\\mathrm{{d}}r = {delta / H_p:+.3f}\\,H_p$', transform=ax.transAxes, fontsize=9, bbox=dict(fc='white', ec='0.7', alpha=0.92))
    r_lo, r_hi = (0.8, rp + 12.0 * H_p)
    vis = (r >= r_lo) & (r <= r_hi)
    for ax, y in ((axes[0], sig_g_rel), (axes[1], Pi), (axes[2], eps)):
        lo, hi = (float(y[vis].min()), float(y[vis].max()))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes:
        ax.axvline(rp, color='gray', lw=1.0, ls=':')
        ax.axvline(r_pmax, color='k', lw=1.1, ls='--')
        ax.set_xlim(r_lo, r_hi)
    y1 = axes[0].get_ylim()[1]
    axes[0].annotate('$r_p$', xy=(rp, y1), xytext=(-16, -13), textcoords='offset points', fontsize=9, color='gray')
    axes[0].annotate('$r_{\\Pi,\\max}$', xy=(r_pmax, y1), xytext=(5, -13), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nGespeichert: {OUT}')

def replot_from_csv():
    import csv as _csv
    base = os.path.join((_HERE_NUM := os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numVal', 'parameterstudie')))
    cols = {k: [] for k in ('r', 'Sigma_g_rel', 'Pi', 'dPi_dr', 'eps_rel')}
    with open(os.path.join(base, 'druckprofil_falle.csv')) as f:
        for row in _csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    a = {k: np.array(v) for k, v in cols.items()}
    kz = {}
    with open(os.path.join(base, 'druckprofil_kennzahlen.csv')) as f:
        for row in _csv.DictReader(f):
            kz[row['groesse']] = float(row['wert'])
    make_figure(a['r'], a['Sigma_g_rel'], a['Pi'], a['dPi_dr'], a['eps_rel'], rp=kz['r_planet'], r_pmax=kz['r_pmax'], r_eps=kz['r_eps_peak'], r_zero=kz['r_dPi_null'], delta=kz['delta_r_eps_minus_r_pmax'], dr_loc=kz['dr_lokal'], H_p=kz['H_p'])

def main():
    res = ps.run_one('druckprofil-fiducial', prog='[Druckprofil 1/1]', **ps.DEFAULTS)
    r = res['r_active']
    sig_g = res['snaps'][-1][1][1:-1]
    sig_d = res['snaps'][-1][2][1:-1]
    cs = d3.cs[1:-1]
    dr = d3.dr[1:-1]
    rp = d3.r_planet
    Pi = cs ** 2 * sig_g
    dPi = np.gradient(Pi, r)
    eps = sig_d / np.maximum(sig_g, d3.SIGMA_FLOOR) / d3.eps0
    H_p = d3.h0 * np.sqrt(d3.GM / d3.r0) * (rp / d3.r0) ** (-0.25) / np.sqrt(d3.GM / rp ** 3)
    win = (r > rp) & (r < rp + 16.0 * H_p)
    i_pmax = int(np.argmax(Pi[win]))
    r_pmax = float(r[win][i_pmax])
    i_eps = int(np.argmax(eps[win]))
    r_eps = float(r[win][i_eps])
    dr_loc = float(dr[win][i_pmax])
    rz = np.nan
    seg = np.where(win)[0]
    for k in seg[:-1]:
        if dPi[k] > 0.0 >= dPi[k + 1]:
            rz = float(r[k] - dPi[k] * (r[k + 1] - r[k]) / (dPi[k + 1] - dPi[k]))
            break
    delta = r_eps - r_pmax
    print('\n Diagnostik')
    print(f'r_pmax = {r_pmax:.5f}')
    print(f'r_0    = {rz:.5f}')
    print(f'r_eps  = {r_eps:.5f}')
    print(f'dr     = {dr_loc:.5f}')
    print(f'H_p    = {H_p:.5f}')
    print(f'r_eps - r_pmax = {delta:+.5f} = {delta / dr_loc:+.2f}dr = {delta / H_p:+.3f}H_p')
    print(f"r_pmax - r_p = {res['w_eff']:.5f}")
    print(f'max ε/ε₀ = {eps.max():.2f}')
    make_figure(r, sig_g / res['S0'], Pi, dPi, eps, rp=rp, r_pmax=r_pmax, r_eps=r_eps, r_zero=rz, delta=delta, dr_loc=dr_loc, H_p=H_p)
    numval.save_rows('parameterstudie', 'druckprofil_falle', ['r', 'Sigma_g_rel', 'Pi', 'dPi_dr', 'eps_rel'], list(zip(r, sig_g / res['S0'], Pi, dPi, eps)))
    numval.save_rows('parameterstudie', 'druckprofil_kennzahlen', ['groesse', 'wert'], [('r_planet', rp), ('r_pmax', r_pmax), ('r_dPi_null', rz), ('r_eps_peak', r_eps), ('dr_lokal', dr_loc), ('H_p', H_p), ('delta_r_eps_minus_r_pmax', delta), ('delta_in_dr', delta / dr_loc), ('delta_in_Hp', delta / H_p), ('w_eff', res['w_eff']), ('max_eps_rel', float(eps.max()))])
    return dict(r_pmax=r_pmax, r_eps=r_eps, r_zero=rz, delta=delta, dr=dr_loc, H_p=H_p, w_eff=res['w_eff'], eps_max=float(eps.max()))
if __name__ == '__main__':
    import sys
    if '--replot' in sys.argv:
        replot_from_csv()
    else:
        main()
