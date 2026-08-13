import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import disk_v3_erweitert_linblad as d3
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.normpath(os.path.join(_HERE, '..', 'thesis', 'figures', 'parameterstudie', 'rwi_criterion.pdf'))
os.makedirs(os.path.dirname(_OUT), exist_ok=True)

def run_gap(q, alpha, N=600, t_orbits=250.0, h0=0.05, St=0.1):
    d3.A_gap = 0.0
    d3.h0 = h0
    d3.q_planet = q
    d3.r_planet = 1.0
    d3.St = St
    d3.alpha = alpha
    d3.C_torque = 0.4478
    d3.rebuild_grid(N)
    sig, mr, L, sd, mrd, Ld = d3.initial_state(nsh_drift_on=True, stationary_visc=True)
    d3.apply_outflow_bc(sig, mr, L)
    d3.apply_outflow_bc(sd, mrd, Ld)
    tgt = (sig.copy(), mr.copy(), L.copy(), sd.copy(), mrd.copy(), Ld.copy())
    dr = d3.build_damping_rate()
    t_end = t_orbits * d3.T_orb
    t = 0.0
    while t < t_end:
        dt = min(d3.compute_dt(sig, mr, sd, mrd), t_end - t)
        sig, mr, L, sd, mrd, Ld = d3.hydro_step(sig, mr, L, sd, mrd, Ld, dt, t_now=t)
        sig, mr, L, sd, mrd, Ld = d3.apply_damping(sig, mr, L, sd, mrd, Ld, tgt, dr, dt)
        t += dt
    return (sig.copy(), L.copy(), sd.copy(), d3.r.copy(), d3.cs.copy())
if __name__ == '__main__':
    sig, L, sd, r, cs = run_gap(0.00045, 0.01)
    s = np.maximum(sig, d3.SIGMA_FLOOR)
    j = L / s
    omega_z = np.gradient(j, r) / r
    vort = omega_z / s
    S0r = d3.Sigma0 * (r / d3.r0) ** (-d3.p_slope)
    sig_rel = sig / S0r
    eps_rel = sd / np.maximum(sig, d3.SIGMA_FLOOR) / d3.eps0
    bg = (r > 1.6) & (r < 2.5)
    vbg = np.median(vort[bg])
    edge = (r > 1.03) & (r < 1.4)
    i = np.argmin(vort[edge])
    r_min = r[edge][i]
    dip = (vbg - vort[edge][i]) / vbg
    print('Vortensitaets-Dip = %.0f%% bei r=%.3f (=%.1f H_p ausserhalb r_p)' % (100 * dip, r_min, (r_min - 1.0) / 0.05), flush=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))
    m = (r > 0.7) & (r < 2.0)
    axA.plot(r[m], sig_rel[m], 'b-', lw=1.8, label='$\\Sigma_g/\\Sigma_0$ (Gas-Lücke)')
    axA.plot(r[m], eps_rel[m], 'g-', lw=1.8, label='$\\varepsilon/\\varepsilon_0$ (Staubfalle)')
    axA.axvline(1.0, ls=':', color='crimson', lw=1)
    axA.axvline(r_min, ls='--', color='0.5', lw=1)
    axA.axhline(1.0, ls=':', color='gray', lw=0.5)
    axA.set_xlabel('r')
    axA.set_ylabel('normiert')
    axA.set_title('Selbstkonsistente Falle (Fiducial)')
    axA.legend(fontsize=9)
    axA.grid(alpha=0.3)
    axB.plot(r[m], (vort / vbg)[m], color='C1', lw=2.0, label='$\\varpi/\\varpi_\\infty$')
    axB.axhline(1.0, ls='--', color='gray', lw=0.6)
    axB.axhspan(0.0, 0.8, color='crimson', alpha=0.08)
    axB.axvline(1.0, ls=':', color='crimson', lw=1)
    axB.axvline(r_min, ls='--', color='0.5', lw=1)
    axB.annotate('Dip $\\approx$%.0f%%\n(notw. RWI-Bed.\nerfuellt)' % (100 * dip), xy=(r_min, vort[edge][i] / vbg), xytext=(1.3, 0.5), fontsize=9, arrowprops=dict(arrowstyle='->', color='0.3'))
    axB.set_xlabel('r')
    axB.set_ylabel('$\\varpi/\\varpi_\\infty$')
    axB.set_title('Vortensität: lokales MIN am Trap-Ort')
    axB.set_ylim(0, 1.6)
    axB.legend(fontsize=9)
    axB.grid(alpha=0.3)
    fig.suptitle('Lovelace-RWI-Kriterium am selbstkonsistenten Lückenrand', fontsize=12)
    plt.tight_layout()
    plt.savefig(_OUT, bbox_inches='tight')
    print(f'Plot: {_OUT}', flush=True)
