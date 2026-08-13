import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import disk_v3_erweitert_linblad as d3
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.normpath(os.path.join(_HERE, '..', 'thesis', 'figures', 'parameterstudie', 'backreaction_compare.pdf'))
os.makedirs(os.path.dirname(_OUT), exist_ok=True)

def run_case(backreact, St, N=400, t_orbits=250.0, q=0.00045, alpha=0.01, h0=0.05):
    d3.DUST_BACKREACTION = backreact
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
    r = d3.r
    idx = int(np.argmin(np.abs(r - 1.0)))
    S0 = d3.Sigma0 * (1.0 / d3.r0) ** (-d3.p_slope)
    eps0 = d3.eps0
    t_end = t_orbits * d3.T_orb
    t = 0.0
    nextlog = 0.0
    traj = []
    while t < t_end:
        dt = min(d3.compute_dt(sig, mr, sd, mrd), t_end - t)
        sig, mr, L, sd, mrd, Ld = d3.hydro_step(sig, mr, L, sd, mrd, Ld, dt, t_now=t)
        sig, mr, L, sd, mrd, Ld = d3.apply_damping(sig, mr, L, sd, mrd, Ld, tgt, dr, dt)
        t += dt
        if t / d3.T_orb >= nextlog:
            eps = sd[1:-1] / np.maximum(sig[1:-1], d3.SIGMA_FLOOR)
            traj.append((t / d3.T_orb, float(eps.max()) / eps0))
            nextlog += 25.0
    eps = sd[1:-1] / np.maximum(sig[1:-1], d3.SIGMA_FLOOR)
    return dict(maxeps=float(eps.max()) / eps0, eps_abs=float(eps.max()), depth=float(sig[idx] / S0), traj=traj, r=r[1:-1].copy(), eps_prof=(eps / eps0).copy())
if __name__ == '__main__':
    print('Rückreaktion: MIT (ON) vs OHNE (OFF, Testflüssigkeit)')
    results = {}
    for St in (0.1, 1.0):
        for flag in (True, False):
            t0 = time.time()
            res = run_case(flag, St)
            el = time.time() - t0
            results[St, flag] = res
            print('St=%.1f  backreact=%-5s  max_eps/eps0=%9.2f  eps_abs=%.3f  depth=%.3f  (%.0fs)' % (St, flag, res['maxeps'], res['eps_abs'], res['depth'], el), flush=True)
    print('\n Effekt der Rückreaktion')
    for St in (0.1, 1.0):
        on = results[St, True]['maxeps']
        off = results[St, False]['maxeps']
        print('St=%.1f:  ON=%.2f  OFF=%.2f  ->  OFF/ON = %.2f  (Rückreaktion senkt Peak um %.0f%%)' % (St, on, off, off / on, 100 * (1 - on / off)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, St in zip(axes, (0.1, 1.0)):
        for flag, c, ls in [(True, 'C0', '-'), (False, 'C3', '--')]:
            tr = results[St, flag]['traj']
            ax.plot([x[0] for x in tr], [x[1] for x in tr], color=c, ls=ls, lw=1.8, label='mit Rückreaktion' if flag else 'ohne (Testflüssigkeit)')
        ax.set_title('St=%.1f' % St)
        ax.set_xlabel('t / T_orb')
        ax.set_ylabel('$\\max_r\\,\\varepsilon/\\varepsilon_0$')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle('Staub-Rückreaktion: voller Drag vs. Testflüssigkeit', fontsize=12)
    plt.tight_layout()
    plt.savefig(_OUT, bbox_inches='tight')
    print(f'\nGespeichert: {_OUT}', flush=True)
