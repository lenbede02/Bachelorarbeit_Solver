import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__))
_NUM = os.path.join(_HERE, 'numVal', 'parameterstudie')
_OUT = os.path.normpath(os.path.join(_HERE, '..', 'thesis', 'figures', 'parameterstudie', 'linblad_peclet_collapse.pdf'))
os.makedirs(os.path.dirname(_OUT), exist_ok=True)
W_EFF_FLOOR = 0.01
SWEEPS = [('summary_St.csv', '$\\mathrm{St}$-Sweep', 'C0', 'o'), ('summary_alpha.csv', '$\\alpha$-Sweep', 'C1', 's'), ('summary_q_planet.csv', '$q$-Sweep', 'C2', '^'), ('summary_Referenz.csv', 'Referenz', 'crimson', '*')]
FIT_PE_MIN, FIT_EPS_MIN = (0.8, 1.5)

def load(fn):
    Pe, eps, weff = ([], [], [])
    path = os.path.join(_NUM, fn)
    if not os.path.exists(path):
        return (np.array([]),) * 3
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                p = float(r['Pe'])
                e = float(r['maxeps_over_eps0'])
                w = float(r.get('w_eff', 'nan'))
            except (KeyError, ValueError):
                continue
            if np.isfinite(p) and np.isfinite(e) and (p > 0) and (e > 0):
                Pe.append(p)
                eps.append(e)
                weff.append(w)
    return (np.array(Pe), np.array(eps), np.array(weff))

def powfit(Pe, eps):
    if Pe.size < 3:
        return None
    k, c = np.polyfit(np.log10(Pe), np.log10(eps), 1)
    z = k * np.log10(Pe) + c
    ss_res = np.sum((np.log10(eps) - z) ** 2)
    ss_tot = np.sum((np.log10(eps) - np.log10(eps).mean()) ** 2)
    return (k, c, 1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
fig, ax = plt.subplots(figsize=(7.4, 5.4))
allPe, allEps, branch_rows = ([], [], [])
n_floor_tot = n_below_tot = 0
for fn, lab, col, mk in SWEEPS:
    Pe, eps, weff = load(fn)
    if Pe.size == 0:
        continue
    floor = weff <= W_EFF_FLOOR
    ok = ~floor & (Pe > FIT_PE_MIN) & (eps > FIT_EPS_MIN)
    below = ~floor & ~ok
    n_floor_tot += int(floor.sum())
    n_below_tot += int(below.sum())
    ax.scatter(Pe[ok], eps[ok], s=120 if mk == '*' else 46, c=col, marker=mk, edgecolor='k', linewidth=0.4, alpha=0.95, label=lab, zorder=3)
    if below.any():
        ax.scatter(Pe[below], eps[below], s=34, facecolor='none', edgecolor=col, marker=mk, linewidth=1.0, alpha=0.8, zorder=3)
    if floor.any():
        ax.scatter(Pe[floor], eps[floor], s=52, c='0.6', marker='x', linewidth=1.3, alpha=0.95, zorder=3)
    fit = powfit(Pe[ok], eps[ok])
    if fit is not None:
        k, c, R2 = fit
        xs = np.geomspace(Pe[ok].min(), Pe[ok].max(), 30)
        ax.plot(xs, 10 ** c * xs ** k, color=col, lw=1.3, ls='-', alpha=0.55, zorder=2)
        branch_rows.append((lab, int(ok.sum()), Pe[ok].min(), Pe[ok].max(), float(np.log10(Pe[ok].max() / Pe[ok].min())), weff[ok].max() / max(weff[ok].min(), 1e-30), k, R2))
    allPe.append(Pe[ok])
    allEps.append(eps[ok])
allPe = np.concatenate(allPe)
allEps = np.concatenate(allEps)
kp, cp, R2p = powfit(allPe, allEps)
xf = np.geomspace(allPe.min(), allPe.max(), 60)
ax.plot(xf, 10 ** cp * xf ** kp, 'k--', lw=1.8, zorder=4, label='gepoolt: $\\propto\\mathrm{Pe}^{%.2f}$ ($R^2=%.2f$)' % (kp, R2p) + '\n' + '(vom $\\mathrm{St}$-Ast getragen)')
ax.axvline(1.0, ls=':', color='0.35', lw=1.2, zorder=1)
ax.text(1.05, allEps.min() * 1.05, 'Pe = 1', fontsize=8.5, color='0.3', va='bottom')
ax.scatter([], [], s=34, facecolor='none', edgecolor='0.4', marker='o', linewidth=1.0, label='unter dem Fit-Fenster')
ax.scatter([], [], s=52, c='0.6', marker='x', linewidth=1.3, label='kein Druckmax. gefunden ($w_{\\mathrm{eff}}$-Floor)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('gemessenes Péclet  $\\mathrm{Pe}=|v_{r,d}|\\,w_{\\mathrm{eff}}/D_d$')
ax.set_ylabel('Anreicherung  $\\varepsilon_{d/g,\\max}/\\varepsilon_{d/g,0}$')
ax.set_title('Anreicherung vs. Péclet — astweise')
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=7.8, loc='upper left', framealpha=0.92)
plt.tight_layout()
plt.savefig(_OUT, bbox_inches='tight')
plt.close()
print('\n──── Péclet: Ast-für-Ast statt gepoolt ────')
print(f"{'Ast':<16s} {'n':>3s} {'Pe von':>8s} {'Pe bis':>8s} {'Dekaden':>8s} {'w_eff-Var':>10s} {'Exponent':>9s} {'R²':>6s}")
for lab, n, p0, p1, dek, wvar, k, R2 in branch_rows:
    clean = lab.replace('$', '').replace('\\mathrm{St}', 'St').replace('\\alpha', 'alpha')
    print(f'{clean:<16s} {n:3d} {p0:8.2f} {p1:8.2f} {dek:8.2f} {wvar:9.2f}x {k:9.2f} {R2:6.3f}')
print(f"{'GEPOOLT':<16s} {allPe.size:3d} {allPe.min():8.2f} {allPe.max():8.2f} {np.log10(allPe.max() / allPe.min()):8.2f} {'—':>10s} {kp:9.2f} {R2p:6.3f}")
print(f'\nVerworfen: {n_below_tot} Punkte unter dem Fit-Fenster (Pe<{FIT_PE_MIN} oder eps<{FIT_EPS_MIN}), {n_floor_tot} Punkte ohne gefundenes Druckmaximum (w_eff-Floor).')
try:
    import numval
    numval.save_rows('parameterstudie', 'peclet_ast_fits', ['ast', 'n', 'Pe_min', 'Pe_max', 'dekaden', 'w_eff_variation', 'exponent', 'R2'], [(lab.replace('$', ''), n, p0, p1, dek, wvar, k, R2) for lab, n, p0, p1, dek, wvar, k, R2 in branch_rows] + [('gepoolt', int(allPe.size), float(allPe.min()), float(allPe.max()), float(np.log10(allPe.max() / allPe.min())), float('nan'), kp, R2p)])
except Exception as exc:
    print(f'(numVal-Export übersprungen: {exc})')
print('\nPlot:', _OUT)
