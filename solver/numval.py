import os
import csv
import re
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_FMT = '%.12g'
_installed = False

def _slug(s, fallback='x'):
    s = '' if s is None else str(s)
    s = re.sub('\\s+', '_', s.strip())
    s = re.sub('[^0-9A-Za-z_.\\-]', '', s)
    return s[:80] or fallback

def base_dir(tag):
    d = os.path.join(_HERE, 'numVal', _slug(tag, 'run'))
    os.makedirs(d, exist_ok=True)
    return d

def _write_columns(path, cols):
    names = list(cols.keys())
    n = max((len(cols[k]) for k in names), default=0)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow(['' if i >= len(cols[k]) else _FMT % float(cols[k][i]) for k in names])

def save_rows(tag, name, header, rows):
    try:
        path = os.path.join(base_dir(tag), _slug(name) + '.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                out = []
                for v in row:
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        out.append(_FMT % float(v))
                    else:
                        out.append('' if v is None else str(v))
                w.writerow(out)
        print(f'[numVal] {path}')
    except Exception as e:
        print(f"[numVal] save_rows('{name}') fehlgeschlagen: {e}")

def save_scalars(tag, name, mapping):
    save_rows(tag, name, ['key', 'value'], list(mapping.items()))

def save_map(tag, name, X, Y, **fields):
    try:
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        keys = list(fields.keys())
        arrs = [np.asarray(fields[k], float) for k in keys]
        path = os.path.join(base_dir(tag), _slug(name) + '_map.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x', 'y'] + keys)
            for idx in np.ndindex(X.shape):
                w.writerow([_FMT % X[idx], _FMT % Y[idx]] + [_FMT % a[idx] for a in arrs])
        print(f'[numVal] {path}')
    except Exception as e:
        print(f"[numVal] save_map('{name}') fehlgeschlagen: {e}")

def _is_reference_line(ln):
    x = np.asarray(ln.get_xdata(), float)
    y = np.asarray(ln.get_ydata(), float)
    if x.size == 2 and np.allclose(x, [0.0, 1.0]):
        return True
    if y.size == 2 and np.allclose(y, [0.0, 1.0]):
        return True
    return False

def _dump_axis_lines(ax, folder, ax_idx):
    cols, k = ({}, 0)
    for ln in ax.get_lines():
        try:
            if _is_reference_line(ln):
                continue
            x = np.asarray(ln.get_xdata(), float)
            y = np.asarray(ln.get_ydata(), float)
        except Exception:
            continue
        if x.size == 0:
            continue
        lab = ln.get_label()
        if not lab or str(lab).startswith('_'):
            lab = f'line{k}'
        lab = _slug(lab, f'line{k}')
        base, j = (lab, 1)
        while f'{lab}_x' in cols:
            lab = f'{base}.{j}'
            j += 1
        cols[f'{lab}_x'] = x
        cols[f'{lab}_y'] = y
        k += 1
    if not cols:
        return None
    title = _slug(ax.get_title(), f'ax{ax_idx}')
    path = os.path.join(folder, f'ax{ax_idx}_{title}.csv')
    _write_columns(path, cols)
    return (ax_idx, ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), ax.get_xscale(), ax.get_yscale())

def _dump_figure(fig, fname, tag):
    if isinstance(fname, (str, os.PathLike)):
        plotname = os.path.splitext(os.path.basename(os.fspath(fname)))[0]
    else:
        plotname = f'figure_{id(fig):x}'
    folder = os.path.join(base_dir(tag), _slug(plotname, 'figure'))
    os.makedirs(folder, exist_ok=True)
    meta = []
    for ax_idx, ax in enumerate(fig.get_axes()):
        row = _dump_axis_lines(ax, folder, ax_idx)
        if row is not None:
            meta.append(row)
    if meta:
        sup = ''
        if getattr(fig, '_suptitle', None) is not None:
            sup = fig._suptitle.get_text()
        with open(os.path.join(folder, 'meta.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['axis', 'title', 'xlabel', 'ylabel', 'xscale', 'yscale', 'suptitle'])
            for r in meta:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], sup])
        print(f'[numVal] {folder}/  ({len(meta)} Achsen)')

def install(tag):
    global _installed
    _tag_holder['tag'] = tag
    base_dir(tag)
    if _installed:
        return
    import matplotlib.figure as mfig
    _orig = mfig.Figure.savefig

    def _patched(self, *args, **kwargs):
        fname = args[0] if args else kwargs.get('fname', kwargs.get('filename'))
        try:
            _dump_figure(self, fname, _tag_holder['tag'])
        except Exception as e:
            print(f'[numVal] Dump fehlgeschlagen für {fname!r}: {e}')
        return _orig(self, *args, **kwargs)
    mfig.Figure.savefig = _patched
    _installed = True
    print(f'[numVal] Recorder aktiv → {base_dir(tag)}')
_tag_holder = {'tag': 'run'}
