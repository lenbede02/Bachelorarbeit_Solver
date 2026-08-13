import sys
import time

def _fmt(sec):
    sec = int(max(sec, 0))
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec // 60}m{sec % 60:02d}s'
    return f'{sec // 3600}h{sec % 3600 // 60:02d}m'

class Bar:

    def __init__(self, label='', width=26, every=0.5):
        self.label = label
        self.width = width
        self.every = every
        self.t0 = time.time()
        self.last = -1000000000.0
        self.last_frac = -1.0
        self.tty = bool(getattr(sys.stdout, 'isatty', lambda: False)())

    def update(self, frac, extra=''):
        frac = min(max(float(frac), 0.0), 1.0)
        now = time.time()
        done = frac >= 1.0
        if not done:
            if self.tty and now - self.last < self.every:
                return
            if not self.tty and frac - self.last_frac < 0.05:
                return
        self.last = now
        self.last_frac = frac
        el = now - self.t0
        eta = el * (1.0 - frac) / max(frac, 1e-09)
        fill = int(round(self.width * frac))
        bar = '█' * fill + '·' * (self.width - fill)
        line = f'{self.label}  |{bar}| {100 * frac:5.1f}%  {_fmt(el)}<{_fmt(eta)}'
        if extra:
            line += f'  {extra}'
        if self.tty:
            sys.stdout.write('\r' + line + ('\n' if done else ''))
        else:
            sys.stdout.write(line + '\n')
        sys.stdout.flush()

    def done(self, extra=''):
        self.update(1.0, extra)

def counter(total, prefix=''):

    def label(i, name=''):
        pre = f'{prefix} ' if prefix else ''
        return f'[{pre}{i}/{total}] {name}'.rstrip()
    return label
