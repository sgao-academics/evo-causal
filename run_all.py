"""Driver for the replication package.

    python run_all.py            run every experiment, then check the archive
    python run_all.py --fast     reduced grid (one seed), for a quick smoke run
    python run_all.py --check    check the shipped result files, no computation
    python run_all.py --figures  redraw the five figures from results/

The --check mode is the one to use when the question is "does the archive agree
with the paper". It touches nothing and finishes in seconds.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, 'scripts')
RESULTS = os.path.join(ROOT, 'results')

STAGES = [
    ('E1-E4  whitening, K, PIC vs raw', 'run_revision.py'),
    ('E5-E9  dose-response, PC, PPA, FCI, n', 'run_revision2.py'),
    ('E1     at the 12-tip tree of Fig. 1', '_e1_n12.py'),
    ('E10    fixed percentile vs top-k', '_e10_threshold.py'),
    ('E11    four families on one grid', '_e11_families.py'),
    ('E11    AUROC for the same grid', '_e11_auroc.py'),
    ('E12    Pearson, Spearman, O(D)GC', '_e12_spearman.py'),
    ('E13    O(D)GC', '_e13_odgc.py'),
]

FIGURES = ['make_fig1.py', 'make_fig2_fig4_fig5.py', 'make_fig3.py']

# Every result file the manuscript reads, with the number of entries expected.
EXPECTED = {
    'e1_whitening.json': 3,
    'e1_whitening_n12.json': 3,
    'e3_blomberg.json': 5,
    'e4_methods.json': 4,
    'e5_k_axis.json': 18,
    'e6_pc.json': 6,
    'e7_ppa.json': 6,
    'e8_fci.json': 6,
    'e9_nsense.json': 10,
    'e10_threshold.json': 5,
    'e11_families.json': 6,
    'e11_auroc.json': 6,
    'e12_spearman.json': 12,
    'e13_odgc.json': 12,
}


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def run_stage(script, fast):
    cmd = [sys.executable, script]
    if fast and script in ('run_revision.py', 'run_revision2.py'):
        cmd += ['--fast']
    print('\n' + '=' * 74)
    print(' '.join(cmd))
    print('=' * 74)
    return subprocess.call(cmd, cwd=SCRIPTS)


def check():
    """Verify the archive: entry counts, then the values the text quotes."""
    bad = []
    print('%-26s %-8s %s' % ('file', 'entries', 'status'))
    for name, n in EXPECTED.items():
        d = load(name)
        if d is None:
            print('%-26s %-8s MISSING' % (name, '-'))
            bad.append(name + ' missing')
            continue
        cells = [k for k in d if isinstance(d[k], dict)]
        ok = len(cells) == n
        print('%-26s %-8s %s' % (name, '%d/%d' % (len(cells), n),
                                 'ok' if ok else 'MISMATCH'))
        if not ok:
            bad.append('%s has %d entries, expected %d' % (name, len(cells), n))

    print('\nvalues quoted in the text')
    e1 = load('e1_whitening_n12.json') or {}
    if e1:
        print('  E1  max|off-diagonal|: BM %.3f, OU(3) %.3f, OU(10) %.3f'
              % (e1['BM']['max_offdiag'], e1['OU_alpha3']['max_offdiag'],
                 e1['OU_alpha10']['max_offdiag']))

    e3 = load('e3_blomberg.json') or {}
    ks = {k: v.get('K') for k, v in e3.items() if isinstance(v, dict)}
    if ks:
        print('  E3  measured K: ' + ', '.join('%s=%.3f' % (k, v)
              for k, v in sorted(ks.items())))

    e4 = load('e4_methods.json') or {}
    hs = [v.get('h_final_mean') for v in e4.values()
          if isinstance(v, dict) and v.get('h_final_mean')]
    if hs:
        print('  E4  acyclicity residual h(W): %.1e to %.1e' % (min(hs), max(hs)))

    e11 = load('e11_families.json') or {}
    fam = [(v['K_measured'], v['f1_corr'], v['f1_notears'], v['f1_pc'], v['f1_fci'])
           for v in e11.values() if isinstance(v, dict) and 'K_measured' in v]
    if fam:
        fam.sort()
        lo, hi = fam[0], fam[-1]
        print('  E11 F1 at K=%.3f: corr %.3f, NOTEARS %.3f, PC %.3f, FCI %.3f'
              % lo)
        print('  E11 F1 at K=%.3f: corr %.3f, NOTEARS %.3f, PC %.3f, FCI %.3f'
              % hi)

    e8 = load('e8_fci.json') or {}
    marks = [v['frac_latent_marks'] for v in e8.values()
             if isinstance(v, dict) and 'frac_latent_marks' in v]
    if marks:
        print('  E8  FCI latent endpoint marks: %.3f to %.3f' % (min(marks), max(marks)))

    e9 = load('e9_nsense.json') or {}
    if e9:
        print('  E9  %d cells over n = 100 to 2000' % len(e9))

    print()
    if bad:
        print('FAILED: %d problem(s)' % len(bad))
        for b in bad:
            print('   ' + b)
        return 1
    print('OK: every result file is present with the expected number of entries')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='one seed, reduced grid')
    ap.add_argument('--check', action='store_true', help='verify the archive only')
    ap.add_argument('--figures', action='store_true', help='redraw the figures')
    args = ap.parse_args()

    if args.check:
        return check()

    if args.figures:
        for s in FIGURES:
            rc = subprocess.call([sys.executable, s], cwd=SCRIPTS)
            if rc:
                print('FAILED: %s' % s)
                return rc
        print('\nfigures written to figures/')
        return 0

    failed = []
    for label, script in STAGES:
        print('\n### %s  (%s)' % (label, script))
        if run_stage(script, args.fast):
            failed.append(script)

    print('\n' + '=' * 74)
    if failed:
        print('FAILED: ' + ', '.join(failed))
        return 1
    print('all stages finished')
    return check()


if __name__ == '__main__':
    sys.exit(main())
