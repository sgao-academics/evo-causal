"""Driver for the replication package.

    python run_all.py            run every experiment, then check the archive
    python run_all.py --fast     reduced grid (one seed), for a quick smoke run
    python run_all.py --check    check the shipped result files, no computation
    python run_all.py --figures  redraw every figure from results/

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
    ('E14-E20 second round: dissociation, screen, deflation', 'exp_new.py'),
    ('E15    screen on the signal arm', 'e15b_signal.py'),
    ('E17    concentration of the admitted set', 'e16b_hub.py'),
    ('E20    four procedures, ten seeds', 'e23_families10.py'),
    ('E21    rank-one residual search, archived trial', 'e21_lrsem.py'),
    ('E25    the ADMG family, maximum likelihood with bidirected edges', 'exp_admg.py'),
    ('E26    real traits on the real synthesis tree', 'exp_real.py'),
    ('E27    the same protocol on birds', 'exp_birds.py'),
    ('E26b   the sparse-graph reference for every block', 'exp_graphnull.py'),
    ('E28    calibration audit and the multi-direction repair', 'exp_calib.py'),
    ('E29    nonlinear additive-noise family', 'exp_resit.py'),
    ('E30    Holm correction and preprocessing sensitivity', 'exp_holm.py'),
    ('E31    the mammal blocks on a dated topology', 'exp_upham.py'),
    ('E32    the k band, and thirty-seed intervals', 'exp_kband.py'),
]

FIGURES = ['make_fig1.py', 'make_fig2_fig4_fig5.py', 'make_fig3.py',
           'make_fig6_fig7.py', 'make_fig_real.py', 'make_fig_resit.py']

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
    'e14_dissociation.json': 32,
    'e15_screen.json': 22,
    'e15b_screen_signal.json': 12,
    'e15c_r_vs_n.json': 10,
    'e16_deflate.json': 18,
    'e16b_hub.json': 5,
    'e17_ppa_order.json': 7,
    'e18_k_vs_n.json': 10,
    'e19_precision_recall.json': 30,
    'e21_lrsem.json': 2,
    'e25_admg.json': 12,
    'e26_real.json': 3,
    'e22_seed_spread.json': 5,
    'e23_families10.json': 6,
    'e27_birds.json': 2,
    'e28_calib.json': 31,
    'e29_resit.json': 6,
    'e30_holm.json': 3,
    'e31_upham.json': 3,
    'e32_kband.json': 12,
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

    e14 = load('e14_dissociation.json') or {}
    if e14:
        a = e14.get('signal_d30_lv6.0', {}); b = e14.get('iid_d30_lv2.0', {})
        d = e14.get('tree_d30_lv6.0', {})
        print('  E14 signal-only  K=%.3f F1=%.2f' % (a.get('K', 0), a.get('f1_pic', 0)))
        print('  E14 confounder, no tree  K=%.3f F1=%.2f' % (b.get('K', 0), b.get('f1_pic', 0)))
        print('  E14 confounder on the tree  K=%.3f F1=%.2f' % (d.get('K', 0), d.get('f1_pic', 0)))

    e15 = load('e15_screen.json') or {}
    if e15:
        print('  E15 screen null: mean %.2f (d=30), %.2f (d=50); 95%% at %.2f and %.2f'
              % (e15['null_d30']['mean'], e15['null_d50']['mean'],
                 e15['null_d30']['q95'], e15['null_d50']['q95']))

    e16 = load('e16_deflate.json') or {}
    if e16:
        v = e16.get('d50_lam4.0', {})
        print('  E16 d=50 lambda=4: plain %.2f, leading component removed %.2f, AUROC %.3f -> %.3f'
              % (v.get('plain', 0), v.get('deflate', 0),
                 v.get('auroc_plain', 0), v.get('auroc_deflate', 0)))

    e23 = load('e23_families10.json') or {}
    if e23:
        lo, hi = e23.get('lam0.0', {}), e23.get('lam3.0', {})
        print('  E20 ten seeds, K=%.3f: corr %.2f, NOTEARS %.2f' %
              (lo.get('K', 0), lo.get('corr', 0), lo.get('notears', 0)))
        print('  E20 ten seeds, K=%.3f: corr %.2f, NOTEARS %.2f' %
              (hi.get('K', 0), hi.get('corr', 0), hi.get('notears', 0)))

    e17 = load('e17_ppa_order.json') or {}
    if e17:
        for lam in ('lam0.0_full', 'lam1.0_full', 'lam2.0_full'):
            v = e17.get(lam, {})
            if v:
                print('  E18 %s: order-free PPA %.3f (exact %.2f), contrasts %.3f'
                      % (lam, v['ppa_f1'], v['ppa_exact'], v['pic_f1']))

    e31 = load('e31_upham.json') or {}
    if e31:
        for blk in ('life_history', 'range_climate'):
            v = e31.get(blk, {})
            if v:
                print('  E31 %-14s A %.2f  B %.2f  C %.2f  (ref %.2f)' %
                      (blk, v['open_tree_grafen'], v['upham_grafen'],
                       v['upham_dated'], v['null']['mean']))

    e32 = load('e32_kband.json') or {}
    if e32:
        for key in ('d30_lam0.0', 'd50_lam2.0'):
            v = e32.get(key, {})
            if v:
                print('  E32 %-12s ' % key + ' '.join(
                    'k=%s %.2f [%.2f,%.2f]' % (k, c['mean'], c['lo'], c['hi'])
                    for k, c in sorted(v.items(), key=lambda kv: int(kv[0]))))

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
