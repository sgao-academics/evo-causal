# -*- coding: utf-8 -*-
"""Audit the revision: every quoted number is recomputed from results/*.json and searched for
in the manuscript. A claim that does not appear in a file fails here rather than in review.

Each check is (label, value_from_json, formatted string that must appear in the text).
"""
import os, io, json, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(BASE), 'results')
TEX = os.path.join(os.path.dirname(BASE), 'paper', 'evo_causal_revision.tex')
CL = os.path.join(os.path.dirname(BASE), 'paper', 'cover_letter_revision.tex')


def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(io.open(p, encoding='utf-8')) if os.path.exists(p) else {}


tex = io.open(TEX, encoding='utf-8').read()
cl = io.open(CL, encoding='utf-8').read()
ok = bad = 0
fails = []


def check(label, s, where='tex', float_tol=0):
    """Does the manuscript contain this string?"""
    global ok, bad
    target = tex if where == 'tex' else cl
    hit = s in target
    if not hit and float_tol:
        # try variations in the number of decimals
        m = re.match(r'^(-?\d+\.\d+)$', s)
        if m:
            v = float(s)
            for d in range(1, 5):
                cand = ('%.' + str(d) + 'f') % v
                if cand in target:
                    hit = True
                    break
    if hit:
        ok += 1
    else:
        bad += 1
        fails.append((label, s, where))
    print(('  ok   ' if hit else '  FAIL ') + '%-52s %s' % (label, s))


e14 = load('e14_dissociation')
e15 = load('e15_screen'); e15b = load('e15b_screen_signal')
e16 = load('e16_deflate'); e16b = load('e16b_hub')
e17 = load('e17_ppa_order'); e18 = load('e18_k_vs_n')
e23 = load('e23_families10'); e19 = load('e19_precision_recall')
e21 = load('e21_lrsem'); e5 = load('e5_k_axis')

print('--- dissociation (E13)')
check('signal arm, top of range', '%.3f' % e14['signal_d30_lv6.0']['K'])
check('signal arm, F1 at top', '%.2f' % e14['signal_d30_lv6.0']['f1_pic'])
check('tree arm, F1 at top', '%.2f' % e14['tree_d30_lv6.0']['f1_pic'])
check('iid arm, K', '%.3f' % e14['iid_d30_lv2.0']['K'])
check('iid arm, F1', '%.2f' % e14['iid_d30_lv2.0']['f1_pic'])
check('iid arm, AUROC', '%.2f' % e14['iid_d30_lv2.0']['auroc'])

print('--- screen (E14)')
check('null mean d=30', '%.2f' % e15['null_d30']['mean'])
check('null q95 d=30', '%.2f' % e15['null_d30']['q95'])
check('null mean d=50', '%.2f' % e15['null_d50']['mean'])
check('null q95 d=50', '%.2f' % e15['null_d50']['q95'])
check('iid lambda=1 r', '%.1f' % e15['iid_d30_lv1.0']['r'])
check('iid lambda=4 r', '%.1f' % e15['iid_d30_lv4.0']['r'])
check('tree lambda=2 r', '%.2f' % e15['tree_d30_lv2.0']['r'])
check('tree lambda=6 r', '%.1f' % e15['tree_d30_lv6.0']['r'])
check('signal lambda=6 r', '%.2f' % e15b['signal_d30_lv6.0']['r'])

print('--- deflation (E16)')
for lam, key in (('2.0', 'd50_lam2.0'), ('3.0', 'd50_lam3.0'), ('4.0', 'd50_lam4.0'), ('6.0', 'd50_lam6.0')):
    check('plain d50 lam=%s' % lam, '%.2f' % e16[key]['plain'])
    check('deflated d50 lam=%s' % lam, '%.2f' % e16[key]['deflate'])
check('AUROC 0.922->0.996', '%.3f' % e16['d50_lam2.0']['auroc_plain'])
check('AUROC deflated 0.996', '%.3f' % e16['d50_lam2.0']['auroc_deflate'])
check('cost d50 lam=0', '%.2f' % e16['d50_lam0.0']['deflate'])
check('cost d30 lam=0', '%.2f' % e16['d30_lam0.0']['deflate'])
check('hub at lam=0', '%.1f' % (e16b['d30_lam0.0']['hub'] * 20 / 30))
check('hub at lam=6', '%.1f' % (e16b['d30_lam6.0']['hub'] * 20 / 30))

print('--- families (E20)')
for lam in ('0.0', '1.0', '3.0'):
    check('corr lam=%s' % lam, '%.2f' % e23['lam' + lam]['corr'])
    check('notears lam=%s' % lam, '%.2f' % e23['lam' + lam]['notears'])
check('pc low', '%.3f' % e23['lam0.0']['pc'])
check('pc high', '%.3f' % e23['lam3.0']['pc'])
check('fci marks low', '%.0f' % (100 * e23['lam0.0']['fci_marks']))
check('fci marks high', '%.0f' % (100 * e23['lam3.0']['fci_marks']))

print('--- ppa (E18)')
for lam in ('0.0', '1.0', '2.0'):
    v = e17['lam%s_full' % lam]
    check('ppa full lam=%s' % lam, '%.3f' % v['ppa_f1'])
    check('pic lam=%s' % lam, '%.3f' % v['pic_f1'])

print('--- sample size (E19)')
check('K lam2 n100', '%.3f' % e18['lam2.0_n100']['K'])
check('K lam2 n1000', '%.3f' % e18['lam2.0_n1000']['K'])

print('--- rank-one residual trial (E21)')
if e21:
    check('lrsem lam0', '%.2f' % e21['lam0.0']['lrsem'])
    check('lrsem plain', '%.2f' % e21['lam0.0']['plain'])

print('--- internal consistency')
n = len(re.findall(r'\b(\d{1,3}(,\d{3})*|\d+)\b', tex))
print('  numbers in manuscript:', n)
for pat, label in ((r'with the submission', 'no "with the submission"'),
                   (r'moves by \$0\.067\$', 'no stale 0.067'),
                   (r'\b1\{,\}760\b', 'no stale 1,760'),
                   (r'O\(D\)GC selected Spearman for', 'ODGC claim present')):
    present = re.search(pat, tex) is not None
    want = label.startswith('no ')
    good = (not present) if want else present
    print(('  ok   ' if good else '  FAIL ') + label)
    ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
    if not good:
        fails.append((label, '', 'tex'))

print()
print('OVERALL: %d passed, %d failed' % (ok, bad))
if fails:
    print('FAILURES:')
    for f in fails:
        print('  ', f)
