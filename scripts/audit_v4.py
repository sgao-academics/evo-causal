# -*- coding: utf-8 -*-
"""Audit the numbers added in this round against results/*.json.

Every claim is written as (string that must appear in the text, value recomputed from the
result file, tolerance). A failure means the prose and the data disagree, which is the only
defect this script exists to catch. Table rows are rebuilt from the JSON in the same format
the generator writes them, so a table cell cannot drift away from its source either.
"""
import io, json, os, re, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(BASE), 'results')
PAPER = os.path.join(os.path.dirname(BASE), 'paper')
sys.path.insert(0, os.path.join(os.path.dirname(BASE), 'scripts'))

M = json.load(io.open(os.path.join(RES, 'e25_admg.json'), encoding='utf-8'))
R = json.load(io.open(os.path.join(RES, 'e26_real.json'), encoding='utf-8'))
B = json.load(io.open(os.path.join(RES, 'e27_birds.json'), encoding='utf-8'))
C = json.load(io.open(os.path.join(RES, 'e28_calib.json'), encoding='utf-8'))

tex = io.open(os.path.join(PAPER, 'evo_causal_revision.tex'), encoding='utf-8').read()
cl = io.open(os.path.join(PAPER, 'cover_letter_revision.tex'), encoding='utf-8').read()
tadmg = io.open(os.path.join(PAPER, 'tab_admg.tex'), encoding='utf-8').read()
treal = io.open(os.path.join(PAPER, 'tab_real.tex'), encoding='utf-8').read()
tcal = io.open(os.path.join(PAPER, 'tab_calib.tex'), encoding='utf-8').read()
tmulti = io.open(os.path.join(PAPER, 'tab_multifac.tex'), encoding='utf-8').read()

ok = fail = 0
fails = []


def flat(s):
    return re.sub(r'\s+', ' ', s).strip()


T, CL = flat(tex), flat(cl)


def chk_absent(where, needle, detail=''):
    """A sentence that must no longer be there."""
    global ok, fail
    if flat(needle) not in where:
        ok += 1
    else:
        fail += 1
        fails.append('%-58s [still present] %s' % (flat(needle)[:58], detail))


def chk(where, needle, cond, detail=''):
    global ok, fail
    present = flat(needle) in where
    if present and cond:
        ok += 1
    else:
        fail += 1
        fails.append('%-58s [present=%s value=%s] %s' % (flat(needle)[:58], present, cond, detail))


# ---------------------------------------------------------------- Table 5 (ADMG)
print('--- table 5: ADMG rows equal e25_admg.json ---')
for tag, d in (('d10', 10), ('d15', 15)):
    for lam in ('0.0', '0.5', '1.0', '1.5', '2.0', '3.0'):
        v = M.get('%s_lam%s' % (tag, lam))
        assert v, tag
        row = '%d & %.1f & %.3f & %.3f & %.3f & %.3f & %.1f \\\\' % (
            d, float(lam), v['K'], v['f1_corr'], v['f1_notears'], v['f1_admg'], v['admg_bidir'])
        chk(tadmg, row, True, tag + lam)

# ---------------------------------------------------------------- Table (real, 5 blocks)
print('--- table 6: five real blocks equal e26/e27 ---')
SHORT = {'adult body mass (g)': 'adult body mass', 'trophic level': 'trophic level',
         'potential evapotranspiration (mm)': 'potential evapotranspiration',
         'invertebrates (% of diet)': 'invertebrate diet', 'ground (%)': 'ground foraging'}
order = [('ecology', R, 'mammals'), ('life_history', R, 'mammals'),
         ('range_climate', R, 'mammals'), ('bird_diet', B, 'birds'),
         ('bird_niche', B, 'birds')]
for key, src, clade in order:
    v = src[key]
    gn = v['graph_null']['sparse']
    hub = SHORT[v['hub']['top_label']]
    row = '%s & %s & %d & %d & %.2f & %.2f & %d & %.2f & %.3f & %s (%d, %.2f) \\\\' % (
        key.replace('_', ' '), clade, v['n'], v['d'], v['K_mean'], v['contrast']['r'][0],
        gn['k'], gn['mean'], gn['p'], hub, v['hub']['top_deg'], v['hub']['share'])
    chk(treal, row, True, key)

# ---------------------------------------------------------------- Table (calibration)
print('--- table 7: calibration cells equal e28_calib.json ---')
for d in (6, 12):
    for n in (40, 80, 150, 200, 400):
        for lam in ('0.0', '2.0'):
            r = C['d%d_n%d_lam%s' % (d, n, lam)]
            row = '%d & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % (
                n, r['iid'], r['fit_half'], r['fit_one'], r['fit_two'], r['random'])
            chk(tcal, row, True, 'd%d n%d lam%s' % (d, n, lam))

# ---------------------------------------------------------------- Table (multi-factor)
print('--- table 8: multi-factor rows equal e28_calib.twofactor ---')
for nf in (1, 2):
    for lam in ('1.5', '3.0'):
        r = C['twofactor']['fac%d_lam%s' % (nf, lam)]
        row = '%d & %s & %.2f & %.2f & %.2f \\\\' % (nf, lam, r['remove0'], r['remove1'],
                                                     r['remove2'])
        chk(tmulti, row, True, 'fac%d %s' % (nf, lam))

# ---------------------------------------------------------------- prose: real data
print('--- prose: real-block K, ratios, references, hubs ---')
ec, lh, rc, bd, bn = R['ecology'], R['life_history'], R['range_climate'], B['bird_diet'], B['bird_niche']
chk(T, 'It averages $1.04$, $1.82$, $0.47$, $1.21$ and $1.26$',
    all(abs(v['K_mean'] - x) < 0.006 for v, x in
        ((ec, 1.04), (lh, 1.82), (rc, 0.47), (bd, 1.21), (bn, 1.26))), 'K means')
tot = sum(v['K_over_0p30'] for v in (ec, lh, rc, bd, bn))
chk(T, '$40$ of the $43$ individual traits', tot == 40, 'traits over 0.30 = %d' % tot)
chk(T, 'exceeds the safe band for $40$ of $43$ traits', tot == 40, 'abstract 40 of 43')
chk(T, '$1.47$, $3.99$, $3.57$, $1.83$ and $1.76$',
    all(abs(v['contrast']['r'][0] - x) < 0.006 for v, x in
        ((ec, 1.47), (lh, 3.99), (rc, 3.57), (bd, 1.83), (bn, 1.76))), 'r1 values')
chk(T, 'against $2.31 \\pm 0.41$, $p = 0.002$',
    abs(lh['graph_null']['sparse']['mean'] - 2.31) < 0.006
    and abs(lh['graph_null']['sparse']['p'] - 0.002) < 0.0006, 'life history sparse null')
chk(T, 'against $2.35 \\pm 0.43$, $p = 0.022$',
    abs(rc['graph_null']['sparse']['mean'] - 2.35) < 0.006
    and abs(rc['graph_null']['sparse']['p'] - 0.022) < 0.0006, 'range sparse null')
chk(T, 'against $2.21 \\pm 0.39$',
    abs(ec['graph_null']['sparse']['mean'] - 2.21) < 0.006, 'ecology sparse null')
chk(T, 'against $2.24 \\pm 0.41$',
    abs(bd['graph_null']['sparse']['mean'] - 2.24) < 0.006, 'bird diet sparse null')
chk(T, 'against $2.29 \\pm 0.40$',
    abs(bn['graph_null']['sparse']['mean'] - 2.29) < 0.006, 'bird niche sparse null')
chk(T, 'which takes a third of the admitted edges',
    abs(lh['hub']['share'] - 1 / 3) < 0.01, 'life history hub share %.3f' % lh['hub']['share'])
chk(T, 'invertebrate diet ($35\\%$)', abs(bd['hub']['share'] - 0.35) < 0.006, 'bird diet hub')
chk(T, 'ground foraging ($28\\%$)', abs(bn['hub']['share'] - 0.28) < 0.006, 'bird niche hub')
chk(T, 'but with only $12\\%$ of edges', abs(rc['hub']['share'] - 0.12) < 0.006, 'range hub')
chk(T, '($4.48$ against $3.57$)', abs(rc['contrast']['r'][1] - 4.48) < 0.006
    and abs(rc['contrast']['r'][0] - 3.57) < 0.006, 'range r2 > r1')
chk(T, 'leaves the next one at $4.92$', abs(rc['r_after_deflation'] - 4.92) < 0.006, 'no collapse')
chk(T, '$0.80$, $0.93$, $0.97$, $0.60$ and $0.61$',
    all(abs(v['tetrad']['reject'] - x) < 0.006 for v, x in
        ((ec, 0.80), (lh, 0.93), (rc, 0.97), (bd, 0.60), (bn, 0.61))), 'tetrad rates')
chk(T, 'measured size of $0.05$--$0.07$',
    all(0.045 <= v['tetrad']['size'] <= 0.075 for v in (ec, lh, rc, bd, bn)),
    'sizes %s' % [v['tetrad']['size'] for v in (ec, lh, rc, bd, bn)])

# ---------------------------------------------------------------- prose: branch lengths
print('--- prose: branch-length and lambda sensitivity ---')
moves = []
for v in (ec, lh, rc, bd, bn):
    vs = v['variants']
    base = vs['grafen_1.0']['r']
    moves.append(max(abs(vs[k]['r'] - base) for k in ('grafen_0.5', 'grafen_2.0', 'equal')))
chk(T, 'moves by at most $0.33$', max(moves) <= 0.335, 'max move %.3f' % max(moves))
chk(T, 'life history, $3.66$ to $3.99$',
    abs(lh['variants']['grafen_0.5']['r'] - 3.66) < 0.006, 'life history grafen 0.5')
chk(T, 'every other block by at most $0.15$',
    max(m for m, v in zip(moves, (ec, lh, rc, bd, bn)) if v is not lh) <= 0.15,
    'others %.3f' % max(moves[0], moves[2], moves[3], moves[4]))

# ---------------------------------------------------------------- prose: calibration + multi
print('--- prose: calibration and multi-directional repair ---')
chk(T, 'inflates the false-positive rate to $0.20$--$0.84$',
    min(C[k]['fit_half'] for k in C if k.startswith('d') and k.endswith('lam0.0')) == 0.20
    and max(C[k]['fit_half'] for k in C if k.startswith('d') and k.endswith('lam0.0')) == 0.84,
    'fit_half size range')
chk(T, 'size is $0.04$--$0.16$ and power $0.88$--$1.00$, close to what the true graph delivers',
    min(C[k]['fit_one'] for k in C if k.startswith('d') and k.endswith('lam0.0')) == 0.04
    and max(C[k]['fit_one'] for k in C if k.startswith('d') and k.endswith('lam0.0')) == 0.16
    and min(C[k]['fit_one'] for k in C if k.startswith('d') and k.endswith('lam2.0')) == 0.88
    and max(C[k]['fit_one'] for k in C if k.startswith('d') and k.endswith('lam2.0')) == 1.00,
    'fit_one size/power')
chk(T, 'power drops to $0.52$ at $n = 40$',
    abs(C['d6_n40_lam2.0']['fit_two'] - 0.52) < 0.006, 'fit_two power n40')
chk(T, 'firing on $68\\%$--$100\\%$ of unconfounded datasets',
    abs(min(C[k]['iid'] for k in C if k.startswith('d') and k.endswith('lam0.0')) - 0.68) < 0.006
    and max(C[k]['iid'] for k in C if k.startswith('d') and k.endswith('lam0.0')) == 1.00,
    'iid size range')
chk(T, 'leaves $F_1$ at $0.02$--$0.06$',
    min(C['twofactor'][k]['remove1'] for k in ('fac2_lam1.5', 'fac2_lam3.0')) == 0.02
    and max(C['twofactor'][k]['remove1'] for k in ('fac2_lam1.5', 'fac2_lam3.0')) == 0.06,
    'two-factor remove1')
chk(T, 'restores it to $0.58$--$0.72$',
    min(C['twofactor'][k]['remove2'] for k in ('fac2_lam1.5', 'fac2_lam3.0')) == 0.58
    and max(C['twofactor'][k]['remove2'] for k in ('fac2_lam1.5', 'fac2_lam3.0')) == 0.72,
    'two-factor remove2')
chk(T, 'which is the chance value $k/M$', True, 'prose')
chk(CL, 'size $0.04$--$0.16$ and power $0.88$--$1.00$', True, 'letter repeats the figures')
chk(CL, '$F_1$ stays at the chance value, $0.02$--$0.06$', True, 'letter multi-factor')

# ---------------------------------------------------------------- prose: abstract
print('--- abstract ---')
chk(T, 'Blomberg\'s $K$ exceeds the safe band for $40$ of $43$ traits ($0.47$ to $1.82$)',
    tot == 40, 'abstract K band')
chk(T, 'removing a single component leaves $F_1$ at the chance value ($0.02$--$0.06$)',
    True, 'abstract multi-factor')

# ---------------------------------------------------------------- permutation invariance
print('--- computed: permutation invariance of the scores ---')
from exp_new import gen, prf
from methods_unified import topk_adj, symmetrise
from run_revision import corr_scores

worst = 0.0
for sd in range(6):
    for lam in (0.0, 2.0):
        X, W, tree, Cv = gen(200, 10, 4, lam, sd)
        k = int((np.abs(W) > 0).sum())
        a = prf(symmetrise(topk_adj(np.abs(corr_scores(X)), k)), W)[2]
        perm = np.random.default_rng(99 + sd).permutation(10)
        b = prf(symmetrise(topk_adj(np.abs(corr_scores(X[:, perm])), k)), W[np.ix_(perm, perm)])[2]
        worst = max(worst, abs(a - b))
chk(T, 'the largest change was $0$', worst == 0.0, 'max |dF1| = %.3g' % worst)
chk(T, 'We checked this directly on twelve datasets', True, 'count')

# ---------------------------------------------------------------- availability
print('--- availability ---')
chk(T, 'https://github.com/sgao-academics/evo-causal', True, 'repo in paper')
chk(CL, 'https://github.com/sgao-academics/evo-causal', True, 'repo in letter')
chk_absent(T, 'available from the corresponding author on reasonable request',
           'stale availability sentence must be gone')

print('\n%d checks passed, %d failed' % (ok, fail))
for x in fails:
    print('  FAIL', x)
print('OVERALL', 'PASS' if fail == 0 else 'FAIL')
