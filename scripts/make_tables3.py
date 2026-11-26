# -*- coding: utf-8 -*-
"""The two tables that stay in the main text, rebuilt from the result files.

Table 2 (procedures) carries the nonlinear additive-noise family in its own column, so the
linearity question is answered inside the table rather than in prose only.
Table 3 (real matrices) carries the Holm-adjusted p-value beside the raw one, because fifteen
one-sided tests were run and the corrected column is the one a reader should use.
"""
import os, sys, json
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)

OUT = os.path.join(os.path.dirname(S), 'paper', 'sagej')

SHORT = {
    'adult body mass (g)': 'adult body mass',
    'trophic level': 'trophic level',
    'potential evapotranspiration (mm)': 'potential evapotranspiration',
    'invertebrates (\\% of diet)': 'invertebrate diet',
    'ground (\\%)': 'ground foraging',
    'habitat breadth': 'habitat breadth',
    'range area (km2)': 'range area',
}


def load(name):
    p = os.path.join(os.path.dirname(S), 'results', name + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def w(f, txt):
    # newline='\n' pins the line ending, so "byte for byte" holds on any platform: the shipped
    # files are LF and a Windows text-mode write would otherwise emit CRLF and differ by one byte
    # per line (2026-11-26: 16 bytes in tab_families, 21 in tab_real, content identical).
    with open(os.path.join(OUT, f), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(txt)
    print('wrote', f)


def esc(s):
    return (s.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_')
            .replace('(g)', '').replace('(mm)', '').strip())


def sd2(x):
    """A standard deviation to two decimals, rounded half up.

    '%.2f' decides exact halves from the binary representation of the value, so 0.125 printed as
    0.12 while 0.165 printed as 0.17: one column, two rounding rules. The half case is decided here
    once, and the audit imports this function rather than repeating the rule.
    """
    return str(Decimal(repr(float(x))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def families():
    e23 = load('e23_families10')
    e29 = load('e29_resit')
    # the NOTEARS ranking on the same grid; without it the table answers "do all methods fail the
    # same way" with two curves instead of three
    e23b = load('e23b_notears_auroc')
    cap = r'''Procedures on one grid: $d = 30$, $n = 200$, ten seeds for the four linear procedures and five for the additive-noise family. Both top-$k$ methods admit exactly ten entries, so precision, recall and $F_1$ coincide; only $F_1$ is listed, with the seed standard deviation in parentheses. PC and FCI choose their own size, so their accuracy sits far below the two rules told how many edges to admit; their adjacencies are in the supplementary table. The last three columns give AUROC for the three rules that rank a continuous score without a cutoff, on the seeds of its own column. RESIT is the nonlinear additive-noise model: its ranking is comparable with the others, its top-$k$ accuracy is not, because a pairwise additive-noise score cannot separate a direct edge from a chain. A search allowing latent common causes and path analysis over candidate models need other grids; both are in the supplementary tables.'''
    # the AUROC group stays on the header row: a second tier of header costs one line, and one line
    # here is the difference between a 20-page and a 21-page SAGE build
    L = [r'\caption{%s}' % cap, r'\label{tab:families}', r'\small',
         r'\setlength{\tabcolsep}{5pt}',
         r'\begin{tabular}{@{}lrrrrrrrrr@{}}', r'\toprule',
         r'$\lambda$ & $K$ & corr. & NOTEARS & PC & FCI & RESIT & corr. & NOTEARS & RESIT \\',
         r'\cmidrule(lr){3-7}\cmidrule(lr){8-10}']
    for key in ('lam0.0', 'lam0.5', 'lam1.0', 'lam1.5', 'lam2.0', 'lam3.0'):
        a, b, c = e23.get(key), e29.get(key), e23b.get(key)
        if not a:
            continue
        L.append(r'%.1f & %.3f & %.3f (%s) & %.3f (%s) & %.3f & %.3f & %s & %.3f & %s & %s \\'
                 % (a['lam'], a['K'], a['corr'], sd2(a['corr_sd']), a['notears'],
                    sd2(a['notears_sd']),
                    a['pc'], a['fci'],
                    ('%.3f' % b['f1']) if b else '--',
                    a['auroc'],
                    ('%.3f' % c['notears_auroc']) if c else '--',
                    ('%.3f' % b['auroc']) if b else '--'))
    L += [r'\bottomrule', r'\end{tabular}']
    w('tab_families.tex', '\n'.join(L) + '\n')


def real5():
    e26, e27 = load('e26_real'), load('e27_birds')
    H = load('e30_holm')
    blocks, holes = H.get('blocks', {}), H.get('holes', {})
    order = [('life_history', e26, 'mammals'), ('range_climate', e26, 'mammals'),
             ('bird_diet', e27, 'birds'), ('bird_niche', e27, 'birds'),
             ('ecology', e26, 'mammals')]
    cap = ('Five published comparative matrices. Traits from PanTHERIA (mammals) and EltonTraits '
           '(birds); topologies from the Open Tree of Life synthesis, branch lengths by Grafen. '
           '$r_1$ is the leading ratio of the contrast correlation matrix, the statistic of the '
           'screen. The reference is a sparse random DAG with the stated number of edges at the '
           'same $n$ and $d$, which is what an ordinary causal structure produces; every block also '
           'rejects the null of independent Brownian traits at $p < 1/4000$, which says only that '
           'the traits are related. The raw $p$ is one-sided against the graph reference; the next '
           'column adjusts it by Holm over the fifteen tests (five blocks by three leading ratios). '
           'The final column names the variable a top-$d$ ranking collapses onto, with its degree '
           'and the share of edge-endpoints incident to it. Probabilities carry four decimals '
           'because the reference uses five thousand draws: $0.0002$ is its resolution limit, not '
           'a measured value.')
    # An eleventh l spec, with nothing to fill it, used to push this table 24 mm past the body edge
    L = [r'\caption{%s}' % cap, r'\label{tab:real}', r'\footnotesize',
         r'\setlength{\tabcolsep}{4.5pt}',
         r'% Two open-ended fields (block label, hub name) become p-columns so they wrap inside',
         r'% \textwidth; the numbers keep their own column separation. Each (deg, share) pair is',
         r'% boxed so a line break cannot land between the number and its closing bracket.',
         r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.165\textwidth} rrrrrr c r '
         r'>{\raggedright\arraybackslash}p{0.175\textwidth}@{}}', r'\toprule',
         r'& & & & & \multicolumn{2}{c}{sparse random DAG} & & & \\',
         r'\cmidrule(lr){6-7}',
         r'block & $n$ & $d$ & $K$ & $r_1$ & $k$ & its $r_1$ & $p$ & Holm & hub (deg, share) \\',
         r'\midrule']
    for key, src, clade in order:
        v, b = src.get(key), blocks.get(key)
        if not v or not b:
            continue
        hub = SHORT.get(v['hub']['top_label'].replace('%', '\\%'), v['hub']['top_label'])
        ph = holes.get('%s:r1' % key, {})
        L.append(r'%s (%s) & %d & %d & %.2f & %.2f & %d & %.2f & %.4f & %.4f & %s \mbox{(%d, %.2f)} \\'
                 % (key.replace('_', ' '), clade, v['n'], v['d'], v['K_mean'],
                    v['contrast']['r'][0], b['k'], b['null_mean'][0], ph.get('p', np.nan),
                    ph.get('holm', np.nan), esc(hub), v['hub']['top_deg'], v['hub']['share']))
    # The tetrad rates used to be typed into this string, which is how they survived a
    # regeneration unchanged while the data underneath them moved.  They are read from the
    # same results the table is built from, and the size range with them.
    tt = [src[key]['tetrad'] for key, src, _ in order if src.get(key)]
    rate = ', '.join('$%.2f$' % t['reject'] for t in tt)
    size_lo = min(t['size'] for t in tt)
    size_hi = max(t['size'] for t in tt)
    key_after = max([(k, src[k]['r_after_deflation']) for k, src, _ in order if src.get(k)],
                    key=lambda kv: kv[1])
    L += [r'\bottomrule', r'\end{tabular}',
          r'\par\smallskip\footnotesize Vanishing-tetrad rejection rates, against a measured size '
          r'of $%.2f$--$%.2f$, are %s in the order of the rows. Removing the leading component '
          r'leaves the next ratio at $%.2f$ in %s, so that block carries more than one direction; '
          r'the per-block spectra and the preprocessing sensitivities are in the supplementary '
          r'material.' % (size_lo, size_hi, rate, key_after[1], key_after[0].replace('_', ' '))]
    w('tab_real.tex', '\n'.join(L) + '\n')


if __name__ == '__main__':
    families()
    real5()
