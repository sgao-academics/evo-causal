# -*- coding: utf-8 -*-
"""The two tables that stay in the main text, rebuilt from the result files.

Table 2 (procedures) carries the nonlinear additive-noise family in its own column, so the
linearity question is answered inside the table rather than in prose only.
Table 3 (real matrices) carries the Holm-adjusted p-value beside the raw one, because fifteen
one-sided tests were run and the corrected column is the one a reader should use.
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)

OUT = os.path.join(os.path.dirname(S), 'paper')

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
    open(os.path.join(OUT, f), 'w', encoding='utf-8').write(txt)
    print('wrote', f)


def esc(s):
    return (s.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_')
            .replace('(g)', '').replace('(mm)', '').strip())


def families():
    e23 = load('e23_families10')
    e29 = load('e29_resit')
    cap = ('Procedures on one grid: $d = 30$, $n = 200$, ten seeds for the four linear procedures '
           'and five for the additive-noise family. Both top-$k$ methods admit exactly ten entries, '
           'so precision, recall and $F_1$ coincide for them and only $F_1$ is listed, with the '
           'standard deviation across seeds in parentheses. PC and FCI choose their own size, which '
           'is why their accuracy sits far below the two rules that are told how many edges to '
           'admit; the adjacencies they return are reported in the supplementary table. RESIT is '
           'the nonlinear additive-noise model: its threshold-free ranking is comparable with the '
           'others, while its top-$k$ accuracy is not, because a pairwise additive-noise score '
           'cannot separate a direct edge from a chain.')
    L = [r'\caption{%s}' % cap, r'\label{tab:families}', r'\small',
         r'\setlength{\tabcolsep}{6pt}',
         r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
         r'$\lambda$ & $K$ & corr. & NOTEARS & PC & FCI & RESIT & RESIT AUROC \\',
         r'\midrule']
    for key in ('lam0.0', 'lam0.5', 'lam1.0', 'lam1.5', 'lam2.0', 'lam3.0'):
        a, b = e23.get(key), e29.get(key)
        if not a:
            continue
        L.append(r'%.1f & %.3f & %.3f (%.2f) & %.3f (%.2f) & %.3f & %.3f & %s & %s \\'
                 % (a['lam'], a['K'], a['corr'], a['corr_sd'], a['notears'], a['notears_sd'],
                    a['pc'], a['fci'],
                    ('%.3f' % b['f1']) if b else '--',
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
           'and the share of admitted edges incident to it. Probabilities carry four decimals '
           'because the reference uses five thousand draws: $0.0002$ is its resolution limit, not '
           'a measured value.')
    L = [r'\caption{%s}' % cap, r'\label{tab:real}', r'\footnotesize',
         r'\setlength{\tabcolsep}{4.5pt}',
         r'\begin{tabular}{@{}lrrrrrrcrrl@{}}', r'\toprule',
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
        L.append(r'%s (%s) & %d & %d & %.2f & %.2f & %d & %.2f & %.4f & %.4f & %s (%d, %.2f) \\'
                 % (key.replace('_', ' '), clade, v['n'], v['d'], v['K_mean'],
                    v['contrast']['r'][0], b['k'], b['null_mean'][0], ph.get('p', np.nan),
                    ph.get('holm', np.nan), esc(hub), v['hub']['top_deg'], v['hub']['share']))
    L += [r'\bottomrule', r'\end{tabular}',
          r'\par\smallskip\footnotesize Vanishing-tetrad rejection rates, against a measured size '
          r'of $0.05$--$0.07$, are $0.93$, $0.97$, $0.60$, $0.61$ and $0.60$ in the order of the '
          r'rows. Removing the leading component leaves the next ratio at $4.92$ in range and '
          r'climate, so that block carries more than one direction; the per-block spectra and the '
          r'preprocessing sensitivities are in the supplementary material.']
    w('tab_real.tex', '\n'.join(L) + '\n')


if __name__ == '__main__':
    families()
    real5()
