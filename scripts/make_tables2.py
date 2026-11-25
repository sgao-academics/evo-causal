# -*- coding: utf-8 -*-
"""Tables for the real-data anchor (v2) and for the calibration audit."""
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


def w(file, txt):
    with open(os.path.join(OUT, file), 'w', encoding='utf-8') as f:
        f.write(txt)
    print('wrote', file)


def esc(s):
    return (s.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_')
            .replace('(g)', '').replace('(mm)', '').strip())


def real5():
    e26 = load('e26_real')
    e27 = load('e27_birds')
    order = [('ecology', e26, 'mammals'), ('life_history', e26, 'mammals'),
             ('range_climate', e26, 'mammals'),
             ('bird_diet', e27, 'birds'), ('bird_niche', e27, 'birds')]
    L = [r'\begin{table}[H]', r'\centering',
         r'\caption{Five real comparative matrices, measured. Traits from PanTHERIA '
         r'(mammals) and EltonTraits (birds); topologies from the Open Tree of Life synthesis '
         r'(Hinchliff et al. 2015), branch lengths by Grafen. $r_1$ is the leading eigenvalue '
         r'of the \emph{contrast} correlation matrix over the mean of the rest, which is the '
         r'statistic of \eqref{eq:screen}. Two references are shown: the null of independent '
         r'Brownian traits, which is what shared history alone produces, and a sparse random '
         r'DAG with the stated number of edges at the same $n$ and $d$, which is what an '
         r'ordinary causal structure produces. Every block sits far above the safe band on '
         r'Blomberg\'s $K$ and above the independent-traits null, but only life history and '
         r'range sit above the sparse-graph reference. The last two columns name the variable '
         r'that a top-$d$ ranking collapses onto and the share of admitted edges incident '
         r'to it.}', r'\label{tab:real}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{@{}llrrrrcrrc@{}}', r'\toprule',
         r'block & clade & $n$ & $d$ & $K$ & $r_1$ & \multicolumn{2}{c}{sparse DAG} & '
         r'$p$ & hub (deg, share) \\',
         r'\cmidrule(lr){7-8}',
         r' & & & & & & edges & its $r_1$ & & \\', r'\midrule']
    for key, src, clade in order:
        v = src.get(key)
        if not v:
            continue
        gn = v['graph_null']['sparse']
        hub = SHORT.get(v['hub']['top_label'].replace('%', '\\%'), v['hub']['top_label'])
        L.append(r'%s & %s & %d & %d & %.2f & %.2f & %d & %.2f & %.3f & %s (%d, %.2f) \\'
                 % (key.replace('_', ' '), clade, v['n'], v['d'], v['K_mean'],
                    v['contrast']['r'][0], gn['k'], gn['mean'], gn['p'],
                    esc(hub), v['hub']['top_deg'], v['hub']['share']))
    L += [r'\bottomrule', r'\end{tabular}',
          r'\par\smallskip\footnotesize The independent-traits null gives $p<1/4000$ for '
          r'every block (4000 draws, none above the observed value), so that column would '
          r'say the same thing five times. The vanishing-tetrad rejection rates, against a '
          r'measured size of $0.05$--$0.07$, are $0.60$, $0.93$, $0.97$, $0.60$ and $0.61$ '
          r'in the order of the rows.',
          r'\end{table}']
    w('tab_real.tex', '\n'.join(L) + '\n')


def calib():
    e = load('e28_calib')
    if not e:
        print('no e28')
        return
    NS = [(40, '40'), (80, '80'), (150, '150'), (200, '200'), (400, '400')]
    cols = [('iid', 'iid'), ('fit_half', r'$\tfrac12k$'), ('fit_one', '$k$'),
            ('fit_two', '$2k$'), ('random', 'random')]
    L = [r'\begin{table}[H]', r'\centering',
         r'\caption{What the screen costs when the calibration graph is not the truth. '
         r'Each cell is the share of runs in which the statistic exceeds the 95th percentile '
         r'of its own calibration null: the false-positive rate in the upper panel (no '
         r'confounder) and the power in the lower panel ($\lambda = 2$). The calibration '
         r'graph is either the null matrix the text warns against, the fitted top-$k$ '
         r'skeleton at half, one or twice the true edge count with parameters estimated from '
         r'the data, or a sparse random DAG with parameters drawn independently of the data. '
         r'$k = d/3$, the density the simulated grids use. Calibrating at a graph sparser '
         r'than the truth inflates the false-positive rate; calibrating at twice the truth '
         r'costs power; calibrating at the fitted skeleton, which is what an analyst actually '
         r'has, behaves like calibrating at the truth.}', r'\label{tab:calib}',
         r'\small', r'\setlength{\tabcolsep}{6pt}',
         r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
         r'& \multicolumn{5}{c}{calibration graph} \\', r'\cmidrule(lr){2-6}',
         r'$n$ & iid & $\tfrac12k$ & $k$ & $2k$ & random \\', r'\midrule']
    for d in (6, 12):
        L.append(r'\multicolumn{6}{@{}l}{\emph{$d = %d$, size (no confounder)}} \\' % d)
        for n, lab in NS:
            r = e.get('d%d_n%d_lam0.0' % (d, n))
            if not r:
                continue
            L.append(r'%s & %s \\' % (lab, ' & '.join('%.2f' % r[c] for c, _ in cols)))
        L += [r'\midrule',
              r'\multicolumn{6}{@{}l}{\emph{$d = %d$, power ($\lambda = 2$)}} \\' % d]
        for n, lab in NS:
            r = e.get('d%d_n%d_lam2.0' % (d, n))
            if not r:
                continue
            L.append(r'%s & %s \\' % (lab, ' & '.join('%.2f' % r[c] for c, _ in cols)))
        if d == 6:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    w('tab_calib.tex', '\n'.join(L) + '\n')


def multifac():
    e = load('e28_calib').get('twofactor')
    if not e:
        print('no twofactor')
        return
    L = [r'\begin{table}[H]', r'\centering',
         r'\caption{Removing one component is enough only when the contamination is one '
         r'direction. $F_1$ of the top-$k$ ranking on the contrast matrix, $d = 30$, '
         r'$k = 10$, $n = 200$, five seeds, as a function of how many leading components '
         r'are removed and how many latent directions the data carry. With two directions, '
         r'removing one --- which is the whole of the current protocol --- leaves the ranking '
         r'at the chance value.}', r'\label{tab:multifac}', r'\small',
         r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
         r'latent directions & $\lambda$ & remove 0 & remove 1 & remove 2 \\', r'\midrule']
    for nf in (1, 2):
        for lam in ('1.5', '3.0'):
            r = e.get('fac%d_lam%s' % (nf, lam))
            if not r:
                continue
            L.append(r'%d & %s & %.2f & %.2f & %.2f \\' % (nf, lam, r['remove0'],
                                                           r['remove1'], r['remove2']))
        L.append(r'\midrule' if nf == 1 else r'')
    if L[-1] == '':
        L.pop()
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    w('tab_multifac.tex', '\n'.join(L) + '\n')


if __name__ == '__main__':
    real5()
    calib()
    multifac()
