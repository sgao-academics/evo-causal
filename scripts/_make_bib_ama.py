# -*- coding: utf-8 -*-
"""Emit AMA-style numeric bibliographies, one per document, ordered by first appearance.

Sage's Evolution Bioinformatics asks for AMA: superscript numerals, numbered in the order the
citations appear. The metadata are the same CrossRef records the earlier author-year list was
built from, so the volumes, pages and DOIs are not retyped here. Journal titles are printed in
full rather than abbreviated, because the abbreviations are the one field CrossRef does not
supply and a hand-built map would introduce errors into the only part of a reference a reader
checks.
"""
import os, io, json, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(BASE), 'paper', 'sagej')
sys.path.insert(0, BASE)

_BIB = [os.path.join(BASE, '_bib_clean.json'),
        os.path.join(BASE, '..', 'revision', '_bib_clean.json')]
_BIB = [p for p in _BIB if os.path.exists(p)]
if not _BIB:
    raise SystemExit('_bib_clean.json not found next to the script or under revision/')
refs = json.load(io.open(_BIB[0], encoding='utf-8'))

# entries resolved from CrossRef while preparing the split submission (verified 2026-11-25)
NEW = [
    dict(key='szekely2013', year=2013,
         authors=['Sz\u00e9kely, G\u00e1bor J.', 'Rizzo, Maria L.'],
         title='The distance correlation t-test of independence in high dimension',
         venue='Journal of Multivariate Analysis', volume='117', page='193-213',
         doi='10.1016/j.jmva.2013.02.012', issue=None),
    dict(key='buhlmann2014', year=2014,
         authors=['B\u00fchlmann, Peter', 'Peters, Jonas', 'Ernest, Jan'],
         title='CAM: causal additive models, high-dimensional order search and penalized regression',
         venue='The Annals of Statistics', volume='42', issue='6', page='2526-2556',
         doi='10.1214/14-aos1260'),
    # added in the second-review round: the two references the novelty claim rests on.
    # Both records were pulled from CrossRef on 2026-11-26, not typed from memory.
    dict(key='freckleton2000', year=2000,
         authors=['Freckleton, Robert P.'],
         title='Phylogenetic tests of ecological and evolutionary hypotheses: checking for '
               'phylogenetic independence',
         venue='Functional Ecology', volume='14', page='129-134',
         doi='10.1046/j.1365-2435.2000.00400.x', issue=None),
    dict(key='schraiber2024', year=2024,
         authors=['Schraiber, Joshua G.', 'Edge, Michael D.', 'Pennell, Matt'],
         title='Unifying approaches from statistical genetics and phylogenetics for mapping '
               'phenotypes in structured populations',
         venue='PLOS Biology', volume='22', page='e3002847',
         doi='10.1371/journal.pbio.3002847', issue=None),
    # upham2019 was in the emitted table but never in _bib_clean.json, so regenerating the
    # bibliography silently dropped it.  Record pulled from CrossRef 2026-11-26.
    dict(key='upham2019', year=2019,
         authors=['Upham, Nathan S.', 'Esselstyn, Jacob A.', 'Jetz, Walter'],
         title='Inferring the mammal tree: species-level sets of phylogenies for questions in '
               'ecology, evolution, and conservation',
         venue='PLOS Biology', volume='17', page='e3000494',
         doi='10.1371/journal.pbio.3000494', issue=None),
]

BY = {r['key']: r for r in refs}
for r in NEW:
    BY.setdefault(r['key'], r)

# CrossRef quirks corrected by hand against the article, carried over from the previous builder
OVERRIDE = {
    'reisach2021': dict(authors=['Reisach, Alexander G.', 'Seiler, Christof', 'Weichwald, Sebastian'],
                        title='Beware of the simulated DAG! Causal discovery benchmarks may be easy to game',
                        venue='arXiv', volume=None, page=None, doi=None, eprint='arXiv:2102.13647'),
    'spirtes2000': dict(year=2000, venue='MIT Press, Cambridge, MA', volume=None, page=None),
    'zheng2018': dict(authors=['Zheng, Xun', 'Aragam, Bryon', 'Ravikumar, Pradeep', 'Xing, Eric P.'],
                      title='DAGs with NO TEARS: continuous optimization for structure learning',
                      venue='Advances in Neural Information Processing Systems',
                      volume='31', page=None, doi=None),
}


def texesc(s):
    """Underscores, ampersands and per cent signs are markup in LaTeX; DOIs and titles carry them."""
    for a, b in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('&', r'\&'), ('%', r'\%'),
                 ('#', r'\#'), ('$', r'\$')):
        s = s.replace(a, b)
    return s.replace(r'\textbackslash{}', '\\')


def initials(given):
    parts = re.split(r'[\s\-]+', given.strip())
    out = []
    for p in parts:
        if not p:
            continue
        out.append(p[0].upper() + '.')
    return ''.join(out)


def ama_author(a):
    if ',' in a:
        fam, giv = a.split(',', 1)
    else:
        bits = a.split()
        fam, giv = bits[-1], ' '.join(bits[:-1])
    return '%s %s' % (fam.strip(), initials(giv))


def entry(key):
    r = dict(BY[key])
    r.update(OVERRIDE.get(key, {}))
    au = [ama_author(a) for a in r.get('authors', [])]
    if len(au) > 6:
        aus = ', '.join(au[:3]) + ', et al'
    else:
        aus = ', '.join(au)
    title = (r.get('title') or '').rstrip('.')
    if title and not title.endswith('?'):
        title += '.'
    venue = r.get('venue') or ''
    bits = '%s. ' % aus if aus else ''
    bits += '%s ' % texesc(title)
    tail = ''
    if venue:
        tail += texesc(venue)
    if r.get('volume'):
        tail += (' ' if tail else '') + str(r['volume'])
        if r.get('issue'):
            tail += '(%s)' % r['issue']
        if r.get('page'):
            tail += ':%s' % r['page']
        tail += '.'
    elif r.get('page'):
        tail += ('. ' if tail else '') + str(r['page']) + '.'
    if r.get('year'):
        tail = ('%s;%s' % (r['year'], tail)) if tail else ('%s.' % r['year'])
    if r.get('eprint'):
        tail += ' %s.' % r['eprint']
    if r.get('doi'):
        d = r['doi']
        tail += r' \href{https://doi.org/%s}{doi:%s}.' % (d, texesc(d))
    return (bits + tail).replace('  ', ' ').strip()


def keys_in(path):
    t = io.open(path, encoding='utf-8').read()
    seen, order = set(), []
    for m in re.finditer(r'\\cite[a-z]*\{([^}]*)\}', t):
        for k in m.group(1).split(','):
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                order.append(k)
    return order


def emit(tex, out):
    if not os.path.exists(tex):
        print('missing', tex)
        return
    order = keys_in(tex)
    missing = [k for k in order if k not in BY]
    L = ['% Generated by _make_bib_ama.py from CrossRef metadata. Do not edit by hand.',
         r'\begin{thebibliography}{%d}' % max(len(order), 1), r'\small']
    for k in order:
        if k in BY:
            L.append(r'\bibitem{%s} %s' % (k, entry(k)))
    L += [r'\end{thebibliography}']
    io.open(os.path.join(PAPER, out), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('%-10s %3d references%s' % (out, len(order) - len(missing),
                                      ('   MISSING: ' + ', '.join(missing)) if missing else ''))


emit(os.path.join(PAPER, 'main_manuscript.tex'), 'refs_ama.tex')
emit(os.path.join(PAPER, 'supplement.tex'), 'refs_si.tex')
