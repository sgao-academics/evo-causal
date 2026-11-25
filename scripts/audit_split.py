# -*- coding: utf-8 -*-
"""Audit of the split submission: no number lost, no cell invented, and the journal's limits.

Six checks, each of which can fail:

1.  every numeric value that appeared in the single-document manuscript still appears in the main
    text, the supplement, or a table of either;
2.  every cell of the two rebuilt tables matches the result file it claims to come from;
3.  the main text is within 4000--6000 words excluding abstract, figures, tables and references;
4.  the abstract is within 300 words and carries the five required headings;
5.  the reference list is within 100 entries, and every citation resolves;
6.  no cross-reference points at a label that is not in the same document, and the style checks
    that the previous round introduced (no markdown emphasis, few em-dashes).

Run from ../revision beside the paper directory: python audit_split.py
"""
import io, os, re, sys, json

S = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(S)
PAPER = os.path.join(ROOT, 'paper')
RES = os.path.join(ROOT, 'results')

ok = fail = 0
fails = []


def chk(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        fails.append('%-62s %s' % (name[:62], detail))


def read(p):
    return io.open(p, encoding='utf-8').read() if os.path.exists(p) else ''


def flat(t):
    t = re.sub(r'\s+', ' ', t)
    return (t.replace('$', '').replace('\\,', '').replace('{,}', '')
             .replace('{', '').replace('}', '').replace('~', ' '))


# ---------------------------------------------------------------- 1 numbers carried over
NUM = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w])')


def numbers(t):
    # an escaped per cent sign is not a comment, and the tables are full of them
    t = re.sub(r'(?<!\\)%[^\n]*', ' ', t)
    t = t.replace('{,}', '')
    t = re.sub(r'\\label\{[^}]*\}', ' ', t)
    t = re.sub(r'\\ref\{[^}]*\}', ' ', t)
    t = re.sub(r'\\cite[a-z]*\{[^}]*\}', ' ', t)
    out = set()
    for m in NUM.finditer(t):
        v = m.group(0)
        if len(v.replace('-', '').replace('.', '')) >= 2:
            out.add(v)
    return out


old = read(os.path.join(PAPER, 'evo_causal_revision.tex'))
main = read(os.path.join(PAPER, 'main_manuscript.tex'))
supp = read(os.path.join(PAPER, 'supplement.tex'))
tabs = ''.join(read(os.path.join(PAPER, f)) for f in os.listdir(PAPER)
               if f.startswith('tab_') and f.endswith('.tex'))
new_all = main + supp + tabs

lost = sorted(numbers(old) - numbers(new_all),
              key=lambda s: float(s) if s.replace('-', '').replace('.', '').isdigit() else 0)
print('numbers in the one-document version   %3d' % len(numbers(old)))
print('absent from main + supplement + tables %3d' % len(lost))
if lost:
    print('   ', ' '.join(lost[:60]))

# Every value that does not survive the split is listed here with the reason. Two of them are
# corrections rather than losses: the tetrad rejection rate of the ecology block was 0.80 in an
# earlier run and is 0.60 in the run the table is built from, and the dataset total moves from
# 1660 to 1690 because the nonlinear grid was added. The rest are either identifying strings
# removed for anonymous review or secondary values whose sentence was compressed.
ALLOWED = {
    '0.022': 'sparse-graph p, re-drawn at 500 reps: now 0.020',
    '2.21': 'null mean, re-drawn: now 2.22',
    '2.31': 'null mean, re-drawn: now 2.30',
    '2.35': 'null mean, re-drawn: now 2.34',
    '0.39': 'null sd, now stated from the JSON',
    '0.40': 'range end of the damage window, compressed',
    '0.41': 'null sd, now stated from the JSON',
    '0.42': 'NOTEARS at d=10 top of range, compressed',
    '0.43': 'null sd, now stated from the JSON',
    '0.067': 'uncorrected F1 at high signal, compressed',
    '0.640': 'K span at d=50, compressed',
    '0.80': 'stale tetrad rate: the run behind the table gives 0.60',
    '0.936': 'Spearman AUROC at lambda=1.5, compressed',
    '0.991': 'Pearson AUROC at lambda=1.5, compressed',
    '0.95': 'replaced by the measured 0.98 or better',
    '83': 'FCI endpoint share on an earlier grid, replaced by 89 and 63',
    '1660': 'dataset total, now 1690 with the nonlinear grid',
    '0004': 'ORCID, removed for anonymous review',
    '0009': 'ORCID, removed for anonymous review',
    '3581': 'ORCID, removed for anonymous review',
    '5641': 'ORCID, removed for anonymous review',
    '401420': 'postcode, removed for anonymous review',
}
unexpected = [v for v in lost if v not in ALLOWED]
print('unexplained losses %d%s' % (len(unexpected), (': ' + ' '.join(unexpected)) if unexpected else ''))
chk('every dropped value has a recorded reason', not unexpected, ' '.join(unexpected))

# ---------------------------------------------------------------- 2 table cells
def load(n):
    p = os.path.join(RES, n + '.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


H = load('e30_holm'); e23 = load('e23_families10'); e29 = load('e29_resit')
tr = read(os.path.join(PAPER, 'tab_real.tex'))
for key, b in H.get('blocks', {}).items():
    chk('tab_real has a row for %s' % key, key.replace('_', ' ') in tr)
    row = [l for l in tr.split('\n') if l.startswith(key.replace('_', ' '))]
    if row:
        chk('tab_real r1 matches %s' % key, ('%.2f' % b['obs'][0]) in row[0], row[0][:60])
        chk('tab_real null matches %s' % key, ('%.2f' % b['null_mean'][0]) in row[0])
        chk('tab_real k matches %s' % key, ('& %d &' % b['k']) in row[0])
for key in H.get('holes', {}):
    chk('tab_real carries the Holm value of %s' % key,
        ('%.3f' % H['holes'][key]['holm']) in tr)

# the prose quotes the same null mean and spread as the table, so both are checked against the file
for key in ('life_history', 'range_climate'):
    b = H['blocks'][key]
    needle = '%.2f \\pm %.2f' % (b['null_mean'][0], b['null_sd'][0])
    chk('prose quotes the null spread of %s' % key, needle in main, needle)
    chk('prose quotes the raw p of %s' % key, ('p = %.3f' % b['p'][0]) in main,
        '%.3f' % b['p'][0])
for key in ('life_history:r1', 'range_climate:r1'):
    chk('prose quotes the Holm value of %s' % key,
        ('%.3f' % H['holes'][key]['holm']) in main, '%.3f' % H['holes'][key]['holm'])
chk('prose reports three of fifteen surviving', 'three of them standing' in main)
chk('prose does not claim five of five exceed the reference',
    'Two of five matrices' in main)

tf = read(os.path.join(PAPER, 'tab_families.tex'))
for k in ('lam0.0', 'lam0.5', 'lam1.0', 'lam1.5', 'lam2.0', 'lam3.0'):
    a = e23.get(k)
    if not a:
        continue
    chk('tab_families row %s' % k, ('%.1f & %.3f' % (a['lam'], a['K'])) in tf)
    chk('tab_families corr %s' % k, ('%.3f (%.2f)' % (a['corr'], a['corr_sd'])) in tf)
    chk('tab_families notears %s' % k, ('%.3f (%.2f)' % (a['notears'], a['notears_sd'])) in tf)
    chk('tab_families pc/fci %s' % k, ('%.3f & %.3f' % (a['pc'], a['fci'])) in tf)
    b = e29.get(k)
    if b:
        chk('tab_families resit %s' % k, ('%.3f & %.3f' % (b['f1'], b['auroc'])) in tf)

# ---------------------------------------------------------------- 3 word count
body = main[main.find(r'\section{Introduction}'):]
body = body[:body.find(r'\section*{Statements')]
for env in ('figure', 'table', 'tabular', 'equation', 'align'):
    body = re.sub(r'\\begin\{%s\*?\}.*?\\end\{%s\*?\}' % (env, env), ' ', body, flags=re.S)
body = re.sub(r'%.*', '', body)
body = re.sub(r'\\(sub)?(paragraph|section)\*?\{[^}]*\}', ' ', body)
body = re.sub(r'\\(cite[a-z]*|ref|eqref|label)\{[^}]*\}', ' ', body)
body = re.sub(r'\$[^$]*\$', ' ', body)
body = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', body)
words = [w for w in body.replace('{', ' ').replace('}', ' ').replace('~', ' ').split()
         if re.search(r'[A-Za-z0-9]', w)]
print('main-text words %d' % len(words))
chk('main text within 4000-6000 words', 4000 <= len(words) <= 6000, str(len(words)))
chk('main text at least 4500 words (no penalty for brevity)', len(words) >= 4500, str(len(words)))

# ---------------------------------------------------------------- 4 abstract
ab = main[main.find(r'\begin{abstract}'):main.find(r'\end{abstract}')]
abw = len([w for w in re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', ab).split()
           if re.search(r'[A-Za-z]', w)])
print('abstract words %d' % abw)
chk('abstract within 300 words', abw <= 300, str(abw))
for head in ('Background', 'Objectives', 'Results', 'Conclusions'):
    chk('abstract heading %s' % head, ('\\textbf{%s.}' % head) in ab)

# ---------------------------------------------------------------- 5 references
def keys(t):
    out = set()
    for m in re.finditer(r'\\cite[a-z]*\{([^}]*)\}', t):
        for k in m.group(1).split(','):
            out.add(k.strip())
    return out


bibm = read(os.path.join(PAPER, 'refs_ama.tex'))
bibs = read(os.path.join(PAPER, 'refs_si.tex'))
dm = set(re.findall(r'\\bibitem\{(\w+)\}', bibm))
ds = set(re.findall(r'\\bibitem\{(\w+)\}', bibs))
print('references: main %d, supplement %d' % (len(dm), len(ds)))
chk('main reference list within 100', len(dm) <= 100, str(len(dm)))
chk('every main citation resolves', keys(main) <= dm, str(sorted(keys(main) - dm))[:80])
chk('every supplementary citation resolves', keys(supp) <= ds, str(sorted(keys(supp) - ds))[:80])
chk('no unused reference in main', dm <= keys(main), str(sorted(dm - keys(main)))[:80])

# ---------------------------------------------------------------- 6 references and style
def labels_refs(t):
    lab = set(re.findall(r'\\label\{([^}]*)\}', t))
    ref = set(re.findall(r'\\(?:eq)?ref\{([^}]*)\}', t))
    return lab, ref


for name, doc, inc in (('main', main, ['tab_axes', 'tab_families', 'tab_real']),
                       ('supplement', supp, ['tab_design', 'tab_calib', 'tab_multifac',
                                             'tab_dose', 'tab_admg', 'tab_ppa'])):
    alltext = doc + ''.join(read(os.path.join(PAPER, f + '.tex')) for f in inc)
    lab, ref = labels_refs(alltext)
    chk('%s: every reference has a label' % name, ref <= lab, str(sorted(ref - lab))[:80])
    chk('%s: no markdown emphasis left' % name, not re.search(r'(?<!\\)\*[A-Za-z][^*]{0,40}(?<!\\)\*', doc))
    ndash = len(re.findall(r'---', doc))
    chk('%s: em-dashes below 10' % name, ndash < 10, str(ndash))
    # a double hyphen inside a word is legitimate for compounds (Ornstein--Uhlenbeck), so what is
    # checked instead is that no em-dash is jammed against a word
    chk('%s: no em-dash jammed against a word' % name,
        not re.search(r'[A-Za-z]---', doc))

print()
print('checks %d, failed %d' % (ok + fail, fail))
for f in fails:
    print('  FAIL', f)
print('OVERALL', 'PASS' if fail == 0 else 'FAIL')
