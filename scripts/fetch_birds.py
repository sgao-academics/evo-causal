# -*- coding: utf-8 -*-
"""A second clade, from a different trait database, on its own synthesis topology.

Traits: EltonTraits 1.0 (Wilman et al. 2014, Ecology 95:2027), 9,995 bird species,
ten diet percentages, seven foraging-stratum percentages, nocturnality, body mass.
Topology: the Open Tree of Life synthesis subtree for Aves, pruned to the species with a
complete trait vector, branch lengths by Grafen (1989) downstream.

Why a second clade. The mammal anchor in exp_real.py is one class and one trait table, so
"real comparative matrices land in the failure region" could be a property of that table.
Birds are a separate radiation with a separate trait compilation and a different topology;
if the same excess appears there, it is a property of comparative data rather than of
PanTHERIA.

Two blocks are built. The ten diet percentages sum to 100, and so do the seven foraging
percentages, so one component is dropped from each to break the closure; keeping all ten
would make the covariance singular for a reason that has nothing to do with phylogeny.

The species set is capped at MAXN by a seeded random draw. The cap is a compute limit
(the null simulation and Blomberg's K are cubic in the number of tips), and it is applied
to the real matrix and its null alike, so the comparison stays internally consistent.
"""
import io, os, json, csv, time
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(S), 'data')
os.makedirs(D, exist_ok=True)

BIRD = os.path.join(D, 'BirdFuncDat.txt')
BIRD_URL = 'https://esapubs.org/archive/ecol/E095/178/BirdFuncDat.txt'
MATCH = os.path.join(D, 'otol_match_birds.json')
NEWICK = os.path.join(D, 'otol_aves.nwk')

MAXN = 1500
SEED = 2024

BLOCKS = {
    'bird_diet': [('BodyMass-Value', 'body mass (g)')] +
                 [('Diet-%s' % c, '%s (%% of diet)' % lab) for c, lab in
                  (('Inv', 'invertebrates'), ('Vend', 'endotherms'), ('Vect', 'ectotherms'),
                   ('Vfish', 'fish'), ('Scav', 'scavenging'), ('Fruit', 'fruit'),
                   ('Nect', 'nectar'), ('Seed', 'seed'), ('PlantO', 'other plant'))],
    'bird_niche': [('BodyMass-Value', 'body mass (g)')] +
                  [('ForStrat-%s' % c, '%s (%%)' % lab) for c, lab in
                   (('wataroundsurf', 'water around surface'), ('ground', 'ground'),
                    ('understory', 'understory'), ('midhigh', 'mid-high'),
                    ('canopy', 'canopy'), ('aerial', 'aerial'))] +
                  [('Nocturnal', 'nocturnality'), ('PelagicSpecialist', 'pelagic specialist')],
}


def fetch_bird():
    if os.path.exists(BIRD) and os.path.getsize(BIRD) > 100000:
        print('elton cached', os.path.getsize(BIRD))
        return
    import urllib.request
    req = urllib.request.Request(BIRD_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=180) as r:
        b = r.read()
    io.open(BIRD, 'wb').write(b)
    print('elton downloaded', len(b))


def read_bird():
    raw = io.open(BIRD, encoding='latin-1').read()
    rd = list(csv.reader(io.StringIO(raw), delimiter='\t'))
    head = [h.strip() for h in rd[0]]
    idx = {h: i for i, h in enumerate(head)}
    rows = [r for r in rd[1:] if len(r) == len(head)]
    sp_i = idx['Scientific']
    want = sorted({c for blk in BLOCKS.values() for c, _ in blk})
    out = []
    for r in rows:
        sp = r[sp_i].strip()
        if not sp or ' ' not in sp:
            continue
        v = {}
        for c in want:
            if c not in idx:
                continue
            try:
                x = float(r[idx[c]])
            except ValueError:
                continue
            if x > -100:
                v[c] = x
        out.append((sp, v))
    return out


def tnrs(names):
    import urllib.request
    body = json.dumps({'names': names, 'do_approximate_matching': False}).encode()
    req = urllib.request.Request('https://api.opentreeoflife.org/v3/tnrs/match_names',
                                 data=body, headers={'Content-Type': 'application/json'})
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:
            print('  tnrs retry', a, type(e).__name__)
            time.sleep(4 * (a + 1))
    return {'results': []}


def match_all(names):
    cache = json.load(io.open(MATCH, encoding='utf-8')) if os.path.exists(MATCH) else {}
    todo = [n for n in names if n not in cache]
    print('to match', len(todo), 'cached', len(cache), flush=True)
    B = 100
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        o = tnrs(chunk)
        got = {}
        for res in o.get('results', []):
            ms = res.get('matches') or []
            if ms:
                t = ms[0]['taxon']
                if not t.get('is_suppressed_from_synth'):
                    got[res['name']] = t['ott_id']
        for n in chunk:
            cache[n] = got.get(n)
        print('  %d/%d matched %d' % (i + len(chunk), len(todo),
                                      sum(1 for v in cache.values() if v)), flush=True)
        time.sleep(0.5)
    json.dump(cache, io.open(MATCH, 'w', encoding='utf-8'))
    return cache


def aves_subtree():
    """The synthesis subtree rooted at Aves, labels as bare binomials."""
    import urllib.request
    if os.path.exists(NEWICK) and os.path.getsize(NEWICK) > 50000:
        print('aves tree cached', os.path.getsize(NEWICK))
        return io.open(NEWICK, encoding='utf-8').read()
    body = json.dumps({'names': ['Aves'], 'do_approximate_matching': True}).encode()
    req = urllib.request.Request('https://api.opentreeoflife.org/v3/tnrs/match_names',
                                 data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        o = json.loads(r.read())
    ott = None
    for res in o.get('results', []):
        for m in res.get('matches') or []:
            ott = m['taxon']['ott_id']
            break
        if ott:
            break
    print('Aves ott_id', ott)
    body = json.dumps({'ott_id': ott, 'label_format': 'name'}).encode()
    req = urllib.request.Request('https://api.opentreeoflife.org/v3/tree_of_life/subtree',
                                 data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        o = json.loads(r.read())
    nw = o.get('newick', '')
    print('aves newick chars', len(nw), 'tips', nw.count(',') + 1)
    io.open(NEWICK, 'w', encoding='utf-8').write(nw)
    return nw


def main():
    fetch_bird()
    rows = read_bird()
    print('bird species with any trait', len(rows))
    cache = match_all([sp for sp, _ in rows])
    ok = [(sp, v) for sp, v in rows if cache.get(sp)]
    print('matched on the synthesis', len(ok))

    out = {}
    rng = np.random.default_rng(SEED)
    for name, blk in BLOCKS.items():
        cols = [(c, lab) for c, lab in blk]
        full = [(sp, v) for sp, v in ok if all(c in v for c, _ in cols)]
        print('block %-12s d=%2d complete %5d' % (name, len(cols), len(full)), flush=True)
        if len(full) > MAXN:
            pick = np.sort(rng.choice(len(full), MAXN, replace=False))
            full = [full[i] for i in pick]
        species = [sp for sp, _ in full]
        X = np.array([[v[c] for c, _ in cols] for _, v in full], float)
        logs = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if col.min() > 0 and col.max() / max(col.min(), 1e-12) > 20:
                X[:, j] = np.log10(col)
                logs.append(cols[j][0])
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        alive = X.std(0) > 1e-8
        X = X[:, alive]
        cols2 = [c for c, a in zip(cols, alive) if a]
        out[name] = dict(species=species, cols=[c for c, _ in cols2],
                         labels=[l for _, l in cols2], logged=logs,
                         dropped=[c for c, a in zip([c for c, _ in cols], alive) if not a],
                         X=X.tolist())
        print('   kept n=%d d=%d logged=%s dropped=%s' % (len(species), X.shape[1], logs,
                                                          out[name]['dropped']))
    json.dump(out, io.open(os.path.join(D, 'bird_blocks.json'), 'w', encoding='utf-8'))
    aves_subtree()


if __name__ == '__main__':
    main()
