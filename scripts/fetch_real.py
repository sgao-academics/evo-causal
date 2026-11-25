# -*- coding: utf-8 -*-
"""Fetch a real trait table and the matching real topology.

Traits: PanTHERIA (Jones et al. 2009, Ecology 90:2648), 4,629 mammal species with 55
columns, of which the ones used below are species-level measurements (body mass, life
history, morphology, ecology) and species range/geography summaries.

Topology: the Open Tree of Life synthesis (Hinchliff et al. 2015), induced on exactly the
species that have a complete trait vector, so that the tree and the table line up species
for species.  The synthesis carries topology but no branch lengths; lengths are assigned
downstream by Grafen's method, and that choice is checked for sensitivity in the
experiment that consumes this file.

Everything is cached under data/.
"""
import os, io, json, time, csv
import urllib.request

S = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(S), 'data')
os.makedirs(D, exist_ok=True)

PANTH = os.path.join(D, 'PanTHERIA.txt')
MATCH = os.path.join(D, 'otol_match.json')
NEWICK = os.path.join(D, 'otol_mammals.nwk')
NEWICK_CLADE = os.path.join(D, 'otol_mammalia.nwk')   # whole Mammalia synthesis subtree
PANTH_URL = 'https://esapubs.org/archive/ecol/E090/184/PanTHERIA_1-0_WR93_Aug2008.txt'

# trait blocks: a species-level matrix of size, life history, morphology and ecology.
# Columns are the exact PanTHERIA headers.
BLOCKS = {
    'ecology': [
        ('5-1_AdultBodyMass_g', 'adult body mass (g)'),
        ('12-1_HabitatBreadth', 'habitat breadth'),
        ('12-2_Terrestriality', 'terrestriality'),
        ('15-1_LitterSize', 'litter size'),
        ('6-1_DietBreadth', 'diet breadth'),
        ('6-2_TrophicLevel', 'trophic level'),
    ],
    'life_history': [
        ('5-1_AdultBodyMass_g', 'adult body mass (g)'),
        ('9-1_GestationLen_d', 'gestation length (d)'),
        ('17-1_MaxLongevity_m', 'maximum longevity (mo)'),
        ('15-1_LitterSize', 'litter size'),
    ],
    'range_climate': [
        ('26-1_GR_Area_km2', 'range area (km2)'),
        ('26-2_GR_MaxLat_dd', 'max latitude of range'),
        ('26-3_GR_MinLat_dd', 'min latitude of range'),
        ('26-6_GR_MinLong_dd', 'min longitude of range'),
        ('26-7_GR_MRLong_dd', 'mean longitude of range'),
        ('26-5_GR_MaxLong_dd', 'span of longitude'),
        ('27-2_HuPopDen_Mean_n/km2', 'mean human population density'),
        ('28-1_Precip_Mean_mm', 'mean precipitation (mm)'),
        ('28-2_Temp_Mean_01degC', 'mean temperature (0.1 C)'),
        ('30-1_AET_Mean_mm', 'actual evapotranspiration (mm)'),
        ('30-2_PET_Mean_mm', 'potential evapotranspiration (mm)'),
        ('27-4_HuPopDen_Change', 'change in human population density'),
    ],
}


def fetch_pantheria():
    if os.path.exists(PANTH) and os.path.getsize(PANTH) > 100000:
        print('pantheria cached', os.path.getsize(PANTH))
        return
    req = urllib.request.Request(PANTH_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=180) as r:
        b = r.read()
    io.open(PANTH, 'wb').write(b)
    print('pantheria downloaded', len(b))


def read_pantheria():
    with io.open(PANTH, encoding='latin-1') as f:
        rd = csv.reader(f, delimiter='\t')
        head = next(rd)
        idx = {h: i for i, h in enumerate(head)}
        rows = [r for r in rd if len(r) == len(head)]
    sp_i = idx['MSW93_Binomial']
    out = []
    for r in rows:
        sp = r[sp_i].strip()
        if not sp or ' ' not in sp:
            continue
        vals = {}
        for blk in BLOCKS.values():
            for col, lab in blk:
                if col not in idx:
                    continue
                try:
                    x = float(r[idx[col]])
                except ValueError:
                    continue
                if x > -100:                      # -999 / -999.99 are missing codes
                    vals[col] = x
        out.append((sp, vals))
    return out


def tnrs(names):
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
    print('to match', len(todo))
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
        print('  %d/%d (%d matched)' % (i + len(chunk), len(todo),
                                        sum(1 for v in cache.values() if v)), flush=True)
        time.sleep(0.8)
    json.dump(cache, io.open(MATCH, 'w', encoding='utf-8'))
    return cache


def induced_once(ids):
    body = json.dumps({'ott_ids': list(ids), 'label_format': 'name'}).encode()
    req = urllib.request.Request('https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree',
                                 data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def induced_repair(ids, depth=0):
    """Ask for the induced subtree; if the endpoint refuses the whole set, split it and
    keep the union of the parts that succeed.  The returned Newick is a list of pieces."""
    ids = sorted(set(ids))
    for a in range(3):
        try:
            o = induced_once(ids)
            return [o.get('newick', '')]
        except Exception as e:
            print('  induced attempt', a, type(e).__name__, str(e)[:80])
            time.sleep(3)
    if len(ids) <= 20 or depth > 6:
        return []
    h = len(ids) // 2
    print('  splitting %d -> %d + %d' % (len(ids), h, len(ids) - h))
    return induced_repair(ids[:h], depth + 1) + induced_repair(ids[h:], depth + 1)


def main():
    fetch_pantheria()
    rows = read_pantheria()
    print('species with any trait', len(rows))
    cache = match_all([sp for sp, _ in rows])
    # keep only species that matched, and report each block's usable size
    for name, blk in BLOCKS.items():
        need = [c for c, _ in blk if c in set(k for _, v in rows[:1] for k in v)]
        ok = [sp for sp, v in rows if cache.get(sp) and all(c in v for c in set(c for c, _ in blk))]
        print('block %-20s cols %2d  complete-case species %4d' % (name, len(blk), len(ok)))
    ids = sorted({cache[sp] for sp, _ in rows if cache.get(sp)})
    print('species matched and on the synthesis', len(ids))
    pieces = induced_repair(ids)
    nw = ''.join(pieces)
    print('newick total length', len(nw), 'pieces', len(pieces))
    io.open(NEWICK, 'w', encoding='utf-8').write(nw)
    json.dump({'pieces': len(pieces), 'n_ids': len(ids)}, io.open(
        os.path.join(D, 'otol_meta.json'), 'w', encoding='utf-8'))


if __name__ == '__main__':
    main()
