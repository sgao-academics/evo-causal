"""
Ensembl is BACK! Full pipeline: download -> PIC -> known pair validation.
"""
import requests, json, numpy as np, time, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from felsenstein_pic import felsenstein_pic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
RES_DIR = os.path.join(ROOT, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

ENSEMBL = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json"}
DELAY = 0.3

GENES = [
    "VEGFA","KDR","FLT1","FLT4","INS","INSR","IGF1","IGF1R",
    "EGF","EGFR","ERBB2","ERBB3","BMP4","BMPR1A","BMPR2",
    "SHH","PTCH1","SMO","NOTCH1","DLL1","HGF","MET",
    "TNF","TNFRSF1A","GAPDH","ACTB","TUBB","TP53","MYC","AKT1","MTOR",
]

TARGET_SP = {
    "homo_sapiens":"Human","pan_troglodytes":"Chimp","gorilla_gorilla":"Gorilla",
    "pongo_abelii":"Orangutan","macaca_mulatta":"Macaque",
    "mus_musculus":"Mouse","rattus_norvegicus":"Rat",
    "oryctolagus_cuniculus":"Rabbit","bos_taurus":"Cow","sus_scrofa":"Pig",
    "canis_familiaris":"Dog","felis_catus":"Cat","equus_caballus":"Horse",
    "monodelphis_domestica":"Opossum","ornithorhynchus_anatinus":"Platypus",
    "gallus_gallus":"Chicken","anolis_carolinensis":"Lizard",
    "xenopus_tropicalis":"Frog","danio_rerio":"Zebrafish","takifugu_rubripes":"Fugu",
}

print(f"=== Ensembl Real Data Pipeline ===\n{GENES} genes x {len(TARGET_SP)} species\n")

# Download ortholog data
ortho = {}  # gene -> {species_name: ortholog_type}
for i, gene in enumerate(GENES):
    url = f"{ENSEMBL}/homology/symbol/homo_sapiens/{gene}"
    url += "?content-type=application/json&format=condensed&type=orthologues"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            sp_found = {}
            for item in data.get("data", []):
                for hom in item.get("homologies", []):
                    sp = hom.get("species", "")
                    otype = hom.get("type", "ortholog")
                    if sp in TARGET_SP:
                        sp_found[sp] = otype
            ortho[gene] = sp_found
            status = f"{len(sp_found)} species" if sp_found else "NO ORTHOLOGS"
        else:
            ortho[gene] = {}
            status = f"HTTP {r.status_code}"
    except Exception as e:
        ortho[gene] = {}
        status = f"ERR: {e}"
    print(f"  [{i+1:2d}/{len(GENES)}] {gene:<12} {status}")
    time.sleep(DELAY)

# Build binary matrix
sp_list = sorted(TARGET_SP.keys())
gene_list = sorted(ortho.keys())
M = np.zeros((len(sp_list), len(gene_list)), dtype=np.float32)
for si, sp in enumerate(sp_list):
    for gi, gene in enumerate(gene_list):
        M[si, gi] = 1.0 if sp in ortho.get(gene, {}) else 0.0

print(f"\nMatrix: {M.shape[0]} sp x {M.shape[1]} genes, total={int(M.sum())}")

# Build tree + PIC
from run_phylo_sweep import build_balanced_tree, _make_ultrametric
tree, _ = build_balanced_tree(len(sp_list), seed=42)
_make_ultrametric(tree)

# Binary->continuous
rng = np.random.RandomState(42)
Xc = M.astype(np.float32)*2.0 + 0.05*rng.randn(*M.shape)
Xc = (Xc-Xc.mean(0))/(Xc.std(0)+1e-8)
Xp = felsenstein_pic(Xc, tree)

# Correlations
Cr, Cp = np.corrcoef(Xc.T), np.corrcoef(Xp.T)
mu = np.triu(np.ones((len(gene_list),len(gene_list))),1).astype(bool)
br, bp = float(np.abs(Cr[mu]).mean()), float(np.abs(Cp[mu]).mean())
print(f"Bkg corr: raw={br:.4f} PIC={bp:.4f} ({(1-bp/max(br,1e-8))*100:.1f}% reduction)")

# Known pairs
KNOWN = [("VEGFA","KDR"),("VEGFA","FLT1"),("INS","INSR"),("IGF1","IGF1R"),
         ("EGF","EGFR"),("BMP4","BMPR1A"),("SHH","PTCH1"),("HGF","MET"),
         ("TNF","TNFRSF1A"),("GAPDH","ACTB")]
gi = {g:i for i,g in enumerate(gene_list)}
allc = np.abs(Cp[mu]); p99 = np.percentile(allc,99)
print(f"\nKnown pairs (p99={p99:.4f}):")
hit=0; tot=0
for g1,g2 in KNOWN:
    if g1 in gi and g2 in gi:
        tot+=1; pc=abs(Cp[gi[g1],gi[g2]]); r=(allc<pc).sum()/len(allc); h=pc>p99
        if h: hit+=1
        print(f"  {g1}-{g2:<15} r={pc:.4f} rank={r:.4f} {'HIT!' if h else ''}")
print(f"\nRecovered: {hit}/{tot} = {100*hit/max(tot,1):.0f}%")

# Save
np.save(os.path.join(DATA_DIR,"ensembl_real_M.npy"),M)
np.save(os.path.join(DATA_DIR,"ensembl_real_X.npy"),Xc)
with open(os.path.join(DATA_DIR,"ensembl_real_genes.json"),"w") as f: json.dump(gene_list,f)
with open(os.path.join(DATA_DIR,"ensembl_real_sp.json"),"w") as f: json.dump(sp_list,f)
print(f"\nDone: {DATA_DIR}/ensembl_real_*")
