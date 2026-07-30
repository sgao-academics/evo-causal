import json

c = json.load(open(r'C:\Users\高帅东\Desktop\evo_causal\results\master_all.json', 'r', encoding='utf-8'))

# P2 Phase diagram
p2 = c['p2']
pic_nz = sum(1 for k, v in p2.items() if v['pic_f1'] > 0)
ct_nz = sum(1 for k, v in p2.items() if v['ct_f1'] > 0)
note_nz = sum(1 for k, v in p2.items() if v['note_f1'] > 0)
pic_best = max(p2.items(), key=lambda x: x[1]['pic_f1'])
ct_best = max(p2.items(), key=lambda x: x[1]['ct_f1'])
note_best = max(p2.items(), key=lambda x: x[1]['note_f1'])

print("=== P2 PHASE DIAGRAM (40 configs: 5d x 8K) ===")
print("PIC+corr: %d/40 non-zero, best %s F1=%.4f" % (pic_nz, pic_best[0], pic_best[1]['pic_f1']))
print("CT:       %d/40 non-zero, best %s F1=%.4f" % (ct_nz, ct_best[0], ct_best[1]['ct_f1']))
print("NOTEARS:  %d/40 non-zero, best %s F1=%.4f" % (note_nz, note_best[0], note_best[1]['note_f1']))

# P3 Architecture
print("\n=== P3 CT ARCHITECTURE ABLATION ===")
for k in sorted(c['p3']['arch'].keys()):
    v = c['p3']['arch'][k]
    print("  %s: F1_mean=%.4f best=%.4f" % (k, v['f1_mean'], v['f1_best']))

print("\n=== P3 N-SCALING ===")
for k in sorted(c['p3']['n_scaling'].keys(), key=lambda x: c['p3']['n_scaling'][x]['n']):
    v = c['p3']['n_scaling'][k]
    print("  n=%d: PIC=%.4f CT=%.4f" % (v['n'], v['pic_f1'], v['ct_f1']))

# P4 DAGMA/GOLEM
p4 = c['p4']
dagma_best_key = max((k for k in p4 if 'dagma_f1' in p4[k]), key=lambda k: p4[k]['dagma_f1'])
dagma_best = p4[dagma_best_key]
print("\n=== P4 DAGMA (40 configs) ===")
print("  Best: %s F1=%.4f" % (dagma_best_key, dagma_best['dagma_f1']))

golem_keys = [k for k in p4 if p4[k].get('golem_f1') is not None]
if golem_keys:
    golem_best_key = max(golem_keys, key=lambda k: p4[k]['golem_f1'])
    print("=== P4 GOLEM (40 configs) ===")
    print("  Best: %s F1=%.4f" % (golem_best_key, p4[golem_best_key]['golem_f1']))
else:
    print("=== P4 GOLEM: not installed ===")

# P5 Head-to-head
p5 = c['p5']
wins = {}
for k, v in p5.items():
    b = v['best']
    wins[b] = wins.get(b, 0) + 1

print("\n=== P5 HEAD-TO-HEAD (15 configs) ===")
print("Best method wins:")
for m, count in sorted(wins.items(), key=lambda x: -x[1]):
    print("  %s: %d/15" % (m, count))

# Show top 3 configs
print("\nTop 3 configs by F1:")
for cfg_key, cfg_val in sorted(p5.items(), key=lambda x: -x[1]['best_f1'])[:3]:
    print("  %s: best=%s F1=%.4f methods=%s" % (
        cfg_key, cfg_val['best'], cfg_val['best_f1'],
        str({k: round(v, 4) for k, v in cfg_val['methods'].items()})
    ))
