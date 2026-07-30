import json
c = json.load(open(r'C:\Users\高帅东\Desktop\evo_causal\results\master_all.json','r',encoding='utf-8'))

# Panel A
ct = c['p1']['ct_trace']
print('=== Panel A: ct_trace ===')
for k in sorted(ct.keys(), key=lambda x: float(x.split('=')[1]))[:5]:
    v = ct[k]
    print(f'  {k}: grad={v.get("grad_last100_norm")} h={v.get("h_plateau")}')

# Panel B
print()
arch = c['p3']['arch']
print('=== Panel B: arch ===')
for k in sorted(arch.keys()):
    print(f'  {k}: f1={arch[k]["f1_mean"]}')

# Panel C
print()
conv = c['p1']['ct_convergence']
print('=== Panel C: convergence ===')
for k in conv:
    v = conv[k]
    t = v.get('trace', {})
    h = t.get('h_W', [])
    e = t.get('edges', [])
    print(f'  {k}: hW_len={len(h)} edges_len={len(e)}')

# Panel D
print()
nsc = c['p3']['n_scaling']
print('=== Panel D: n-scaling ===')
for k in sorted(nsc.keys(), key=lambda x: nsc[x]['n']):
    print(f'  n={nsc[k]["n"]}: PIC={nsc[k]["pic_f1"]} CT={nsc[k]["ct_f1"]}')
