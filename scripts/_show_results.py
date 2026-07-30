import json
c = json.load(open(r'C:\Users\高帅东\Desktop\evo_causal\results\nonlinear_wall.json', 'r', encoding='utf-8'))

# === EXP A ===
a = c['exp_a_nonlinear']
print('=== EXP A: Nonlinear SEM ===')
print('Config        | Linear   | MLP      | Sigmoid  | Consistent?')
print('-' * 72)
for d in [30, 50, 100]:
    for alpha in [0.02, 0.10]:
        base = f'd={d}_a={alpha}_n=200'
        lin = a.get(base + '_linear', {}).get('pic_f1', -1)
        mlp = a.get(base + '_mlp', {}).get('pic_f1', -1)
        sig = a.get(base + '_sigmoid', {}).get('pic_f1', -1)
        ok = (lin > 0.1) == (mlp > 0.1) == (sig > 0.1)
        status = 'OK' if ok else '*** DIVERGENT ***'
        print(f'd={d:3d} a={alpha:.2f}  | {lin:.4f}    | {mlp:.4f}    | {sig:.4f}    | {status}')

# === EXP B ===
b = c['exp_b_penetration']
print()
print('=== EXP B: Wall Penetration ===')
penetrated = []
for n in [100, 200, 500, 1000, 2000, 5000]:
    for alpha in [0.01, 0.005, 0.001, 0.0001, 0.0]:
        key = f'n={n}_a={alpha}'
        v = b.get(key, {})
        f1 = v.get('pic_mean', -1)
        pen = v.get('penetrated', False)
        if f1 > 0.001 or n <= 200:
            marker = 'PENETRATED' if pen else 'no'
            print(f'n={n:5d} a={alpha:.4f}  F1={f1:.4f}  {marker}')
        if pen:
            penetrated.append((key, f1))

print()
if penetrated:
    print(f'WALL PENETRATED at {len(penetrated)} configs:')
    for k, f1 in sorted(penetrated, key=lambda x: -x[1])[:5]:
        print(f'  {k}: F1={f1:.4f}')
else:
    print('WALL NOT PENETRATED.')
    best_key = max(b.items(), key=lambda x: x[1]['pic_mean'])
    print(f'Best: {best_key[0]} F1={best_key[1]["pic_mean"]:.4f}')
