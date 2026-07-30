import json
nl = json.load(open(r'C:\Users\高帅东\Desktop\evo_causal\results\nonlinear_wall.json','r',encoding='utf-8'))

# Check exp_b_penetration keys
b = nl['exp_b_penetration']
print('=== Panel B: penetration keys (sample) ===')
for k in list(b.keys())[:5]:
    print(f'  {k}: pic_mean={b[k]["pic_mean"]:.4f}')
print(f'  Total: {len(b)} configs')

# Check exp_a_nonlinear keys
print()
a = nl['exp_a_nonlinear']
print('=== Panel C: nonlinear keys (sample) ===')
for k in list(a.keys())[:5]:
    v = a[k]
    print(f'  {k}: pic_f1={v.get("pic_f1")}')
print(f'  Total: {len(a)} configs')
