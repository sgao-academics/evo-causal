import json
c = json.load(open(r'C:\Users\高帅东\Desktop\evo_causal\results\master_all.json', 'r', encoding='utf-8'))

# Check ct_trace keys
ct = c['p1']['ct_trace']
print('=== ct_trace sample ===')
for k in list(ct.keys())[:1]:
    print(f'Key: {k}')
    print(f'All keys: {list(ct[k].keys())}')

# Check convergence trace
print()
print('=== convergence sample ===')
conv = c['p1']['ct_convergence']
for k in list(conv.keys())[:1]:
    v = conv[k]
    print(f'Key: {k}')
    print(f'Conv entry keys: {list(v.keys())}')
    t = v.get('trace', {})
    print(f'Trace keys: {list(t.keys())}')
    for tk, tv in t.items():
        tp = type(tv).__name__
        ln = len(tv) if hasattr(tv, '__len__') else 'scalar'
        print(f'  {tk}: type={tp}, value={tv if tp=="float" else str(tv)[:50]}')
