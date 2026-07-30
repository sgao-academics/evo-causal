import re, os
path = r'C:\Users\高帅东\Desktop\evo_causal\paper\evo_causal_paper.tex'
with open(path, 'r', encoding='utf-8') as f:
    tex = f.read()

print('=== 1. Brace balance ===')
print(f'  {{: {tex.count("{")}, }}: {tex.count("}")}, diff: {tex.count("{")-tex.count("}")}')
print(f'  [: {tex.count("[")}, ]: {tex.count("]")}')

print()
print('=== 2. Math integrity ===')
lines = tex.split('\n')
odd = 0
for i, line in enumerate(lines):
    if 'bibitem' in line or line.strip().startswith('%'):
        continue
    if line.count('$') % 2 != 0:
        odd += 1
        if odd <= 5:
            print(f'  L{i+1} ODD: {line.strip()[:100]}')
print(f'  Total odd: {odd}')

print()
print('=== 3. References ===')
bibs = re.findall(r'bibitem\[([^\]]*)\]\{([^}]+)\}', tex)
cited = set()
for m in re.finditer(r'citep?\{(.+?)\}', tex):
    for k in m.group(1).split(','):
        cited.add(k.strip())
uncited = [b[1] for b in bibs if b[1] not in cited]
self_cites = [b[1] for b in bibs if 'gao' in b[1].lower()]
print(f'  Total: {len(bibs)}, Uncited: {len(uncited)}, Self: {len(self_cites)}')
if uncited:
    print(f'  UNCITED: {uncited}')

print()
print('=== 4. Cross-references ===')
labels = set(re.findall(r'label\{([^}]+)\}', tex))
refs = set(re.findall(r'ref\{([^}]+)\}', tex))
orphan = labels - refs
undefined = refs - labels
print(f'  Labels: {len(labels)}, Ref usages: {len(refs)}')
print(f'  Orphan labels: {orphan if orphan else "none"}')
print(f'  Undefined refs: {undefined if undefined else "none"}')

# Duplicate labels
all_labs = re.findall(r'label\{([^}]+)\}', tex)
dupes = [l for l in set(all_labs) if all_labs.count(l) > 1]
if dupes:
    print(f'  DUPLICATE: {dupes}')

print()
print('=== 5. Structure ===')
figs = len(re.findall(r'begin\{figure\}', tex))
tabs = len(re.findall(r'begin\{table\}', tex))
secs = len(re.findall(r'section\{', tex))
subs = len(re.findall(r'subsection\{', tex))
print(f'  {figs} figures, {tabs} tables, {secs} sections, {subs} subsections')

print()
print('=== 6. Content checks ===')
# Check all \ref targets
for m in re.finditer(r'ref\{([^}]+)\}', tex):
    ref_target = m.group(1)
    if ref_target not in labels:
        print(f'  UNDEFINED REF: {ref_target}')

# Check figure filenames exist
import os
fig_dir = r'C:\Users\高帅东\Desktop\evo_causal\figures'
figs_in_tex = re.findall(r'includegraphics[^}]*\{([^}]+)\}', tex)
for f in figs_in_tex:
    fp = os.path.join(fig_dir, f)
    if not os.path.exists(fp):
        print(f'  MISSING FIGURE: {f}')

print()
print('=== 7. Key data claims ===')
claims = ['0.305', '0.306', '0.222', '14/15', '1.24', '950']
for c in claims:
    found = c in tex
    print(f'  "{c}": {"found" if found else "MISSING"}')

print()
print('=== ALL CHECKS DONE ===')
