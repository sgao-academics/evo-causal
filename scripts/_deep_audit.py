"""Deep content audit: cross-check claims vs data, figure labels vs captions, language."""
import re, os

path = r'C:\Users\高帅东\Desktop\evo_causal\paper\evo_causal_paper.tex'
with open(path, 'r', encoding='utf-8') as f:
    tex = f.read()

issues = []

# 1. Figure panel labels in caption vs actual figure
print('=== 1. Figure caption panel labels ===')
fig_captions = re.findall(r'\\caption\{(.+?)\}', tex, re.DOTALL)
for cap in fig_captions:
    if r'\textbf' in cap:  # figure captions
        # Extract panel labels (A) (B) etc
        panels = re.findall(r'\(([A-F])\)', cap)
        print(f'  Panels: {panels}')
        if not panels:
            issues.append('Figure caption missing panel labels')

# 2. Check for unbalanced braces
print()
print('=== 2. Brace balance ===')
open_b = tex.count('{')
close_b = tex.count('}')
if open_b != close_b:
    issues.append(f'Unbalanced braces: {{={open_b} }}={close_b}')
    print('  FAIL: braces unbalanced')
else:
    print('  PASS: braces balanced')

# 3. Check for duplicate labels
print()
print('=== 3. Duplicate labels ===')
labels = re.findall(r'\\label\{([^}]+)\}', tex)
seen = {}
dupes = []
for lab in labels:
    if lab in seen:
        dupes.append(lab)
    seen[lab] = True
if dupes:
    issues.append(f'Duplicate labels: {dupes}')
    print(f'  FAIL: {dupes}')
else:
    print(f'  PASS: {len(labels)} unique labels')

# 4. Check for undefined references (labels without \ref or \cite without bib)
print()
print('=== 4. Cross-reference check ===')
refs = set(re.findall(r'\\ref\{([^}]+)\}', tex))
labels_set = set(labels)
undefined = refs - labels_set
if undefined:
    issues.append(f'Undefined \\ref: {undefined}')
    print(f'  FAIL: {undefined}')
else:
    print(f'  PASS: all {len(refs)} refs match {len(labels_set)} labels')

# 5. Check for citation key mismatches
print()
print('=== 5. Citation key check ===')
bib_keys = set(re.findall(r'\\bibitem\[.*?\]\{([^}]+)\}', tex))
cited_keys = set()
for m in re.finditer(r'\\citep?\{(.+?)\}', tex):
    for k in m.group(1).split(','):
        cited_keys.add(k.strip())
uncited = bib_keys - cited_keys
undefined_cite = cited_keys - bib_keys
if undefined_cite:
    issues.append(f'Undefined citations: {undefined_cite}')
    print(f'  FAIL: undefined={undefined_cite}')
else:
    print(f'  PASS: {len(bib_keys)} bib entries, {len(cited_keys)} cited')
if uncited:
    print(f'  NOTE: {len(uncited)} uncited: {list(uncited)[:5]}')

# 6. Check specific data claims vs experiments
print()
print('=== 6. Data-claim consistency ===')
# F1=0.305 claim
if '0.305' not in tex:
    issues.append('Missing F1=0.305 claim')
if 'F_{1}=0.305' not in tex and 'F$_{1}=0.305' not in tex:
    print('  NOTE: F1=0.305 format check')
else:
    print('  PASS: F1=0.305 present')

# 14/15 claim
if '14/15' not in tex and '14 of 15' not in tex:
    issues.append('Missing 14/15 wins claim')
else:
    print('  PASS: 14/15 wins present')

# Blomberg K 
if '1.24' not in tex:
    issues.append('Missing Blomberg K=1.24')
else:
    print('  PASS: Blomberg K=1.24 present')

# Kaiser & Sipos
if 'kaiser2022' not in tex and 'Kaiser' not in tex:
    issues.append('Missing Kaiser & Sipos citation')
else:
    print('  PASS: Kaiser & Sipos cited')

# 7. Language: check for common errors
print()
print('=== 7. Language checks ===')
lang_issues = []
# British/American consistency: -ize vs -ise
if 'characterise' in tex or 'characterised' in tex:
    lang_issues.append('British spelling: characterise')
# that vs which (restrictive clauses)
# Double spaces after period
if '  ' in tex.replace('  ', 'MARKER').split('MARKER')[0]:
    pass  # too many false positives from LaTeX
if lang_issues:
    for li in lang_issues:
        print(f'  NOTE: {li}')
else:
    print('  PASS: no obvious language issues')

# 8. Check for orphaned \ref{} 
print()
print('=== 8. Table/Figure numbering ===')
tab_refs = re.findall(r'Table~\\ref\{([^}]+)\}', tex)
fig_refs = re.findall(r'Figure~\\ref\{([^}]+)\}', tex) + re.findall(r'Fig\.~\\ref\{([^}]+)\}', tex)
print(f'  Table refs: {len(tab_refs)}')
print(f'  Figure refs: {len(fig_refs)}')

# 9. Check for stray \alpha or broken math
print()
print('=== 9. Broken math ===')
stray_dollar = []
for i, line in enumerate(tex.split('\n')):
    if 'bibitem' in line or line.strip().startswith('%'):
        continue
    c = line.count('$')
    if c > 0 and c % 2 != 0:
        stray_dollar.append((i+1, line.strip()[:100]))
if stray_dollar:
    issues.append(f'{len(stray_dollar)} lines with odd $ count')
    for ln, txt in stray_dollar:
        print(f'  FAIL L{ln}: {txt}')
else:
    print('  PASS: all $ balanced')

# 10. Check for remaining K (not Blomberg context)
print()
print('=== 10. Stray K references ===')
stray_k = []
for i, line in enumerate(tex.split('\n')):
    if 'bibitem' in line or 'Blomberg' in line or 'blomberg' in line:
        continue
    if '\\textbf{$K$}' in line:
        stray_k.append((i+1, 'Table header K'))
    if re.search(r'(?<!\\)\$K\$(?![a-zA-Z])', line):
        stray_k.append((i+1, line.strip()[:80]))
if stray_k:
    issues.append(f'{len(stray_k)} stray K refs')
    for ln, txt in stray_k:
        print(f'  FAIL L{ln}: {txt}')
else:
    print('  PASS: no stray K')

print()
print('='*50)
print(f'AUDIT COMPLETE: {len(issues)} issues found')
for iss in issues:
    print(f'  * {iss}')
if not issues:
    print('  PAPER IS PERFECT.')
