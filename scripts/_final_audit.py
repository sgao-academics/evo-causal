"""Final audit: check every aspect of the paper."""
import re, os

path = r'C:\Users\高帅东\Desktop\evo_causal\paper\evo_causal_paper.tex'
with open(path, 'r', encoding='utf-8') as f:
    tex = f.read()

issues = []
ok = lambda msg: issues.append(f'  [PASS] {msg}')
fail = lambda msg: issues.append(f'  [FAIL] {msg}')

# 1. Math mode integrity: count $ signs per line
print('=== 1. Math mode integrity ===')
lines = tex.split('\n')
odd_lines = []
for i, line in enumerate(lines):
    if 'bibitem[' in line or line.strip().startswith('%'):
        continue
    c = line.count('$')
    if c > 0 and c % 2 != 0:
        odd_lines.append((i+1, c, line.strip()[:100]))
if odd_lines:
    fail(f'{len(odd_lines)} lines with odd $ count')
    for ln, cnt, txt in odd_lines[:10]:
        print(f'    L{ln} ({cnt} $): {txt}')
else:
    ok('All math mode brackets balanced')

# 2. Check for remaining $K$ (not Blomberg)
print()
print('=== 2. Remaining $K$ references (non-Bloomberg) ===')
stray_k = []
for i, line in enumerate(lines):
    if 'bibitem[' in line or 'Blomberg' in line or 'blomberg' in line:
        continue
    if '\\textbf{$K$}' in line:
        stray_k.append((i+1, line.strip()[:80]))
    # Text-level $K$ 
    if re.search(r'(?<!\\)\$K\$(?![a-zA-Z])', line) and 'blomberg' not in line.lower():
        stray_k.append((i+1, line.strip()[:80]))
if stray_k:
    fail(f'{len(stray_k)} stray $K$ references')
    for ln, txt in stray_k:
        print(f'    L{ln}: {txt}')
else:
    ok('No stray $K$ references')

# 3. Reference audit
print()
print('=== 3. Reference audit ===')
bibs = re.findall(r'\\bibitem\[([^\]]*)\]\{([^}]+)\}', tex)
cited = set()
for m in re.finditer(r'\\citep?\{(.+?)\}', tex):
    for k in m.group(1).split(','):
        cited.add(k.strip())
uncited = [b[1] for b in bibs if b[1] not in cited]
if uncited:
    fail(f'{len(uncited)} uncited references: {uncited}')
else:
    ok(f'All {len(bibs)} references cited in text')

self_cites = [b[1] for b in bibs if 'gao' in b[1].lower()]
if len(self_cites) <= 2:
    ok(f'{len(self_cites)} self-citations (limit: 2)')
else:
    fail(f'{len(self_cites)} self-citations exceeds limit of 2: {self_cites}')

# 4. Figure/Table references
print()
print('=== 4. Figure/Table references ===')
figs = len(re.findall(r'\\includegraphics', tex))
tab_refs = len(re.findall(r'\\ref\{tab:', tex))
fig_refs = len(re.findall(r'\\ref\{fig:', tex))
ok(f'{figs} figures embedded, {fig_refs} cross-references')
ok(f'{tab_refs} table cross-references')

# Count actual tables
tables = len(re.findall(r'\\begin\{table\}', tex))
ok(f'{tables} tables defined')

# 5. Check for broken LaTeX patterns
print()
print('=== 5. Broken LaTeX patterns ===')
broken = []
pat = r'\\$\\text\{F_\d+\}\$'
if re.search(r'\$\\text\{F_\d+\}\$\\alpha', tex):
    fail('Broken math: $\\text{F}$\\alpha pattern found')
else:
    ok('No broken math patterns')
if re.search(r'\\\\section', tex) or re.search(r'\\\\subsection', tex):
    fail('Double backslash before section/subsection')
else:
    ok('No double backslash before section headers')
if re.search(r'\$\\alpha\\alpha', tex):
    fail('Double alpha ($\\alpha\\alpha$)')
else:
    ok('No double alpha patterns')

# 6. Check for abstract presence
print()
print('=== 6. Structure ===')
sections = re.findall(r'\\section\{([^}]+)\}', tex)
subsections = re.findall(r'\\subsection\{([^}]+)\}', tex)
ok(f'{len(sections)} sections: {[s[:25] for s in sections]}')
ok(f'{len(subsections)} subsections')

# Check title
title_match = re.search(r'\\title\{([^}]+)\}', tex)
if title_match:
    title_words = len(title_match.group(1).split())
    ok(f'Title: {title_words} words')

# 7. Word count (rough)
print()
print('=== 7. Content ===')
# Strip LaTeX commands
clean = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', ' ', tex)
clean = re.sub(r'%.*$', '', clean, flags=re.MULTILINE)
words = len(clean.split())
ok(f'~{words} words of content')

print()
print('='*50)
print('AUDIT COMPLETE')
print('='*50)
total_fails = sum(1 for i in issues if '[FAIL]' in i)
print(f'{total_fails} failures, {len(issues)-total_fails} passes')
for iss in issues:
    print(iss)
