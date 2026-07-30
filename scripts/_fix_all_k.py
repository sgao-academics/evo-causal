import re

path = r'C:\Users\高帅东\Desktop\evo_causal\paper\evo_causal_paper.tex'
with open(path, 'r', encoding='utf-8') as f:
    tex = f.read()

lines = tex.split('\n')
fixed = 0

for i in range(len(lines)):
    line = lines[i]
    if 'bibitem[' in line:
        continue
    
    orig = line
    
    # Pattern 1: $K$\alpha -> $\alpha (stray K in math mode)
    line = line.replace('$K$\\alpha', '$\\alpha')
    line = line.replace('$K$\\alpha=', '$\\alpha=')
    
    # Pattern 2: $K= -> $\alpha=
    line = line.replace('$K=0', '$\\alpha=0')
    line = line.replace('$K = 0', '$\\alpha = 0')
    line = line.replace('$K =', '$\\alpha =')
    
    # Pattern 3: $K\in -> $\alpha\in  
    line = line.replace('$K \\in', '$\\alpha \\in')
    
    # Pattern 4: $K\approx -> $\alpha\approx
    line = line.replace('$K \\approx', '$\\alpha \\approx')
    
    # Pattern 5: $K\ge -> $\alpha\ge
    line = line.replace('$K \\ge', '$\\alpha \\ge')
    line = line.replace('$K \\geq', '$\\alpha \\geq')
    line = line.replace('$K >', '$\\alpha >')
    
    # Pattern 6: Delta K -> Delta\alpha
    line = line.replace('Delta K', 'Delta\\alpha')
    
    # Pattern 7: function of $K$ -> function of $\alpha$
    line = line.replace('function of $K$', 'function of $\\alpha$')
    
    # Pattern 8: between $K$ and -> between $\alpha$ and
    line = line.replace('between $K$ and', 'between $\\alpha$ and')
    
    # Pattern 9: K$\alpha=0.15$ -> $\alpha=0.15$
    # (handled by pattern 1)
    
    if line != orig:
        fixed += 1
        print(f'L{i+1}: {orig.strip()[:100]}')
        print(f'  -> {line.strip()[:100]}')
    
    lines[i] = line

new_tex = '\n'.join(lines)
with open(path, 'w', encoding='utf-8') as f:
    f.write(new_tex)

print(f'\nTotal fixes: {fixed}')
