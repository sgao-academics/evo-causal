with open(r'C:\Users\高帅东\Desktop\evo_causal\paper\evo_causal_paper.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Replace filenames
tex = tex.replace('fig1_phase_heatmap.pdf', 'fig1_phylogenetic_wall.pdf')
tex = tex.replace('fig3_head2head.pdf', 'fig3_method_landscape.pdf')

# Update Figure 1 caption
old1 = r'\caption{\textbf{The phylogenetic wall across five methods.} Heatmaps show $\text{F}_1$ as a function of variable dimension $d$ (rows) and phylogenetic signal strength $\alpha$ (columns). The red dashed line marks $\alpha=0.05$, above which all methods yield $\text{F}_1=0$. Non-zero cells are annotated. PIC+correlation achieves the highest $\text{F}_1=0.305$ at $d=30,\alpha=0.02$, but collapses at $\alpha\ge0.05$ for all dimensions. CT and NOTEARS produce non-zero F$_1$ in fewer than 5\% of all configurations.}'
new1 = r'\caption{\textbf{The phylogenetic wall across all five methods.} Each panel shows F$_{1}$ as a function of variable dimension $d$ (rows) and phylogenetic signal strength $\alpha$ (columns). The dashed line marks the wall at $\alpha=0.05$; non-zero cells are annotated. (A) PIC+correlation achieves the highest F$_{1}=0.305$ at $d=30,\alpha=0.02$ but collapses at $\alpha\ge0.05$. (B--E) CT, NOTEARS, DAGMA, and GOLEM produce non-zero F$_{1}$ in fewer than 12\% of configurations. All five methods yield F$_{1}=0$ above $\alpha=0.05$ for $d\ge50$.}'
tex = tex.replace(old1, new1)

# Update Figure 2 caption
old2 = r'\caption{\textbf{CT failure is not due to vanishing gradients, insufficient capacity, or premature stopping.} (A) Gradient norms remain $\approx0.04$ across all $\alpha$, while the DAG constraint $h(W)$ converges reliably. Gradients do not vanish---edges simply never lift above threshold. (B) All six architecture variants ($d_{\text{model}}\in\{32,64,128\}$, $L\in\{2,4\}$) yield $\text{F}_1=0$, ruling out capacity limitations. (C) Extending training to 3,000 epochs reduces $h(W)$ from $2\times10^{-4}$ to $8\times10^{-5}$ but produces zero edges throughout.}'
new2 = r'\caption{\textbf{Anatomy of CT failure.} (A) Gradient norms $|\nabla W|$ remain stable at $\approx0.04$ across all $\alpha$, while the DAG constraint $h(W)$ converges to $\sim2\times10^{-4}$---gradients persist but edges never lift above threshold. (B) All six architecture variants ($d_{\text{model}}\in\{32,64,128\}$, $L\in\{2,4\}$) yield F$_{1}=0$, ruling out capacity as the limiting factor. (C) Convergence curves show $h(W)$ reaching $8\times10^{-5}$ at 3,000 epochs with zero edges throughout (inset). (D) Sample size scaling from $n=50$ to $n=800$ at $d=100,\alpha=0.15$ does not rescue CT; wall persists for both methods.}'
tex = tex.replace(old2, new2)

# Update Figure 3 caption
old3 = r'\caption{\textbf{Head-to-head comparison and sample size scaling.} (A) PIC+correlation thresholding wins 14 of 15 head-to-head configurations; all other methods (NOTEARS, GOLEM, CT, DAGMA) win at most 1. The dashed line marks chance level (3/15). (B) Sample size scaling from $n=50$ to $n=800$ at $d=100,\alpha=0.15$ does not rescue CT: PIC+correlation achieves $\text{F}_1\approx0.006$ while CT remains at $\text{F}_1=0$ across all sample sizes.}'
new3 = r'\caption{\textbf{Method comparison, wall boundaries, and robustness.} (A) Head-to-head wins: PIC+corr wins 14 of 15 configurations; all other methods win at most 1 (chance: 3/15). (B) Wall penetration boundary at $d=50$: F$_{1}>0.1$ (blue) requires $n\ge200$ \textit{and} $\alpha\le0.01$. (C) Nonlinear robustness: PIC+corr F$_{1}$ across linear, MLP, and sigmoid data generators at five key configurations---the wall conclusion is consistent across all generators. (D) Best F$_{1}$ across all 40 phase-diagram configurations for each method; all fall below the actionable threshold.}'
tex = tex.replace(old3, new3)

with open(r'C:\Users\高帅东\Desktop\evo_causal\paper\evo_causal_paper.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
print('Figure refs + captions updated')
