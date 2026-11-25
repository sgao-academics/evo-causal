# -*- coding: utf-8 -*-
"""Screen statistic for the arm that raises K without a confounder.

Written to its own file so it cannot race the main run writing e15_screen.json.
"""
import os, sys, json
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(S), 'scripts'))
sys.path.insert(0, S)
from exp_new import gen, lead_ratio, OUT

name = 'e15b_screen_signal'
res = {}
nulls = {}
for d in (30, 50):
    for lev in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        rs = []
        for sd in range(10):
            X, W, tree, C = gen(200, d, 10, 0.0, sd, trait_signal=lev)
            rs.append(lead_ratio(__import__('run_revision').pic_of(X, tree))[0])
        key = 'signal_d%d_lv%.1f' % (d, lev)
        res[key] = dict(latent='signal', d=d, level=lev, seeds=10,
                        r=round(float(np.mean(rs)), 3),
                        r_sd=round(float(np.std(rs)), 3))
        print('  %-16s r=%.3f' % (key, res[key]['r']), flush=True)

json.dump(res, open(os.path.join(OUT, name + '.json'), 'w', encoding='utf-8'), indent=1)
print('saved', name)
