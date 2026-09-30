"""Reproduces the random samples used for human evaluation (Section 3.7).
  Step 1 (seed 2026):     100 of the 300 QA items; 100 sentiment reviews stratified by star label (50/50).
  Step 2 (seed 20260925): 60 closed-book items and, independently, 40 context-supported items drawn
                          from the 100; then 20 of the 60 closed-book items for double annotation.
The random calls are made in the same order as in the original annotation-sheet builders, so the
item indices are identical. Model responses were shown blinded and in random order; the blinding key
and the annotation sheets are not public.
Run:  python human_eval_sampling.py  ->  human_eval_sample.csv
"""
import numpy as np, pandas as pd
from load import sa

s = sa()['gpt-4o']
rng = np.random.default_rng(2026)
qi = np.sort(rng.choice(300, 100, replace=False))
pos = np.where(s.gold == 1)[0]; neg = np.where(s.gold == 0)[0]
si = np.sort(np.r_[rng.choice(pos, 50, replace=False), rng.choice(neg, 50, replace=False)])

rng = np.random.default_rng(20260925)
items = np.array(sorted(qi))
cb_items = np.sort(rng.choice(items, 60, replace=False))
cx_items = np.sort(rng.choice(items, 40, replace=False))
cb_double = np.sort(rng.choice(cb_items, 20, replace=False))

rows = [dict(task='closed-book QA', item=int(i), double_annotated=int(i in cb_double)) for i in cb_items]
rows += [dict(task='QA with context', item=int(i), double_annotated=1) for i in cx_items]
rows += [dict(task='sentiment', item=int(i), double_annotated=1) for i in si]
out = pd.DataFrame(rows)
out.to_csv('human_eval_sample.csv', index=False)
print(out.groupby('task').agg(items=('item', 'size'), double=('double_annotated', 'sum')))
print('closed-book / context overlap:', len(set(cb_items) & set(cx_items)), 'items')
print('sentiment star labels:', s.gold.iloc[si].value_counts().to_dict())
