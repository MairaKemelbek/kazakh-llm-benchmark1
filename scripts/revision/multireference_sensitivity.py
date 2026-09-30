"""Sensitivity of the QA results to multiple-reference handling (Section 4.6; Supplementary Table S1).
Three rules are compared:
  max   - score against every reference span and keep the maximum (main analysis, SQuAD approach);
  first - score against the first reference span only;
  all   - on the 27 multiple-reference questions, Containment requires every reference span to occur
          in the response; single-reference questions keep the original bidirectional rule.
Run:  python multireference_sensitivity.py  ->  multireference_sensitivity.xlsx
"""
import string
from collections import Counter
import numpy as np, pandas as pd
from load import qa, M, NM
from extract import refs
from stats import table

# Same normalization as the main scoring code (scripts/kaz_qad_experiment.py).
def norm(s):
    s = str(s).lower().replace("«", " ").replace("»", " ")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def em(p, rs):
    return int(any(norm(p) == norm(r) for r in rs))

def f1(p, rs):
    best = 0.0
    for r in rs:
        a, b = norm(p).split(), norm(r).split()
        ns = sum((Counter(a) & Counter(b)).values())
        if ns:
            pr, rc = ns / len(a), ns / len(b); best = max(best, 2 * pr * rc / (pr + rc))
    return best

def cont_both(p, rs):
    n = norm(p)
    return int(bool(n) and any((norm(r) in n) or (n in norm(r)) for r in rs))

def cont_all(p, rs):
    if len(rs) == 1:
        return cont_both(p, rs)
    n = norm(p)
    return int(bool(n) and all(norm(r) and norm(r) in n for r in rs))

rows, sig = [], []
outs = {}
for reg, t in [('ctx', 'test300'), ('cb', 'cb300')]:
    d = qa(t)
    for m in M:
        x = d[m]; P = x.raw.astype(str).tolist(); G = x.gold.map(refs).tolist()
        multi = np.array([len(g) > 1 for g in G])
        cmax = np.array([cont_both(p, g) for p, g in zip(P, G)])
        cfirst = np.array([cont_both(p, g[:1]) for p, g in zip(P, G)])
        call = np.array([cont_all(p, g) for p, g in zip(P, G)])
        emax = np.array([em(p, g) for p, g in zip(P, G)])
        efirst = np.array([em(p, g[:1]) for p, g in zip(P, G)])
        outs[(reg, 'max', m)], outs[(reg, 'first', m)], outs[(reg, 'all', m)] = cmax, cfirst, call
        rows.append(dict(regime=reg, model=NM[m],
            EM_max=round(emax.mean(), 3), EM_first=round(efirst.mean(), 3),
            F1_max=round(np.mean([f1(p, g) for p, g in zip(P, G)]), 3),
            F1_first=round(np.mean([f1(p, g[:1]) for p, g in zip(P, G)]), 3),
            Cont_max=round(cmax.mean(), 3), Cont_first=round(cfirst.mean(), 3),
            n_items_Cont_changed=int((cmax != cfirst).sum()), n_items_EM_changed=int((emax != efirst).sum()),
            Cont_on27_max=round(cmax[multi].mean(), 3), Cont_on27_first=round(cfirst[multi].mean(), 3),
            Cont_allparts=round(call.mean(), 3)))
for rule, label in [('max', 'max over refs (main)'), ('first', 'first reference only'), ('all', 'all parts required (Containment)')]:
    r = {}
    for reg in ['ctx', 'cb']:
        t = table({m: outs[(reg, rule, m)] for m in M})
        r[reg] = f"{(t.p_raw < .05).sum()}/{(t.p_holm < .05).sum()}"
        r[reg + '_pairs'] = '; '.join(t.pair[t.p_holm < .05])
    sig.append({'analysis': label, 'ctx sig raw/Holm': r['ctx'], 'cb sig raw/Holm': r['cb'],
                'cb Holm-significant pairs': r['cb_pairs']})
R = {'by_model': pd.DataFrame(rows), 'significance': pd.DataFrame(sig)}
with pd.ExcelWriter('multireference_sensitivity.xlsx') as w:
    for k, v in R.items(): v.to_excel(w, sheet_name=k, index=False)
for k, v in R.items(): print('\n##', k); print(v.to_string(index=False))
