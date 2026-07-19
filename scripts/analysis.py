# -*- coding: utf-8 -*-
"""
Барлық статистика мен кестелерді қайта құру
============================================
Мақаладағы Table 2, 4, 6, 7 және Fig. 2–5 үшін қажет барлық есептеу.

    python scripts/analysis.py

Алдын ала үш режим де жүргізілген болуы керек (results/ ішінде):
    kazsandra_test300_<model>.csv
    kazqad_test300_<model>.csv
    kazqad_cb300_<model>.csv
"""

import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import accuracy_score, f1_score

RESULTS = Path("results")
MODELS = ["gpt-4o", "claude-sonnet", "gemini-flash", "gpt-4o-mini", "claude-haiku"]
LABELS = {"gpt-4o": "GPT-4o", "claude-sonnet": "Claude Sonnet",
          "gemini-flash": "Gemini Flash", "gpt-4o-mini": "GPT-4o-mini",
          "claude-haiku": "Claude Haiku"}


# ---------------------------------------------------------------- helpers
def holm(pvals):
    """Holm–Bonferroni step-down түзетуі."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, prev = len(pvals), [0.0]*len(pvals), 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, pvals[i] * (m - rank))
        v = max(v, prev); prev = v
        adj[i] = v
    return adj


def odds_ratio_ci(b, c, conf=0.95):
    """OR = b/c + Clopper–Pearson дәл сенімділік аралығы."""
    n = b + c
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    lo_p, hi_p = binomtest(b, n, 0.5).proportion_ci(confidence_level=conf, method="exact")
    orr = b / c if c else float("inf")
    lo = lo_p / (1 - lo_p) if lo_p < 1 else float("inf")
    hi = hi_p / (1 - hi_p) if hi_p < 1 else float("inf")
    return orr, (lo, hi)


def load_correct(stage, model):
    """Әр элемент бойынша дұрыс/қате векторын қайтарады."""
    if stage == "sentiment":
        d = pd.read_csv(RESULTS / f"kazsandra_test300_{model}.csv")
        d = d[d.pred != -1]
        return d.set_index("text").apply(lambda r: int(r.gold == r.pred), axis=1)
    tag = "kazqad_test300" if stage == "qa_context" else "kazqad_cb300"
    d = pd.read_csv(RESULTS / f"{tag}_{model}.csv")
    d = d[~d.raw.astype(str).str.startswith("ERROR")]
    return d.set_index("question")["contain"]


# ---------------------------------------------------------------- tables
def table_main():
    print("\n" + "="*78)
    print("TABLE 2 — Main results (5 models x 3 regimes)")
    print("="*78)
    rows = []
    for m in MODELS:
        d = pd.read_csv(RESULTS / f"kazsandra_test300_{m}.csv")
        ok = d[d.pred != -1]
        rc = pd.read_csv(RESULTS / f"kazqad_test300_{m}.csv")
        cb = pd.read_csv(RESULTS / f"kazqad_cb300_{m}.csv")
        rows.append({
            "Model": LABELS[m],
            "Sent_Acc": round(accuracy_score(ok.gold, ok.pred), 3),
            "Sent_F1": round(f1_score(ok.gold, ok.pred, average="macro"), 3),
            "QActx_EM": round(rc.em.mean(), 3),
            "QActx_Cont": round(rc.contain.mean(), 3),
            "QAcb_EM": round(cb.em.mean(), 3),
            "QAcb_Cont": round(cb.contain.mean(), 3),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv("table2_main.csv", index=False)
    return t


def mcnemar_regime(stage, label):
    print("\n" + "-"*78)
    print(f"McNemar — {label}")
    print("-"*78)
    vec = {m: load_correct(stage, m) for m in MODELS}
    pairs, ps = [], []
    for a, b in itertools.combinations(MODELS, 2):
        j = pd.concat([vec[a], vec[b]], axis=1, join="inner")
        j.columns = ["a", "b"]
        n_a = int(((j.a == 1) & (j.b == 0)).sum())
        n_b = int(((j.a == 0) & (j.b == 1)).sum())
        p = binomtest(n_a, n_a + n_b, 0.5).pvalue if (n_a + n_b) else 1.0
        pairs.append((a, b, n_a, n_b)); ps.append(p)

    adj = holm(ps)
    rows = []
    for (a, b, n_a, n_b), p, pa in zip(pairs, ps, adj):
        orr, (lo, hi) = odds_ratio_ci(n_a, n_b)
        rows.append({
            "Pair": f"{LABELS[a]} vs. {LABELS[b]}",
            "b/c": f"{n_a}/{n_b}",
            "OR": round(orr, 2),
            "CI95": f"[{lo:.2f}, {hi:.2f}]",
            "p_raw": round(p, 4),
            "p_Holm": round(pa, 4),
            "sig": "*" if pa < 0.05 else "",
        })
    t = pd.DataFrame(rows).sort_values("OR", ascending=False)
    print(t.to_string(index=False))
    n_sig_raw = sum(p < .05 for p in ps)
    n_sig_adj = sum(p < .05 for p in adj)
    print(f"\n  significant: raw {n_sig_raw}/10, Holm-adjusted {n_sig_adj}/10")
    return t, n_sig_raw, n_sig_adj


def table_regimes(counts):
    print("\n" + "="*78)
    print("TABLE 6 — Significant pairs per regime")
    print("="*78)
    t = pd.DataFrame([
        {"Regime": "Sentiment (understanding)", "raw": f"{counts['sentiment'][0]} / 10",
         "Holm": f"{counts['sentiment'][1]} / 10"},
        {"Regime": "Reading-comprehension QA", "raw": f"{counts['qa_context'][0]} / 10",
         "Holm": f"{counts['qa_context'][1]} / 10"},
        {"Regime": "Closed-book QA (knowledge)", "raw": f"{counts['qa_closed'][0]} / 10",
         "Holm": f"{counts['qa_closed'][1]} / 10"},
    ])
    print(t.to_string(index=False))
    t.to_csv("table6_regimes.csv", index=False)


def rank_correlations(t):
    print("\n" + "="*78)
    print("Spearman rank correlations between regimes")
    print("="*78)
    axes = {"Sentiment": t.Sent_Acc.tolist(),
            "QA-context": t.QActx_Cont.tolist(),
            "QA-closed": t.QAcb_Cont.tolist()}
    for (n1, v1), (n2, v2) in itertools.combinations(axes.items(), 2):
        rho, p = spearmanr(v1, v2)
        print(f"  {n1:12s} vs {n2:12s}: rho = {rho:+.2f}  (p = {p:.3f})")

    print("\n  Context dividend (QA-context − QA-closed), by model:")
    for i, m in enumerate(t.Model):
        print(f"    {m:15s} {t.QActx_Cont[i] - t.QAcb_Cont[i]:+.3f}")


if __name__ == "__main__":
    t = table_main()
    counts = {}
    for stage, label in [("sentiment", "Sentiment (accuracy)"),
                         ("qa_context", "QA with context (Containment)"),
                         ("qa_closed", "QA closed-book (Containment)")]:
        tbl, raw, adj = mcnemar_regime(stage, label)
        counts[stage] = (raw, adj)
        if stage == "qa_closed":
            tbl.to_csv("table7_mcnemar_closedbook.csv", index=False)
    table_regimes(counts)
    rank_correlations(t)
    print("\nDone. CSVs written to the current directory.")
