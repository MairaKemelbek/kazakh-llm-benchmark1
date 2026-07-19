# -*- coding: utf-8 -*-
"""
Метрика салыстыру — ШЫНАЙЫ құралдармен
======================================
  EM         — SQuAD стандарты (Rajpurkar et al., 2016)
  LemmaEM    — Stanza лемматизаторымен (Qi et al., 2020) — Maxutov et al. (2024) қолданған тәсіл
  Token-F1   — SQuAD стандарты
  chrF       — sacrebleu, Popović (2015) ресми имплементациясы
  Containment — біздің метрика

Colab:
  !pip install -q sacrebleu stanza
  !unzip -o results_backup_ALL.zip
  !python metric_compare_real.py

Ескерту: Stanza қазақ моделін бір рет жүктейді (~1-2 мин).
"""

import string
from collections import Counter
from pathlib import Path
import pandas as pd

RESULTS = Path("results")
MODELS = ["gpt-4o", "claude-sonnet", "gemini-flash", "gpt-4o-mini", "claude-haiku"]
LABELS = {"gpt-4o": "GPT-4o", "claude-sonnet": "Claude Sonnet",
          "gemini-flash": "Gemini Flash", "gpt-4o-mini": "GPT-4o-mini",
          "claude-haiku": "Claude Haiku"}


def normalize(s):
    s = str(s).lower().replace("«", " ").replace("»", " ").replace("—", " ")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


# ---------- Stanza лемматизаторы ----------
_nlp = None
_cache = {}

def get_nlp():
    global _nlp
    if _nlp is None:
        import stanza
        try:
            stanza.download("kk", verbose=False)
        except Exception as e:
            print("Stanza жүктеу қатесі:", e)
        _nlp = stanza.Pipeline("kk", processors="tokenize,lemma",
                               verbose=False, use_gpu=False)
    return _nlp


def lemmatize(text):
    t = normalize(text)
    if not t:
        return ""
    if t in _cache:
        return _cache[t]
    doc = get_nlp()(t)
    out = " ".join(w.lemma if w.lemma else w.text
                   for sent in doc.sentences for w in sent.words)
    out = normalize(out)
    _cache[t] = out
    return out


# ---------- Метрикалар ----------
def em(pred, golds):
    p = normalize(pred)
    return int(any(p == normalize(g) for g in golds))


def lemma_em(pred, golds):
    p = lemmatize(pred)
    return int(any(p == lemmatize(g) for g in golds))


def token_f1(pred, golds):
    def f1(p, g):
        pt, gt = normalize(p).split(), normalize(g).split()
        if not pt or not gt:
            return 0.0
        c = Counter(pt) & Counter(gt); ns = sum(c.values())
        if ns == 0:
            return 0.0
        prec, rec = ns/len(pt), ns/len(gt)
        return 2*prec*rec/(prec+rec)
    return max(f1(pred, g) for g in golds)


def chrf_official(pred, golds):
    """Popović (2015) — sacrebleu ресми имплементациясы (chrF, n=6, beta=2)."""
    from sacrebleu.metrics import CHRF
    global _chrf
    try:
        _chrf
    except NameError:
        _chrf = CHRF()
    return _chrf.sentence_score(normalize(pred),
                                [normalize(g) for g in golds]).score / 100.0


def containment(pred, golds):
    p = normalize(pred)
    if not p:
        return 0
    return int(any(p in normalize(g) or normalize(g) in p for g in golds))


def main():
    rows = []
    for m in MODELS:
        f = RESULTS / f"kazqad_test300_{m}.csv"
        if not f.exists():
            print(f"! жоқ: {f}"); continue
        d = pd.read_csv(f)
        d = d[~d.raw.astype(str).str.startswith("ERROR")]
        preds = d.raw.astype(str).tolist()
        golds = [str(g).split("|||") for g in d.gold]
        n = len(d)
        print(f"{m}: {n} жауап өңделуде...")

        rows.append({
            "Model": LABELS[m],
            "EM":          round(sum(em(p,g)        for p,g in zip(preds,golds))/n, 3),
            "LemmaEM":     round(sum(lemma_em(p,g)  for p,g in zip(preds,golds))/n, 3),
            "Token-F1":    round(sum(token_f1(p,g)  for p,g in zip(preds,golds))/n, 3),
            "chrF":        round(sum(chrf_official(p,g) for p,g in zip(preds,golds))/n, 3),
            "Containment": round(sum(containment(p,g)   for p,g in zip(preds,golds))/n, 3),
        })

    t = pd.DataFrame(rows)
    t.to_csv("metric_table_real.csv", index=False)
    print("\n=== Бес критерий × бес модель (KazQAD, n=300) ===\n")
    print(t.to_string(index=False))
    print("\nОрташа алшақтық:")
    print("  Containment − EM      :", round((t.Containment - t.EM).mean(), 3))
    print("  LemmaEM    − EM       :", round((t.LemmaEM - t.EM).mean(), 3),
          "  ← лемматизация нені қайтарады")
    print("  Containment − LemmaEM :", round((t.Containment - t.LemmaEM).mean(), 3),
          "  ← лемматизациядан кейін де қалатыны")
    print("  Containment − chrF    :", round((t.Containment - t["chrF"]).mean(), 3))


if __name__ == "__main__":
    main()
