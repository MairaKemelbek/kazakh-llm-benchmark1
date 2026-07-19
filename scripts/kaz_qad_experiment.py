# -*- coding: utf-8 -*-
"""
KazQAD эксперименті: LLM-дердің қазақша сұраққа жауап беруі (reading comprehension)
====================================================================================
Пилот:  python kaz_qad_experiment.py --data kazqad_pilot20.csv --models claude-haiku
Толық:  python kaz_qad_experiment.py --data kazqad_test300.csv
"""

import argparse
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TEMPERATURE = 0.0
MAX_TOKENS = 50
SLEEP = 4

MODELS = {
    "gpt-4o-mini":   {"provider": "openai",    "name": "gpt-4o-mini"},
    "gpt-4o":        {"provider": "openai",    "name": "gpt-4o"},
    "claude-haiku":  {"provider": "anthropic", "name": "claude-haiku-4-5-20251001"},
    "claude-sonnet": {"provider": "anthropic", "name": "claude-sonnet-4-6"},
    "gemini-flash":  {"provider": "google",    "name": "gemini-flash-lite-latest"},
}

PROMPT = (
    "Төмендегі мәтінге сүйеніп, сұраққа жауап бер.\n"
    "Жауап мәтіннің ішінен алынған ең қысқа сөз немесе сөз тіркесі болсын.\n"
    "Түсіндірме жазба, тек жауаптың өзін жаз.\n\n"
    "Мәтін: {context}\n\nСұрақ: {question}\n\nЖауап:"
)


def call_model(model_key: str, prompt: str) -> str:
    cfg = MODELS[model_key]
    p = cfg["provider"]

    if p == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=cfg["name"], temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

    if p == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=cfg["name"], temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text.strip()

    if p == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        r = genai.GenerativeModel(cfg["name"]).generate_content(
            prompt, generation_config={"temperature": TEMPERATURE,
                                       "max_output_tokens": MAX_TOKENS})
        return r.text.strip()

    raise ValueError(p)


# --- Қазақшаға бейімделген нормализация мен метрикалар ---
def normalize(s: str) -> str:
    s = str(s).lower().replace("«", " ").replace("»", " ")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def em_score(pred: str, golds: list) -> int:
    p = normalize(pred)
    return int(any(p == normalize(g) for g in golds))


def f1_score_qa(pred: str, golds: list) -> float:
    def f1(p, g):
        pt, gt = normalize(p).split(), normalize(g).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            return 0.0
        prec, rec = ns / len(pt), ns / len(gt)
        return 2 * prec * rec / (prec + rec)
    return max(f1(pred, g) for g in golds)


def containment(pred: str, golds: list) -> int:
    """Жалғамалы тіл үшін жұмсақ метрика: біреуі екіншісінің ішінде болса — дұрыс.
    Мысалы gold='Ұлтабарға', pred='ұлтабар' → дұрыс."""
    p = normalize(pred)
    if not p:
        return 0
    for g in golds:
        gn = normalize(g)
        if p in gn or gn in p:
            return 1
    return 0


def run(model_key: str, data_path: str) -> dict:
    df = pd.read_csv(data_path)
    rows = []
    for _, ex in tqdm(df.iterrows(), total=len(df), desc=model_key):
        golds = str(ex["answers"]).split("|||")
        try:
            raw = call_model(model_key, PROMPT.format(
                context=ex["context"], question=ex["question"]))
        except Exception as e:
            raw = f"ERROR: {e}"
        rows.append({"question": ex["question"], "gold": ex["answers"],
                     "raw": raw,
                     "em": em_score(raw, golds),
                     "f1": round(f1_score_qa(raw, golds), 4),
                     "contain": containment(raw, golds)})
        time.sleep(SLEEP)

    out = pd.DataFrame(rows)
    tag = Path(data_path).stem
    out.to_csv(RESULTS_DIR / f"{tag}_{model_key}.csv", index=False)

    errors = int(out["raw"].str.startswith("ERROR").sum())
    ok = out[~out["raw"].str.startswith("ERROR")]
    return {"data": tag, "model": model_key, "n": len(out),
            "exact_match": round(ok["em"].mean(), 4) if len(ok) else None,
            "f1": round(ok["f1"].mean(), 4) if len(ok) else None,
            "containment": round(ok["contain"].mean(), 4) if len(ok) else None,
            "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kazqad_test300.csv")
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    args = ap.parse_args()

    summaries = [run(m, args.data) for m in args.models]
    sdf = pd.DataFrame(summaries)
    out = RESULTS_DIR / "summary_qad.csv"
    if out.exists():
        sdf = pd.concat([pd.read_csv(out), sdf], ignore_index=True)
    sdf.to_csv(out, index=False)
    print("\n===== ҚОРЫТЫНДЫ =====")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
