# -*- coding: utf-8 -*-
"""
Қазақша sentiment-эксперимент: LLM-дерді KazSAnDRA датасетінде салыстыру
=========================================================================
Орнату:   pip install openai anthropic google-generativeai scikit-learn pandas tqdm
Пилот:    python kaz_experiment.py --data kazsandra_pilot20.csv --models claude-haiku
Толық:    python kaz_experiment.py --data kazsandra_test300.csv
Тексеру:  python kaz_experiment.py --data kazsandra_pilot20.csv --models dry-run
"""

import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TEMPERATURE = 0.0
MAX_TOKENS = 10   # бір сөз ғана керек
SLEEP = 0.3

MODELS = {
    "dry-run":       {"provider": "mock"},   # API-сыз тексеру режимі
    "gpt-4o-mini":   {"provider": "openai",     "name": "gpt-4o-mini"},
    "gpt-4o":        {"provider": "openai",     "name": "gpt-4o"},
    "claude-haiku":  {"provider": "anthropic",  "name": "claude-haiku-4-5-20251001"},
    "claude-sonnet": {"provider": "anthropic",  "name": "claude-sonnet-4-6"},
    "gemini-flash":  {"provider": "google",     "name": "gemini-2.0-flash"},
    "llama-3.1-8b":  {"provider": "openrouter", "name": "meta-llama/llama-3.1-8b-instruct"},
    "qwen-2.5-7b":   {"provider": "openrouter", "name": "qwen/qwen-2.5-7b-instruct"},
}

# Барлық модельге бірдей қазақша промпт (әдістемелік талап)
PROMPT = (
    "Төмендегі пікірдің көңіл-күйін (сентиментін) анықта.\n"
    "Жауап ретінде тек бір сөз жаз: позитив немесе негатив.\n\n"
    "Пікір: {text}\n\nЖауап:"
)


def call_model(model_key: str, prompt: str) -> str:
    cfg = MODELS[model_key]
    p = cfg["provider"]

    if p == "mock":  # кездейсоқ жауап — тек конвейерді тексеру үшін
        return random.choice(["позитив", "негатив"])

    if p in ("openai", "openrouter"):
        from openai import OpenAI
        client = OpenAI() if p == "openai" else OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"])
        r = client.chat.completions.create(
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


def parse_answer(answer: str) -> str:
    """Модель жауабын 1/0 label-ге айналдыру (қазақша да, ағылшынша да)."""
    a = answer.lower()
    if "позитив" in a or "positive" in a or "оң" in a:
        return 1
    if "негатив" in a or "negative" in a or "теріс" in a:
        return 0
    return -1  # түсініксіз жауап


def run(model_key: str, data_path: str) -> dict:
    df = pd.read_csv(data_path)
    rows = []
    for _, ex in tqdm(df.iterrows(), total=len(df), desc=model_key):
        try:
            raw = call_model(model_key, PROMPT.format(text=ex["text"]))
        except Exception as e:
            raw = f"ERROR: {e}"
        rows.append({"text": ex["text"], "domain": ex["domain"],
                     "gold": ex["label"], "raw": raw,
                     "pred": parse_answer(raw)})
        if MODELS[model_key]["provider"] != "mock":
            time.sleep(SLEEP)

    out = pd.DataFrame(rows)
    tag = Path(data_path).stem
    out.to_csv(RESULTS_DIR / f"{tag}_{model_key}.csv", index=False)

    ok = out[out["pred"] != -1]
    summary = {
        "data": tag, "model": model_key, "n": len(out),
        "accuracy": round(accuracy_score(ok["gold"], ok["pred"]), 4) if len(ok) else None,
        "f1_macro": round(f1_score(ok["gold"], ok["pred"], average="macro"), 4) if len(ok) else None,
        "parse_fail": int((out["pred"] == -1).sum()),
        "errors": int(out["raw"].str.startswith("ERROR").sum()),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kazsandra_test300.csv")
    ap.add_argument("--models", nargs="*",
                    default=[m for m in MODELS if m != "dry-run"])
    args = ap.parse_args()

    summaries = [run(m, args.data) for m in args.models]
    sdf = pd.DataFrame(summaries)
    out = RESULTS_DIR / "summary.csv"
    if out.exists():
        sdf = pd.concat([pd.read_csv(out), sdf], ignore_index=True)
    sdf.to_csv(out, index=False)
    print("\n===== ҚОРЫТЫНДЫ =====")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
