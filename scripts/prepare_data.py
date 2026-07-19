# -*- coding: utf-8 -*-
"""
Іріктемелерді ресми дереккөздерден қайта жасау (seed=42)
=========================================================
Датасеттер қайта таратылмайды — бұл скрипт оларды ресми көзден жүктеп,
мақалада қолданылған дәл сол іріктемелерді қайта құрады.

    python scripts/prepare_data.py

Шығатыны:
    kazsandra_test300.csv   — 150 позитив / 150 негатив (теңгерілген)
    kazqad_test300.csv      — 300 сұрақ (контекстімен)

KazQAD үшін HF_TOKEN керек әрі датасетке қолжетімділік алынған болуы тиіс:
    https://huggingface.co/datasets/issai/kazqad
"""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

SEED = 42
KAZSANDRA_URL = ("https://raw.githubusercontent.com/IS2AI/KazSAnDRA/"
                 "main/dataset/05_pc_test.zip")


def prepare_kazsandra(out="kazsandra_test300.csv"):
    print("KazSAnDRA: ресми репозиторийден жүктелуде...")
    r = requests.get(KAZSANDRA_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(name))
    print(f"  толық тест сплиті: {len(df)} пікір")

    pos = df[df.label == 1].sample(150, random_state=SEED)
    neg = df[df.label == 0].sample(150, random_state=SEED)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.to_csv(out, index=False)
    print(f"  ✓ {out}: {len(sample)} ({(sample.label==1).sum()} поз / "
          f"{(sample.label==0).sum()} нег)")


def prepare_kazqad(out="kazqad_test300.csv"):
    from datasets import load_dataset
    if not os.environ.get("HF_TOKEN"):
        print("! HF_TOKEN орнатылмаған — KazQAD gated датасет.")
        print("  https://huggingface.co/datasets/issai/kazqad бетінен рұқсат алыңыз.")
        return
    print("KazQAD: Hugging Face-тен жүктелуде...")
    ds = load_dataset("issai/kazqad", split="test")
    print(f"  толық тест сплиті: {len(ds)} сұрақ")

    sample = ds.shuffle(seed=SEED).select(range(300))
    rows = [{"id": ex["id"],
             "question": ex["question"],
             "context": ex["context"],
             "answers": "|||".join(ex["answers"]["text"])} for ex in sample]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  ✓ {out}: {len(rows)} сұрақ")


if __name__ == "__main__":
    prepare_kazsandra()
    prepare_kazqad()
    print("\nДайын. Келесі қадам: scripts/kaz_experiment.py")
