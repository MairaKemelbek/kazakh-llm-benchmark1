"""Shared loaders for the revision analyses.
Reads the raw model outputs in <repo>/results/ (15 CSV files)."""
from pathlib import Path
import pandas as pd, numpy as np, re, string

RESULTS = Path(__file__).resolve().parents[2] / "results"
M = ['gpt-4o', 'claude-sonnet', 'gemini-flash', 'gpt-4o-mini', 'claude-haiku']
NM = {'gpt-4o': 'GPT-4o', 'claude-sonnet': 'Claude Sonnet', 'gemini-flash': 'Gemini Flash Lite',
      'gpt-4o-mini': 'GPT-4o-mini', 'claude-haiku': 'Claude Haiku'}

def qa(t):
    """t = 'test300' (QA with context) or 'cb300' (closed-book QA)."""
    return {m: pd.read_csv(RESULTS / f'kazqad_{t}_{m}.csv') for m in M}

def sa():
    return {m: pd.read_csv(RESULTS / f'kazsandra_test300_{m}.csv') for m in M}
