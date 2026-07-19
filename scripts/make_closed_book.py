# -*- coding: utf-8 -*-
"""
Closed-book нұсқасын жасау
==========================
kaz_qad_experiment.py файлындағы промптан контексті алып тастап,
kaz_qad_cb.py файлын жасайды.

    python scripts/make_closed_book.py
    python scripts/kaz_qad_cb.py --data kazqad_test300.csv
"""

from pathlib import Path

SRC = Path(__file__).parent / "kaz_qad_experiment.py"
DST = Path(__file__).parent / "kaz_qad_cb.py"

OLD = '''PROMPT = (
    "Төмендегі мәтінге сүйеніп, сұраққа жауап бер.\\n"
    "Жауап мәтіннің ішінен алынған ең қысқа сөз немесе сөз тіркесі болсын.\\n"
    "Түсіндірме жазба, тек жауаптың өзін жаз.\\n\\n"
    "Мәтін: {context}\\n\\nСұрақ: {question}\\n\\nЖауап:"
)'''

NEW = '''PROMPT = (
    "Сұраққа қазақ тілінде жауап бер.\\n"
    "Жауап ең қысқа сөз немесе сөз тіркесі болсын. Түсіндірме жазба.\\n\\n"
    "Сұрақ: {question}\\n\\nЖауап:"
)'''

src = SRC.read_text(encoding="utf-8")
assert OLD in src, "Промпт табылмады — kaz_qad_experiment.py өзгерген бе?"
DST.write_text(src.replace(OLD, NEW), encoding="utf-8")
print(f"✓ {DST.name} жасалды (контекстсіз промпт)")
