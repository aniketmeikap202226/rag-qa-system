"""Check that every keyword in eval/questions.json really appears in your docs.

    python check_questions.py          # report only
    python check_questions.py --fix    # also remove questions whose keyword is missing
"""
import glob
import json
import sys

QFILE = "eval/questions.json"

text = ""
for path in glob.glob("docs/*"):
    if path.lower().endswith((".txt", ".md")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text += " " + " ".join(f.read().split())
text = text.lower()

with open(QFILE, encoding="utf-8") as f:
    items = json.load(f)

good, bad = [], []
for it in items:
    n = text.count(it["keyword"].lower())
    (good if n else bad).append(it)
    print(f"{'OK     ' if n else 'MISSING'} {n:3d}  {it['keyword']}")

print(f"\n{len(good)} OK, {len(bad)} missing")
if "--fix" in sys.argv and bad:
    with open(QFILE, "w", encoding="utf-8") as f:
        json.dump(good, f, indent=2, ensure_ascii=False)
    print(f"Removed {len(bad)} questions. {len(good)} remain in {QFILE}.")
elif bad:
    print("Run with --fix to remove the missing ones.")
