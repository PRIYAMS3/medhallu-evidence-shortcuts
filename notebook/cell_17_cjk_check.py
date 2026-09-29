# ============================================================
# CELL 17 — How many rewrites came out in the wrong language?
#            CPU only, seconds. Run Cell 1 first.
# ============================================================
import re, pandas as pd
OUT = "/content/drive/MyDrive/medhallu_v2/review_v16"
cjk = re.compile(r"[　-鿿가-힯]")
for f in ["p3_correct_assertive.csv", "p3_halluc_hedged.csv"]:
    d = pd.read_csv(f"{OUT}/{f}")
    bad = d[d['text'].astype(str).str.contains(cjk)]
    print(f"{f}: {len(bad)} of {len(d)} contain Chinese/Japanese/Korean characters")
    print("   item numbers:", bad['i'].tolist()[:40])
