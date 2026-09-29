# ============================================================
# CELL 21 — Export the result files for GitHub as one zip.
# Any runtime (CPU is fine). Takes about 1 minute.
# Downloads:  medhallu_evidence_shortcuts_results.zip
# ============================================================
from google.colab import drive, files
drive.mount('/content/drive')

import os, zipfile

DRIVE = "/content/drive/MyDrive/medhallu_v2"
assert os.path.isdir(DRIVE), "Drive folder medhallu_v2 not found"

OUT_ZIP = "/content/medhallu_evidence_shortcuts_results.zip"
MAX_MB = 24           # GitHub's browser upload limit is 25 MB per file
KEEP_EXT = (".csv", ".json", ".npz", ".txt")

# result files in the main folder that the paper uses
ROOT_FILES = [
    "test_predictions.csv", "phase1_probs.npz",
    "all_encoder_seeds.csv", "all_encoder_predictions.npz",
    "type_cv_predictions.csv", "type_cv_cis.json",
    "type_predictions_labeled_v3.csv",
    "shortcut_summary.json", "shortcut_full.json",
    "shortcut_answer_only.json", "shortcut_shuffled.json",
    "shortcut_evidence_only.json",
    "external_halueval.json", "severity_final.csv",
    "figure_data_final.npz", "multiseed_v2.csv",
]
REVIEW_DIRS = ["review_v14", "review_v16", "review_v19", "review_v20"]

# folders holding model weights or tokenizers are never exported
SKIP_WORDS = ("checkpoint", "model", "_tok", "tokenizer")

added, skipped_big = [], []

def consider(path, arcname):
    name = os.path.basename(path).lower()
    if not name.endswith(KEEP_EXT):
        return
    if "paired_bootstrap" in name or "results_v3" in name:
        return
    mb = os.path.getsize(path) / 1e6
    if mb > MAX_MB:
        skipped_big.append((arcname, round(mb, 1)))
        return
    z.write(path, arcname)
    added.append((arcname, round(mb, 2)))

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ROOT_FILES:
        p = os.path.join(DRIVE, f)
        if os.path.isfile(p):
            consider(p, f"results/{f}")
    for d in REVIEW_DIRS:
        base = os.path.join(DRIVE, d)
        if not os.path.isdir(base):
            print("missing folder:", d)
            continue
        for root, dirs, fs in os.walk(base):
            # do not descend into model folders
            dirs[:] = [x for x in dirs
                       if not any(w in x.lower() for w in SKIP_WORDS)]
            for f in fs:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, DRIVE)
                consider(p, f"results/{rel}")

print(f"Added {len(added)} files")
for a, mb in added:
    print(f"  {mb:>7} MB  {a}")
if skipped_big:
    print("\nNOT added (too big for GitHub, send these names to Claude):")
    for a, mb in skipped_big:
        print(f"  {mb:>7} MB  {a}")
print(f"\nZip size: {os.path.getsize(OUT_ZIP)/1e6:.1f} MB")

files.download(OUT_ZIP)
