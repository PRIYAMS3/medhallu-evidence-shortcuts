# ============================================================
# CELL 13 — Regenerate every figure in the paper
#
# IEEE style: serif face matching the body text, greyscale-safe
# palette, 600+ dpi. Column figures 3.5 in, page-wide 7.16 in.
# ============================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, json, pandas as pd
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

FIGS = "/content/figures"; os.makedirs(FIGS, exist_ok=True)
d = np.load(f"{DRIVE}/figure_data_final.npz", allow_pickle=True)
seeds_df = pd.read_csv(f"{DRIVE}/all_encoder_seeds.csv")
short = json.load(open(f"{DRIVE}/shortcut_summary.json"))
cvcis = json.load(open(f"{DRIVE}/type_cv_cis.json"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 750, "savefig.dpi": 750, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6})
GREY, DARK, OURS, GRID = "#B0B0B0", "#606060", "#1A1A1A", "#DDDDDD"
COL, PAGE = 3.5, 7.16
def save(f, n): f.savefig(f"{FIGS}/{n}", facecolor="white"); plt.close(f); print("wrote", n)

# ---- shortcut diagnostics (principal result) -----------------------
conds = ['full', 'answer_only', 'shuffled', 'evidence_only']
labs  = ['Full\n(q + evidence)', 'Answer\nonly', 'Shuffled\nevidence',
         'Evidence only\n(control)']
vals  = [short[c]['macro_f1'] for c in conds]
fig, ax = plt.subplots(figsize=(COL, 2.2))
b = ax.bar(range(4), vals, color=[OURS, DARK, DARK, GREY],
           edgecolor="black", linewidth=.5, width=.65)
for bar, v in zip(b, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+.015, f"{v:.3f}",
            ha="center", fontsize=6)
ax.axhline(0.5, color="black", ls=":", lw=.8)
ax.text(3.4, 0.52, "chance", fontsize=5.5, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=6)
ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1.08)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, "shortcut_chart.png")

# ---- encoder comparison, 5 seeds -----------------------------------
order = ['Bio_ClinicalBERT', 'RoBERTa', 'BioBERT', 'BiomedBERT']
piv = seeds_df.pivot(index='seed', columns='encoder', values='macro_f1')
means = [piv[e].mean() for e in order]
sds   = [piv[e].std(ddof=1) for e in order]
fig, ax = plt.subplots(figsize=(COL, 2.1))
b = ax.bar(range(4), means, yerr=sds, capsize=2.5,
           color=[GREY]*3+[OURS], edgecolor="black", linewidth=.5, width=.6,
           error_kw=dict(elinewidth=.6))
ax.set_xticks(range(4))
ax.set_xticklabels([e.replace('_','-') for e in order], rotation=20,
                   ha="right", fontsize=6)
ax.set_ylabel("Macro F1"); ax.set_ylim(0.93, 0.98)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, "comparison_chart.png")

# ---- multi-seed stability ------------------------------------------
sv = piv['BiomedBERT'].values
hv = seeds_df.pivot(index='seed', columns='encoder',
                    values='hard_f1')['BiomedBERT'].values
av = seeds_df.pivot(index='seed', columns='encoder',
                    values='auroc')['BiomedBERT'].values
fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.0))
for ax, (v, lab) in zip(axes, [(sv, "Macro F1"), (hv, "Hard-case F1"),
                               (av, "AUROC")]):
    m, s = v.mean(), v.std(ddof=1)
    ax.bar(range(5), v, color=GREY, edgecolor="black", linewidth=.5, width=.65)
    ax.axhline(m, color="black", ls="--", lw=.8)
    ax.axhspan(m-s, m+s, color="#EEEEEE", zorder=0)
    ax.set_xticks(range(5))
    ax.set_xticklabels([str(x) for x in piv.index], fontsize=6)
    ax.set_xlabel("Random seed"); ax.set_ylabel(lab)
    ax.set_ylim(max(0, v.min()-4*s), min(1.002, v.max()+3*s))
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title(f"{lab}: {m:.4f} $\\pm$ {s:.4f}", fontsize=7)
fig.tight_layout(); save(fig, "multiseed_chart.png")

# ---- type classification, cross-validated --------------------------
types = ["Misinterpretation", "Mechanism\nmisattribution",
         "Incomplete\ninformation", "Evidence\nfabrication"]
f1s = [0.8309, 0.5901, 0.5012, 0.0400]
los = [cvcis["Misinterpretation"][0], cvcis["Mechanism misattr."][0],
       cvcis["Incomplete info"][0], cvcis["Evidence fabric."][0]]
his = [cvcis["Misinterpretation"][1], cvcis["Mechanism misattr."][1],
       cvcis["Incomplete info"][1], cvcis["Evidence fabric."][1]]
ns  = [6848, 1096, 1010, 46]
fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2))
y = np.arange(4)[::-1]
ax = axes[0]
err = np.array([np.array(f1s)-np.array(los), np.array(his)-np.array(f1s)])
ax.barh(y, f1s, xerr=err, capsize=2.5, color=GREY, edgecolor="black",
        linewidth=.5, height=.55, error_kw=dict(elinewidth=.6))
for yi, v in zip(y, f1s):
    ax.text(v+0.03, yi, f"{v:.2f}", va="center", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels(types, fontsize=6)
ax.set_xlabel("F1 score (5-fold CV, 95% CI)"); ax.set_xlim(0, 1.0)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax = axes[1]
ax.barh(y, ns, color=DARK, edgecolor="black", linewidth=.5, height=.55)
for yi, n in zip(y, ns):
    ax.text(n+90, yi, str(n), va="center", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels([])
ax.set_xlabel("Training instances"); ax.set_xlim(0, 8000)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "type_chart.png")

# ---- severity -------------------------------------------------------
sev, sd = d['severity'], d['sev_difficulty']
fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2))
bands = ["LOW", "MODERATE", "HIGH"]
cnt = [int((d['sev_band'] == b).sum()) for b in bands]
ax = axes[0]
bb = ax.bar([b.capitalize() for b in bands], cnt, color=GREY,
            edgecolor="black", linewidth=.5, width=.6)
for bar, c in zip(bb, cnt):
    ax.text(bar.get_x()+bar.get_width()/2, c+8, str(c), ha="center", fontsize=6)
ax.set_ylabel("Count"); ax.set_title("Severity band distribution", fontsize=7)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax = axes[1]
bp = ax.boxplot([sev[sd == t] for t in ['easy','medium','hard']],
                tick_labels=["Easy","Medium","Hard"], widths=.5,
                patch_artist=True,
                flierprops=dict(marker="o", markersize=1.5,
                                markerfacecolor="none", markeredgewidth=.4))
for p in bp["boxes"]:
    p.set_facecolor("#EEEEEE"); p.set_edgecolor("black"); p.set_linewidth(.6)
for el in ["whiskers","caps","medians"]:
    for ln in bp[el]: ln.set_color("black"); ln.set_linewidth(.6)
ax.set_ylabel("Severity score $S$")
ax.set_title("Severity by difficulty (internal consistency)", fontsize=7)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "severity_chart.png")

# ---- contradiction signal -------------------------------------------
ph, pc = d['phi_hall'], d['phi_clean']
fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2))
ax = axes[0]; bins = np.linspace(0, 1, 26)
ax.hist(pc, bins=bins, density=True, color="#DDDDDD", edgecolor="black",
        linewidth=.4, label="Non-hallucinated")
ax.hist(ph, bins=bins, density=True, color=DARK, edgecolor="black",
        linewidth=.4, alpha=.75, label="Hallucinated")
ax.set_xlabel("Contradiction score"); ax.set_ylabel("Density")
ax.legend(frameon=False); ax.set_title("Score distribution", fontsize=7)
ax = axes[1]
bp = ax.boxplot([pc, ph], tick_labels=["Non-hallucinated","Hallucinated"],
                widths=.45, patch_artist=True,
                flierprops=dict(marker="o", markersize=1.5,
                                markerfacecolor="none", markeredgewidth=.4))
for p in bp["boxes"]:
    p.set_facecolor("#EEEEEE"); p.set_edgecolor("black"); p.set_linewidth(.6)
for el in ["whiskers","caps","medians"]:
    for ln in bp[el]: ln.set_color("black"); ln.set_linewidth(.6)
ax.set_ylabel("Contradiction score")
ax.set_title("Separation by class", fontsize=7)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "contradiction_chart.png")

# ---- ROC / PR --------------------------------------------------------
yv, p1, p2 = d['y_true'], d['y_score_p1'], d['y_score_p2']
fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.4))
ax = axes[0]
for s, lab, st in [(p1, "Phase 1 (held out)", "-"),
                   (p2, "Phase 2 (CV on test)", "--")]:
    fpr, tpr, _ = roc_curve(yv, s)
    ax.plot(fpr, tpr, st, color="black",
            label=f"{lab} (AUC {auc(fpr, tpr):.4f})")
ax.plot([0,1],[0,1], ":", color=GREY, label="Random")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Receiver operating characteristic", fontsize=7)
ax = axes[1]
for s, lab, st in [(p1, "Phase 1 (held out)", "-"),
                   (p2, "Phase 2 (CV on test)", "--")]:
    pr, rc, _ = precision_recall_curve(yv, s)
    ax.plot(rc, pr, st, color="black",
            label=f"{lab} (AUPRC {average_precision_score(yv, s):.4f})")
ax.axhline(.5, ls=":", color=GREY, label="Random")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(.45, 1.02)
ax.legend(frameon=False, loc="lower left")
ax.set_title("Precision-recall", fontsize=7)
fig.tight_layout(); save(fig, "roc_pr_curves.png")

from PIL import Image
print("\nfigure sizes:")
for f in sorted(os.listdir(FIGS)):
    print(f"  {f:28s} {Image.open(f'{FIGS}/{f}').size}")
print("\nDownload with:  from google.colab import files")
print("                for f in os.listdir(FIGS): files.download(f'{FIGS}/{f}')")