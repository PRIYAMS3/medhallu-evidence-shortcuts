# ============================================================
# CELL 15 — New Figure 3 (evidence ablation, five seeds)
#
# Replaces shortcut_chart.png. Page-wide, two panels:
#   (a) all four conditions on 0-1, control annotated
#   (b) the three answer-bearing conditions zoomed, seeds as dots
# Run after Cell 14. Needs only the files Cell 14 wrote.
# ============================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json, os

DRIVE = "/content/drive/MyDrive/medhallu_v2"
FIGS = "/content/figures"; os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "savefig.dpi": 750, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6})
GREY, DARK, OURS, GRID = "#B0B0B0", "#606060", "#1A1A1A", "#DDDDDD"

df = pd.read_csv(f"{DRIVE}/review_v14/a2_multiseed.csv")
ctrl = json.load(open(f"{DRIVE}/shortcut_evidence_only.json"))['macro_f1']
conds = ['full', 'answer_only', 'shuffled']
labs = ['Full\n(q + evidence)', 'Answer\nonly', 'Shuffled\nevidence']
vals = {c: df[df.condition == c]['macro_f1'].values for c in conds}
means = [vals[c].mean() for c in conds]
sds = [vals[c].std(ddof=1) for c in conds]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.3),
                             gridspec_kw={"width_ratios": [1.15, 1]})

# (a) all four conditions
x = np.arange(4)
h = means + [ctrl]
a1.bar(x, h, yerr=sds + [0], capsize=2.5, width=.62,
       color=[OURS, DARK, DARK, GREY], edgecolor="black", linewidth=.5,
       error_kw=dict(elinewidth=.6))
for xi, v in zip(x, h):
    a1.text(xi, v + .03, f"{v:.3f}", ha="center", fontsize=6)
a1.text(3, ctrl / 2, "single-class\nprediction", ha="center", va="center",
        fontsize=5.5)
a1.set_xticks(x); a1.set_xticklabels(labs + ['Evidence only\n(control)'],
                                     fontsize=6)
a1.set_ylabel("Macro F1"); a1.set_ylim(0, 1.08)
a1.yaxis.grid(True, color=GRID); a1.set_axisbelow(True)
a1.set_title("(a) All conditions", fontsize=7)

# (b) zoom, seeds as dots
rng = np.random.default_rng(0)
for i, c in enumerate(conds):
    a2.bar(i, means[i], width=.55, color="white", edgecolor="black",
           linewidth=.6)
    a2.errorbar(i, means[i], yerr=sds[i], color="black", capsize=3,
                elinewidth=.7)
    jit = rng.uniform(-.12, .12, len(vals[c]))
    a2.scatter(i + jit, vals[c], s=9, color=OURS if c == 'full' else DARK,
               zorder=3, linewidths=0)
    a2.text(i, max(vals[c].max(), means[i] + sds[i]) + .0012,
            f"{means[i]:.4f}", ha="center",
            fontsize=6)
a2.set_xticks(range(3)); a2.set_xticklabels(labs, fontsize=6)
a2.set_ylabel("Macro F1"); a2.set_ylim(.94, .98)
a2.yaxis.grid(True, color=GRID); a2.set_axisbelow(True)
a2.set_title("(b) Answer-bearing conditions, five seeds (y axis from 0.94)",
             fontsize=7)

fig.tight_layout()
fig.savefig(f"{FIGS}/shortcut_chart.png", facecolor="white")
plt.close(fig)
print("wrote shortcut_chart.png")
from google.colab import files
files.download(f"{FIGS}/shortcut_chart.png")
