"""Figures 1 and 2 for the paper, drawn with matplotlib in the same
greyscale serif style as the other figures. Page-wide (7.16 in), 600 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7})
INK, MID, LIGHT, FILL = "#1A1A1A", "#606060", "#B0B0B0", "#F2F2F2"


def box(ax, x, y, w, h, text, fill="white", bold=False, size=6.4, lw=0.7, ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fill, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=7, lw=0.7, color=INK, ls=ls,
                                 shrinkA=0, shrinkB=0))


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                                fc=FILL, ec=LIGHT, lw=0.6))
    ax.text(x + 0.012, y + h - 0.035, title, ha="left", va="top", fontsize=7,
            fontweight="bold", color=INK)


# =================================================================== FIG 1
fig = plt.figure(figsize=(7.16, 3.35))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# input strip
box(ax, 0.02, 0.83, 0.96, 0.13,
    "Each MedHallu item: question $q$  +  source passage $\\kappa$  +  answer $a$  "
    "(correct or hallucinated)\n"
    "Detector input:  [CLS]  $q \\oplus \\kappa$  [SEP]  $a$  [SEP]   "
    "→   P(hallucinated)", fill="white", size=6.8)

PW, PH, PY = 0.228, 0.74, 0.04
xs = [0.02, 0.265, 0.51, 0.755]
titles = ["(a) Training-time ablation", "(b) Test-time ablation",
          "(c) Surface classifiers", "(d) Style-controlled test"]
for x, t in zip(xs, titles):
    panel(ax, x, PY, PW, PH, t)
    arrow(ax, x + PW / 2, 0.83, x + PW / 2, PY + PH)

# (a)
x = xs[0] + 0.012; w = PW - 0.024
conds = ["Full:  $q \\oplus \\kappa$  vs  $a$",
         "Answer only:  [EMPTY]  vs  $a$",
         "Shuffled:  $q' \\oplus \\kappa'$  vs  $a$",
         "Evidence only (control):\n$q \\oplus \\kappa$  vs  [EMPTY]"]
ys = [0.60, 0.52, 0.44, 0.34]
hs = [0.065, 0.065, 0.065, 0.085]
for c, yy, hh in zip(conds, ys, hs):
    box(ax, x, yy, w, hh, c, size=6.1)
box(ax, x, 0.17, w, 0.12,
    "Same encoder, same recipe\nBiomedBERT, 3 epochs,\n5 seeds per condition",
    fill="#E4E4E4", size=6.1)
for yy in [0.34]:
    arrow(ax, x + w / 2, yy, x + w / 2, 0.29)
box(ax, x, 0.07, w, 0.07, "Macro F1 on 2000 test pairs", size=6.1)
arrow(ax, x + w / 2, 0.17, x + w / 2, 0.14)

# (b)
x = xs[1] + 0.012; w = PW - 0.024
box(ax, x, 0.60, w, 0.085, "5 detectors trained on\nfull inputs (not retrained)",
    fill="#E4E4E4", size=6.1)
tests = ["Correct passage", "Passage removed", "Unrelated passage", "Answer removed"]
for i, t in enumerate(tests):
    yy = 0.49 - i * 0.075
    box(ax, x + 0.03, yy, w - 0.03, 0.06, t, size=6.1)
    ax.plot([x + 0.015, x + 0.015], [0.60, yy + 0.03], color=INK, lw=0.7)
    arrow(ax, x + 0.015, yy + 0.03, x + 0.03, yy + 0.03)
box(ax, x, 0.07, w, 0.12,
    "Share of correct answers\nflagged; pairwise accuracy\n(hallucinated scored higher?)",
    size=6.0)
arrow(ax, x + w / 2 + 0.015, 0.265, x + w / 2 + 0.015, 0.19)

# (c)
x = xs[2] + 0.012; w = PW - 0.024
box(ax, x, 0.60, w, 0.085, "Answer text only\n(no question, no passage)",
    fill="#E4E4E4", size=6.1)
feats = ["Length in words", "11 style features\n(hedging, negation, ...)",
         "TF-IDF word n-grams"]
ys = [0.49, 0.385, 0.30]; hs = [0.06, 0.08, 0.06]
for f, yy, hh in zip(feats, ys, hs):
    box(ax, x, yy, w, hh, f, size=6.1)
arrow(ax, x + w / 2, 0.60, x + w / 2, 0.55)
box(ax, x, 0.20, w, 0.065, "Overlap with passage\n(reads the evidence)", size=6.0,
    ec=MID)
box(ax, x, 0.07, w, 0.09, "Logistic regression\nMacro F1, top terms", size=6.1)
arrow(ax, x + w / 2, 0.20, x + w / 2, 0.16)

# (d)
x = xs[3] + 0.012; w = PW - 0.024
box(ax, x, 0.60, w, 0.085, "Rewrite test answers with\nQwen2.5-7B and Phi-4",
    fill="#E4E4E4", size=6.1)
box(ax, x, 0.475, w, 0.09, "Correct answers restyled:\nshort, direct, no hedging",
    size=6.1)
box(ax, x, 0.35, w, 0.09, "Both restyled: hallucinated\nanswers also hedged", size=6.1)
arrow(ax, x + w / 2, 0.60, x + w / 2, 0.565)
arrow(ax, x + w / 2, 0.475, x + w / 2, 0.44)
box(ax, x, 0.22, w, 0.09, "Meaning check: entailment\nmodel + 2 × 50 read by hand", size=6.0,
    ec=MID)
box(ax, x, 0.07, w, 0.11,
    "Re-score (a)–(c) models (5 seeds)\nand two prompted LLMs with the\ncorrect or an unrelated passage", size=5.9)
arrow(ax, x + w / 2, 0.22, x + w / 2, 0.18)

fig.savefig("figures/pipeline_architecture.png", dpi=600,
            facecolor="white")
plt.close(fig)

# =================================================================== FIG 2
fig = plt.figure(figsize=(7.16, 2.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# MedHallu
panel(ax, 0.015, 0.05, 0.60, 0.90, "MedHallu (biomedical, 10 000 items)")
box(ax, 0.03, 0.62, 0.27, 0.18,
    "pqa_artificial\n9000 items", fill="#E4E4E4", bold=True, size=6.6)
box(ax, 0.33, 0.62, 0.27, 0.18,
    "pqa_labeled\n1000 items", fill="#E4E4E4", bold=True, size=6.6)
box(ax, 0.03, 0.34, 0.27, 0.18,
    "Training\n18 000 pairs\n(1 correct + 1 hallucinated each)", size=6.3)
box(ax, 0.33, 0.34, 0.27, 0.18,
    "Evaluation\n2000 pairs\n548 easy / 636 medium / 816 hard", size=6.3)
arrow(ax, 0.165, 0.62, 0.165, 0.52)
arrow(ax, 0.465, 0.62, 0.465, 0.52)
# overlap check
ax.plot([0.30, 0.33], [0.43, 0.43], color=INK, lw=0.7, ls=(0, (2, 1.5)))
ax.text(0.315, 0.555, "question\noverlap = 0", ha="center", va="center",
        fontsize=5.8, color=INK)
box(ax, 0.03, 0.08, 0.57, 0.19,
    "Correct answers: PubMedQA long answers (abstract conclusions), verified for all 1000\n"
    "Hallucinated answers: one generator, Qwen2.5-14B\n"
    "Style-controlled copies of the evaluation split: Qwen2.5-7B (993 items), Phi-4 (1000 items)",
    size=6.0)

# HaluEval
panel(ax, 0.635, 0.05, 0.35, 0.90, "HaluEval QA (general, 10 000 items)")
box(ax, 0.65, 0.62, 0.32, 0.18,
    "Split by question\n8000 / 2000, overlap = 0", fill="#E4E4E4", bold=True, size=6.4)
box(ax, 0.65, 0.34, 0.15, 0.18, "Training\n16 000\npairs", size=6.3)
box(ax, 0.82, 0.34, 0.15, 0.18, "Evaluation\n4000\npairs", size=6.3)
arrow(ax, 0.725, 0.62, 0.725, 0.52)
arrow(ax, 0.895, 0.62, 0.895, 0.52)
box(ax, 0.65, 0.08, 0.32, 0.19,
    "Replication: same controls, RoBERTa\n"
    "Transfer: MedHallu detector applied\nzero-shot to 4000 HaluEval pairs",
    size=6.0)

fig.savefig("figures/dataset_flow.png", dpi=600, facecolor="white")
plt.close(fig)
print("figures written")
