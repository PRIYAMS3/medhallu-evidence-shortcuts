# How Much Evidence Does a Hallucination Detector Read?

Code, prompts, per-seed results and manual-check sheets for

> P. Nambiar and Ashwath Rao B, "How Much Evidence Does a Hallucination Detector Read? Answer-Level Shortcuts in Biomedical Benchmarks," submitted to *IEEE Access*, 2026.

The paper tests whether a fine-tuned biomedical hallucination detector's accuracy on MedHallu depends on the evidence passage. It uses training-time and test-time evidence ablations, answer-only surface classifiers, a style-controlled test set produced by two independent rewriting models, two zero-shot prompted detectors, and a replication on HaluEval QA.

## Repository layout

```
notebook/        The full Colab notebook (MedHallu_Full_Pipeline.ipynb) and the
                 revision experiments as standalone cells (cell_14 ... cell_21)
prompts/         The rewriting and prompted-detector prompts (also in Appendix C)
manual_check/    Row-level manual checks of 50 rewrites per rewriter
results/         Per-seed results, per-item predictions and rewritten test sets
figures/         Script for Figures 1 and 2 (Figures 3-7 come from cells 13 and 15)
```

## Environment

All experiments were run on Google Colab. Encoder comparison and auxiliary analyses: NVIDIA T4. Five-seed ablations, rewriting, style-controlled evaluation and prompted detectors: NVIDIA A100 (40 GB).

```
pip install -r requirements.txt
```

Every cell reads and writes a Google Drive folder, `MyDrive/medhallu_v2/`. Run **Cell 1** first in every session. It mounts Drive and asserts that the mount succeeded. Cells 14–20 are standalone: each needs only Cell 1, and caches its work so it can resume after a disconnect.

## Data

- **MedHallu** (Pandit et al., EMNLP 2025), `UTAustin-AIHealth/MedHallu` on Hugging Face:
  - training uses `pqa_artificial` (9,000 items, 18,000 answer pairs)
  - evaluation uses `pqa_labeled` (1,000 items, 2,000 answer pairs)
  - no question appears in both splits
- **HaluEval QA** (Li et al., EMNLP 2023): 10,000 items, split by question 8,000 / 2,000.

## Which cell produces which result

| Paper | Content | Cell(s) | Main output in `results/` |
|---|---|---|---|
| Table 1 | Research gaps | (text only) | — |
| Fig. 1, Fig. 2 | Experiment and data overview | `figures/make_fig1_fig2.py` | `figures/*.png` |
| Table 2, Fig. 3 | Training-time evidence ablation, 5 seeds | 8–9 (seed 42, evidence-only control), 14 (A2), 15 | `review_v14/a2_summary.json`, `a2_multiseed.csv`, `shortcut_evidence_only.json` |
| Table 3 | Test-time ablation, 5 seeds | 14 (A1, seed 42), 19 (Q1) | `review_v14/a1_test_time.json`, `review_v19/` |
| Tables 4–5 | Answer-only surface classifiers, top terms | 14 (A9), 16 (P1, length-matched) | `review_v14/a9_baselines.json`, `review_v16/` |
| Table 6 | Style-controlled test, Qwen2.5-7B rewrites | 16 (P3 rewriting), 17–18 (language check), 19 (Q1) | `review_v16/p3_*.csv`, `review_v19/` |
| Table 7 | Style-controlled test, Phi-4 rewrites | 20 (R1, R2) | `review_v20/phi_*.csv`, `review_v20/` |
| Table 8 | Prompted detectors (Qwen2.5-7B, OLMo-2-7B) | 19 (Q2), 20 (R3) | `review_v19/`, `review_v20/` |
| Table 9 | Ablation within HaluEval QA | 16 (P4) | `review_v16/p4_*.json` |
| Table 10 | Transfer to HaluEval | 6–7 | `external_halueval.json` |
| Tables 11–12 | Encoder comparison and paired tests | 4–7, 11 | `all_encoder_seeds.csv`, `all_encoder_predictions.npz` |
| Table 13 | Recommended controls | (text only) | — |
| Table 14 | Entailment ensemble (Appendix A) | 2–3 | `phase1_probs.npz`, `nli_test.csv` |
| Tables 15–16 | Type classification (Appendix B) | 6–7 | `type_cv_predictions.csv`, `type_cv_cis.json` |
| Figs. 4–7 | Encoder, severity, contradiction, ROC/PR charts | 12, 13 | `severity_final.csv`, `figure_data_final.npz` |

## Notes on reproduction

- **Run order.** In `MedHallu_Full_Pipeline.ipynb`, run cells 1–13 in order. Cells 4 and 6 both define `jobA` and `jobB`, and only Cell 6 checks its cache, so Cell 6 must follow Cell 4.
- **Superseded cells.** Cells 4–5 are superseded by Cell 6 but are kept: Cell 12 uses the type predictions that Cell 4's `job0` produces.
- **Exactness.** Training uses fp16 mixed precision, so reruns match the reported numbers to within seed-level variation, not bit for bit. Reported values come from the saved outputs in `results/`.
- **Language filter.** Seven Qwen2.5-7B rewrites contained Chinese text. They are excluded from Table 6 (993 items remain; Cells 17–18). All 1,000 Phi-4 rewrites are in English.
- **Superseded result files.** `paired_bootstrap.json` and `results_v3_SUPERSEDED.json` are superseded and are not used by the paper.

## Manual checks

`manual_check/` holds 50 randomly drawn items per rewriter. Each rewritten correct answer is classed as:

- `same`: same claim, same strength
- `stronger`: same finding stated with less hedging, which is the intended effect
- `changed`: the claim itself changed

Each hallucinated rewrite is marked for whether it kept the original, incorrect claim.

| Rewriter | same | stronger | changed | excluded | hallucinated claim kept |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 28 | 18 | 3 | 1 (Chinese text) | 49/49 |
| Phi-4 | 35 | 13 | 2 | 0 | 50/50 |

## License

Code: MIT (see `LICENSE`). MedHallu and HaluEval, and text derived from them (including the rewritten test sets), remain under their original licenses.

## Citation

See `CITATION.cff`. Please also cite MedHallu and HaluEval if you use the data.
