# ============================================================
# CELL 12 — Severity and contradiction analysis (paper Section VII)
#
# Uses the seed 42 predictions from Cell 6's matched-protocol run,
# so these figures share one execution with Table II.
# ============================================================
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import (f1_score, roc_auc_score, recall_score,
                             precision_score, average_precision_score)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

test_df = pd.read_csv(f"{DRIVE}/test_df.csv")
te_nli  = pd.read_csv(f"{DRIVE}/nli_test.csv")
tp      = pd.read_csv(f"{DRIVE}/type_predictions_labeled_v3.csv")
te_p1   = np.load(f"{DRIVE}/all_encoder_predictions.npz")['BiomedBERT_42']
y = test_df['label'].values

# ---- ensemble, cross-validated on the test features ----------------
# NOTE: this is NOT a held-out result. Each fold's classifier sees 80%
# of the test labels. Reported in the paper as a negative result only.
F = ['contradiction', 'neutral', 'entailment', 'phi']
X = StandardScaler().fit_transform(
        np.column_stack([te_p1] + [te_nli[c].values for c in F]))
p2 = cross_val_predict(
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42),
        X, y, cv=5, method='predict_proba')[:, 1]

print("condition        macroF1   sens    spec    AUROC")
for n, s in [('Phase 1 (held out)', te_p1), ('Phase 2 (CV on test)', p2)]:
    pr = (s >= .5).astype(int)
    print(f"{n:22s} {f1_score(y, pr, average='macro'):.4f} "
          f"{recall_score(y, pr):.4f} {recall_score(y, pr, pos_label=0):.4f} "
          f"{roc_auc_score(y, s):.4f}")

print("\nper difficulty      n   Phase1   Phase2")
for d in ['easy', 'medium', 'hard']:
    m = (test_df['difficulty'] == d).values
    print(f"  {d:8s} {m.sum():>6d}   "
          f"{f1_score(y[m], (te_p1[m]>=.5).astype(int), average='macro'):.4f}   "
          f"{f1_score(y[m], (p2[m]>=.5).astype(int), average='macro'):.4f}")

# ---- severity ------------------------------------------------------
# R is MedHallu's difficulty label: benchmark metadata, not derivable
# from (q, a, k). The score is therefore an analysis of labelled data,
# not a deployable component.
h = np.where(y == 1)[0]
W = {0: .7, 1: .4, 2: .8, 3: 1.}
R = {'hard': 1., 'medium': .6, 'easy': .2}
sev = pd.DataFrame({'difficulty': test_df.loc[h, 'difficulty'].values,
                    'p1': te_p1[h],
                    'contra': te_nli['contradiction'].values[h]})
sev['Wt'] = [W[i] for i in tp['pred_tid'].values]
sev['severity'] = np.minimum(
    sev['Wt'] * (.6*sev['p1'] + .4*sev['contra']) *
    sev['difficulty'].map(R), 1.)
sev['band'] = pd.cut(sev['severity'], [-.01, .3, .7, 1.01],
                     labels=['LOW', 'MODERATE', 'HIGH'])
rho, pv = spearmanr(sev['severity'],
                    sev['difficulty'].map({'easy':0,'medium':1,'hard':2}))
print(f"\nseverity: rho = {rho:.4f} (p = {pv:.2e})")
print("  NOTE: difficulty is a factor inside S, so this correlation is")
print("        an internal-consistency check, not validation.")
print("  bands:", sev['band'].value_counts().to_dict())
print("  means:", {d: round(sev[sev.difficulty==d]['severity'].mean(), 4)
                   for d in ['easy','medium','hard']})

# ---- contradiction signal ------------------------------------------
ph = te_nli['contradiction'].values[y == 1]
pc = te_nli['contradiction'].values[y == 0]
sp = np.sqrt(((len(ph)-1)*ph.std(ddof=1)**2 +
              (len(pc)-1)*pc.std(ddof=1)**2) / (len(ph)+len(pc)-2))
d = (ph.mean() - pc.mean()) / sp
print(f"\ncontradiction: hallucinated {ph.mean():.4f} vs clean "
      f"{pc.mean():.4f}, Cohen's d = {d:.4f}, "
      f"p = {mannwhitneyu(ph, pc, alternative='greater').pvalue:.2e}")

sev.to_csv(f"{DRIVE}/severity_final.csv", index=False)
np.savez(f"{DRIVE}/figure_data_final.npz", y_true=y, y_score_p1=te_p1,
         y_score_p2=p2, difficulty=test_df['difficulty'].values,
         severity=sev['severity'].values,
         sev_band=sev['band'].astype(str).values,
         sev_difficulty=sev['difficulty'].values,
         phi_hall=ph, phi_clean=pc)
print("\nsaved figure_data_final.npz")