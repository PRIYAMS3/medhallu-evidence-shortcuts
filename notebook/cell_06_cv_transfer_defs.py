# ============================================================
# CELL 6 - All-encoder seeds, paired test, type CV, external transfer.
# ============================================================
import os, json, itertools
import numpy as np, pandas as pd, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                             average_precision_score, confusion_matrix)
from sklearn.model_selection import GroupKFold

DRIVE = "/content/drive/MyDrive/medhallu_v2"
BIOMEDBERT = ("microsoft/BiomedNLP-BiomedBERT-base-"
              "uncased-abstract-fulltext")
ENCODERS = {
    'BiomedBERT':       BIOMEDBERT,
    'BioBERT':          'dmis-lab/biobert-base-cased-v1.2',
    'Bio_ClinicalBERT': 'emilyalsentzer/Bio_ClinicalBERT',
    'RoBERTa':          'roberta-base',
}
SEEDS = [42, 123, 456, 789, 1234]
TYPES = ['Misinterpretation of #Question#', 'Incomplete Information',
         'Mechanism and Pathway Misattribution',
         'Methodological and Evidence Fabrication']
TYPE2ID = {t: i for i, t in enumerate(TYPES)}
SHORT = ['Misinterpretation', 'Incomplete info',
         'Mechanism misattr.', 'Evidence fabric.']

train_df = pd.read_csv(f"{DRIVE}/train_df.csv")
test_df  = pd.read_csv(f"{DRIVE}/test_df.csv")
y_te = test_df['label'].values
hard = (test_df['difficulty'] == 'hard').values


class PairDS(torch.utils.data.Dataset):
    def __init__(self, df, tok, col='label'):
        self.df = df.reset_index(drop=True); self.tok = tok; self.col = col
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(f"{r['question']} {r['knowledge']}", r['answer'],
                       truncation='longest_first', max_length=512,
                       padding='max_length', return_tensors='pt')
        it = {k: v.squeeze(0) for k, v in enc.items()}
        it['labels'] = torch.tensor(int(r[self.col]))
        return it


def train_and_predict(model_name, seed, train_data, predict_data,
                      n_labels=2, epochs=3, label_col='label',
                      class_weights=None):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels)

    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float).cuda()
        class T(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                lab = inputs.pop("labels")
                o = model(**inputs)
                loss = torch.nn.functional.cross_entropy(o.logits, lab, weight=w)
                return (loss, o) if return_outputs else loss
        TrainerCls = T
    else:
        TrainerCls = Trainer

    args = TrainingArguments(
        output_dir=f"/content/tmp_{seed}", num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="no", save_strategy="no",
        warmup_steps=500, logging_steps=1000,
        fp16=True, seed=seed, report_to="none")
    t = TrainerCls(model=model, args=args,
                   train_dataset=PairDS(train_data, tok, label_col))
    t.train()
    logits = t.predict(PairDS(predict_data, tok, label_col)).predictions
    probs = torch.softmax(torch.tensor(logits), -1).numpy()
    del model, t; torch.cuda.empty_cache()
    return probs


# =====================================================================
# JOB A — all encoders x 5 seeds
# =====================================================================
def jobA():
    path = f"{DRIVE}/all_encoder_seeds.csv"
    pred_path = f"{DRIVE}/all_encoder_predictions.npz"
    if os.path.exists(path) and os.path.exists(pred_path):
        print("JOB A cached"); return pd.read_csv(path)

    rows, preds = [], {}
    for name, mpath in ENCODERS.items():
        for seed in SEEDS:
            print(f"\n--- JOB A {name} seed {seed} ---", flush=True)
            p = train_and_predict(mpath, seed, train_df, test_df)[:, 1]
            pred = (p >= 0.5).astype(int)
            preds[f"{name}_{seed}"] = p
            rows.append({'encoder': name, 'seed': seed,
                         'macro_f1': f1_score(y_te, pred, average='macro'),
                         'hard_f1': f1_score(y_te[hard], pred[hard],
                                             average='macro'),
                         'auroc': roc_auc_score(y_te, p),
                         'auprc': average_precision_score(y_te, p)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)
            np.savez(pred_path, **preds)

    df = pd.DataFrame(rows)
    print("\n=== JOB A summary ===")
    for enc in ENCODERS:
        d = df[df['encoder'] == enc]
        for m in ['macro_f1', 'hard_f1', 'auroc']:
            v = d[m].values; mu, sd = v.mean(), v.std(ddof=1)
            ci = 1.96 * sd / np.sqrt(len(v))
            print(f"{enc:18s} {m:9s} {mu:.4f} +/- {sd:.4f} "
                  f"CI [{mu-ci:.4f}, {mu+ci:.4f}]")
    return df


# =====================================================================
# JOB A2 — paired bootstrap between encoders
# =====================================================================
def jobA2(n_boot=10000, seed=0):
    preds = np.load(f"{DRIVE}/all_encoder_predictions.npz")
    rng = np.random.default_rng(seed)
    n = len(y_te)

    def mean_pred(enc):
        cols = [preds[f"{enc}_{s}"] for s in SEEDS if f"{enc}_{s}" in preds]
        return np.mean(cols, axis=0)

    out = {}
    base = 'BiomedBERT'
    pb = mean_pred(base)
    for other in [e for e in ENCODERS if e != base]:
        po = mean_pred(other)
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            diffs[b] = (f1_score(y_te[idx], (pb[idx] >= .5).astype(int),
                                 average='macro') -
                        f1_score(y_te[idx], (po[idx] >= .5).astype(int),
                                 average='macro'))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = float((diffs <= 0).mean() * 2)
        out[f"{base} vs {other}"] = {
            'mean_diff': round(float(diffs.mean()), 4),
            'ci95': [round(float(lo), 4), round(float(hi), 4)],
            'p_two_sided': round(min(p, 1.0), 4)}
        print(f"{base} vs {other}: {out[f'{base} vs {other}']}", flush=True)

    json.dump(out, open(f"{DRIVE}/paired_bootstrap.json", "w"), indent=2)
    return out


# =====================================================================
# JOB B — 5-fold CV type classification
#         Every Evidence Fabrication example gets predicted once.
# =====================================================================
def jobB():
    path = f"{DRIVE}/type_cv_predictions.csv"
    if os.path.exists(path):
        print("JOB B cached"); return pd.read_csv(path)

    hall = train_df[train_df['label'] == 1].copy()
    hall['tid'] = hall['htype'].map(TYPE2ID)
    hall = hall.dropna(subset=['tid']).reset_index(drop=True)
    print("total hallucinated:", len(hall))
    print(hall['htype'].value_counts().to_dict())

    gkf = GroupKFold(n_splits=5)
    all_true, all_pred, all_fold = [], [], []

    for fold, (tr, te) in enumerate(
            gkf.split(hall, groups=hall['question'])):
        print(f"\n--- JOB B fold {fold+1}/5 ---", flush=True)
        d_tr, d_te = hall.iloc[tr], hall.iloc[te]
        print("  fold test support:", d_te['htype'].value_counts().to_dict())
        counts = d_tr['tid'].value_counts().sort_index().values
        w = np.sqrt(len(d_tr) / (4 * counts))
        probs = train_and_predict(BIOMEDBERT, 42, d_tr, d_te,
                                  n_labels=4, epochs=2, label_col='tid',
                                  class_weights=w)
        all_true.extend(d_te['tid'].astype(int).values)
        all_pred.extend(probs.argmax(-1))
        all_fold.extend([fold] * len(d_te))

    res = pd.DataFrame({'true_tid': all_true, 'pred_tid': all_pred,
                        'fold': all_fold})
    res.to_csv(path, index=False)

    print("\n=== JOB B: pooled 5-fold type classification ===")
    print(classification_report(res['true_tid'], res['pred_tid'],
                                target_names=SHORT, zero_division=0,
                                digits=4))
    print("confusion matrix (rows true, cols predicted):")
    print(confusion_matrix(res['true_tid'], res['pred_tid']))

    # bootstrap CIs per class
    rng = np.random.default_rng(0)
    t, p = res['true_tid'].values, res['pred_tid'].values
    cis = {}
    for k, s in enumerate(SHORT):
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(t), len(t))
            boots.append(f1_score(t[idx] == k, p[idx] == k, zero_division=0))
        cis[s] = [round(float(np.percentile(boots, 2.5)), 4),
                  round(float(np.percentile(boots, 97.5)), 4)]
    print("\nper-class F1 95% CI:", json.dumps(cis, indent=2))
    json.dump(cis, open(f"{DRIVE}/type_cv_cis.json", "w"), indent=2)
    return res


# =====================================================================
# JOB C — external evaluation, different generator and domain
# =====================================================================
def jobC():
    path = f"{DRIVE}/external_halueval.json"
    if os.path.exists(path):
        print("JOB C cached"); return json.load(open(path))

    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa")['data']
    print("HaluEval QA:", len(ds), ds.column_names)

    rows = []
    for ex in ds.select(range(2000)):
        base = {'question': ex['question'], 'knowledge': ex['knowledge']}
        rows.append({**base, 'answer': ex['right_answer'], 'label': 0})
        rows.append({**base, 'answer': ex['hallucinated_answer'],
                     'label': 1})
    ext = pd.DataFrame(rows)
    print("external pairs:", len(ext))

    tok = AutoTokenizer.from_pretrained(f"{DRIVE}/biomedbert_v2_tok")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{DRIVE}/biomedbert_v2").cuda().eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(ext), 64):
            c = ext.iloc[i:i+64]
            enc = tok([f"{q} {k}" for q, k in zip(c['question'],
                                                  c['knowledge'])],
                      list(c['answer']), truncation='longest_first',
                      max_length=512, padding=True,
                      return_tensors='pt').to('cuda')
            probs.append(torch.softmax(model(**enc).logits, -1)
                         .cpu().numpy()[:, 1])
            if i % 640 == 0: print(f"  {i}/{len(ext)}", flush=True)

    p = np.concatenate(probs)
    yv = ext['label'].values
    pred = (p >= 0.5).astype(int)
    out = {'n_pairs': len(ext),
           'macro_f1': round(f1_score(yv, pred, average='macro'), 4),
           'auroc': round(float(roc_auc_score(yv, p)), 4),
           'auprc': round(float(average_precision_score(yv, p)), 4)}
    print("\n=== JOB C: HaluEval QA (zero-shot transfer) ===")
    print(json.dumps(out, indent=2))
    print("NOTE: HaluEval QA is built from HotpotQA, so this shifts both")
    print("generator AND domain. Interpret as a joint shift, not a clean")
    print("generator-only test.")
    ext['proba'] = p
    ext[['label', 'proba']].to_csv(f"{DRIVE}/external_halueval_preds.csv",
                                   index=False)
    json.dump(out, open(path, "w"), indent=2)
    return out