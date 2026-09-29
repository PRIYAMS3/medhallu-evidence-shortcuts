# ============================================================
# CELL 4 - Multi-seed and encoder baseline definitions.
# ============================================================
import os, json
import numpy as np, pandas as pd, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DRIVE = "/content/drive/MyDrive/medhallu_v2"
BIOMEDBERT = ("microsoft/BiomedNLP-BiomedBERT-base-"
              "uncased-abstract-fulltext")
TYPES = ['Misinterpretation of #Question#', 'Incomplete Information',
         'Mechanism and Pathway Misattribution',
         'Methodological and Evidence Fabrication']
TYPE2ID = {t: i for i, t in enumerate(TYPES)}
SHORT = ['Misinterpretation', 'Incomplete info',
         'Mechanism misattr.', 'Evidence fabric.']

train_df = pd.read_csv(f"{DRIVE}/train_df.csv")
test_df  = pd.read_csv(f"{DRIVE}/test_df.csv")
tr_nli   = pd.read_csv(f"{DRIVE}/nli_train.csv")
te_nli   = pd.read_csv(f"{DRIVE}/nli_test.csv")

y_tr = train_df['label'].values
y_te = test_df['label'].values
NLI_F = ['contradiction', 'neutral', 'entailment', 'phi']

results = json.load(open(f"{DRIVE}/results_v2.json"))


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


def macro_metrics(p):
    return {'macro_f1': f1_score(p.label_ids, p.predictions.argmax(-1),
                                 average='macro')}


# =====================================================================
# JOB 0 — type classifier, model selection on macro F1
# =====================================================================
def job0():
    hall_tr = train_df[train_df['label'] == 1].copy()
    hall_tr['tid'] = hall_tr['htype'].map(TYPE2ID)
    hall_tr = hall_tr.dropna(subset=['tid'])

    qs = hall_tr['question'].unique()
    np.random.default_rng(42).shuffle(qs)
    cut = int(0.85 * len(qs))
    art_train = hall_tr[hall_tr['question'].isin(set(qs[:cut]))]
    art_held  = hall_tr[hall_tr['question'].isin(set(qs[cut:]))]

    hall_te = test_df[test_df['label'] == 1].copy()
    hall_te['tid'] = hall_te['htype'].map(TYPE2ID)
    hall_te = hall_te.dropna(subset=['tid'])

    tok = AutoTokenizer.from_pretrained(BIOMEDBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BIOMEDBERT, num_labels=4)

    counts = art_train['tid'].value_counts().sort_index().values
    # sqrt-tempered weights: full inverse-frequency was over-correcting
    # and destroying recall on the dominant class.
    w_raw = len(art_train) / (4 * counts)
    w = torch.tensor(np.sqrt(w_raw), dtype=torch.float).cuda()
    print("class weights:", w.cpu().numpy().round(3))

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            o = model(**inputs)
            loss = torch.nn.functional.cross_entropy(o.logits, labels,
                                                     weight=w)
            return (loss, o) if return_outputs else loss

    args = TrainingArguments(
        output_dir="/content/type_out3", num_train_epochs=5,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, warmup_steps=300,
        fp16=True, seed=42, report_to="none")

    t = WTrainer(model=model, args=args,
                 train_dataset=PairDS(art_train, tok, 'tid'),
                 eval_dataset=PairDS(art_held, tok, 'tid'),
                 compute_metrics=macro_metrics)
    t.train()

    out = {}
    for df, name in [(hall_te, 'labeled'), (art_held, 'artificial')]:
        pred = t.predict(PairDS(df, tok, 'tid')).predictions.argmax(-1)
        true = df['tid'].astype(int).values
        print(f"\n=== JOB 0 type: {name} ===")
        print(classification_report(true, pred, target_names=SHORT,
                                    zero_division=0, digits=4))
        rep = classification_report(true, pred, target_names=SHORT,
                                    zero_division=0, output_dict=True)
        d = df[['htype', 'difficulty']].copy()
        d['true_tid'] = true; d['pred_tid'] = pred
        d.to_csv(f"{DRIVE}/type_predictions_{name}_v3.csv", index=False)
        out[name] = {'macro_f1': round(rep['macro avg']['f1-score'], 4),
                     'per_class': {s: {k: round(rep[s][k], 4)
                                       for k in ['precision', 'recall',
                                                 'f1-score']}
                                   | {'n': int(rep[s]['support'])}
                                   for s in SHORT}}
    model.save_pretrained(f"{DRIVE}/type_model_v4")
    return out


# =====================================================================
# JOB A — multi-seed. Split is fixed; this isolates initialisation.
# =====================================================================
def run_encoder(model_name, seed, tag):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=f"/content/{tag}_out", num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="no", save_strategy="no",
        warmup_steps=500, logging_steps=500,
        fp16=True, seed=seed, report_to="none")
    t = Trainer(model=model, args=args,
                train_dataset=PairDS(train_df, tok))
    t.train()
    tr_p = torch.softmax(torch.tensor(
        t.predict(PairDS(train_df, tok)).predictions), -1).numpy()[:, 1]
    te_p = torch.softmax(torch.tensor(
        t.predict(PairDS(test_df, tok)).predictions), -1).numpy()[:, 1]
    del model, t; torch.cuda.empty_cache()
    return tr_p, te_p


def fit_ensemble(tr_p, te_p, seed):
    X_tr = np.column_stack([tr_p] + [tr_nli[c].values for c in NLI_F])
    X_te = np.column_stack([te_p] + [te_nli[c].values for c in NLI_F])
    sc = StandardScaler().fit(X_tr)
    g = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=seed)
    g.fit(sc.transform(X_tr), y_tr)
    return (g.predict(sc.transform(X_te)),
            g.predict_proba(sc.transform(X_te))[:, 1])


def jobA():
    rows = []
    hard = (test_df['difficulty'] == 'hard').values
    for seed in [42, 123, 456, 789, 1234]:
        print(f"\n--- JOB A seed {seed} ---")
        tr_p, te_p = run_encoder(BIOMEDBERT, seed, f"ms{seed}")
        pred, proba = fit_ensemble(tr_p, te_p, seed)
        rows.append({
            'seed': seed,
            'macro_f1': round(f1_score(y_te, pred, average='macro'), 4),
            'hard_f1': round(f1_score(y_te[hard], pred[hard],
                                      average='macro'), 4),
            'auroc': round(roc_auc_score(y_te, proba), 4)})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(f"{DRIVE}/multiseed_v2.csv", index=False)
    summ = {}
    for c in ['macro_f1', 'hard_f1', 'auroc']:
        v = df[c].values
        m, s = v.mean(), v.std(ddof=1)
        ci = 1.96 * s / np.sqrt(len(v))
        summ[c] = {'mean': round(m, 4), 'std': round(s, 4),
                   'ci95': [round(m-ci, 4), round(m+ci, 4)]}
    print(json.dumps(summ, indent=2))
    return {'per_seed': rows, 'summary': summ}


# =====================================================================
# JOB B — encoder baselines, same split, Phase 1 style (no ensemble)
# =====================================================================
def jobB():
    models = {
        'RoBERTa': 'roberta-base',
        'BioBERT': 'dmis-lab/biobert-base-cased-v1.2',
        'Bio_ClinicalBERT': 'emilyalsentzer/Bio_ClinicalBERT',
    }
    hard = (test_df['difficulty'] == 'hard').values
    out = {}
    for name, path in models.items():
        print(f"\n--- JOB B {name} ---")
        try:
            _, te_p = run_encoder(path, 42, name.lower())
            pred = (te_p >= 0.5).astype(int)
            out[name] = {
                'macro_f1': round(f1_score(y_te, pred, average='macro'), 4),
                'hard_f1': round(f1_score(y_te[hard], pred[hard],
                                          average='macro'), 4),
                'auroc': round(roc_auc_score(y_te, te_p), 4),
                'auprc': round(average_precision_score(y_te, te_p), 4)}
            print(name, out[name])
        except Exception as e:
            print(f"{name} FAILED: {e}")
            out[name] = {'error': str(e)}
    return out