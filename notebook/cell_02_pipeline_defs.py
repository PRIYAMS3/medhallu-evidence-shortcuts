# ============================================================
# CELL 2 - Data prep, Phase 1, NLI, ensemble, type classifier.
#          Definitions only. Nothing runs yet.
# ============================================================
import os, json, random
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DRIVE = "/content/drive/MyDrive/medhallu_v2"
os.makedirs(DRIVE, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BIOMEDBERT = ("microsoft/BiomedNLP-BiomedBERT-base-"
              "uncased-abstract-fulltext")
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

TYPES = ['Misinterpretation of #Question#',
         'Incomplete Information',
         'Mechanism and Pathway Misattribution',
         'Methodological and Evidence Fabrication']
TYPE2ID = {t: i for i, t in enumerate(TYPES)}


def stage_done(name):
    return os.path.exists(f"{DRIVE}/{name}")


# =====================================================================
# STAGE 0 — build dataframes, verify no question overlap
# =====================================================================
def build_data():
    if stage_done("train_df.csv") and stage_done("test_df.csv"):
        print("stage 0: loading cached")
        return (pd.read_csv(f"{DRIVE}/train_df.csv"),
                pd.read_csv(f"{DRIVE}/test_df.csv"))

    def to_rows(ds):
        rows = []
        for ex in ds:
            k = (" ".join(ex['Knowledge'])
                 if isinstance(ex['Knowledge'], list) else str(ex['Knowledge']))
            base = {'question': ex['Question'], 'knowledge': k,
                    'difficulty': ex['Difficulty Level'],
                    'htype': ex['Category of Hallucination']}
            rows.append({**base, 'answer': ex['Ground Truth'], 'label': 0})
            rows.append({**base, 'answer': ex['Hallucinated Answer'], 'label': 1})
        return pd.DataFrame(rows)

    art = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")['train']
    lab = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")['train']

    train_df = to_rows(art)
    test_df = to_rows(lab)

    # CRITICAL CHECK — the two splits must not share questions.
    overlap = set(train_df['question']) & set(test_df['question'])
    print(f"question overlap between splits: {len(overlap)}")
    if overlap:
        print("!! removing overlapping questions from TRAIN")
        train_df = train_df[~train_df['question'].isin(overlap)]
        train_df = train_df.reset_index(drop=True)

    train_df.to_csv(f"{DRIVE}/train_df.csv", index=False)
    test_df.to_csv(f"{DRIVE}/test_df.csv", index=False)
    print(f"train {len(train_df)} pairs | test {len(test_df)} pairs")
    return train_df, test_df


# =====================================================================
# STAGE 1 — BiomedBERT binary classifier
# =====================================================================
class PairDS(torch.utils.data.Dataset):
    def __init__(self, df, tok, label_col='label'):
        self.df = df.reset_index(drop=True); self.tok = tok
        self.label_col = label_col
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(f"{r['question']} {r['knowledge']}", r['answer'],
                       truncation='longest_first', max_length=512,
                       padding='max_length', return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(int(r[self.label_col]))
        return item


def metrics_fn(p):
    preds = p.predictions.argmax(-1)
    return {'macro_f1': f1_score(p.label_ids, preds, average='macro')}


def train_phase1(train_df, test_df):
    if stage_done("phase1_probs.npz"):
        print("stage 1: loading cached")
        d = np.load(f"{DRIVE}/phase1_probs.npz")
        return d['train_p1'], d['test_p1']

    tok = AutoTokenizer.from_pretrained(BIOMEDBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BIOMEDBERT, num_labels=2)

    args = TrainingArguments(
        output_dir="/content/p1_out",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no",
        warmup_steps=500, logging_steps=200,
        fp16=True, seed=SEED, report_to="none")

    tr = Trainer(model=model, args=args,
                 train_dataset=PairDS(train_df, tok),
                 eval_dataset=PairDS(test_df, tok),
                 compute_metrics=metrics_fn)
    tr.train()

    model.save_pretrained(f"{DRIVE}/biomedbert_v2")
    tok.save_pretrained(f"{DRIVE}/biomedbert_v2_tok")

    train_p1 = torch.softmax(torch.tensor(
        tr.predict(PairDS(train_df, tok)).predictions), -1).numpy()[:, 1]
    test_p1 = torch.softmax(torch.tensor(
        tr.predict(PairDS(test_df, tok)).predictions), -1).numpy()[:, 1]

    np.savez(f"{DRIVE}/phase1_probs.npz",
             train_p1=train_p1, test_p1=test_p1)
    return train_p1, test_p1


# =====================================================================
# STAGE 2 — DeBERTa NLI scores  (slowest stage)
# =====================================================================
def nli_scores(df, tag, batch=64):
    path = f"{DRIVE}/nli_{tag}.csv"
    if os.path.exists(path):
        print(f"stage 2 [{tag}]: loading cached")
        return pd.read_csv(path)

    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL).cuda().eval().half()

    # label order for this checkpoint: contradiction, entailment, neutral
    id2label = model.config.id2label
    print("NLI label map:", id2label)

    out = []
    with torch.no_grad():
        for i in range(0, len(df), batch):
            chunk = df.iloc[i:i+batch]
            enc = tok(list(chunk['knowledge']), list(chunk['answer']),
                      truncation=True, max_length=512,
                      padding=True, return_tensors='pt').to('cuda')
            probs = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
            out.append(probs)
            if i % (batch * 50) == 0:
                print(f"  {tag}: {i}/{len(df)}")

    probs = np.vstack(out)
    cols = {id2label[j].lower(): probs[:, j] for j in range(probs.shape[1])}
    res = pd.DataFrame(cols)
    res['phi'] = res['contradiction'] - res['entailment']
    res.to_csv(path, index=False)
    return res


# =====================================================================
# STAGE 3 — gradient boosting ensemble
# =====================================================================
def train_ensemble(train_df, test_df, train_p1, test_p1, tr_nli, te_nli):
    feats = ['contradiction', 'neutral', 'entailment', 'phi']
    X_tr = np.column_stack([train_p1] + [tr_nli[c].values for c in feats])
    X_te = np.column_stack([test_p1] + [te_nli[c].values for c in feats])

    sc = StandardScaler().fit(X_tr)
    ens = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=SEED)
    ens.fit(sc.transform(X_tr), train_df['label'].values)

    pred = ens.predict(sc.transform(X_te))
    proba = ens.predict_proba(sc.transform(X_te))[:, 1]

    res = test_df[['label', 'difficulty', 'htype']].copy()
    res['phase1_pred'] = (test_p1 >= 0.5).astype(int)
    res['phase1_proba'] = test_p1
    res['phase2_pred'] = pred
    res['phase2_proba'] = proba
    res.to_csv(f"{DRIVE}/test_predictions.csv", index=False)
    return res


# =====================================================================
# STAGE 4 — type classifier (now with real support per class)
# =====================================================================
def train_type_classifier(train_df, test_df):
    if stage_done("type_predictions.csv"):
        print("stage 4: loading cached")
        return pd.read_csv(f"{DRIVE}/type_predictions.csv")

    tr = train_df[train_df['label'] == 1].copy()
    te = test_df[test_df['label'] == 1].copy()
    tr['tid'] = tr['htype'].map(TYPE2ID)
    te['tid'] = te['htype'].map(TYPE2ID)
    tr = tr.dropna(subset=['tid']); te = te.dropna(subset=['tid'])
    print("type train support:", tr['htype'].value_counts().to_dict())

    tok = AutoTokenizer.from_pretrained(BIOMEDBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BIOMEDBERT, num_labels=4)

    counts = tr['tid'].value_counts().sort_index().values
    w = torch.tensor(len(tr) / (4 * counts), dtype=torch.float).cuda()

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels,
                                                     weight=w)
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir="/content/type_out", num_train_epochs=4,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no",
        warmup_steps=300, fp16=True, seed=SEED, report_to="none")

    t = WTrainer(model=model, args=args,
                 train_dataset=PairDS(tr, tok, 'tid'),
                 eval_dataset=PairDS(te, tok, 'tid'))
    t.train()

    preds = t.predict(PairDS(te, tok, 'tid')).predictions.argmax(-1)
    out = te[['htype', 'difficulty']].copy()
    out['true_tid'] = te['tid'].astype(int).values
    out['pred_tid'] = preds
    out.to_csv(f"{DRIVE}/type_predictions.csv", index=False)
    model.save_pretrained(f"{DRIVE}/type_model_v2")
    return out


# =====================================================================
# RUN
# =====================================================================