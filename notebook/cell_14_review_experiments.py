# ============================================================
# CELL 14 — Reviewer-response experiments
#
#   A9  answer-only surface baselines (length, style, TF-IDF)   CPU, ~3 min
#   A1  test-time ablation of the saved full-input model        GPU, ~5 min
#   A2  training-time evidence ablation across all five seeds   GPU, ~2 h
#
# Run order: Cell 1 (mount), then this cell. Nothing else needed.
# Every unit caches to Drive under medhallu_v2/review_v14/, so if
# Colab disconnects, just re-run Cell 1 and this cell: finished
# units are skipped and only the interrupted run repeats.
#
# At the end it prints a block starting "PASTE THIS BACK". Send
# that block with the results.
# ============================================================
import os, json, re, time
import numpy as np, pandas as pd, torch
from scipy import stats
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DRIVE = "/content/drive/MyDrive/medhallu_v2"
OUT = f"{DRIVE}/review_v14"
assert os.path.exists(f"{DRIVE}/train_df.csv"), "Run Cell 1 (Drive mount) first"
os.makedirs(OUT, exist_ok=True)

BIOMEDBERT = ("microsoft/BiomedNLP-BiomedBERT-base-"
              "uncased-abstract-fulltext")
SEEDS = [42, 123, 456, 789, 1234]

train_df = pd.read_csv(f"{DRIVE}/train_df.csv")
test_df = pd.read_csv(f"{DRIVE}/test_df.csv")
for d in (train_df, test_df):
    n_nan = d[['question', 'knowledge', 'answer']].isna().sum().sum()
    if n_nan:
        print(f"note: {n_nan} empty text fields filled with ''")
    for c in ['question', 'knowledge', 'answer']:
        d[c] = d[c].fillna('').astype(str)

y_tr = train_df['label'].values
y_te = test_df['label'].values
hard = (test_df['difficulty'] == 'hard').values


def score(p, y=y_te, mask=hard):
    """Macro F1, hard-tier macro F1 and AUROC from P(hallucinated)."""
    pred = (np.asarray(p) >= 0.5).astype(int)
    out = {'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
           'auroc': round(float(roc_auc_score(y, p)), 4)}
    if mask is not None:
        out['hard_f1'] = round(float(f1_score(y[mask], pred[mask],
                                              average='macro')), 4)
    return out


def shuffle_df(df, rng):
    """Give every row the question and passage of a different row."""
    d = df.copy().reset_index(drop=True)
    n = len(d)
    perm = rng.permutation(n)
    while True:
        same = perm == np.arange(n)
        if not same.any():
            break
        perm[same] = (perm[same] + 1) % n
    d['knowledge'] = d['knowledge'].values[perm]
    d['question'] = d['question'].values[perm]
    return d


# =====================================================================
# A9 — answer-only surface baselines.  No GPU, no evidence (except the
#      overlap feature, which is labelled as evidence-using).
# =====================================================================
HEDGE = (r"\b(may|might|could|possibly|potentially|suggest(s|ed|ing)?|"
         r"likely|appear(s|ed)?|seem(s|ed)?|unclear)\b")
NEG = r"\b(not|no|neither|nor|without|never)\b|n't"
CERT = r"\b(clearly|definitely|significantly|strongly|always|demonstrates?|proves?|confirms?|establish(es|ed)?)\b"
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on and or to for with is are was were be been by as at "
           "that this from it its which these those than into".split())


def style_features(df):
    a = df['answer']
    low = a.str.lower()
    n_words = a.str.split().str.len().fillna(0)
    f = pd.DataFrame({
        'log_words': np.log1p(n_words),
        'log_chars': np.log1p(a.str.len()),
        'n_sents': a.str.count(r"[.!?](\s|$)"),
        'avg_word_len': a.str.replace(r"\s+", "", regex=True).str.len()
                        / n_words.clip(lower=1),
        'digits': low.str.count(r"\d"),
        'commas': a.str.count(","),
        'parens': a.str.count(r"\("),
        'hedges': low.str.count(HEDGE),
        'negations': low.str.count(NEG),
        'certainty': low.str.count(CERT),
        'starts_yes_no': low.str.match(r"^\s*(yes|no)\b").astype(int),
    })
    return f.fillna(0)


def overlap_feature(df):
    """Share of the answer's content words that appear in the passage.
    This one DOES read the evidence, so it is reported separately."""
    vals = []
    for a, k in zip(df['answer'], df['knowledge']):
        aw = {w for w in TOKEN.findall(a.lower()) if w not in STOP}
        kw = set(TOKEN.findall(k.lower()))
        vals.append(len(aw & kw) / len(aw) if aw else 0.0)
    return np.array(vals).reshape(-1, 1)


def load_halueval():
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa")['data'].select(range(2000))
    rows = []
    for ex in ds:
        base = {'question': ex['question'], 'knowledge': ex['knowledge']}
        rows.append({**base, 'answer': ex['right_answer'], 'label': 0})
        rows.append({**base, 'answer': ex['hallucinated_answer'], 'label': 1})
    return pd.DataFrame(rows)


def a9_baselines():
    path = f"{OUT}/a9_baselines.json"
    if os.path.exists(path):
        print("A9: cached")
        return json.load(open(path))
    t0 = time.time()
    res = {}

    # descriptive length statistics
    desc = {}
    for name, d in [('train', train_df), ('test', test_df)]:
        w = d['answer'].str.split().str.len()
        desc[name] = {'correct_median_words': float(w[d.label == 0].median()),
                      'halluc_median_words': float(w[d.label == 1].median()),
                      'correct_mean_words': round(float(w[d.label == 0].mean()), 1),
                      'halluc_mean_words': round(float(w[d.label == 1].mean()), 1)}
    res['length_stats'] = desc

    lr = lambda: LogisticRegression(max_iter=5000, C=1.0)

    # 1) length only
    Xtr, Xte = style_features(train_df), style_features(test_df)
    m = make_pipeline(StandardScaler(), lr()).fit(Xtr[['log_words']], y_tr)
    res['length_only'] = score(m.predict_proba(Xte[['log_words']])[:, 1])

    # 2) all hand-crafted style features
    m_style = make_pipeline(StandardScaler(), lr()).fit(Xtr, y_tr)
    res['style_features'] = score(m_style.predict_proba(Xte)[:, 1])
    coefs = m_style[-1].coef_[0]
    res['style_coefficients'] = {c: round(float(v), 3)
                                 for c, v in sorted(zip(Xtr.columns, coefs),
                                                    key=lambda t: -abs(t[1]))}

    # 3) TF-IDF bag of words on the answer alone
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=50000, lowercase=True)
    Atr = vec.fit_transform(train_df['answer'])
    Ate = vec.transform(test_df['answer'])
    m_tfidf = LogisticRegression(max_iter=5000, C=1.0).fit(Atr, y_tr)
    p_tfidf = m_tfidf.predict_proba(Ate)[:, 1]
    res['tfidf_answer_only'] = score(p_tfidf)
    np.save(f"{OUT}/a9_tfidf_test_probs.npy", p_tfidf)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(m_tfidf.coef_[0])
    res['tfidf_top_hallucinated'] = names[order[::-1][:30]].tolist()
    res['tfidf_top_correct'] = names[order[:30]].tolist()

    # 4) lexical overlap with the passage (uses evidence)
    Otr, Ote = overlap_feature(train_df), overlap_feature(test_df)
    m = LogisticRegression(max_iter=5000).fit(Otr, y_tr)
    res['overlap_with_evidence'] = score(m.predict_proba(Ote)[:, 1])
    res['overlap_means'] = {
        'correct': round(float(Ote[y_te == 0].mean()), 4),
        'hallucinated': round(float(Ote[y_te == 1].mean()), 4)}

    # 5) does the TF-IDF shortcut transfer to HaluEval?
    try:
        ext = load_halueval()
        pe = m_tfidf.predict_proba(vec.transform(ext['answer']))[:, 1]
        res['tfidf_halueval'] = score(pe, y=ext['label'].values, mask=None)
    except Exception as e:
        res['tfidf_halueval'] = {'error': str(e)}

    json.dump(res, open(path, "w"), indent=2)
    print(f"A9 done in {time.time()-t0:.0f}s")
    return res


# =====================================================================
# A1 — test-time ablation of the SAVED full-input model (biomedbert_v2).
#      No training. Answers: does the model that was trained WITH
#      evidence use the evidence when it predicts?
# =====================================================================
@torch.no_grad()
def predict_saved(model, tok, df, batch=64):
    seg1 = [f"{q} {k}" if (q or k) else "" for q, k in
            zip(df['question'], df['knowledge'])]
    seg2 = list(df['answer'])
    probs = []
    for i in range(0, len(df), batch):
        enc = tok(seg1[i:i+batch], seg2[i:i+batch],
                  truncation='longest_first', max_length=512,
                  padding=True, return_tensors='pt').to('cuda')
        probs.append(torch.softmax(model(**enc).logits.float(), -1)
                     .cpu().numpy()[:, 1])
    return np.concatenate(probs)


def a1_test_time():
    path = f"{OUT}/a1_test_time.json"
    if os.path.exists(path):
        print("A1: cached")
        return json.load(open(path))
    assert torch.cuda.is_available(), "A1 needs a GPU runtime"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(f"{DRIVE}/biomedbert_v2_tok")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{DRIVE}/biomedbert_v2").cuda().eval()

    variants = {
        'full': test_df,
        'evidence_removed': test_df.assign(question='', knowledge=''),
        'evidence_shuffled': shuffle_df(test_df, np.random.default_rng(0)),
        'answer_removed': test_df.assign(answer=''),
    }
    res, P = {}, {}
    for name, d in variants.items():
        P[name] = predict_saved(model, tok, d)
        res[name] = score(P[name])
        print(f"  A1 {name:18s} {res[name]}", flush=True)
    base = (P['full'] >= .5)
    for name in ['evidence_removed', 'evidence_shuffled']:
        res[name]['pred_changed_vs_full_pct'] = round(
            float(((P[name] >= .5) != base).mean() * 100), 2)
        res[name]['mean_abs_prob_change'] = round(
            float(np.abs(P[name] - P['full']).mean()), 4)
    np.savez(f"{OUT}/a1_test_time_probs.npz", **P)
    json.dump(res, open(path, "w"), indent=2)
    del model; torch.cuda.empty_cache()
    print(f"A1 done in {time.time()-t0:.0f}s")
    return res


# =====================================================================
# A2 — training-time ablation, five seeds.
#   full        : the five BiomedBERT runs already in
#                 all_encoder_predictions.npz (same recipe, same seeds),
#                 so the paper has ONE full-input number everywhere.
#   answer_only : seed 42 from Cell 8 cache, seeds 123..1234 trained here
#   shuffled    : seed 42 from Cell 8 cache, seeds 123..1234 trained here
#   evidence_only (control): single run from Cell 8, reported as is.
# Same recipe as Cell 8 ("[EMPTY]" placeholder for removed segments).
# Padding is dynamic instead of fixed 512: masked positions do not
# affect the output, it only makes the short answer-only runs faster.
# =====================================================================
class AblDS(torch.utils.data.Dataset):
    def __init__(self, df, tok):
        self.a = [(f"{q} {k}".strip() or "[EMPTY]")
                  for q, k in zip(df['question'], df['knowledge'])]
        self.b = [(str(x).strip() or "[EMPTY]") for x in df['answer']]
        self.y = df['label'].astype(int).values
        self.tok = tok
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        enc = self.tok(self.a[i], self.b[i], truncation='longest_first',
                       max_length=512)
        enc['labels'] = int(self.y[i])
        return enc


def train_ablation(cond, seed):
    path = f"{OUT}/a2_{cond}_{seed}.json"
    if os.path.exists(path):
        print(f"A2 {cond} seed {seed}: cached")
        return json.load(open(path))
    assert torch.cuda.is_available(), "A2 needs a GPU runtime"
    t0 = time.time()
    if cond == 'answer_only':
        tr = train_df.assign(question='', knowledge='')
        te = test_df.assign(question='', knowledge='')
    elif cond == 'shuffled':
        rng = np.random.default_rng(seed)
        tr, te = shuffle_df(train_df, rng), shuffle_df(test_df, rng)
    else:
        raise ValueError(cond)

    tok = AutoTokenizer.from_pretrained(BIOMEDBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BIOMEDBERT, num_labels=2)
    args = TrainingArguments(
        output_dir=f"/content/a2_{cond}_{seed}", num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="no", save_strategy="no", warmup_steps=500,
        logging_steps=1000, fp16=True, seed=seed, report_to="none")
    t = Trainer(model=model, args=args, train_dataset=AblDS(tr, tok),
                data_collator=DataCollatorWithPadding(tok))
    t.train()
    logits = t.predict(AblDS(te, tok)).predictions
    p = torch.softmax(torch.tensor(logits), -1).numpy()[:, 1]
    res = {'condition': cond, 'seed': seed, **score(p),
           'minutes': round((time.time() - t0) / 60, 1)}
    np.save(f"{OUT}/a2_{cond}_{seed}_probs.npy", p)
    json.dump(res, open(path, "w"), indent=2)
    del model, t; torch.cuda.empty_cache()
    print(f"A2 {cond} seed {seed}: {res}", flush=True)
    return res


def a2_multiseed():
    rows = []
    enc = np.load(f"{DRIVE}/all_encoder_predictions.npz")
    for s in SEEDS:
        rows.append({'condition': 'full', 'seed': s,
                     **score(enc[f"BiomedBERT_{s}"])})
    for cond in ['answer_only', 'shuffled']:
        for s in SEEDS:
            cached = f"{DRIVE}/shortcut_{cond}.json"
            if s == 42 and os.path.exists(cached):
                r = json.load(open(cached))
                rows.append({'condition': cond, 'seed': 42,
                             'macro_f1': r['macro_f1'],
                             'hard_f1': r['hard_f1'], 'auroc': r['auroc']})
            else:
                r = train_ablation(cond, s)
                rows.append({k: r[k] for k in
                             ['condition', 'seed', 'macro_f1', 'hard_f1',
                              'auroc']})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/a2_multiseed.csv", index=False)

    tcrit = stats.t.ppf(.975, len(SEEDS) - 1)
    summ = {}
    for cond in ['full', 'answer_only', 'shuffled']:
        d = df[df.condition == cond].set_index('seed').loc[SEEDS]
        summ[cond] = {}
        for m in ['macro_f1', 'hard_f1', 'auroc']:
            v = d[m].values
            mu, sd = v.mean(), v.std(ddof=1)
            h = tcrit * sd / np.sqrt(len(v))
            summ[cond][m] = {'mean': round(mu, 4), 'sd': round(sd, 4),
                             'ci95': [round(mu - h, 4), round(mu + h, 4)]}
    full = df[df.condition == 'full'].set_index('seed').loc[SEEDS]['macro_f1'].values
    paired = {}
    for cond in ['answer_only', 'shuffled']:
        v = df[df.condition == cond].set_index('seed').loc[SEEDS]['macro_f1'].values
        diff = full - v
        h = tcrit * diff.std(ddof=1) / np.sqrt(len(diff))
        tstat, p = stats.ttest_rel(full, v)
        paired[cond] = {
            'mean_drop': round(float(diff.mean()), 4),
            'ci95': [round(float(diff.mean() - h), 4),
                     round(float(diff.mean() + h), 4)],
            'p': round(float(p), 4),
            'pct_of_full': round(float((v / full).mean() * 100), 1),
            'pct_of_above_chance': round(float(((v - .5) / (full - .5)).mean()
                                               * 100), 1)}
    ctrl = json.load(open(f"{DRIVE}/shortcut_evidence_only.json"))
    out = {'per_seed': rows, 'summary': summ, 'paired_vs_full': paired,
           'evidence_only_control_seed42': ctrl}
    json.dump(out, open(f"{OUT}/a2_summary.json", "w"), indent=2)
    return out


# =====================================================================
# RUN — fastest first, so useful results arrive early.
# =====================================================================
r9 = a9_baselines()
print(json.dumps({k: r9[k] for k in ['length_only', 'style_features',
                                     'tfidf_answer_only']}, indent=1))
r1 = a1_test_time()
r2 = a2_multiseed()

print("\n" + "=" * 64)
print("PASTE THIS BACK")
print("=" * 64)
print(json.dumps({'A9': r9, 'A1': r1,
                  'A2_summary': r2['summary'],
                  'A2_paired_vs_full': r2['paired_vs_full'],
                  'A2_per_seed': r2['per_seed'],
                  'A2_control': r2['evidence_only_control_seed42']},
                 indent=1))
