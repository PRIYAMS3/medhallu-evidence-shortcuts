# ============================================================
# CELL 16 — New experiments for the revision.  STANDALONE.
#
# Needs only: Cell 1 (Drive mount) run first, and an A100 runtime.
# Does NOT need any earlier cell to have been run in this session.
#
#   P0  fact checks (CPU, ~2 min)
#       - are MedHallu's correct answers PubMedQA's long answers?
#       - how the random / zero-shot NLI baselines were defined
#       - HaluEval answer lengths (why the TF-IDF cues reverse)
#       - AUROC of the contradiction score (rank effect size)
#   P1  length-matched subset (CPU, seconds; uses saved predictions)
#   P2  answer-only detector, retrained once and SAVED (GPU, ~3 min)
#   P3  style-controlled test set: an LLM rewrites answers so both
#       classes share one register, then every model is re-scored
#       (GPU, ~30-40 min)
#   P4  the same ablation inside HaluEval QA (GPU, ~25 min)
#
# Everything caches to Drive under medhallu_v2/review_v16/. If Colab
# disconnects: run Cell 1, then this cell again. Finished parts are
# skipped; an interrupted rewrite resumes where it stopped.
#
# At the end it prints a "PASTE THIS BACK" results summary,
# plus the file review_v16/p3_manual_check.csv (50 rewrites for you
# to read, see the instructions printed at the end).
# ============================================================
import os, json, re, time, gc
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorWithPadding)
from datasets import load_dataset

DRIVE = "/content/drive/MyDrive/medhallu_v2"
OUT = f"{DRIVE}/review_v16"
assert os.path.exists(f"{DRIVE}/train_df.csv"), "Run Cell 1 (Drive mount) first"
os.makedirs(OUT, exist_ok=True)
BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
REWRITER = "Qwen/Qwen2.5-7B-Instruct"
RESULTS = {}


def load_split(name):
    d = pd.read_csv(f"{DRIVE}/{name}_df.csv")
    for c in ['question', 'knowledge', 'answer']:
        d[c] = d[c].fillna('').astype(str)
    return d


train_df, test_df = load_split("train"), load_split("test")
y_te = test_df['label'].values
assert (y_te[0::2] == 0).all() and (y_te[1::2] == 1).all(), \
    "expected rows to alternate correct / hallucinated per item"


def cached(name, fn):
    path = f"{OUT}/{name}.json"
    if os.path.exists(path):
        print(f"[{name}] cached")
        return json.load(open(path))
    t0 = time.time()
    r = fn()
    json.dump(r, open(path, "w"), indent=1)
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    return r


def score(p, y):
    p = np.asarray(p, float); y = np.asarray(y)
    pred = (p >= .5).astype(int)
    out = {'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
           'auroc': round(float(roc_auc_score(y, p)), 4)}
    if len(y) % 2 == 0 and (y[0::2] == 0).all() and (y[1::2] == 1).all():
        out['pairwise_acc'] = round(float((p[1::2] > p[0::2]).mean()), 4)
        out['correct_flagged'] = round(float(pred[y == 0].mean()), 4)
    return out


def words(s):
    return len(str(s).split())


# =====================================================================
# P0 — fact checks
# =====================================================================
def p0():
    r = {}
    # (a) provenance of MedHallu's correct answers
    pq = load_dataset("qiaojin/PubMedQA", "pqa_labeled")['train']
    la = {q.strip(): a.strip() for q, a in zip(pq['question'], pq['long_answer'])}
    corr = test_df[test_df.label == 0]
    hit = [la.get(q.strip()) for q in corr['question']]
    found = [h is not None for h in hit]
    exact = [h is not None and h == a.strip() for h, a in zip(hit, corr['answer'])]
    r['provenance'] = {
        'questions_found_in_pubmedqa': int(sum(found)),
        'correct_answer_equals_long_answer': int(sum(exact)),
        'n': int(len(corr)),
        'example_medhallu': corr['answer'].iloc[0][:300],
        'example_pubmedqa_long_answer': (hit[0] or '')[:300]}
    # (b) baseline definitions
    nli = pd.read_csv(f"{DRIVE}/nli_test.csv")
    cand = {'phi_gt_0': (nli['phi'] > 0).astype(int),
            'argmax_is_contradiction':
                (nli[['contradiction', 'entailment', 'neutral']].values.argmax(1) == 0).astype(int),
            'not_entailment_argmax':
                (nli[['contradiction', 'entailment', 'neutral']].values.argmax(1) != 1).astype(int)}
    r['zero_shot_nli_candidates'] = {k: round(float(f1_score(y_te, v, average='macro')), 4)
                                     for k, v in cand.items()}
    r['contradiction_score_auroc'] = round(float(roc_auc_score(y_te, nli['contradiction'])), 4)
    for f in ['results_v3_SUPERSEDED.json', 'results_v3.json', 'results_v2.json']:
        pth = f"{DRIVE}/{f}"
        if os.path.exists(pth):
            j = json.load(open(pth))
            r['stored_baselines'] = j.get('baselines_v2', j.get('baselines', 'not found'))
            r['stored_baselines_file'] = f
            break
    # (c) HaluEval lengths
    he = load_dataset("pminervini/HaluEval", "qa")['data']
    rl = np.array([words(a) for a in he['right_answer']])
    hl = np.array([words(a) for a in he['hallucinated_answer']])
    r['halueval_lengths'] = {'right_median_words': float(np.median(rl)),
                             'halluc_median_words': float(np.median(hl)),
                             'halluc_longer_pct': round(float((hl > rl).mean() * 100), 1)}
    return r


# =====================================================================
# P1 — length-matched subset, from predictions already on Drive
# =====================================================================
def p1():
    lc = test_df['answer'].iloc[0::2].map(words).values
    lh = test_df['answer'].iloc[1::2].map(words).values
    diff = np.abs(lc - lh)
    enc = np.load(f"{DRIVE}/all_encoder_predictions.npz")
    rv = f"{DRIVE}/review_v14"
    models = {
        'full_mean5': np.mean([enc[f"BiomedBERT_{s}"] for s in [42, 123, 456, 789, 1234]], 0),
        'answer_only_mean5': np.mean(
            [np.load(f"{DRIVE}/shortcut_answer_only_probs.npy")] +
            [np.load(f"{rv}/a2_answer_only_{s}_probs.npy") for s in [123, 456, 789, 1234]], 0),
        'tfidf_answer_only': np.load(f"{rv}/a9_tfidf_test_probs.npy"),
        'full_detector_seed42_saved': np.load(f"{rv}/a1_test_time_probs.npz")['full'],
    }
    r = {'items_total': int(len(diff))}
    for tol in [3, 5]:
        keep = diff <= tol
        rows = np.repeat(keep, 2)
        r[f'within_{tol}_words'] = {'items': int(keep.sum())}
        for k, p in models.items():
            r[f'within_{tol}_words'][k] = score(p[rows], y_te[rows])
    return r


# =====================================================================
# Shared model helpers
# =====================================================================
class PairDS(torch.utils.data.Dataset):
    def __init__(self, df, tok):
        self.a = [(f"{q} {k}".strip() or "[EMPTY]") for q, k in zip(df['question'], df['knowledge'])]
        self.b = [(str(x).strip() or "[EMPTY]") for x in df['answer']]
        self.y = df['label'].astype(int).values
        self.tok = tok
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        e = self.tok(self.a[i], self.b[i], truncation='longest_first', max_length=512)
        e['labels'] = int(self.y[i])
        return e


def train_model(model_name, tr, seed, out_dir=None):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=f"/content/tmp_{seed}", num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, weight_decay=0.01, eval_strategy="no",
        save_strategy="no", warmup_steps=500, logging_steps=1000,
        fp16=True, seed=seed, report_to="none")
    t = Trainer(model=model, args=args, train_dataset=PairDS(tr, tok),
                data_collator=DataCollatorWithPadding(tok))
    t.train()
    if out_dir:
        model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return model.eval(), tok


@torch.no_grad()
def predict(model, tok, df, batch=64):
    model = model.cuda().eval()
    a = [(f"{q} {k}".strip() or "[EMPTY]") for q, k in zip(df['question'], df['knowledge'])]
    b = [(str(x).strip() or "[EMPTY]") for x in df['answer']]
    out = []
    for i in range(0, len(df), batch):
        e = tok(a[i:i+batch], b[i:i+batch], truncation='longest_first',
                max_length=512, padding=True, return_tensors='pt').to('cuda')
        out.append(torch.softmax(model(**e).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def shuffle_df(df, seed=0):
    d = df.copy().reset_index(drop=True); n = len(d)
    perm = np.random.default_rng(seed).permutation(n)
    while (perm == np.arange(n)).any():
        same = perm == np.arange(n); perm[same] = (perm[same] + 1) % n
    d['knowledge'] = d['knowledge'].values[perm]; d['question'] = d['question'].values[perm]
    return d


def free():
    gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# P2 — answer-only BiomedBERT, trained once and saved for reuse
# =====================================================================
AO_DIR = f"{OUT}/answer_only_seed42"


def p2():
    tr = train_df.assign(question='', knowledge='')
    m, t = train_model(BIOMEDBERT, tr, 42, AO_DIR)
    r = score(predict(m, t, test_df.assign(question='', knowledge='')), y_te)
    del m; free()
    return r


# =====================================================================
# P3 — style-controlled test set
# =====================================================================
TO_ASSERTIVE = """Rewrite the answer below to a biomedical research question.

Keep the meaning and the conclusion exactly the same. Do not add or remove any fact, number, drug, gene, population, mechanism or outcome, and do not correct anything.
Change only the style: write it as one or two short, direct, declarative sentences in the third person, stating the finding plainly. Do not use first-person words (we, our, us) or phrases such as "this study", "these results", "these findings", "further studies", "may", "might", "could" or "suggest".
Aim for about {n} words.

Answer:
{answer}

Rewritten answer:"""

TO_HEDGED = """Rewrite the answer below to a biomedical research question.

Keep the meaning and every claim exactly the same, including any claim that is wrong. Do not add or remove any fact, number, drug, gene, population, mechanism or outcome, and do not correct anything.
Change only the style: write it the way the authors of a study would phrase the conclusion of their abstract, in the first person plural, with cautious academic hedging (for example "our results suggest", "may", "these findings indicate", "further studies are needed").
Aim for about {n} words.

Answer:
{answer}

Rewritten answer:"""


def rewrite_all(texts, targets, template, cache_csv, batch=24):
    done = pd.read_csv(cache_csv) if os.path.exists(cache_csv) else pd.DataFrame(columns=['i', 'text'])
    start = len(done)
    if start >= len(texts):
        return done.sort_values('i')['text'].tolist()
    tok = AutoTokenizer.from_pretrained(REWRITER); tok.padding_side = 'left'
    lm = AutoModelForCausalLM.from_pretrained(REWRITER, torch_dtype=torch.bfloat16,
                                              device_map='cuda').eval()
    rows = done.to_dict('records')
    for i in range(start, len(texts), batch):
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": template.format(answer=t, n=n)}],
            tokenize=False, add_generation_prompt=True)
            for t, n in zip(texts[i:i+batch], targets[i:i+batch])]
        enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=160, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        outs = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for j, o in enumerate(outs):
            o = o.strip().strip('"').strip()
            o = re.sub(r'^(Rewritten answer:\s*)', '', o, flags=re.I).strip()
            rows.append({'i': i + j, 'text': o})
        pd.DataFrame(rows).to_csv(cache_csv, index=False)
        print(f"  {os.path.basename(cache_csv)}: {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    del lm; free()
    return pd.DataFrame(rows).sort_values('i')['text'].tolist()


HEDGE = r"\b(may|might|could|suggest\w*|our|we|us|these (results|findings)|this study|further (studies|research))\b"


@torch.no_grad()
def nli_entails(premises, hyps, batch=32):
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    m = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).cuda().eval().half()
    ent = [i for i, l in m.config.id2label.items() if l.lower() == 'entailment'][0]
    out = []
    for i in range(0, len(premises), batch):
        e = tok(premises[i:i+batch], hyps[i:i+batch], truncation=True, max_length=512,
                padding=True, return_tensors='pt').to('cuda')
        out.append(torch.softmax(m(**e).logits.float(), -1).argmax(-1).cpu().numpy() == ent)
    del m; free()
    return np.concatenate(out)


def p3():
    corr = test_df.iloc[0::2].reset_index(drop=True)
    hall = test_df.iloc[1::2].reset_index(drop=True)
    new_corr = rewrite_all(corr['answer'].tolist(), hall['answer'].map(words).tolist(),
                           TO_ASSERTIVE, f"{OUT}/p3_correct_assertive.csv")
    new_hall = rewrite_all(hall['answer'].tolist(), corr['answer'].map(words).tolist(),
                           TO_HEDGED, f"{OUT}/p3_halluc_hedged.csv")

    def interleave(c_ans, h_ans):
        d = test_df.copy()
        d.loc[d.index[0::2], 'answer'] = list(c_ans)
        d.loc[d.index[1::2], 'answer'] = list(h_ans)
        return d

    sets = {'original': test_df,
            'correct_restyled': interleave(new_corr, hall['answer']),
            'both_swapped': interleave(new_corr, new_hall)}

    r = {'style_stats': {}}
    for name, d in sets.items():
        c, h = d['answer'].iloc[0::2], d['answer'].iloc[1::2]
        r['style_stats'][name] = {
            'correct_median_words': float(c.map(words).median()),
            'halluc_median_words': float(h.map(words).median()),
            'correct_hedge_terms_per_answer': round(float(c.str.lower().str.count(HEDGE).mean()), 2),
            'halluc_hedge_terms_per_answer': round(float(h.str.lower().str.count(HEDGE).mean()), 2)}

    # meaning preservation: does the original entail the rewrite, and vice versa?
    f1 = nli_entails(corr['answer'].tolist(), list(new_corr))
    b1 = nli_entails(list(new_corr), corr['answer'].tolist())
    f2 = nli_entails(hall['answer'].tolist(), list(new_hall))
    b2 = nli_entails(list(new_hall), hall['answer'].tolist())
    r['meaning_preserved_pct'] = {
        'correct_orig_entails_rewrite': round(float(f1.mean() * 100), 1),
        'correct_mutual_entailment': round(float((f1 & b1).mean() * 100), 1),
        'halluc_orig_entails_rewrite': round(float(f2.mean() * 100), 1),
        'halluc_mutual_entailment': round(float((f2 & b2).mean() * 100), 1)}
    keep = f1 & b1 & f2 & b2
    r['items_both_rewrites_mutually_entailed'] = int(keep.sum())

    # 50 random items for Priya to read by hand
    idx = np.random.default_rng(0).choice(len(corr), 50, replace=False)
    pd.DataFrame({'item': idx, 'question': corr['question'].values[idx],
                  'correct_original': corr['answer'].values[idx],
                  'correct_rewritten': np.array(new_corr)[idx],
                  'halluc_original': hall['answer'].values[idx],
                  'halluc_rewritten': np.array(new_hall)[idx],
                  'correct_meaning_kept (Y/N)': '', 'halluc_meaning_kept (Y/N)': ''}
                 ).to_csv(f"{OUT}/p3_manual_check.csv", index=False)

    # models to score
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=50000, lowercase=True)
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(
        vec.fit_transform(train_df['answer']), train_df['label'].values)
    full_tok = AutoTokenizer.from_pretrained(f"{DRIVE}/biomedbert_v2_tok")
    full_m = AutoModelForSequenceClassification.from_pretrained(f"{DRIVE}/biomedbert_v2")
    ao_tok = AutoTokenizer.from_pretrained(AO_DIR)
    ao_m = AutoModelForSequenceClassification.from_pretrained(AO_DIR)

    rows_keep = np.repeat(keep, 2)
    for name, d in sets.items():
        res = {
            'tfidf_answer_only': score(lr.predict_proba(vec.transform(d['answer']))[:, 1], y_te),
            'biomedbert_answer_only': score(predict(ao_m, ao_tok, d.assign(question='', knowledge='')), y_te),
            'biomedbert_full': score(pf := predict(full_m, full_tok, d), y_te),
            'biomedbert_full_shuffled_evidence': score(predict(full_m, full_tok, shuffle_df(d)), y_te),
        }
        res['biomedbert_full_meaning_kept_subset'] = score(pf[rows_keep], y_te[rows_keep])
        r[name] = res
        print(name, json.dumps(res), flush=True)
    del full_m, ao_m; free()
    return r


# =====================================================================
# P4 — the same controls inside HaluEval QA (train and test on HaluEval)
# =====================================================================
def p4():
    he = load_dataset("pminervini/HaluEval", "qa")['data'].to_pandas()
    qs = he['question'].unique()
    rng = np.random.default_rng(42); rng.shuffle(qs)
    test_q = set(qs[:2000])
    rows = []
    for _, ex in he.iterrows():
        base = {'question': ex['question'], 'knowledge': ex['knowledge'],
                'split': 'test' if ex['question'] in test_q else 'train'}
        rows.append({**base, 'answer': ex['right_answer'], 'label': 0})
        rows.append({**base, 'answer': ex['hallucinated_answer'], 'label': 1})
    d = pd.DataFrame(rows)
    for c in ['question', 'knowledge', 'answer']:
        d[c] = d[c].fillna('').astype(str)
    tr = d[d.split == 'train'].reset_index(drop=True)
    te = d[d.split == 'test'].reset_index(drop=True)
    yt = te['label'].values
    r = {'n_train_pairs': int(len(tr)), 'n_test_pairs': int(len(te)),
         'question_overlap': len(set(tr.question) & set(te.question))}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=50000, lowercase=True)
    lr = LogisticRegression(max_iter=5000).fit(vec.fit_transform(tr['answer']), tr['label'])
    r['tfidf_answer_only'] = score(lr.predict_proba(vec.transform(te['answer']))[:, 1], yt)
    L = lambda x: np.log1p(x['answer'].map(words).values).reshape(-1, 1)
    r['length_only'] = score(LogisticRegression().fit(L(tr), tr['label']).predict_proba(L(te))[:, 1], yt)

    conds = {'full': lambda x: x,
             'answer_only': lambda x: x.assign(question='', knowledge=''),
             'shuffled': lambda x: shuffle_df(x, 1)}
    MODEL = "roberta-base"   # general-domain corpus, general-domain encoder
    per = []
    for cond, fn in conds.items():
        for seed in [42, 123, 456]:
            cp = f"{OUT}/p4_{cond}_{seed}.json"
            if os.path.exists(cp):
                per.append(json.load(open(cp))); continue
            m, t = train_model(MODEL, fn(tr), seed)
            res = {'condition': cond, 'seed': seed, **score(predict(m, t, fn(te)), yt)}
            if cond == 'full' and seed == 42:
                res['test_time_shuffled'] = score(predict(m, t, shuffle_df(te, 2)), yt)
                res['test_time_evidence_removed'] = score(
                    predict(m, t, te.assign(question='', knowledge='')), yt)
            json.dump(res, open(cp, 'w')); per.append(res)
            print(res, flush=True)
            del m; free()
    r['per_run'] = per
    df = pd.DataFrame(per)
    r['summary'] = {c: {'macro_f1_mean': round(df[df.condition == c].macro_f1.mean(), 4),
                        'macro_f1_sd': round(df[df.condition == c].macro_f1.std(ddof=1), 4),
                        'auroc_mean': round(df[df.condition == c].auroc.mean(), 4)}
                    for c in conds}
    return r


# =====================================================================
# RUN
# =====================================================================
assert torch.cuda.is_available(), "Runtime > Change runtime type > A100 GPU"
print("GPU:", torch.cuda.get_device_name(0))
RESULTS['P0'] = cached('p0_fact_checks', p0)
RESULTS['P1'] = cached('p1_length_matched', p1)
RESULTS['P2'] = cached('p2_answer_only_saved', p2)
RESULTS['P3'] = cached('p3_style_controlled', p3)
RESULTS['P4'] = cached('p4_halueval_within', p4)

print("\n" + "=" * 64 + "\nPASTE THIS BACK\n" + "=" * 64)
print(json.dumps(RESULTS, indent=1))
print("\nALSO: open  MyDrive/medhallu_v2/review_v16/p3_manual_check.csv ,")
print("read the 50 rows, fill the two Y/N columns (did the rewrite keep the")
print("meaning?), and send the file back.")
