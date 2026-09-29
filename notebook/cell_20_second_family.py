# ============================================================
# CELL 20 — Independence from the Qwen family.  STANDALONE.
# A100 runtime. Run Cell 1 (Drive mount) first, then this cell.
# About 90 minutes. Resumable: re-run Cell 1 + this cell after a
# disconnect; finished parts are skipped.
#
#   R1  Rewrite the test answers with a second model family
#       (microsoft/phi-4), check language + meaning, and export
#       50 items for a manual check.
#   R2  Score the five BiomedBERT seeds (full + answer-only) and
#       TF-IDF on the Phi-4 rewrites; gaps with bootstrap CIs.
#   R3  A second prompted detector from a third family
#       (allenai/OLMo-2-1124-7B-Instruct) on all test sets, and the
#       Qwen detector on the Phi-4 rewrites.
#
# Outputs go to medhallu_v2/review_v20/. Prints "PASTE THIS BACK".
# ============================================================
import os, re, json, time, gc
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorWithPadding)

DRIVE = "/content/drive/MyDrive/medhallu_v2"
V16, V19, OUT = f"{DRIVE}/review_v16", f"{DRIVE}/review_v19", f"{DRIVE}/review_v20"
assert os.path.exists(f"{DRIVE}/train_df.csv"), "Run Cell 1 (Drive mount) first"
assert torch.cuda.is_available(), "Runtime > Change runtime type > A100"
os.makedirs(OUT, exist_ok=True)
BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
REWRITER2 = "microsoft/phi-4"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
OLMO = "allenai/OLMo-2-1124-7B-Instruct"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
SEEDS = [42, 123, 456, 789, 1234]
print("GPU:", torch.cuda.get_device_name(0))


def load(n):
    d = pd.read_csv(f"{DRIVE}/{n}_df.csv")
    for c in ['question', 'knowledge', 'answer']:
        d[c] = d[c].fillna('').astype(str)
    return d


train_df, test_df = load("train"), load("test")
y_all = test_df['label'].values
assert (y_all[0::2] == 0).all() and (y_all[1::2] == 1).all()
NONLATIN = re.compile(r"[Ѐ-ӿ؀-ۿ　-鿿가-힯]")


def words(s): return len(str(s).split())


def free(): gc.collect(); torch.cuda.empty_cache()


def score(p, y):
    p = np.asarray(p, float); pred = (p >= .5).astype(int)
    return {'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
            'auroc': round(float(roc_auc_score(y, p)), 4),
            'pairwise_acc': round(float((p[1::2] > p[0::2]).mean()), 4),
            'correct_flagged': round(float(pred[y == 0].mean()), 4)}


def shuffle_df(df, seed=0):
    d = df.copy().reset_index(drop=True); n = len(d)
    perm = np.random.default_rng(seed).permutation(n)
    while (perm == np.arange(n)).any():
        s = perm == np.arange(n); perm[s] = (perm[s] + 1) % n
    d['knowledge'] = d['knowledge'].values[perm]; d['question'] = d['question'].values[perm]
    return d


def inter(c, h):
    d = test_df.copy()
    d.loc[d.index[0::2], 'answer'] = list(c)
    d.loc[d.index[1::2], 'answer'] = list(h)
    return d


def cached(name, fn):
    path = f"{OUT}/{name}.json"
    if os.path.exists(path):
        print(f"[{name}] cached"); return json.load(open(path))
    t0 = time.time(); r = fn()
    json.dump(r, open(path, "w"), indent=1)
    print(f"[{name}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return r


# =====================================================================
# R1 — rewrite with Phi-4 (same instructions as the Qwen rewrite)
# =====================================================================
TO_ASSERTIVE = """Rewrite the answer below to a biomedical research question.

Keep the meaning and the conclusion exactly the same. Do not add or remove any fact, number, drug, gene, population, mechanism or outcome, and do not correct anything.
Change only the style: write it as one or two short, direct, declarative sentences in the third person, stating the finding plainly. Do not use first-person words (we, our, us) or phrases such as "this study", "these results", "these findings", "further studies", "may", "might", "could" or "suggest".
Aim for about {n} words. Reply in English with the rewritten answer only.

Answer:
{answer}

Rewritten answer:"""

TO_HEDGED = """Rewrite the answer below to a biomedical research question.

Keep the meaning and every claim exactly the same, including any claim that is wrong. Do not add or remove any fact, number, drug, gene, population, mechanism or outcome, and do not correct anything.
Change only the style: write it the way the authors of a study would phrase the conclusion of their abstract, in the first person plural, with cautious academic hedging (for example "our results suggest", "may", "these findings indicate", "further studies are needed").
Aim for about {n} words. Reply in English with the rewritten answer only.

Answer:
{answer}

Rewritten answer:"""


def rewrite_all(texts, targets, template, cache_csv, batch=16):
    done = pd.read_csv(cache_csv) if os.path.exists(cache_csv) else pd.DataFrame(columns=['i', 'text'])
    if len(done) >= len(texts):
        return done.sort_values('i')['text'].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(REWRITER2); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(REWRITER2, dtype=torch.bfloat16,
                                              device_map='cuda').eval()
    rows = done.to_dict('records')
    for i in range(len(done), len(texts), batch):
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": template.format(answer=t, n=n)}],
            tokenize=False, add_generation_prompt=True)
            for t, n in zip(texts[i:i+batch], targets[i:i+batch])]
        enc = tok(prompts, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=160, do_sample=False,
                              pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for j, o in enumerate(outs):
            o = re.sub(r'^(Rewritten answer:\s*)', '', o.strip(), flags=re.I).strip().strip('"').strip()
            rows.append({'i': i + j, 'text': o})
        pd.DataFrame(rows).to_csv(cache_csv, index=False)
        print(f"  {os.path.basename(cache_csv)}: {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    del lm; free()
    return pd.DataFrame(rows).sort_values('i')['text'].astype(str).tolist()


@torch.no_grad()
def nli_entails(prem, hyp, batch=32):
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    m = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).cuda().eval().half()
    ent = [i for i, l in m.config.id2label.items() if l.lower() == 'entailment'][0]
    out = []
    for i in range(0, len(prem), batch):
        e = tok(prem[i:i+batch], hyp[i:i+batch], truncation=True, max_length=512,
                padding=True, return_tensors='pt').to('cuda')
        out.append(m(**e).logits.float().argmax(-1).cpu().numpy() == ent)
    del m; free()
    return np.concatenate(out)


HEDGE = r"\b(may|might|could|suggest\w*|our|we|us|these (results|findings)|this study|further (studies|research))\b"

corr = test_df.iloc[0::2].reset_index(drop=True)
hall = test_df.iloc[1::2].reset_index(drop=True)


def r1():
    pc = rewrite_all(corr['answer'].tolist(), hall['answer'].map(words).tolist(),
                     TO_ASSERTIVE, f"{OUT}/phi_correct_assertive.csv")
    ph = rewrite_all(hall['answer'].tolist(), corr['answer'].map(words).tolist(),
                     TO_HEDGED, f"{OUT}/phi_halluc_hedged.csv")
    bad = np.array([bool(NONLATIN.search(a) or NONLATIN.search(b) or not a.strip() or not b.strip())
                    for a, b in zip(pc, ph)])
    np.save(f"{OUT}/phi_keep_items.npy", ~bad)
    ent = nli_entails(corr['answer'].tolist(), pc)
    idx = np.random.default_rng(1).choice(np.where(~bad)[0], 50, replace=False)
    pd.DataFrame({'item': idx, 'question': corr['question'].values[idx],
                  'correct_original': corr['answer'].values[idx],
                  'correct_rewritten': np.array(pc)[idx],
                  'halluc_original': hall['answer'].values[idx],
                  'halluc_rewritten': np.array(ph)[idx],
                  'correct_meaning_kept (Y/N)': '', 'halluc_meaning_kept (Y/N)': ''}
                 ).to_csv(f"{OUT}/phi_manual_check.csv", index=False)
    c = pd.Series(pc)[~bad]; h = pd.Series(ph)[~bad]
    return {'items_excluded_nonenglish_or_empty': int(bad.sum()),
            'items_kept': int((~bad).sum()),
            'correct_orig_entails_rewrite_pct': round(float(ent[~bad].mean() * 100), 1),
            'correct_rewrite_median_words': float(c.map(words).median()),
            'halluc_rewrite_median_words': float(h.map(words).median()),
            'correct_rewrite_hedges_per_answer': round(float(c.str.lower().str.count(HEDGE).mean()), 2),
            'halluc_rewrite_hedges_per_answer': round(float(h.str.lower().str.count(HEDGE).mean()), 2)}


# =====================================================================
# Shared: test sets
# =====================================================================
def phi_sets():
    pc = pd.read_csv(f"{OUT}/phi_correct_assertive.csv").sort_values('i')['text'].astype(str).tolist()
    ph = pd.read_csv(f"{OUT}/phi_halluc_hedged.csv").sort_values('i')['text'].astype(str).tolist()
    keep = np.load(f"{OUT}/phi_keep_items.npy")
    return {'phi_correct_restyled': inter(pc, hall['answer'].tolist()),
            'phi_both_swapped': inter(pc, ph)}, keep


def qwen_sets():
    cjk = re.compile(r"[　-鿿가-힯]")
    nc = pd.read_csv(f"{V16}/p3_correct_assertive.csv").sort_values('i')['text'].astype(str).tolist()
    nh = pd.read_csv(f"{V16}/p3_halluc_hedged.csv").sort_values('i')['text'].astype(str).tolist()
    keep = np.array([not (cjk.search(a) or cjk.search(b)) for a, b in zip(nc, nh)])
    return {'original': test_df,
            'correct_restyled': inter(nc, hall['answer'].tolist()),
            'both_swapped': inter(nc, nh)}, keep


# =====================================================================
# R2 — five BiomedBERT seeds + TF-IDF on the Phi-4 rewrites
# =====================================================================
class PairDS(torch.utils.data.Dataset):
    def __init__(self, df, tok):
        self.a = [(f"{q} {k}".strip() or "[EMPTY]") for q, k in zip(df['question'], df['knowledge'])]
        self.b = [(str(x).strip() or "[EMPTY]") for x in df['answer']]
        self.y = df['label'].astype(int).values; self.tok = tok
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        e = self.tok(self.a[i], self.b[i], truncation='longest_first', max_length=512)
        e['labels'] = int(self.y[i]); return e


def train(tr, seed):
    tok = AutoTokenizer.from_pretrained(BIOMEDBERT)
    m = AutoModelForSequenceClassification.from_pretrained(BIOMEDBERT, num_labels=2)
    args = TrainingArguments(output_dir=f"/content/r2_{seed}", num_train_epochs=3,
                             per_device_train_batch_size=16, learning_rate=2e-5,
                             weight_decay=0.01, eval_strategy="no", save_strategy="no",
                             warmup_steps=500, logging_steps=1000, fp16=True, seed=seed,
                             report_to="none")
    Trainer(model=m, args=args, train_dataset=PairDS(tr, tok),
            data_collator=DataCollatorWithPadding(tok)).train()
    return m.eval(), tok


@torch.no_grad()
def predict(m, t, d, b=64):
    m = m.cuda().eval()
    a = [(f"{q} {k}".strip() or "[EMPTY]") for q, k in zip(d['question'], d['knowledge'])]
    c = [(str(x).strip() or "[EMPTY]") for x in d['answer']]
    out = []
    for i in range(0, len(d), b):
        e = t(a[i:i+b], c[i:i+b], truncation='longest_first', max_length=512,
              padding=True, return_tensors='pt').to('cuda')
        out.append(torch.softmax(m(**e).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def r2_seed(seed, SETS):
    path = f"{OUT}/r2_seed{seed}.npz"
    if os.path.exists(path):
        print(f"R2 seed {seed}: cached"); return dict(np.load(path))
    t0 = time.time(); P = {}
    if seed == 42:
        fm = AutoModelForSequenceClassification.from_pretrained(f"{DRIVE}/biomedbert_v2")
        ft = AutoTokenizer.from_pretrained(f"{DRIVE}/biomedbert_v2_tok")
    else:
        fm, ft = train(train_df, seed)
    for name, d in SETS.items():
        P[f'{name}__full_correct'] = predict(fm, ft, d)
        P[f'{name}__full_unrelated'] = predict(fm, ft, shuffle_df(d))
    del fm; free()
    if seed == 42:
        am = AutoModelForSequenceClassification.from_pretrained(f"{V16}/answer_only_seed42")
        at = AutoTokenizer.from_pretrained(f"{V16}/answer_only_seed42")
    else:
        am, at = train(train_df.assign(question='', knowledge=''), seed)
    for name, d in SETS.items():
        P[f'{name}__answer_only'] = predict(am, at, d.assign(question='', knowledge=''))
    del am; free()
    np.savez(path, **P)
    print(f"R2 seed {seed}: done in {(time.time()-t0)/60:.1f} min", flush=True)
    return P


def r2():
    SETS, keep = phi_sets()
    rows = np.repeat(keep, 2); y = y_all[rows]; n_items = int(keep.sum())
    allp = {s: r2_seed(s, SETS) for s in SEEDS}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
    lr = LogisticRegression(max_iter=5000).fit(vec.fit_transform(train_df['answer']), train_df['label'])
    res = {'items': n_items}
    rng = np.random.default_rng(0)
    for name, d in SETS.items():
        r = {'tfidf_answer_only': score(lr.predict_proba(vec.transform(d['answer']))[:, 1][rows], y)}
        for mdl in ['full_correct', 'full_unrelated', 'answer_only']:
            per = [score(allp[s][f'{name}__{mdl}'][rows], y) for s in SEEDS]
            r[mdl] = {m: [round(float(np.mean([p[m] for p in per])), 4),
                          round(float(np.std([p[m] for p in per], ddof=1)), 4)] for m in per[0]}
        w = {m: np.stack([(allp[s][f'{name}__{m}'][rows][1::2] > allp[s][f'{name}__{m}'][rows][0::2]
                           ).astype(float) for s in SEEDS]) for m in ['full_correct', 'full_unrelated']}
        diff = w['full_correct'] - w['full_unrelated']
        boots = [diff[:, rng.integers(0, n_items, n_items)].mean() for _ in range(5000)]
        r['gap'] = {'mean': round(float(diff.mean(1).mean()), 4),
                    'sd_over_seeds': round(float(diff.mean(1).std(ddof=1)), 4),
                    'per_seed': [round(float(g), 4) for g in diff.mean(1)],
                    'ci95': [round(float(np.percentile(boots, 2.5)), 4),
                             round(float(np.percentile(boots, 97.5)), 4)]}
        res[name] = r
    return res


# =====================================================================
# R3 — prompted detectors: OLMo-2 on everything, Qwen on Phi-4 sets
# =====================================================================
WITH_PASSAGE = """You are checking an answer to a biomedical research question against a source passage.

Passage:
{k}

Question:
{q}

Answer:
{a}

Is the answer correct and supported by the passage? Reply with Yes or No only."""

NO_PASSAGE = """You are checking an answer to a biomedical research question.

Question:
{q}

Answer:
{a}

Is the answer correct? Reply with Yes or No only."""


@torch.no_grad()
def llm_scores(lm, tok, d, with_passage, b=16):
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    prompts = []
    for q, k, a in zip(d['question'], d['knowledge'], d['answer']):
        k = " ".join(k.split()[:450])
        body = WITH_PASSAGE.format(k=k, q=q, a=a) if with_passage else NO_PASSAGE.format(q=q, a=a)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": body}],
                                               tokenize=False, add_generation_prompt=True))
    out = []
    for i in range(0, len(prompts), b):
        e = tok(prompts[i:i+b], return_tensors='pt', padding=True,
                add_special_tokens=False).to('cuda')
        logits = lm(**e).logits[:, -1, :].float()
        out.append(torch.softmax(logits[:, [yes, no]], -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def run_llm(model_name, tag, sets_and_keep):
    tok = AutoTokenizer.from_pretrained(model_name); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lm = None; res = {}
    for SETS, keep in sets_and_keep:
        rows = np.repeat(keep, 2); y = y_all[rows]
        for name, d in SETS.items():
            dd = d[rows].reset_index(drop=True); res[name] = {}
            for cond, x, wp in [('correct_passage', dd, True),
                                ('unrelated_passage', shuffle_df(dd), True),
                                ('no_passage', dd, False)]:
                path = f"{OUT}/r3_{tag}_{name}_{cond}.npy"
                if os.path.exists(path):
                    p = np.load(path)
                else:
                    if lm is None:
                        lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                                  device_map='cuda').eval()
                    print(f"  R3 {tag} {name} / {cond}", flush=True)
                    p = llm_scores(lm, tok, x, wp); np.save(path, p)
                res[name][cond] = score(p, y)
    if lm is not None:
        del lm; free()
    return res


def r3():
    return {'olmo2_7b': run_llm(OLMO, 'olmo', [qwen_sets(), phi_sets()]),
            'qwen2.5_7b_on_phi_sets': run_llm(QWEN, 'qwen', [phi_sets()])}


R = {'R1': cached('r1_phi_rewrite', r1), 'R2': cached('r2_bert_on_phi', r2),
     'R3': cached('r3_prompted', r3)}
print("\n" + "=" * 64 + "\nPASTE THIS BACK\n" + "=" * 64)
print(json.dumps(R, indent=1))
print("\nALSO: MyDrive/medhallu_v2/review_v20/phi_manual_check.csv -> fill Y/N, send back.")
