# ============================================================
# CELL 19 — Final experiments.  STANDALONE.  A100 runtime.
# Run Cell 1 (Drive mount) first, then this cell.  ~60-75 min.
#
#   Q1  Five-seed robustness of the test-time ablation and the
#       style-controlled test (retrains 4 full + 4 answer-only
#       BiomedBERT models; seed 42 reuses the saved models), with
#       bootstrap confidence intervals over items.
#   Q2  A prompted LLM detector (Qwen2.5-7B-Instruct, zero-shot)
#       on the original and restyled test sets, with the correct
#       passage, an unrelated passage, and no passage.
#
# Everything caches per seed / per condition under
# medhallu_v2/review_v19/. If Colab disconnects: Cell 1, then this
# cell again; finished parts are skipped.
# Prints "PASTE THIS BACK" at the end.
# ============================================================
import os, re, json, time, gc
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorWithPadding)

DRIVE = "/content/drive/MyDrive/medhallu_v2"
V16 = f"{DRIVE}/review_v16"
OUT = f"{DRIVE}/review_v19"
assert os.path.exists(f"{DRIVE}/train_df.csv"), "Run Cell 1 (Drive mount) first"
assert torch.cuda.is_available(), "Runtime > Change runtime type > A100"
os.makedirs(OUT, exist_ok=True)
BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
LLM = "Qwen/Qwen2.5-7B-Instruct"
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

# ---- the three test sets, restricted to the 993 clean items ----------
cjk = re.compile(r"[　-鿿가-힯]")
nc = pd.read_csv(f"{V16}/p3_correct_assertive.csv").sort_values('i')['text'].astype(str).tolist()
nh = pd.read_csv(f"{V16}/p3_halluc_hedged.csv").sort_values('i')['text'].astype(str).tolist()
keep_items = np.array([not (cjk.search(a) or cjk.search(b)) for a, b in zip(nc, nh)])
ROWS = np.repeat(keep_items, 2)
print("clean items:", int(keep_items.sum()))


def inter(c, h):
    d = test_df.copy()
    d.loc[d.index[0::2], 'answer'] = list(c)
    d.loc[d.index[1::2], 'answer'] = list(h)
    return d


SETS = {'original': test_df,
        'correct_restyled': inter(nc, test_df['answer'].iloc[1::2].tolist()),
        'both_swapped': inter(nc, nh)}


def shuffle_df(df, seed=0):
    d = df.copy().reset_index(drop=True); n = len(d)
    perm = np.random.default_rng(seed).permutation(n)
    while (perm == np.arange(n)).any():
        s = perm == np.arange(n); perm[s] = (perm[s] + 1) % n
    d['knowledge'] = d['knowledge'].values[perm]; d['question'] = d['question'].values[perm]
    return d


def score(p, y):
    p = np.asarray(p, float); pred = (p >= .5).astype(int)
    return {'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
            'auroc': round(float(roc_auc_score(y, p)), 4),
            'pairwise_acc': round(float((p[1::2] > p[0::2]).mean()), 4),
            'correct_flagged': round(float(pred[y == 0].mean()), 4)}


def free():
    gc.collect(); torch.cuda.empty_cache()


# ---- encoder helpers --------------------------------------------------
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
    args = TrainingArguments(output_dir=f"/content/q1_{seed}", num_train_epochs=3,
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


# =====================================================================
# Q1 — five seeds
# =====================================================================
def q1_seed(seed):
    path = f"{OUT}/q1_seed{seed}.npz"
    if os.path.exists(path):
        print(f"Q1 seed {seed}: cached"); return dict(np.load(path))
    t0 = time.time(); P = {}
    # full-input detector
    if seed == 42:
        fm = AutoModelForSequenceClassification.from_pretrained(f"{DRIVE}/biomedbert_v2")
        ft = AutoTokenizer.from_pretrained(f"{DRIVE}/biomedbert_v2_tok")
    else:
        fm, ft = train(train_df, seed)
    P['tt_full'] = predict(fm, ft, test_df)
    P['tt_removed'] = predict(fm, ft, test_df.assign(question='', knowledge=''))
    P['tt_shuffled'] = predict(fm, ft, shuffle_df(test_df))
    P['tt_answer_removed'] = predict(fm, ft, test_df.assign(answer=''))
    for name, d in SETS.items():
        P[f'{name}__full_correct'] = predict(fm, ft, d)
        P[f'{name}__full_unrelated'] = predict(fm, ft, shuffle_df(d))
    del fm; free()
    # answer-only detector
    if seed == 42:
        am = AutoModelForSequenceClassification.from_pretrained(f"{V16}/answer_only_seed42")
        at = AutoTokenizer.from_pretrained(f"{V16}/answer_only_seed42")
    else:
        am, at = train(train_df.assign(question='', knowledge=''), seed)
    for name, d in SETS.items():
        P[f'{name}__answer_only'] = predict(am, at, d.assign(question='', knowledge=''))
    del am; free()
    np.savez(path, **P)
    print(f"Q1 seed {seed}: done in {(time.time()-t0)/60:.1f} min", flush=True)
    return P


def q1():
    allp = {s: q1_seed(s) for s in SEEDS}
    res = {'test_time': {}, 'style': {}, 'gaps': {}}
    # test-time ablation, all 2000 pairs
    for k in ['tt_full', 'tt_removed', 'tt_shuffled', 'tt_answer_removed']:
        per = [score(allp[s][k], y_all) for s in SEEDS]
        res['test_time'][k] = {m: [round(float(np.mean([p[m] for p in per])), 4),
                                   round(float(np.std([p[m] for p in per], ddof=1)), 4)]
                               for m in per[0]}
    # style-controlled, 993 clean items
    y = y_all[ROWS]
    for name in SETS:
        res['style'][name] = {}
        for mdl in ['full_correct', 'full_unrelated', 'answer_only']:
            per = [score(allp[s][f'{name}__{mdl}'][ROWS], y) for s in SEEDS]
            res['style'][name][mdl] = {m: [round(float(np.mean([p[m] for p in per])), 4),
                                           round(float(np.std([p[m] for p in per], ddof=1)), 4)]
                                       for m in per[0]}
    # gap = pairwise(correct passage) - pairwise(unrelated passage),
    # per seed, and a bootstrap CI over items of the seed-averaged gap
    rng = np.random.default_rng(0)
    n_items = int(keep_items.sum())
    for name in SETS:
        wins = {}
        for mdl in ['full_correct', 'full_unrelated']:
            w = np.stack([(allp[s][f'{name}__{mdl}'][ROWS][1::2] >
                           allp[s][f'{name}__{mdl}'][ROWS][0::2]).astype(float) for s in SEEDS])
            wins[mdl] = w                      # seeds x items
        diff = wins['full_correct'] - wins['full_unrelated']
        per_seed = diff.mean(1)
        boots = [diff[:, rng.integers(0, n_items, n_items)].mean() for _ in range(5000)]
        res['gaps'][name] = {
            'gap_mean_over_seeds': round(float(per_seed.mean()), 4),
            'gap_sd_over_seeds': round(float(per_seed.std(ddof=1)), 4),
            'gap_per_seed': [round(float(g), 4) for g in per_seed],
            'bootstrap_ci95_items': [round(float(np.percentile(boots, 2.5)), 4),
                                     round(float(np.percentile(boots, 97.5)), 4)]}
    return res


# =====================================================================
# Q2 — prompted LLM detector (zero-shot, Qwen2.5-7B-Instruct)
# P(hallucinated) = P("No") / (P("Yes") + P("No")) for the next token
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
        k = " ".join(k.split()[:450])            # keep prompts within budget
        body = (WITH_PASSAGE.format(k=k, q=q, a=a) if with_passage
                else NO_PASSAGE.format(q=q, a=a))
        prompts.append(tok.apply_chat_template([{"role": "user", "content": body}],
                                               tokenize=False, add_generation_prompt=True))
    out = []
    for i in range(0, len(prompts), b):
        e = tok(prompts[i:i+b], return_tensors='pt', padding=True).to('cuda')
        logits = lm(**e).logits[:, -1, :].float()
        two = torch.softmax(logits[:, [yes, no]], -1)
        out.append(two[:, 1].cpu().numpy())
        if i % (b * 40) == 0:
            print(f"    {i}/{len(prompts)}", flush=True)
    return np.concatenate(out)


def q2():
    tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = 'left'
    lm = None
    y = y_all[ROWS]; res = {}
    for name, d in SETS.items():
        d = d[ROWS].reset_index(drop=True)
        res[name] = {}
        for cond, dd, wp in [('correct_passage', d, True),
                             ('unrelated_passage', shuffle_df(d), True),
                             ('no_passage', d, False)]:
            path = f"{OUT}/q2_{name}_{cond}.npy"
            if os.path.exists(path):
                p = np.load(path)
            else:
                if lm is None:
                    lm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16,
                                                              device_map='cuda').eval()
                print(f"  Q2 {name} / {cond}", flush=True)
                p = llm_scores(lm, tok, dd, wp); np.save(path, p)
            res[name][cond] = score(p, y)
    if lm is not None:
        del lm; free()
    return res


def cached(name, fn):
    path = f"{OUT}/{name}.json"
    if os.path.exists(path):
        print(f"[{name}] cached"); return json.load(open(path))
    r = fn(); json.dump(r, open(path, "w"), indent=1); return r


R = {'Q1': cached('q1_summary', q1), 'Q2': cached('q2_summary', q2)}
print("\n" + "=" * 64 + "\nPASTE THIS BACK\n" + "=" * 64)
print(json.dumps(R, indent=1))
