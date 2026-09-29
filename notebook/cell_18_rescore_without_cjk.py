# ============================================================
# CELL 18 — Re-score the style-controlled test WITHOUT the 7 items
#            whose rewrites came out partly in Chinese.
# Run Cell 1 first. Any GPU (T4 is fine). ~3 minutes.
# ============================================================
import re, json, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DRIVE = "/content/drive/MyDrive/medhallu_v2"; OUT = f"{DRIVE}/review_v16"
cjk = re.compile(r"[　-鿿가-힯]")


def load(n):
    d = pd.read_csv(f"{DRIVE}/{n}_df.csv")
    for c in ['question', 'knowledge', 'answer']:
        d[c] = d[c].fillna('').astype(str)
    return d


train_df, test_df = load("train"), load("test")
y = test_df['label'].values
nc = pd.read_csv(f"{OUT}/p3_correct_assertive.csv").sort_values('i')['text'].astype(str).tolist()
nh = pd.read_csv(f"{OUT}/p3_halluc_hedged.csv").sort_values('i')['text'].astype(str).tolist()
bad = np.array([bool(cjk.search(a)) or bool(cjk.search(b)) for a, b in zip(nc, nh)])
keep_items = ~bad
rows = np.repeat(keep_items, 2)
print("items dropped:", int(bad.sum()), "| items kept:", int(keep_items.sum()))


def inter(c, h):
    d = test_df.copy()
    d.loc[d.index[0::2], 'answer'] = list(c)
    d.loc[d.index[1::2], 'answer'] = list(h)
    return d


hall = test_df['answer'].iloc[1::2].tolist()
sets = {'original': test_df, 'correct_restyled': inter(nc, hall), 'both_swapped': inter(nc, nh)}


def score(p, yy):
    pred = (p >= .5).astype(int)
    return {'macro_f1': round(float(f1_score(yy, pred, average='macro')), 4),
            'auroc': round(float(roc_auc_score(yy, p)), 4),
            'pairwise_acc': round(float((p[1::2] > p[0::2]).mean()), 4)}


@torch.no_grad()
def predict(m, t, d, b=64):
    m = m.cuda().eval()
    a = [(f"{q} {k}".strip() or "[EMPTY]") for q, k in zip(d['question'], d['knowledge'])]
    c = [(x.strip() or "[EMPTY]") for x in d['answer']]
    out = []
    for i in range(0, len(d), b):
        e = t(a[i:i+b], c[i:i+b], truncation='longest_first', max_length=512,
              padding=True, return_tensors='pt').to('cuda')
        out.append(torch.softmax(m(**e).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def shuffle_df(df, seed=0):
    d = df.copy().reset_index(drop=True); n = len(d)
    perm = np.random.default_rng(seed).permutation(n)
    while (perm == np.arange(n)).any():
        s = perm == np.arange(n); perm[s] = (perm[s] + 1) % n
    d['knowledge'] = d['knowledge'].values[perm]; d['question'] = d['question'].values[perm]
    return d


vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
lr = LogisticRegression(max_iter=5000).fit(vec.fit_transform(train_df['answer']), train_df['label'])
ft = AutoTokenizer.from_pretrained(f"{DRIVE}/biomedbert_v2_tok")
fm = AutoModelForSequenceClassification.from_pretrained(f"{DRIVE}/biomedbert_v2")
at = AutoTokenizer.from_pretrained(f"{OUT}/answer_only_seed42")
am = AutoModelForSequenceClassification.from_pretrained(f"{OUT}/answer_only_seed42")

res = {'items_dropped': int(bad.sum()), 'items_kept': int(keep_items.sum())}
for name, d in sets.items():
    P = {'tfidf_answer_only': lr.predict_proba(vec.transform(d['answer']))[:, 1],
         'biomedbert_answer_only': predict(am, at, d.assign(question='', knowledge='')),
         'biomedbert_full': predict(fm, ft, d),
         'biomedbert_full_shuffled_evidence': predict(fm, ft, shuffle_df(d))}
    res[name] = {k: score(v[rows], y[rows]) for k, v in P.items()}
json.dump(res, open(f"{OUT}/p3_without_cjk.json", "w"), indent=1)
print("\nPASTE THIS BACK\n" + json.dumps(res, indent=1))
