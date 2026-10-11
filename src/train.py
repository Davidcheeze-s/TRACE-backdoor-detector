"""
TRACE training / evaluation.

  python train.py --files data/tiny_none.jsonl data/tiny_T1.jsonl data/tiny_T2.jsonl data/tiny_T3.jsonl \
                  --unseen T3

Splits are grouped by prompt_id so the same prompt never appears in train and test.
Unseen-trigger samples are excluded from train/val and only used in the unseen test.
Reports Behavioral-only, Generation-only, and Full TRACE (ablation).
"""
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def load(files):
    recs = [json.loads(l) for f in files for l in open(f)]
    return [r for r in recs if r["label"] >= 0]  # drop failed-trigger samples (recorded separately)


def split(recs, unseen, rng):
    pids = sorted({r["prompt_id"] for r in recs})
    rng.shuffle(pids)
    n = len(pids)
    tr, va = set(pids[: int(.7 * n)]), set(pids[int(.7 * n): int(.85 * n)])
    out = {"train": [], "val": [], "test_seen": [], "test_unseen": []}
    for r in recs:
        p, t = r["prompt_id"], r["trigger_id"]
        if t == unseen:
            if p not in tr and p not in va:
                out["test_unseen"].append(r)
        else:
            k = "train" if p in tr else "val" if p in va else "test_seen"
            out[k].append(r)
        if t == "none" and p not in tr and p not in va:  # clean controls also form the unseen-test normal class
            out["test_unseen"].append(r)
    return out


def supcon_loss(z, y, t=0.1):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / t
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    has = pos.sum(1) > 0
    loss = -(logprob * pos).sum(1) / pos.sum(1).clamp(min=1)
    return loss[has].mean()


def train_head(X, y, dim=128, epochs=50, bs=256, lr=1e-3):
    head = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, dim))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    X, y = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            if len(idx) < 4: continue
            loss = supcon_loss(head(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def best_threshold(y, s):
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return th[np.argmax(f1[:-1])]


def metrics(y, s, th):
    pred = (s >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9), fpr=fp / max(fp + tn, 1),
                auroc=roc_auc_score(y, s) if len(set(y)) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--unseen", required=True, help="trigger_id held out for unseen test")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()

    S = split(load(a.files), a.unseen, np.random.RandomState(SEED))
    for k, v in S.items():
        print(k, len(v), "pos:", sum(r["label"] for r in v))

    enc = SentenceTransformer(a.encoder)
    def E(rs):
        txt = [f"[PROMPT] {r['prompt']}\n[RESPONSE] {r['response']}" for r in rs]
        return enc.encode(txt, batch_size=64, show_progress_bar=False).astype("float32")
    def Y(rs): return np.array([r["label"] for r in rs])
    def G(rs): return np.array([r["gen_features"] for r in rs])

    emb = {k: E(v) for k, v in S.items()}
    head = train_head(emb["train"], Y(S["train"]))                       # contrastive: train split only
    proj = lambda X: F.normalize(head(torch.tensor(X)), dim=1).detach().numpy()

    sc = StandardScaler().fit(G(S["train"]))                             # scaler: train split only
    feats = {
        "Behavioral": lambda k: proj(emb[k]),
        "Generation": lambda k: sc.transform(G(S[k])),
        "TRACE (fused)": lambda k: np.hstack([proj(emb[k]), sc.transform(G(S[k]))]),
    }
    for name, fx in feats.items():
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(fx("train"), Y(S["train"]))
        th = best_threshold(Y(S["val"]), clf.predict_proba(fx("val"))[:, 1])  # threshold: val split only
        for split_name in ("test_seen", "test_unseen"):
            m = metrics(Y(S[split_name]), clf.predict_proba(fx(split_name))[:, 1], th)
            print(f"{name:14s} {split_name:12s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()