"""Probes, baselines, and metrics for Research B evaluation."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


def ridge_spearman(Xtr, ytr, Xte, yte, alpha=10.0):
    if len(Xtr) < 5 or len(Xte) < 3:
        return float("nan"), None
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    return spearmanr(pred, yte).correlation, pred


def logistic_auroc(Xtr, ytr, Xte, yte, C=1.0):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p), p


def cca_transform(Xtr_a, Xtr_b, Xte_a, Xte_b, n_comp=16):
    if min(Xtr_a.shape[0], Xte_a.shape[0]) < 3:
        raise ValueError("too few samples for CCA")
    n_comp = min(n_comp, Xtr_a.shape[1], Xtr_b.shape[1], Xtr_a.shape[0] - 1)
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(Xtr_a, Xtr_b)
    Za_tr, _ = cca.transform(Xtr_a, Xtr_b)
    Za_te, Zb_te = cca.transform(Xte_a, Xte_b)
    return cca, Za_tr, Za_te, Zb_te


def retrieval_recall(za, zb, ks=(1, 5, 10)):
    """Row i of za should retrieve row i of zb. Returns dict Recall@k and MRR."""
    za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-8)
    zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-8)
    sim = za @ zb.T
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
    out = {f"R@{k}": float((ranks < k).mean()) for k in ks}
    out["MRR"] = float((1.0 / (ranks + 1)).mean())
    out["median_rank"] = float(np.median(ranks) + 1)
    out["n"] = n
    return out
