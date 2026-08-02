#!/usr/bin/env python3
"""Reviewer-requested rigor: (1) sparse-code vs alignment-head vs pure deep-CCA ablation,
(2) bootstrap CIs over test examples + paired tests vs CCA, (3) PLS baseline, (4) probe CIs
and class balance, (5) shared code vs classical VEP predictors. CPU-only."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cdd.crosscoder.data import load_paired
from cdd.crosscoder.model import CrosscoderConfig, SharedPrivateCrosscoder, info_nce
from cdd.eval.probes import retrieval_recall, cca_transform

dev = "cpu"
rng = np.random.default_rng(0)
pdd = load_paired("outputs/act/evo2", "outputs/act/esm", "blocks.24.mlp.l3", "L33",
                  pooling="local", n_pca=128)
m = pdd.meta
tr = (m.split_position == "train").to_numpy(); te = (m.split_position == "test").to_numpy()
y = m.dms_score.to_numpy(); keep = ~np.isnan(y)
Xd, Xp = pdd.dna, pdd.prot


def ridge_sp(A, ytr, B, yte):
    sc = StandardScaler().fit(A); mdl = Ridge(alpha=10).fit(sc.transform(A), ytr)
    return sc, mdl, mdl.predict(sc.transform(B))


def boot_r1(za, zb, nb=1000):
    za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-8)
    zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-8)
    sim = za @ zb.T; n = len(za)
    hit1 = (sim.argmax(1) == np.arange(n)).astype(float)
    order = np.argsort(-sim, 1); rank = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
    hit10 = (rank < 10).astype(float)
    b1 = [hit1[rng.integers(0, n, n)].mean() for _ in range(nb)]
    b10 = [hit10[rng.integers(0, n, n)].mean() for _ in range(nb)]
    return hit1.mean(), np.percentile(b1, [2.5, 97.5]), hit10.mean(), np.percentile(b10, [2.5, 97.5])


def boot_spear(pred, truth, nb=1000):
    n = len(pred); base = spearmanr(pred, truth).correlation
    bs = [spearmanr(pred[idx], truth[idx]).correlation for idx in (rng.integers(0, n, n) for _ in range(nb))]
    return base, np.nanpercentile(bs, [2.5, 97.5])


# ---- load trained crosscoder (seed 0) ----
ck = torch.load("outputs/b_mvp/crosscoder.pt", map_location=dev, weights_only=False)
model = SharedPrivateCrosscoder(CrosscoderConfig(**ck["cfg"])); model.load_state_dict(ck["state_dict"]); model.eval()
with torch.no_grad():
    a = model.encode_all(torch.tensor(Xd), torch.tensor(Xp))
align = (a["align_dna"].numpy(), a["align_prot"].numpy())
sparse = (a["shared_dna"].numpy(), a["shared_prot"].numpy())


# ---- pure deep-CCA baseline: align heads only, contrastive loss, no reconstruction ----
def train_deepcca(seed=0, steps=4000, dalign=64):
    torch.manual_seed(seed); np.random.seed(seed)
    Ed = torch.nn.Linear(128, dalign); Ep = torch.nn.Linear(128, dalign)
    opt = torch.optim.Adam(list(Ed.parameters()) + list(Ep.parameters()), 1e-3)
    Xdt = torch.tensor(Xd[tr]); Xpt = torch.tensor(Xp[tr]); n = Xdt.shape[0]
    for s in range(steps):
        idx = torch.randint(0, n, (256,))
        loss = info_nce(Ed(Xdt[idx]), Ep(Xpt[idx]), temp=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return Ed(torch.tensor(Xd)).numpy(), Ep(torch.tensor(Xp)).numpy()

dcca = train_deepcca()

res = {}
# ---- retrieval + DMS for each representation, with CIs and paired test vs CCA ----
_, cca_tr, cca_a, cca_b = cca_transform(Xd[tr], Xp[tr], Xd[te], Xp[te], n_comp=32)
reps = {"align_head": align, "sparse_shared": sparse, "deep_cca": dcca,
        "linear_CCA": (None, None)}
r1_hits = {}
for name, (zd, zp) in reps.items():
    if name == "linear_CCA":
        za, zb = cca_a, cca_b
    else:
        za, zb = zd[te], zp[te]
    r1, r1ci, r10, r10ci = boot_r1(za, zb)
    # DMS from concat rep
    if name == "linear_CCA":
        A, B = cca_tr, cca_a
        sc, mdl, pred = ridge_sp(A[keep[tr]], y[tr & keep], B[keep[te]], y[te & keep])
    else:
        A = np.concatenate([zd, zp], 1)
        sc, mdl, pred = ridge_sp(A[tr & keep], y[tr & keep], A[te & keep], y[te & keep])
    dms, dmsci = boot_spear(pred, y[te & keep])
    res[name] = dict(R1=round(r1, 3), R1_ci=[round(x, 3) for x in r1ci],
                     R10=round(r10, 3), R10_ci=[round(x, 3) for x in r10ci],
                     DMS=round(dms, 3), DMS_ci=[round(x, 3) for x in dmsci])
    # store hit vectors for paired test
    zan = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-8)
    zbn = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-8)
    sim = zan @ zbn.T
    r1_hits[name] = (sim.argmax(1) == np.arange(len(za))).astype(float)

# paired bootstrap: align_head vs linear_CCA on R@1
def paired(a, b, nb=2000):
    d = a - b; n = len(d)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(nb)]
    return float(d.mean()), float((np.array(bs) <= 0).mean())  # one-sided p that align<=CCA

diff, p = paired(r1_hits["align_head"], r1_hits["linear_CCA"])
res["paired_align_vs_cca_R1"] = dict(diff=round(diff, 3), p_one_sided=round(p, 4))

# ---- PLS ----
from sklearn.cross_decomposition import PLSRegression
cc = np.concatenate([Xd, Xp], 1)
pls = PLSRegression(n_components=32).fit(cc[tr & keep], y[tr & keep])
res["PLS_DMS"] = round(float(spearmanr(pls.predict(cc[te & keep]).ravel(), y[te & keep]).correlation), 3)

# ---- classical VEP predictors vs shared (residue-disjoint test) ----
vep = {}
for col, sign in [("sift", -1), ("cadd", 1), ("phylop", 1), ("polyphen2", 1)]:
    v = np.asarray([float(x) if str(x) not in ("nan", "None", "") else np.nan for x in m[col]])
    mask = (te & keep) & ~np.isnan(v)
    if mask.sum() > 20:
        vep[col] = round(abs(spearmanr(v[mask], y[mask]).correlation), 3)
res["vep_abs_spearman"] = vep

# ---- probe CIs + class balance ----
def auroc_ci(feat, lab, mask, nb=1000):
    trm = tr & mask; tem = te & mask
    sc = StandardScaler().fit(feat[trm])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(feat[trm]), lab[trm])
    p = clf.predict_proba(sc.transform(feat[tem]))[:, 1]
    base = roc_auc_score(lab[tem], p)
    n = tem.sum(); yy = lab[tem]
    bs = []
    for _ in range(nb):
        ii = rng.integers(0, n, n)
        if len(np.unique(yy[ii])) < 2: continue
        bs.append(roc_auc_score(yy[ii], p[ii]))
    return round(base, 3), [round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)]

probe = {}
for lname, lab, mask in [("LOF", (m.func_class == "LOF").astype(float).to_numpy(), m.func_class.isin(["LOF", "FUNC"]).to_numpy()),
                         ("RING", (m.domain == "RING").astype(float).to_numpy(), m.domain.isin(["RING", "BRCT"]).to_numpy())]:
    probe[f"{lname}_n_test"] = int((te & mask).sum())
    probe[f"{lname}_pos_rate"] = round(float(lab[te & mask].mean()), 3)
    for fn, feat in [("shared", np.concatenate(align, 1)), ("priv_prot", a["priv_prot"].numpy())]:
        b, ci = auroc_ci(feat, lab, mask)
        probe[f"{lname}/{fn}"] = dict(auroc=b, ci=ci)
res["probe"] = probe

json.dump(res, open("outputs/b_mvp/ablations.json", "w"), indent=2)
print(json.dumps(res, indent=2))
