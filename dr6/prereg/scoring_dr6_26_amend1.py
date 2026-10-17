"""dr6 scoring, AMENDMENT 1 (written after unblinding; see prereg/amendment_1.md).

Identical to the registered dr6-score-1.0.0 except for one change, applied only to text that
contains a closing reasoning tag: everything up to and including the last "</think>" is removed
before the reply is parsed (forced choice and self-pass) or graded by the fixed rule. Qwen3.8
emits its reasoning followed by "</think>" with no opening tag, which the registered parser did
not strip, so it read the example JSON quoted in the reasoning instead of the final answer.
A Qwen reply with no closing tag (the output ceiling was reached mid-reasoning) has no final
answer and is treated as unreadable. Text without the tag is handled exactly as before, so API
replies are unaffected (notebook 27 checks this). Jev outputs and human grades are unchanged. The rule grade is recomputed from the stage-1 text instead of rule_grades.csv.

Original header follows.

dr6 scoring (registered before the stage-2 freeze; run only after it).

Implements SAP_dr6_v1 sections 6 and 7. Every choice the SAP leaves open is fixed here, before the
answer key is opened:

* Image sets. PRIMARY and KEY SECONDARY use the common set: gradable images (reference ICDR 0 to 4)
  from the five public datasets, read by all five image models. Qwen on all six datasets (adding
  BRSET) is reported as a secondary estimate.
* Index test. Forced-choice grade (parsed icdr_grade, normalised). Positive call = grade 2, 3 or 4,
  or "ungradable", or no usable grade (unreadable reply or failed call). Sensitivity analysis drops
  the ungradable and unreadable calls instead. The three repeats are averaged within image.
* Free-text arms. Grade from each mapper (Jev choice, self-pass JSON, fixed rule) applied to the
  repeat-0 description, with the same positive-call rule. A missing description or mapper reply
  counts as a referral.
* Weighting. Within dataset, each image gets weight N(dataset, grade) / n(dataset, grade): the
  dataset's full gradable population count over the sampled count. Sensitivity and specificity are
  computed within dataset and then averaged with equal weight per dataset. Balanced accuracy is the
  mean of the two. Unweighted estimates are reported as secondary.
* Inference. 10,000 bootstrap resamples of images, stratified by dataset and reference grade,
  seed 20261016; percentile intervals. Paired comparisons reuse the same resampled images.
  Primary: one-sided P for sensitivity > 0.85 and for specificity > 0.825, P = (1 + #{boot <= thr})
  / (B + 1); model P = the larger; Holm across the five models. A model meets the benchmark if both
  one-sided 97.5% lower bounds (2.5th percentiles) exceed the thresholds.
  Key secondary: balanced accuracy, Jev-mapped free text minus forced choice repeat 0, pooled over
  the five models (mean of the per-image indicators across models, which equals the mean of the
  model estimates), two-sided P = min(1, 2 x min(share <= 0, share >= 0)).
* Other secondaries are estimates with 95% intervals, no tests. Five-level agreement, calibration,
  AUC and repeatability are unweighted, on the common set unless stated.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_VERSION = "dr6-score-1.1.0-amendment1"
B = 10_000
SENS_THR, SPEC_THR = 0.85, 0.825
MODELS = ["anthropic", "openai", "google", "xai", "qwen"]
PUBLIC = ["aptos", "eyepacs", "idrid", "ddr", "sustech"]
POS = {"2", "3", "4"}


# ---------------------------------------------------------------------------------------------
# Reading replies
# ---------------------------------------------------------------------------------------------
def final_answer(t):
    """Amendment 1: drop reasoning that ends with a closing tag (no opening tag needed)."""
    return t.rsplit("</think>", 1)[1] if isinstance(t, str) and "</think>" in t else t


def _parsed(r):
    t = r.get("text")
    if r.get("vendor") == "qwen" and isinstance(t, str) and "</think>" not in t:
        return None
    if isinstance(t, str) and "</think>" in t:
        return L.parse_json_reply(final_answer(t))[0] if r.get("ok") else None
    p = r.get("parsed")
    if isinstance(p, dict):
        return p
    if r.get("ok") and t:
        return L.parse_json_reply(t)[0]
    return None


def grade_from_json(r) -> str | None:
    p = _parsed(r) if r else None
    return L.norm_grade(p.get("icdr_grade")) if isinstance(p, dict) else None


def confidence_from_json(r) -> float | None:
    p = _parsed(r) if r else None
    if not isinstance(p, dict):
        return None
    try:
        c = float(p.get("confidence"))
    except (TypeError, ValueError):
        return None
    if 1 < c <= 100:
        c /= 100
    return c if 0 <= c <= 1 else None


def _jev_key(x) -> str | None:
    if x is None:
        return None
    g = L.norm_grade(x)
    if g:
        return g
    s = str(x).strip().lower()
    for k, v in L.ICDR_TEXT.items():
        if s == k or s == v.lower() or (len(s) > 12 and v.lower().startswith(s[:40])):
            return k
    return None


def jev_answer(r):
    """(grade, {grade: probability}) from a Jev record, or (None, {})."""
    if not r or not r.get("ok") or not r.get("text"):
        return None, {}
    try:
        a = json.loads(r["text"])
    except Exception:
        return None, {}
    g = _jev_key(a.get("choice"))
    probs, raw = {}, a.get("probabilities")
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = [((d.get("choice") or d.get("option") or d.get("label") or d.get("key")),
                  (d.get("probability", d.get("p")))) for d in raw if isinstance(d, dict)]
    else:
        items = []
    for k, v in items:
        kk = _jev_key(k)
        try:
            if kk:
                probs[kk] = probs.get(kk, 0.0) + float(v)
        except (TypeError, ValueError):
            pass
    tot = sum(probs.values())
    if tot > 0:
        probs = {k: v / tot for k, v in probs.items()}
    return g, probs


def positive(g: str | None, drop_unusable: bool = False) -> float:
    if g in POS:
        return 1.0
    if g in ("0", "1"):
        return 0.0
    return np.nan if drop_unusable else 1.0


# ---------------------------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------------------------
def load_all(R: Path, key_dir: Path, human_path: Path):
    s1 = list(L.load_done(R / "stage1_api" / "calls.jsonl").values()) + \
         list(L.load_done(R / "stage1_qwen" / "calls.jsonl").values())
    s2 = list(L.load_done(R / "stage2" / "calls.jsonl").values()) + \
         list(L.load_done(R / "stage2_qwen" / "calls.jsonl").values())
    ji = list(L.load_done(R / "stage2_jev_image" / "calls.jsonl").values()) \
        if (R / "stage2_jev_image" / "calls.jsonl").exists() else []
    rule = pd.read_csv(R / "stage2" / "rule_grades.csv", dtype=str)
    key = pd.read_csv(key_dir / "key.csv", dtype=str)
    key["grade"] = key.grade.astype(int)
    cov = pd.read_csv(key_dir / "covariates.csv", dtype=str) if (key_dir / "covariates.csv").exists() else None
    human = pd.read_csv(human_path, dtype=str) if human_path.exists() else None
    link = pd.read_csv(key_dir / "text_grading_link.csv", dtype=str) if (key_dir / "text_grading_link.csv").exists() else None
    return s1, s2, ji, rule, key, cov, human, link


def call_table(s1, s2, rule):
    """One row per (model, image) with forced reps, free-text task, and each mapper's grade."""
    forced, free = {}, {}
    for r in s1:
        k = (r["vendor"], r["image_id"])
        if r.get("prompt_id") == "forced":
            forced.setdefault(k, {})[int(r.get("rep", 0))] = r
        elif r.get("prompt_id") == "free" and int(r.get("rep", 0)) == 0:
            free[k] = r
    s2_by = {(r["arm"], r["source_task_id"]): r for r in s2 if r.get("source_task_id")}
    rule_by = dict(zip(rule.source_task_id, rule.rule_grade))
    rows = []
    for (v, iid), reps in forced.items():
        fr = free.get((v, iid))
        stid = fr["task_id"] if fr else None
        row = {"model": v, "image_id": iid, "dataset": next(iter(reps.values()))["dataset"]}
        for rep in range(3):
            rr = reps.get(rep)
            row[f"f{rep}"] = grade_from_json(rr) if rr and rr.get("ok") else None
            row[f"c{rep}"] = confidence_from_json(rr) if rr and rr.get("ok") else None
        row["free_ok"] = bool(fr and fr.get("ok"))
        jg, jp = jev_answer(s2_by.get(("jev_text", stid))) if stid else (None, {})
        row["jev"] = jg
        row["jev_pref"] = sum(jp.get(g, 0.0) for g in POS) if jp else np.nan
        row["jev_probs"] = json.dumps(jp) if jp else None
        sp = s2_by.get(("self_text", stid)) if stid else None
        row["self"] = grade_from_json(sp) if sp and sp.get("ok") else None
        ft = fr.get("text") if (fr and fr.get("ok")) else None
        if v == "qwen" and isinstance(ft, str) and "</think>" not in ft:
            ft = None
        row["rule"] = L.rule_grade(final_answer(ft))[0] if ft else None
        row["rule_registered"] = rule_by.get(stid) if stid else None
        row["free_task_id"] = stid
        rows.append(row)
    return pd.DataFrame(rows)


def population_weights(key: pd.DataFrame, pop: dict) -> pd.Series:
    """weight per image_id: N(dataset, grade) / n(dataset, grade), gradable images only."""
    g = key[key.grade <= 4]
    n = g.groupby(["dataset", "grade"]).size()
    w = {}
    for r in g.itertuples():
        N = pop[r.dataset][str(r.grade)]
        w[r.image_id] = N / n[(r.dataset, r.grade)]
    return pd.Series(w)


# ---------------------------------------------------------------------------------------------
# Weighted estimates and the stratified bootstrap
# ---------------------------------------------------------------------------------------------
class Design:
    """Images of one analysis set, grouped into (dataset, grade) strata, with a fixed bootstrap."""

    def __init__(self, img: pd.DataFrame, weights: pd.Series | None, seed: int, b: int = B):
        img = img.sort_values(["dataset", "grade", "image_id"]).reset_index(drop=True)
        self.img = img
        self.ids = img.image_id.tolist()
        self.pos = {i: k for k, i in enumerate(self.ids)}
        self.ds = sorted(img.dataset.unique())
        self.ref = (img.grade >= 2).to_numpy()
        self.w = (weights.reindex(img.image_id).to_numpy() if weights is not None else np.ones(len(img)))
        rng = np.random.default_rng(seed)
        idx = np.empty((b, len(img)), dtype=np.int64)
        for (_, _), grp in img.groupby(["dataset", "grade"]):
            ii = grp.index.to_numpy()
            idx[:, ii] = ii[rng.integers(0, len(ii), size=(b, len(ii)))]
        self.idx = idx

    def vec(self, s: pd.Series) -> np.ndarray:
        """Per-image values aligned to this design (missing image -> NaN)."""
        return s.reindex(self.ids).to_numpy(dtype=float)

    def _est(self, y, rows):
        """sens, spec for value matrix y (n,) or (b, n) given resample rows (same shape)."""
        ref = self.ref[rows]
        w = self.w[rows] * ~np.isnan(y)
        y0 = np.nan_to_num(y)
        ds_arr = self.img.dataset.to_numpy()[rows]
        se, sp = [], []
        for d in self.ds:
            m = ds_arr == d
            wr, wn = w * (m & ref), w * (m & ~ref)
            se.append((wr * y0).sum(-1) / wr.sum(-1))
            sp.append((wn * (1 - y0)).sum(-1) / wn.sum(-1))
        se, sp = np.nanmean(np.array(se), 0), np.nanmean(np.array(sp), 0)
        return se, sp

    def point(self, y):
        rows = np.arange(len(self.ids))
        return self._est(y, rows)

    def boot(self, y, chunk: int = 1000):
        se, sp = [], []
        for s in range(0, len(self.idx), chunk):
            rows = self.idx[s:s + chunk]
            a, b_ = self._est(y[rows], rows)
            se.append(a); sp.append(b_)
        return np.concatenate(se), np.concatenate(sp)


def ci(x, lo=2.5, hi=97.5):
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [np.nan, np.nan]


def summary(D: Design, y):
    se, sp = D.point(y)
    bse, bsp = D.boot(y)
    ba, bba = (se + sp) / 2, (bse + bsp) / 2
    return {"sens": float(se), "sens_ci": ci(bse), "spec": float(sp), "spec_ci": ci(bsp),
            "bal_acc": float(ba), "bal_acc_ci": ci(bba)}, (bse, bsp, bba)


def diff(D: Design, y1, y0):
    s1, p1 = D.point(y1); s0, p0 = D.point(y0)
    b1s, b1p = D.boot(y1); b0s, b0p = D.boot(y0)
    d = ((s1 + p1) - (s0 + p0)) / 2
    bd = ((b1s + b1p) - (b0s + b0p)) / 2
    out = {"d_sens": float(s1 - s0), "d_sens_ci": ci(b1s - b0s), "d_spec": float(p1 - p0),
           "d_spec_ci": ci(b1p - b0p), "d_bal_acc": float(d), "d_bal_acc_ci": ci(bd)}
    return out, bd


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def qwk(a, b, k=5):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return np.nan
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else np.nan


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    m = ~np.isnan(s)
    s, l = s[m], l[m]
    if l.sum() == 0 or (~l).sum() == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[m], y[m]
    if not len(p):
        return np.nan
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).sum() for k in range(bins) if (b == k).any()) / len(p))


# ---------------------------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------------------------
def run(lib, R: Path, key_dir: Path, human_path: Path, pop: dict, seed: int, b: int = B) -> dict:
    global L
    L = lib
    s1, s2, ji, rule, key, cov, human, link = load_all(R, key_dir, human_path)
    T = call_table(s1, s2, rule)
    T = T.merge(key[["image_id", "grade"]], on="image_id", how="left")
    assert T.grade.notna().all(), "image without a reference grade"
    W = population_weights(key, pop)
    res = {"score_version": SCORE_VERSION, "B": b, "seed": seed, "n_calls_stage1": len(s1)}

    # per-image indicators -------------------------------------------------------------------
    def ind(df, cols, drop=False):
        m = np.column_stack([[positive(g, drop) for g in df[c]] for c in cols])
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            v = np.nanmean(m, 1) if m.size else np.nan
        return pd.Series(v, index=df.image_id.to_numpy())

    grad = key[key.grade <= 4]
    common_ids = set(grad[grad.dataset.isin(PUBLIC)].image_id)
    for v in MODELS:
        common_ids &= set(T[T.model == v].image_id)
    common = grad[grad.image_id.isin(common_ids)][["image_id", "dataset", "grade"]]
    Dc = Design(common, W, seed, b)
    Du = Design(common, None, seed, b)
    res["common_set"] = {"n": len(common), "referable": int((common.grade >= 2).sum()),
                         "by_dataset": common.dataset.value_counts().to_dict()}

    # primary --------------------------------------------------------------------------------
    prim, pvals = {}, {}
    for v in MODELS:
        tv = T[T.model == v]
        y = Dc.vec(ind(tv, ["f0", "f1", "f2"]))
        est, (bse, bsp, _) = summary(Dc, y)
        p_se = (1 + np.sum(bse <= SENS_THR)) / (len(bse) + 1)
        p_sp = (1 + np.sum(bsp <= SPEC_THR)) / (len(bsp) + 1)
        est.update({"p_sens": float(p_se), "p_spec": float(p_sp),
                    "meets_benchmark": bool(est["sens_ci"][0] > SENS_THR and est["spec_ci"][0] > SPEC_THR)})
        pvals[v] = max(p_se, p_sp)
        # sensitivity analysis: ungradable / unreadable calls excluded
        ys = Dc.vec(ind(tv, ["f0", "f1", "f2"], drop=True))
        est["excl_unusable"], _ = summary(Dc, ys)
        est["unweighted"], _ = summary(Du, Du.vec(ind(tv, ["f0", "f1", "f2"])))
        prim[v] = est
    for v, p in holm(pvals).items():
        prim[v]["p_holm"] = float(p)
    res["primary"] = prim

    # Qwen on all six datasets (secondary)
    qw = T[T.model == "qwen"]
    qset = grad[grad.image_id.isin(set(qw.image_id))][["image_id", "dataset", "grade"]]
    Dq = Design(qset, W, seed, b)
    res["qwen_all_six"], _ = summary(Dq, Dq.vec(ind(qw, ["f0", "f1", "f2"])))

    # key secondary: Jev free text minus forced rep 0, pooled over models -----------------------
    yj = np.nanmean([Dc.vec(ind(T[T.model == v], ["jev"])) for v in MODELS], 0)
    yf = np.nanmean([Dc.vec(ind(T[T.model == v], ["f0"])) for v in MODELS], 0)
    ks, bd = diff(Dc, yj, yf)
    ks["p_two_sided"] = float(min(1.0, 2 * min(np.mean(bd <= 0), np.mean(bd >= 0))))
    ks["jev"], _ = summary(Dc, yj)
    ks["forced_rep0"], _ = summary(Dc, yf)
    res["key_secondary"] = ks

    # other secondaries ----------------------------------------------------------------------
    sec = {}
    # 1. model versus model (forced, three repeats), and open-weight versus each closed model
    ys = {v: Dc.vec(ind(T[T.model == v], ["f0", "f1", "f2"])) for v in MODELS}
    sec["model_vs_model"] = {f"{a}-{c}": diff(Dc, ys[a], ys[c])[0]
                             for i, a in enumerate(MODELS) for c in MODELS[i + 1:]}
    # 2 and 3. free-text arms, each mapper, per model, and differences
    arms = {}
    for v in MODELS:
        tv = T[T.model == v]
        yv = {m: Dc.vec(ind(tv, [m])) for m in ("jev", "self", "rule")}
        yv["forced_rep0"] = Dc.vec(ind(tv, ["f0"]))
        arms[v] = {m: summary(Dc, y)[0] for m, y in yv.items()}
        arms[v]["jev_minus_forced"] = diff(Dc, yv["jev"], yv["forced_rep0"])[0]
        arms[v]["jev_minus_self"] = diff(Dc, yv["jev"], yv["self"])[0]
        arms[v]["jev_minus_rule"] = diff(Dc, yv["jev"], yv["rule"])[0]
        arms[v]["self_minus_forced"] = diff(Dc, yv["self"], yv["forced_rep0"])[0]
    sec["free_text_arms"] = arms
    # 4. by dataset (each dataset's own weighting), and spread across datasets
    byds = {}
    for v in MODELS:
        tv = T[T.model == v]
        byds[v] = {}
        for d in sorted(set(grad.dataset)):
            sub = grad[(grad.dataset == d) & grad.image_id.isin(set(tv.image_id))][["image_id", "dataset", "grade"]]
            if len(sub) == 0:
                continue
            Dd = Design(sub, W, seed, b)
            byds[v][d] = summary(Dd, Dd.vec(ind(tv, ["f0", "f1", "f2"])))[0]
        ss = [x["sens"] for x in byds[v].values()]
        pp = [x["spec"] for x in byds[v].values()]
        byds[v]["_spread"] = {"sens_range": [min(ss), max(ss)], "spec_range": [min(pp), max(pp)]}
    sec["by_dataset"] = byds
    # 5. gradability
    ug = key[key.grade == 5]
    grad_all = {}
    for v in MODELS:
        tv = T[T.model == v].set_index("image_id")
        on_u = tv.reindex([i for i in ug.image_id if i in tv.index])
        on_g = tv.reindex([i for i in grad.image_id if i in tv.index])
        f = ["f0", "f1", "f2"]
        grad_all[v] = {
            "ungradable_images": int(len(on_u)),
            "called_ungradable_on_ungradable": float(np.mean([g == "ungradable" for c in f for g in on_u[c]])) if len(on_u) else None,
            "called_ungradable_on_gradable": float(np.mean([g == "ungradable" for c in f for g in on_g[c]])),
            "unreadable_calls": float(np.mean([g is None for c in f for g in on_g[c]])),
        }
    sec["gradability"] = grad_all
    # 6. five-level agreement (forced rep 0, gradable common set, unweighted)
    five = {}
    for v in MODELS:
        tv = T[(T.model == v) & T.image_id.isin(common_ids)]
        ok = tv[tv.f0.isin(list("01234"))]
        a, p = ok.grade.astype(int).to_numpy(), ok.f0.astype(int).to_numpy()
        five[v] = {"n_graded": int(len(ok)), "n": int(len(tv)), "qwk": qwk(a, p), "exact": float(np.mean(a == p)),
                   "within_one": float(np.mean(abs(a - p) <= 1)), "share_grade2": float(np.mean(p == 2)),
                   "reference_share_grade2": float(np.mean(a == 2))}
    sec["five_level"] = five
    # 7. calibration and AUC
    cal = {}
    for v in MODELS:
        tv = T[(T.model == v) & T.image_id.isin(common_ids)]
        refv = (tv.grade >= 2).astype(float).to_numpy()
        # stated confidence vs correctness of the referable call, all three repeats
        cs, cy = [], []
        for rep in range(3):
            g = tv[f"f{rep}"]
            for gg, cc, rr in zip(g, tv[f"c{rep}"], refv):
                if gg in list("01234") and cc is not None and not (isinstance(cc, float) and math.isnan(cc)):
                    cs.append(cc); cy.append(float((gg in POS) == bool(rr)))
        pj = tv.jev_pref.to_numpy(dtype=float)
        cal[v] = {"confidence_ece": ece(cs, cy), "confidence_mean": float(np.mean(cs)) if cs else None,
                  "accuracy_when_confident": float(np.mean(cy)) if cy else None,
                  "jev_pref_ece": ece(pj, refv), "jev_pref_brier": float(np.nanmean((pj - refv) ** 2)),
                  "jev_pref_auc": auc(pj, refv)}
    sec["calibration"] = cal
    # 8. repeatability
    sec["repeatability"] = {v: float(np.mean([(a == b_ == c) for a, b_, c in
                                              T[T.model == v][["f0", "f1", "f2"]].itertuples(index=False)]))
                            for v in MODELS}
    # 9. native resolution tertile (field-of-view diameter, within dataset)
    if cov is not None and "fov_diameter" in cov.columns:
        cv = cov[["image_id", "fov_diameter"]].copy()
        cv["fov_diameter"] = pd.to_numeric(cv.fov_diameter, errors="coerce")
        cc = common.merge(cv, on="image_id", how="left")
        cc["tertile"] = cc.groupby("dataset").fov_diameter.transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
        tert = {}
        for t in ["low", "mid", "high"]:
            sub = cc[cc.tertile == t][["image_id", "dataset", "grade"]]
            Dt = Design(sub, W, seed, b)
            tert[t] = {v: summary(Dt, Dt.vec(ind(T[T.model == v], ["f0", "f1", "f2"])))[0] for v in MODELS}
        sec["resolution_tertile"] = tert
    # 10. mapper fidelity against the blind human text grades
    if human is not None and link is not None:
        h = human.merge(link, on="item")
        h["human"] = h.icdr_grade.map(L.norm_grade)
        t2 = T.set_index("free_task_id")
        fid = {"n_items": int(len(h)), "n_human_graded": int(h.human.notna().sum())}
        for m in ("jev", "self", "rule"):
            mg = h.task_id.map(t2[m])
            both = h.human.isin(list("01234")) & mg.isin(list("01234"))
            hh, mm = h.human[both].astype(int), mg[both].astype(int)
            fid[m] = {"n": int(both.sum()), "exact": float(np.mean(hh == mm)) if both.any() else None,
                      "qwk": qwk(hh, mm), "referable_agreement":
                          float(np.mean([positive(a) == positive(c) for a, c in zip(h.human, mg)]))}
        sec["mapper_fidelity"] = fid
    res["secondary"] = sec

    # exploratory: Jev reads the image -------------------------------------------------------
    if ji:
        jr = pd.DataFrame([{"image_id": r["image_id"], "dataset": r["dataset"], "ok": r["ok"],
                            "jev": jev_answer(r)[0] if r["ok"] else None} for r in ji])
        jr = jr.merge(key[["image_id", "grade"]], on="image_id")
        acc = jr[jr.ok & (jr.grade <= 4)]
        De = Design(acc[["image_id", "dataset", "grade"]], W, seed, b)
        res["exploratory_jev_image"] = {
            "sent": int(len(jr)), "accepted": int(jr.ok.sum()),
            "rejected_by_dataset": jr[~jr.ok].dataset.value_counts().to_dict(),
            "rejected_by_reference_grade": jr[~jr.ok].grade.value_counts().sort_index().to_dict(),
            "on_accepted_gradable": summary(De, De.vec(ind(acc.rename(columns={}), ["jev"])))[0]}
    return res
