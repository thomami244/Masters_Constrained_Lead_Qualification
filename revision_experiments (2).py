"""
DeSE2026 revision experiments for
"From Prediction to Action: Policy-Constrained ML for Lead Qualification Under Limited Sales Capacity"

Reproduces the as-submitted pipeline, then adds every analysis the three reviews ask for:

  R1.1 / R2.3 / R3.3  paired bootstrap CI for S2-S1 and S2-S3 (Precision@K and converter count)
  R1.2 / R3.1 / R3.2  nested train / validation / test: model, signals, tier cap, blend weight
                      selected on train+val only; untouched test set for final numbers
  R1.3 / R2.5         retrained full-information XGBoost baseline (S4) ranked at the same K
  R1.4 / R2.4         SMS leakage: quantify overlap, rerun with Last Notable Activity removed
  R1.5                repeated stratified splits (25 seeds): mean / spread / sign stability
  R1.minor / R3.4     P1 & P2 conversion rates in training data, leads removed per filter,
                      class balancing confirmed training-only

v2 (after independent critique):
  * every repeated split re-tunes LR and both XGBoost models on its own training rows (no frozen hyperparameters)
  * the predictive engine actually used is the validation-AUC winner (LR vs XGBoost), not XGBoost regardless
  * the >40%-missing column rule is applied on training rows of each fold, not on the full dataset
  * leakage assertions on the restricted feature list; provenance (dataset sha256, versions, split-index hashes)

Usage:  python revision_experiments.py --data "Lead Scoring.csv" --out results/
Every number printed is also written to results/*.csv and results/summary.json
"""

import argparse, json, os, warnings, time, hashlib, platform
import sklearn, xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ----------------------------------------------------------------------------- constants
LEAKAGE_COLUMNS = ["Tags", "Lead Quality", "Asymmetrique Activity Index", "Asymmetrique Profile Index",
                   "Asymmetrique Activity Score", "Asymmetrique Profile Score"]
ID_COLUMNS = ["Prospect ID", "Lead Number"]
POLICY_COLUMNS = ["Do Not Email", "Do Not Call", "What is your current occupation", "Last Activity"]
OCC_COL, ACT_COL, TARGET = "What is your current occupation", "Last Activity", "Converted"
MODEL_EXCLUDE = POLICY_COLUMNS            # withheld from the *initial* (restricted) model
SMS_LEAK_COL = "Last Notable Activity"    # the feature Review 1 flags as leaky

CAP_GRID = [0.20, 0.40, 0.60, 1.00]
BLEND_GRID = [0.95, 0.90, 0.80]
K_FRACTION = 0.20
MIN_GROUP_N = 50
SIGNAL_LIFT_RULE = 1.25   # a category becomes a qualification signal if train conv rate >= 1.25 x train baseline


# ----------------------------------------------------------------------------- data
HIGH_MISSING_PCT = 40


def high_missing_columns(frame, pct=HIGH_MISSING_PCT):
    m = frame.isnull().mean() * 100
    return m[m > pct].index.tolist()


def load_and_clean(path, drop_high_missing_globally=True):
    """Replicates notebook cells 02-04. drop_high_missing_globally=True reproduces the notebook (rule applied on the
    full dataset); False defers the rule to each fold's training rows (nested_run), which is the v2 primary path."""
    df_raw = pd.read_csv(path)
    policy_full = df_raw[POLICY_COLUMNS].copy()                      # raw values, before 'Select'->NaN
    df = df_raw.drop(columns=[c for c in LEAKAGE_COLUMNS + ID_COLUMNS if c in df_raw.columns]).copy()
    df = df.replace("Select", np.nan)
    df = df.drop_duplicates()
    if drop_high_missing_globally:
        df = df.drop(columns=high_missing_columns(df))
    df["Engagement Intensity"] = (df["Total Time Spent on Website"] / df["TotalVisits"].replace(0, np.nan))
    df["Engagement Intensity"] = df["Engagement Intensity"].replace([np.inf, -np.inf], np.nan)
    policy_full = policy_full.loc[df.index].copy()
    df = df.reset_index(drop=True); policy_full = policy_full.reset_index(drop=True)
    for c in ["Do Not Email", "Do Not Call"]:
        policy_full[c] = policy_full[c].map({"Yes": True, "No": False, 1: True, 0: False, True: True, False: False})
    return df, policy_full


def make_preprocessor(X):
    num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def feature_frame(X, spec, include_qual=False):
    """spec: 'orig' keeps Last Notable Activity; 'nosms' drops it (all SMS-related features gone).
       include_qual=True keeps occupation + Last Activity (full-information baseline S4)."""
    drop = ["Do Not Email", "Do Not Call"] if include_qual else list(MODEL_EXCLUDE)
    if spec == "nosms":
        drop = drop + [SMS_LEAK_COL]
    return X.drop(columns=[c for c in drop if c in X.columns])


# ----------------------------------------------------------------------------- models
XGB_PARAMS = {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4, 5],
              "model__learning_rate": [0.01, 0.05, 0.1], "model__subsample": [0.7, 0.8, 1.0],
              "model__colsample_bytree": [0.7, 0.8, 1.0]}


def xgb_pipeline(X, y, params=None):
    spw = (y == 0).sum() / (y == 1).sum()          # class balancing computed on TRAINING labels only
    model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, scale_pos_weight=spw, n_jobs=2)
    pipe = Pipeline([("preprocessor", make_preprocessor(X)), ("model", model)])
    if params:
        pipe.set_params(**params)
    return pipe


def tune_xgb(X, y, n_iter=20):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    s = RandomizedSearchCV(xgb_pipeline(X, y), XGB_PARAMS, n_iter=n_iter, scoring="roc_auc", cv=cv,
                           random_state=RANDOM_STATE, n_jobs=1, verbose=0)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


def tune_logit(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocessor", make_preprocessor(X)),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))])
    s = GridSearchCV(pipe, {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}, scoring="roc_auc", cv=cv, n_jobs=1)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


# ----------------------------------------------------------------------------- policy layer
def select_signals(df_train_policy, y_train, rule=SIGNAL_LIFT_RULE, min_n=MIN_GROUP_N):
    """Data-driven signal selection on TRAINING rows only (addresses R1.2 / R3.2).
       A category qualifies if n >= min_n and conv_rate >= rule x training baseline."""
    base = y_train.mean()
    out = {}
    for col in [OCC_COL, ACT_COL]:
        g = pd.DataFrame({"c": df_train_policy[col].values, "y": y_train.values}).groupby("c")["y"].agg(["mean", "count"])
        g = g[(g["count"] >= min_n)]
        out[col] = g[g["mean"] >= rule * base].index.tolist()
        out[col + "_table"] = g.sort_values("mean", ascending=False).round(4).reset_index().to_dict("records")
    return out


def rank_systems(res, p1, p2, cap_frac, blend_w, cutoff, k_frac=K_FRACTION):
    """Vectorised re-implementation of notebook apply_systems (identical ordering semantics).
       res needs: Do Not Email, Do Not Call, occupation, Last Activity, actual, ml_score, [full_score]."""
    out = res.copy()
    h1 = out["Do Not Email"].eq(True) & out["Do Not Call"].eq(True)
    tier = np.full(len(out), 3)
    tier[out[ACT_COL].isin(p2).values] = 2
    tier[out[OCC_COL].isin(p1).values] = 1
    demoted = (out["ml_score"] < cutoff).values & (tier < 3)
    tier[demoted] = 3
    out["tier"] = tier
    sig = out[OCC_COL].isin(p1).astype(float) * 0.5 + out[ACT_COL].isin(p2).astype(float) * 0.5
    out["blended"] = blend_w * out["ml_score"] + (1 - blend_w) * sig
    pool = out[~h1].copy()
    K = int(len(res) * k_frac); cap = int(K * cap_frac)
    s1 = pool.sort_values("ml_score", ascending=False)
    t1 = pool[pool.tier == 1].sort_values("ml_score", ascending=False)
    t2 = pool[pool.tier == 2].sort_values("ml_score", ascending=False)
    t3 = pool[pool.tier == 3]
    overflow = pd.concat([t1.iloc[cap:], t2.iloc[cap:], t3]).sort_values("ml_score", ascending=False)
    s2 = pd.concat([t1.head(cap), t2.head(cap), overflow])
    s3 = pool.sort_values("blended", ascending=False)
    systems = {"S1": s1, "S2": s2, "S3": s3}
    if "full_score" in pool.columns:
        systems["S4"] = pool.sort_values("full_score", ascending=False)
    diag = {"K": K, "cap": cap, "h1_removed": int(h1.sum()), "pool": int(len(pool)),
            "p1_in_pool": int((pool.tier == 1).sum()), "p2_in_pool": int((pool.tier == 2).sum()),
            "demoted_by_cutoff": int(demoted[~h1.values].sum()),
            "s2_from_p1": int(min(cap, len(t1))), "s2_from_p2": int(min(cap, len(t2))),
            "s2_from_residual": int(K - min(cap, len(t1)) - min(cap, len(t2)))}
    return systems, diag


def precision_at_k(systems, K):
    return {n: (int(s.head(K)["actual"].sum()), float(s.head(K)["actual"].mean())) for n, s in systems.items()}


def choose_policy_on_val(res_val, p1, p2, cutoff):
    """Grid over tier cap and blend weight; pick by validation Precision@K (R1.2 / R3.2)."""
    K = int(len(res_val) * K_FRACTION)
    best_cap, best_cap_p = None, -1
    for cap in CAP_GRID:
        sysd, _ = rank_systems(res_val, p1, p2, cap, 0.9, cutoff)
        p = precision_at_k(sysd, K)["S2"][1]
        if p > best_cap_p + 1e-12:               # strict '>' => first (smallest) cap wins ties
            best_cap, best_cap_p = cap, p
    best_w, best_w_p = None, -1
    for w in BLEND_GRID:
        sysd, _ = rank_systems(res_val, p1, p2, best_cap, w, cutoff)
        p = precision_at_k(sysd, K)["S3"][1]
        if p > best_w_p + 1e-12:
            best_w, best_w_p = w, p
    return best_cap, best_w, best_cap_p, best_w_p


def paired_bootstrap(res, p1, p2, cap, w, cutoff, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    K = int(len(res) * K_FRACTION)
    rows = []
    for i in range(n):
        samp = res.iloc[rng.choice(len(res), len(res), replace=True)].reset_index(drop=True)
        sysd, _ = rank_systems(samp, p1, p2, cap, w, cutoff)
        pk = precision_at_k(sysd, K)
        rows.append({f"{k}_conv": v[0] for k, v in pk.items()} | {f"{k}_prec": v[1] for k, v in pk.items()})
    b = pd.DataFrame(rows)
    out = {}
    pairs = [("S2", "S1"), ("S2", "S3"), ("S3", "S1")] + ([("S2", "S4"), ("S4", "S1")] if "S4_prec" in b else [])
    for a, c in pairs:
        for m in ["prec", "conv"]:
            d = b[f"{a}_{m}"] - b[f"{c}_{m}"]
            out[f"{a}-{c}_{m}"] = {"mean": float(d.mean()), "lo95": float(d.quantile(0.025)),
                                   "hi95": float(d.quantile(0.975)), "p_gt0": float((d > 0).mean()),
                                   "p_ge0": float((d >= 0).mean())}
    for s in ["S1", "S2", "S3", "S4"]:
        if f"{s}_prec" in b:
            out[f"{s}_prec_CI"] = [float(b[f"{s}_prec"].quantile(0.025)), float(b[f"{s}_prec"].quantile(0.975))]
    return out, b


# ----------------------------------------------------------------------------- one full nested run
def nested_run(df, policy, seed, spec, xgb_params_r=None, xgb_params_f=None, tune=False, log=print):
    """60/20/20 nested split. Returns dict of everything needed for the tables."""
    y = df[TARGET]; X = df.drop(columns=[TARGET])
    X_tr80, X_te, y_tr80, y_te, P_tr80, P_te = train_test_split(X, y, policy, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va, P_tr, P_va = train_test_split(X_tr80, y_tr80, P_tr80, test_size=0.25, random_state=seed, stratify=y_tr80)

    # preprocessing isolation: the >40%-missing column rule is decided on TRAINING rows only
    fold_drop = high_missing_columns(X_tr)
    X_tr, X_va, X_te = (a.drop(columns=fold_drop) for a in (X_tr, X_va, X_te))

    Xr_tr, Xr_va, Xr_te = (feature_frame(a, spec) for a in (X_tr, X_va, X_te))
    Xf_tr, Xf_va, Xf_te = (feature_frame(a, spec, include_qual=True) for a in (X_tr, X_va, X_te))

    idx_hash = lambda ix: hashlib.sha256(",".join(map(str, sorted(ix.tolist()))).encode()).hexdigest()[:16]
    r = {"seed": seed, "spec": spec, "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
         "train_rate": float(y_tr.mean()), "val_rate": float(y_va.mean()), "test_rate": float(y_te.mean()),
         "fold_dropped_high_missing": fold_drop,
         "train_idx_sha256": idx_hash(X_tr.index), "val_idx_sha256": idx_hash(X_va.index), "test_idx_sha256": idx_hash(X_te.index),
         "restricted_features": Xr_tr.columns.tolist(), "full_features": Xf_tr.columns.tolist()}
    # leakage audit on the restricted (initial) model
    banned = [OCC_COL, ACT_COL] + ([SMS_LEAK_COL] if spec == "nosms" else [])
    leaks = [c for c in Xr_tr.columns if any(c.startswith(b) for b in banned)]
    assert not leaks, f"leakage audit failed: {leaks}"
    r["leakage_audit"] = {"banned_columns": banned, "present_in_restricted": leaks,
                          "sms_bearing_columns_remaining": [c for c in Xr_tr.columns if Xr_tr[c].astype(str).str.contains("SMS", case=False).any()]}

    # --- restricted (initial) model: LR and XGBoost tuned on TRAIN via CV; the VALIDATION-AUC winner is the engine
    if tune:
        xgb_r, r["xgb_params_r"] = tune_xgb(Xr_tr, y_tr)
        lr_r, r["lr_params"] = tune_logit(Xr_tr, y_tr)
        xgb_f, r["xgb_params_f"] = tune_xgb(Xf_tr, y_tr)
    else:  # exploratory frozen-hyperparameter path, kept only for comparison with v1; never the primary result
        xgb_r = xgb_pipeline(Xr_tr, y_tr, xgb_params_r).fit(Xr_tr, y_tr)
        lr_r, r["lr_params"] = tune_logit(Xr_tr, y_tr)
        xgb_f = xgb_pipeline(Xf_tr, y_tr, xgb_params_f).fit(Xf_tr, y_tr)
        r["xgb_params_r"], r["xgb_params_f"] = xgb_params_r, xgb_params_f
    r["val_auc_logit"] = float(roc_auc_score(y_va, lr_r.predict_proba(Xr_va)[:, 1]))
    r["val_auc_xgb"] = float(roc_auc_score(y_va, xgb_r.predict_proba(Xr_va)[:, 1]))
    r["selected_engine"] = "XGBoost" if r["val_auc_xgb"] >= r["val_auc_logit"] else "LogisticRegression"
    engine = xgb_r if r["selected_engine"] == "XGBoost" else lr_r
    r["test_auc_engine"] = float(roc_auc_score(y_te, engine.predict_proba(Xr_te)[:, 1]))
    r["test_auc_xgb"] = float(roc_auc_score(y_te, xgb_r.predict_proba(Xr_te)[:, 1]))
    r["test_auc_logit"] = float(roc_auc_score(y_te, lr_r.predict_proba(Xr_te)[:, 1]))
    r["val_auc_full"] = float(roc_auc_score(y_va, xgb_f.predict_proba(Xf_va)[:, 1]))
    r["test_auc_full"] = float(roc_auc_score(y_te, xgb_f.predict_proba(Xf_te)[:, 1]))

    # --- signals selected on TRAIN rows only
    sig = select_signals(P_tr, y_tr)
    p1, p2 = sig[OCC_COL], sig[ACT_COL]
    r["p1"], r["p2"] = p1, p2
    r["p1_train_rate"] = float(y_tr[P_tr[OCC_COL].isin(p1).values].mean()) if p1 else None
    r["p2_train_rate"] = float(y_tr[P_tr[ACT_COL].isin(p2).values].mean()) if p2 else None
    r["p1_train_n"] = int(P_tr[OCC_COL].isin(p1).sum()); r["p2_train_n"] = int(P_tr[ACT_COL].isin(p2).sum())
    r["occupation_train_table"] = sig[OCC_COL + "_table"]; r["activity_train_table"] = sig[ACT_COL + "_table"]
    cutoff = float(y_tr.mean())                     # baseline cutoff = TRAINING conversion rate (was full-data rate)
    r["cutoff"] = cutoff

    def res_frame(P, yy, Xr, Xf):
        d = P.reset_index(drop=True).copy()
        d["actual"] = yy.values
        d["ml_score"] = engine.predict_proba(Xr)[:, 1]
        d["full_score"] = xgb_f.predict_proba(Xf)[:, 1]
        return d
    res_va, res_te = res_frame(P_va, y_va, Xr_va, Xf_va), res_frame(P_te, y_te, Xr_te, Xf_te)

    # --- policy parameters chosen on VAL
    cap, w, cap_vp, w_vp = choose_policy_on_val(res_va, p1, p2, cutoff)
    r["cap"], r["blend_w"], r["val_prec_S2_at_cap"], r["val_prec_S3_at_w"] = cap, w, cap_vp, w_vp

    # --- untouched TEST
    K = int(len(res_te) * K_FRACTION)
    sysd, diag = rank_systems(res_te, p1, p2, cap, w, cutoff)
    pk = precision_at_k(sysd, K)
    r["K"] = K; r["diag_test"] = diag
    for s, (c, p) in pk.items():
        r[f"{s}_conv"], r[f"{s}_prec"] = c, p
    # sensitivity on test (reported, not used for selection)
    r["cap_sens_test"] = {str(c): precision_at_k(rank_systems(res_te, p1, p2, c, w, cutoff)[0], K)["S2"][1] for c in CAP_GRID}
    r["k_sens_test"] = {str(kf): {s: v[1] for s, v in precision_at_k(rank_systems(res_te, p1, p2, cap, w, cutoff, k_frac=kf)[0],
                                                                      int(len(res_te) * kf)).items()} for kf in [0.1, 0.2, 0.3]}
    r["_res_te"], r["_res_va"], r["_P_tr"], r["_y_tr"] = res_te, res_va, P_tr, y_tr
    return r


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=25); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S = {}
    t0 = time.time()
    S["provenance"] = {"dataset_sha256": hashlib.sha256(open(a.data, "rb").read()).hexdigest(), "python": platform.python_version(),
                       "xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                       "random_state": RANDOM_STATE, "k_fraction": K_FRACTION, "cap_grid": CAP_GRID, "blend_grid": BLEND_GRID,
                       "signal_rule": f"n>={MIN_GROUP_N} and train conv rate >= {SIGNAL_LIFT_RULE} x train baseline",
                       "bootstrap_n": a.boot, "repeated_seeds": a.seeds, "repeated_splits_retuned": True}
    df, policy = load_and_clean(a.data)                                   # notebook-identical (global rule) for the reproduction
    df_fold, policy_fold = load_and_clean(a.data, drop_high_missing_globally=False)   # fold-wise rule for nested runs
    S["provenance"]["global_high_missing_dropped"] = [c for c in df_fold.columns if c not in df.columns]
    print(f"Clean dataset: {df.shape}, conversion {df[TARGET].mean():.4f}; versions {S['provenance']['xgboost']} / {S['provenance']['sklearn']}")

    # ===================== 0. Reproduce as-submitted (80/20, seed 42, full-data cutoff, test-selected params)
    y = df[TARGET]; X = df.drop(columns=[TARGET])
    X_tr, X_te, y_tr, y_te, P_tr, P_te = train_test_split(X, y, policy, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    Xr_tr, Xr_te = feature_frame(X_tr, "orig"), feature_frame(X_te, "orig")
    xgb0, params0 = tune_xgb(Xr_tr, y_tr)
    res0 = P_te.reset_index(drop=True).copy(); res0["actual"] = y_te.values; res0["ml_score"] = xgb0.predict_proba(Xr_te)[:, 1]
    sys0, diag0 = rank_systems(res0, ["Working Professional"], ["SMS Sent"], 0.40, 0.90, float(df[TARGET].mean()))
    pk0 = precision_at_k(sys0, diag0["K"])
    S["repro"] = {"test_auc": float(roc_auc_score(y_te, res0["ml_score"])), "params": params0, "diag": diag0,
                  "precision": {k: v[1] for k, v in pk0.items()}, "converters": {k: v[0] for k, v in pk0.items()}}
    print("\n=== 0. REPRODUCTION (as submitted) ===")
    print(f"XGB test AUC {S['repro']['test_auc']:.4f}  params {params0}")
    print({k: v for k, v in pk0.items()}, diag0)
    # paired bootstrap on the as-submitted configuration (R1.1 answered even for the original design)
    pb0, _ = paired_bootstrap(res0, ["Working Professional"], ["SMS Sent"], 0.40, 0.90, float(df[TARGET].mean()), n=a.boot)
    S["repro"]["paired_bootstrap"] = pb0
    print("paired bootstrap (as-submitted):"); [print(f"  {k}: {v}") for k, v in pb0.items()]

    # ===================== SMS overlap (R1.4)
    ov = {}
    for name, P_, X_ in [("train", P_tr, X_tr), ("test", P_te, X_te), ("all", policy, X)]:
        la = P_[ACT_COL].eq("SMS Sent").values; lna = X_[SMS_LEAK_COL].eq("SMS Sent").values
        ov[name] = {"n": int(len(P_)), "last_activity_sms": int(la.sum()), "last_notable_sms": int(lna.sum()),
                    "both": int((la & lna).sum()), "P2_with_notable_sms_pct": float((la & lna).sum() / max(la.sum(), 1)),
                    "notable_sms_that_are_P2_pct": float((la & lna).sum() / max(lna.sum(), 1)),
                    "jaccard": float((la & lna).sum() / max((la | lna).sum(), 1))}
    S["sms_overlap"] = ov
    print("\n=== SMS OVERLAP ==="); [print(k, v) for k, v in ov.items()]
    S["sms_features_in_model"] = [c for c in Xr_tr.columns if "sms" in c.lower()] + \
        [f"{SMS_LEAK_COL}=SMS Sent (one-hot category)"]

    # ===================== 1. Nested primary runs, seed 42, both specs (tuned)
    primary = {}
    for spec in ["orig", "nosms"]:
        print(f"\n=== 1. NESTED PRIMARY RUN  spec={spec} ===")
        r = nested_run(df_fold, policy_fold, RANDOM_STATE, spec, tune=True)
        pb, braw = paired_bootstrap(r["_res_te"], r["p1"], r["p2"], r["cap"], r["blend_w"], r["cutoff"], n=a.boot)
        r["paired_bootstrap"] = pb
        braw.to_csv(f"{a.out}/bootstrap_raw_{spec}.csv", index=False)
        # H1 / cutoff filter counts on train as well
        _, dtr = rank_systems(r["_res_va"], r["p1"], r["p2"], r["cap"], r["blend_w"], r["cutoff"])
        r["diag_val"] = dtr
        primary[spec] = r
        keep = {k: v for k, v in r.items() if not k.startswith("_")}
        S[f"primary_{spec}"] = keep
        print(json.dumps({k: keep[k] for k in ["selected_engine", "val_auc_logit", "val_auc_xgb", "test_auc_engine", "val_auc_full", "fold_dropped_high_missing", "leakage_audit",
                                               "test_auc_full", "p1", "p2", "p1_train_rate", "p2_train_rate", "cutoff", "cap", "blend_w",
                                               "K", "S1_conv", "S2_conv", "S3_conv", "S4_conv", "S1_prec", "S2_prec", "S3_prec", "S4_prec",
                                               "diag_test", "cap_sens_test", "k_sens_test"]}, indent=1, default=str))
        print("paired bootstrap:"); [print(f"  {k}: {v}") for k, v in pb.items()]

    # ===================== 2. Repeated stratified splits (R1.5): FULLY RE-TUNED inside every split, both specs
    rows = []
    for spec in ["orig", "nosms"]:
        for seed in range(a.seeds):
            r = nested_run(df_fold, policy_fold, seed, spec, tune=True)
            rows.append({k: r[k] for k in ["seed", "spec", "p1", "p2", "cap", "blend_w", "cutoff", "K", "selected_engine", "val_auc_logit", "val_auc_xgb",
                                           "test_auc_engine", "test_auc_full", "fold_dropped_high_missing", "test_idx_sha256",
                                           "S1_conv", "S2_conv", "S3_conv", "S4_conv", "S1_prec", "S2_prec", "S3_prec", "S4_prec"]})
            print(f"  seed {seed:2d} {spec:5s} eng={r['selected_engine']} cap={r['cap']} w={r['blend_w']} p1={r['p1']} p2={r['p2']} "
                  f"S1={r['S1_prec']:.4f} S2={r['S2_prec']:.4f} S3={r['S3_prec']:.4f} S4={r['S4_prec']:.4f}", flush=True)
    rep = pd.DataFrame(rows); rep["p1"] = rep["p1"].astype(str); rep["p2"] = rep["p2"].astype(str)
    rep["fold_dropped_high_missing"] = rep["fold_dropped_high_missing"].astype(str)
    rep["S2-S1"] = rep["S2_prec"] - rep["S1_prec"]; rep["S2-S3"] = rep["S2_prec"] - rep["S3_prec"]
    rep["S2-S4"] = rep["S2_prec"] - rep["S4_prec"]; rep["S4-S1"] = rep["S4_prec"] - rep["S1_prec"]
    rep.to_csv(f"{a.out}/repeated_splits.csv", index=False)
    summ = {}
    for spec, g in rep.groupby("spec"):
        summ[spec] = {c: {"mean": float(g[c].mean()), "sd": float(g[c].std()), "min": float(g[c].min()), "max": float(g[c].max())}
                      for c in ["S1_prec", "S2_prec", "S3_prec", "S4_prec", "S2-S1", "S2-S3", "S2-S4", "S4-S1", "test_auc_engine", "test_auc_full"]}
        summ[spec]["frac_S2_gt_S3"] = float((g["S2-S3"] > 0).mean()); summ[spec]["frac_S3_gt_S1"] = float(((g["S3_prec"] - g["S1_prec"]) > 0).mean())
        summ[spec]["engine_chosen"] = g["selected_engine"].value_counts().to_dict()
        summ[spec]["fold_dropped_high_missing"] = g["fold_dropped_high_missing"].value_counts().to_dict()
        summ[spec]["frac_S2_gt_S1"] = float((g["S2-S1"] > 0).mean()); summ[spec]["frac_S2_ge_S1"] = float((g["S2-S1"] >= 0).mean())
        summ[spec]["frac_S2_gt_S4"] = float((g["S2-S4"] > 0).mean()); summ[spec]["frac_S4_gt_S1"] = float((g["S4-S1"] > 0).mean())
        summ[spec]["cap_chosen"] = g["cap"].value_counts().to_dict(); summ[spec]["w_chosen"] = g["blend_w"].value_counts().to_dict()
        summ[spec]["p1_chosen"] = g["p1"].value_counts().to_dict(); summ[spec]["p2_chosen"] = g["p2"].value_counts().to_dict()
    S["repeated_splits"] = summ
    print("\n=== 2. REPEATED SPLITS SUMMARY ==="); print(json.dumps(summ, indent=1, default=str))

    S["runtime_sec"] = time.time() - t0
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1, default=str)
    print(f"\nDone in {S['runtime_sec']/60:.1f} min. Wrote {a.out}/summary.json")


if __name__ == "__main__":
    main()
