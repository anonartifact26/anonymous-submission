# -*- coding: utf-8 -*-
"""
Appendix robustness experiment:
Impact-scale misspecification under fixed line-graph OSUB-KL

Design choice:
- Keep the paper's KL-UCB / OSUB construction unchanged:
    * Bernoulli survival KL-UCB
    * scaled index = r_hat[k] * q_k
    * empirical leader + periodic leader sampling
    * fixed line graph N(k) = {k-1, k, k+1}
- Only the learner's impact scale is misspecified.
- True environment remains unchanged.
- Regret is always evaluated under the true environment.

This version enriches the appendix visualization layout:
- keep one profile figure
- keep one win-rate heatmap
- add opt-arm shift heatmap
- add regret-gap boxplot
- add arm-frequency heatmaps

This avoids overusing line charts in the appendix.
"""

import os
import re
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from scipy.stats import ttest_rel, wilcoxon, t

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

ROOT = r"F:\NIPS"
OUTDIR = os.path.join(ROOT, "outputs_appendix_impact_misspec_fixed_linegraph")
os.makedirs(OUTDIR, exist_ok=True)

SEEDS = list(range(30))
DATASET_ORDER = ["Adult", "BankMarketing"]

K = 11
TAU = 20
T_BATCHES = 140
T_PULLS = T_BATCHES * TAU

Q_DEFENSE = 0.15
N_BASE_ATTACK_SAMPLES = 3000
ALGORITHMS = ["UCB", "KLUCB_OSUB"]
GAMMA_LEAD = 2

ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
MISSPEC_MODE = "adversarial_peak_shift"

COLOR_MAP = {
    "UCB": "#1f77b4",
    "KLUCB_OSUB": "#d62728",
}
LABEL_MAP = {
    "UCB": "UCB",
    "KLUCB_OSUB": "KL-UCB",
}

BOOTSTRAP_B = 10000
BOOTSTRAP_RANDOM_SEED = 12345

# for arm-frequency visualization
TAIL_BATCHES_FOR_FREQ = 40

np.set_printoptions(precision=4, suppress=True)

print(f"[INFO] ROOT={ROOT}")
print(f"[INFO] OUTDIR={OUTDIR}")


# ============================================================
# Utilities
# ============================================================

def safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except Exception:
        return []


def find_file_fuzzy(root: str, include_keywords: List[str]) -> Optional[str]:
    if not os.path.exists(root):
        return None

    candidates = []
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)
        if not os.path.isfile(fpath):
            continue
        low = fname.lower()
        if all(k.lower() in low for k in include_keywords):
            candidates.append(fpath)

    if not candidates:
        return None

    def score(path: str):
        low = os.path.basename(path).lower()
        s = 0
        if ".arff" in low:
            s += 6
        if ".csv" in low:
            s += 5
        if ".data" in low:
            s += 4
        if ".txt" in low:
            s += 3
        return -s

    return sorted(candidates, key=score)[0]


def preview_file(path: str, n_lines: int = 6):
    print(f"[INFO] Preview: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(n_lines):
                line = f.readline()
                if not line:
                    break
                print(f"  L{i+1:02d}: {repr(line[:220])}")
    except Exception as e:
        print(f"[WARN] preview failed: {e}")


def read_head_text(path: str, n_chars: int = 5000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(n_chars)
    except Exception:
        return ""


def looks_like_arff(path: str) -> bool:
    text = read_head_text(path, 5000).lower()
    return ("@relation" in text) and ("@attribute" in text) and ("@data" in text)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def mean_ci95(arr: np.ndarray, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[axis]
    mean = np.mean(arr, axis=axis)
    if n <= 1:
        return mean, np.zeros_like(mean)
    std = np.std(arr, axis=axis, ddof=1)
    ci = 2.045 * std / np.sqrt(n)
    return mean, ci


def monotone_nonincreasing(x: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(np.all(np.diff(np.asarray(x, dtype=float)) <= tol))


def monotone_nondecreasing(x: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(np.all(np.diff(np.asarray(x, dtype=float)) >= -tol))


def is_unimodal(x: np.ndarray, tol: float = 1e-12) -> Tuple[bool, int]:
    x = np.asarray(x, dtype=float)
    peak = int(np.argmax(x))
    left_ok = np.all(np.diff(x[:peak + 1]) >= -tol)
    right_ok = np.all(np.diff(x[peak:]) <= tol)
    return bool(left_ok and right_ok), peak


def kl_bernoulli(p: float, q: float) -> float:
    eps = 1e-12
    p = min(max(float(p), eps), 1 - eps)
    q = min(max(float(q), eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def bootstrap_ci_mean(x: np.ndarray, B: int = 10000, alpha: float = 0.05, seed: int = 12345) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), float(x[0])

    rng = np.random.RandomState(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        means[b] = float(np.mean(x[idx]))
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def t_ci_mean(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), float(x[0])

    mu = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    crit = float(t.ppf(1 - alpha / 2, df=n - 1))
    half = crit * s / math.sqrt(n)
    return mu - half, mu + half


def safe_paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return np.nan, np.nan
    stat, p = ttest_rel(a, b)
    return float(stat), float(p)


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 1:
        return np.nan, np.nan
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


# ============================================================
# ARFF parsing
# ============================================================

def split_arff_row(line: str) -> List[str]:
    parts, cur = [], []
    in_quote = False
    quote_char = None
    for ch in line:
        if ch in ["'", '"']:
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
                quote_char = None
            cur.append(ch)
        elif ch == "," and not in_quote:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def parse_arff_simple(path: str) -> pd.DataFrame:
    attrs = []
    rows = []
    in_data = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue

            low = line.lower()
            if low.startswith("@attribute"):
                m = re.match(r'@attribute\s+(".*?"|\'.*?\'|\S+)\s+(.*)', line, flags=re.I)
                if m:
                    name = m.group(1).strip().strip('"').strip("'")
                    attrs.append(name)
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                parts = split_arff_row(line)
                if len(parts) == len(attrs):
                    cleaned = []
                    for x in parts:
                        x = x.strip().strip('"').strip("'")
                        if x == "?":
                            x = np.nan
                        cleaned.append(x)
                    rows.append(cleaned)

    if not attrs or not rows:
        raise ValueError(f"Failed to parse ARFF file: {path}")

    df = pd.DataFrame(rows, columns=attrs)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


# ============================================================
# Adult loading
# ============================================================

def normalize_adult_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        low = str(c).strip().lower().replace("-", "_")
        if low in ["class", "income", "salary", "target"]:
            rename_map[c] = "income"
        else:
            rename_map[c] = low
    df = df.rename(columns=rename_map)
    if "income" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "income"})
    return df


def parse_adult_raw(path: str) -> pd.DataFrame:
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("@") or low.startswith("%") or low.startswith("|"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 15:
                rows.append(parts)

    if not rows:
        raise ValueError(f"No valid Adult rows parsed from {path}")

    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace("?", np.nan)

    num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip().str.lower()
    df["income"] = df["income"].map({"<=50k": 0, ">50k": 1, "0": 0, "1": 1})
    df = df.dropna().reset_index(drop=True)
    df["income"] = df["income"].astype(int)
    return df


def load_adult(root: str) -> pd.DataFrame:
    path = find_file_fuzzy(root, ["adult"])
    if path is None:
        raise FileNotFoundError(f"Adult file not found under {root}. Files={safe_listdir(root)}")
    preview_file(path)

    if looks_like_arff(path):
        df = parse_arff_simple(path)
        df = normalize_adult_columns(df)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip().replace("?", np.nan)
        for c in ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip().str.lower()
        mapped = df["income"].map({"<=50k": 0, ">50k": 1, "<=50k.": 0, ">50k.": 1, "0": 0, "1": 1})
        if mapped.notna().sum() == 0:
            codes, uniques = pd.factorize(df["income"])
            if len(uniques) != 2:
                raise ValueError(f"Adult target not binary: {list(uniques)}")
            df["income"] = codes.astype(int)
        else:
            df["income"] = mapped
        df = df.dropna().reset_index(drop=True)
        df["income"] = df["income"].astype(int)
    else:
        df = parse_adult_raw(path)

    print(f"[INFO] Adult loaded: {df.shape}")
    return df


# ============================================================
# Bank Marketing loading
# ============================================================

def normalize_bank_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        low = str(c).strip().lower().replace("-", "_").replace(" ", "_")
        if low in ["y", "class", "target", "label"]:
            rename_map[c] = "target"
        else:
            rename_map[c] = low
    df = df.rename(columns=rename_map)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    return df


def load_bank_marketing(root: str) -> pd.DataFrame:
    path = find_file_fuzzy(root, ["bank", "marketing"])
    if path is None:
        path = find_file_fuzzy(root, ["bank"])
    if path is None:
        raise FileNotFoundError(f"Bank Marketing file not found under {root}. Files={safe_listdir(root)}")

    preview_file(path)

    if looks_like_arff(path):
        df = parse_arff_simple(path)
    else:
        try:
            df = pd.read_csv(path, sep=";")
            if df.shape[1] < 2:
                raise ValueError("semicolon parse failed")
        except Exception:
            df = pd.read_csv(path)

    df = normalize_bank_columns(df)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace("?", np.nan)

    df["target"] = df["target"].astype(str).str.strip().str.lower()
    mapped = df["target"].map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if mapped.notna().sum() == 0:
        codes, uniques = pd.factorize(df["target"])
        if len(uniques) != 2:
            raise ValueError(f"Bank target not binary: {list(uniques)}")
        df["target"] = codes.astype(int)
    else:
        df["target"] = mapped

    df = df.dropna().reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    print(f"[INFO] BankMarketing loaded: {df.shape}")
    return df


# ============================================================
# Dataset preparation
# ============================================================

@dataclass
class PreparedDataset:
    name: str
    raw_df: pd.DataFrame
    X_proc: np.ndarray
    y: np.ndarray


def prepare_dataset(df: pd.DataFrame, name: str, target_col: str) -> PreparedDataset:
    X = df.drop(columns=[target_col]).copy()
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe())
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_proc = pre.fit_transform(X).astype(float)
    print(f"[INFO] Prepared {name}: n={X_proc.shape[0]}, d={X_proc.shape[1]}")
    return PreparedDataset(name=name, raw_df=df, X_proc=X_proc, y=y)


# ============================================================
# Stationary simulator construction
# ============================================================

@dataclass
class StationaryBatchedEnv:
    name: str
    K: int
    tau: int
    q_defense: float
    arms: np.ndarray
    impacts_true: np.ndarray
    success_probs: np.ndarray
    mean_rewards_true: np.ndarray
    opt_arm_true: int
    opt_mean_true: float
    threshold: float
    direction_name: str
    base_step: float


def anomaly_score(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    Z = (X - center) / np.maximum(scale, 1e-8)
    return np.sqrt(np.sum(Z ** 2, axis=1))


def impact_scale(arms: np.ndarray) -> np.ndarray:
    return 0.12 + 0.88 * np.power(arms, 0.8)


def candidate_directions(X: np.ndarray, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    dirs = {}

    n, d = X.shape
    n_comp = min(3, d, max(1, min(n - 1, 3)))
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X)

    for i, v in enumerate(pca.components_):
        vv = v / (np.linalg.norm(v) + 1e-12)
        dirs[f"pca_{i+1}"] = vv

    absmean = np.mean(np.abs(X), axis=0)
    absmean = absmean / (np.linalg.norm(absmean) + 1e-12)
    dirs["absmean"] = absmean

    for j in range(4):
        v = rng.normal(size=d)
        v = v / (np.linalg.norm(v) + 1e-12)
        dirs[f"rand_{j+1}"] = v

    return dirs


def build_stationary_batched_env(
    prepared: PreparedDataset,
    K: int,
    tau: int,
    q_defense: float,
    n_base_attack_samples: int,
    seed: int = 123
) -> StationaryBatchedEnv:
    rng = np.random.RandomState(seed)
    X = prepared.X_proc
    n, d = X.shape

    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0) + 1e-8
    ref_scores = anomaly_score(X, center, scale)
    threshold = float(np.quantile(ref_scores, 1.0 - q_defense))

    arms = np.linspace(0.0, 1.0, K)
    impacts = impact_scale(arms)

    idx = rng.choice(n, size=min(n_base_attack_samples, n), replace=(n_base_attack_samples > n))
    X_base = X[idx]

    feature_std = float(np.std(X))
    base_step = 0.55 * feature_std + 0.25

    best_obj = -1e18
    best_pack = None
    diagnostics = []

    for dname, direction in candidate_directions(X, rng).items():
        ps = []
        for a in arms:
            step = base_step * (0.10 + 1.65 * a)
            X_att = X_base + step * direction[None, :]
            scores = anomaly_score(X_att, center, scale)
            p = float(np.mean(scores <= threshold))
            ps.append(p)

        ps = np.asarray(ps, dtype=float)
        ps = np.minimum.accumulate(ps)
        means = impacts * ps
        uni_ok, peak = is_unimodal(means)

        obj = 0.0
        if monotone_nonincreasing(ps):
            obj += 20.0
        else:
            obj -= 200.0

        if monotone_nondecreasing(impacts):
            obj += 10.0

        if uni_ok:
            obj += 18.0
            if 0 < peak < K - 1:
                obj += 4.0
        else:
            obj -= 20.0

        obj += 6.0 * float(means.max() - means.min())
        obj += 2.0 * float(ps[0] - ps[-1])

        diagnostics.append({
            "dataset": prepared.name,
            "direction": dname,
            "score": obj,
            "unimodal": uni_ok,
            "peak": int(peak),
            "p0": float(ps[0]),
            "pK": float(ps[-1]),
            "mu_max": float(means.max()),
            "mu_min": float(means.min()),
        })

        if obj > best_obj:
            best_obj = obj
            best_pack = (dname, ps, means)

    pd.DataFrame(diagnostics).to_csv(
        os.path.join(OUTDIR, f"{prepared.name}_env_direction_diagnostics.csv"), index=False
    )

    dname, success_probs, means_true = best_pack
    opt_arm_true = int(np.argmax(means_true))
    opt_mean_true = float(means_true[opt_arm_true])

    env = StationaryBatchedEnv(
        name=prepared.name,
        K=K,
        tau=tau,
        q_defense=q_defense,
        arms=arms,
        impacts_true=impacts,
        success_probs=success_probs,
        mean_rewards_true=means_true,
        opt_arm_true=opt_arm_true,
        opt_mean_true=opt_mean_true,
        threshold=threshold,
        direction_name=dname,
        base_step=base_step
    )

    print(f"[INFO] Built env for {prepared.name}")
    print(f"       direction={env.direction_name}, opt_arm_true={env.opt_arm_true}, opt_mean_true={env.opt_mean_true:.4f}")
    print(f"       impacts_true={np.round(env.impacts_true, 4)}")
    print(f"       success_probs={np.round(env.success_probs, 4)}")
    print(f"       mean_rewards_true={np.round(env.mean_rewards_true, 4)}")
    return env


# ============================================================
# Impact misspecification
# ============================================================

def build_misspecified_impacts(impacts_true: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    r = np.asarray(impacts_true, dtype=float).copy()
    K_local = len(r)
    if alpha <= 0:
        return r.copy()

    x = np.linspace(0.0, 1.0, K_local)

    if mode == "adversarial_peak_shift":
        z = np.cos(2.0 * np.pi * x)
        z = z / (np.max(np.abs(z)) + 1e-12)
        raw = r * (1.0 + 0.95 * z)

    elif mode == "wiggle":
        z = np.sin(2.5 * np.pi * x)
        z = z / (np.max(np.abs(z)) + 1e-12)
        raw = r * (1.0 + 1.10 * z)

    elif mode == "reverse_tilt":
        z = np.linspace(-1.0, 1.0, K_local)
        raw = r * (1.0 - 1.35 * z)

    else:
        raise ValueError(f"Unknown misspecification mode: {mode}")

    raw = np.maximum(raw, 1e-6)
    r_hat = (1.0 - alpha) * r + alpha * raw
    r_hat = np.maximum(r_hat, 1e-6)
    return r_hat


def impact_error_metrics(r_true: np.ndarray, r_hat: np.ndarray) -> Dict[str, float]:
    r_true = np.asarray(r_true, dtype=float)
    r_hat = np.asarray(r_hat, dtype=float)

    rel_l2 = float(np.linalg.norm(r_hat - r_true) / max(np.linalg.norm(r_true), 1e-12))
    max_abs = float(np.max(np.abs(r_hat - r_true)))
    mean_abs = float(np.mean(np.abs(r_hat - r_true)))

    rank_true = np.argsort(np.argsort(r_true))
    rank_hat = np.argsort(np.argsort(r_hat))
    rank_corr = float(np.corrcoef(rank_true, rank_hat)[0, 1]) if len(r_true) >= 2 else 1.0

    return {
        "rel_l2_error": rel_l2,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rank_corr": rank_corr,
    }


# ============================================================
# Algorithms
# ============================================================

class UCBTrimmedBatch:
    def __init__(self, impacts_for_learning: np.ndarray):
        self.impacts = np.asarray(impacts_for_learning, dtype=float)
        self.K = len(self.impacts)
        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.batch_t = 0

    def select_arm(self) -> int:
        self.batch_t += 1
        for k in range(self.K):
            if self.N[k] == 0:
                return k

        t_cur = self.batch_t
        p_hat = self.S / np.maximum(self.N, 1)
        mu_hat = self.impacts * p_hat
        u = mu_hat + self.impacts * np.sqrt(2.0 * np.log(max(t_cur, 2)) / np.maximum(self.N, 1))
        return int(np.argmax(u))

    def update_batch(self, arm: int, successes: int, tau: int):
        self.N[arm] += tau
        self.S[arm] += successes


class OSUBKLTrimmedBatch:
    """
    Keep the paper's OSUB-KL construction:
    fixed line graph, local neighborhood, leader sampling, scaled Bernoulli KL index.
    Only impacts used by the learner may be misspecified.
    """
    def __init__(self, impacts_for_learning: np.ndarray, gamma_lead: int = 2):
        self.impacts = np.asarray(impacts_for_learning, dtype=float)
        self.K = len(self.impacts)
        self.gamma_lead = gamma_lead

        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.L = np.zeros(self.K, dtype=int)
        self.batch_t = 0

    def _kl_ucb_survival(self, p_hat: float, N_k: int, t_cur: int) -> float:
        if N_k <= 0:
            return 1.0

        rhs = np.log(max(t_cur, 2))
        if t_cur >= 3:
            rhs += 3.0 * np.log(np.log(t_cur))
        rhs /= N_k

        lo = min(max(float(p_hat), 0.0), 1.0)
        hi = 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if kl_bernoulli(p_hat, mid) <= rhs:
                lo = mid
            else:
                hi = mid
        return lo

    def select_arm(self) -> int:
        self.batch_t += 1
        for k in range(self.K):
            if self.N[k] == 0:
                return k

        t_cur = self.batch_t
        p_hat = self.S / np.maximum(self.N, 1)
        mu_hat = self.impacts * p_hat

        leader = int(np.argmax(mu_hat))
        self.L[leader] += 1

        if self.L[leader] % self.gamma_lead == 1:
            return leader

        neigh = [leader]
        if leader - 1 >= 0:
            neigh.append(leader - 1)
        if leader + 1 < self.K:
            neigh.append(leader + 1)

        best_arm = neigh[0]
        best_idx = -1e18
        for k in neigh:
            q = self._kl_ucb_survival(float(p_hat[k]), int(self.N[k]), int(t_cur))
            idx = self.impacts[k] * q
            if idx > best_idx:
                best_idx = idx
                best_arm = k

        return int(best_arm)

    def update_batch(self, arm: int, successes: int, tau: int):
        self.N[arm] += tau
        self.S[arm] += successes


def make_agent(algo_name: str, impacts_for_learning: np.ndarray):
    if algo_name == "UCB":
        return UCBTrimmedBatch(impacts_for_learning)
    elif algo_name == "KLUCB_OSUB":
        return OSUBKLTrimmedBatch(impacts_for_learning, gamma_lead=GAMMA_LEAD)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ============================================================
# Experiment runner
# ============================================================

@dataclass
class RunResult:
    chosen_arms: np.ndarray
    reward_per_batch: np.ndarray
    realized_regret_curve: np.ndarray
    pseudo_regret_curve: np.ndarray
    final_realized_regret: float
    final_pseudo_regret: float


def run_single_seed_stationary_batched_misspecified(
    env: StationaryBatchedEnv,
    algo_name: str,
    impacts_for_learning: np.ndarray,
    seed: int,
    T_batches: int
) -> RunResult:
    rng = np.random.RandomState(seed)
    agent = make_agent(algo_name, impacts_for_learning)

    chosen_arms = []
    rewards = []
    realized_regrets = []
    pseudo_regrets = []

    cum_realized_regret = 0.0
    cum_pseudo_regret = 0.0

    true_batch_oracle = env.tau * env.opt_mean_true

    for _ in range(T_batches):
        arm = agent.select_arm()

        successes = int(rng.binomial(env.tau, env.success_probs[arm]))
        batch_reward = env.impacts_true[arm] * successes

        agent.update_batch(arm, successes, env.tau)

        batch_true_mean = env.tau * env.mean_rewards_true[arm]

        cum_realized_regret += (true_batch_oracle - batch_reward)
        cum_pseudo_regret += (true_batch_oracle - batch_true_mean)

        chosen_arms.append(arm)
        rewards.append(batch_reward)
        realized_regrets.append(cum_realized_regret)
        pseudo_regrets.append(cum_pseudo_regret)

    return RunResult(
        chosen_arms=np.asarray(chosen_arms, dtype=int),
        reward_per_batch=np.asarray(rewards, dtype=float),
        realized_regret_curve=np.asarray(realized_regrets, dtype=float),
        pseudo_regret_curve=np.asarray(pseudo_regrets, dtype=float),
        final_realized_regret=float(realized_regrets[-1]),
        final_pseudo_regret=float(pseudo_regrets[-1]),
    )


# ============================================================
# Build datasets
# ============================================================

def load_all_prepared_datasets() -> Dict[str, PreparedDataset]:
    print("[INFO] Files under ROOT:")
    for x in safe_listdir(ROOT):
        print("   ", x)

    adult_df = load_adult(ROOT)
    bank_df = load_bank_marketing(ROOT)

    adult = prepare_dataset(adult_df, "Adult", "income")
    bank = prepare_dataset(bank_df, "BankMarketing", "target")

    return {"Adult": adult, "BankMarketing": bank}


# ============================================================
# Plotting helpers
# ============================================================

def _ensure_alpha_sorted(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["alpha"] = out["alpha"].astype(float)
    out = out.sort_values(["dataset", "alpha"]).reset_index(drop=True)
    return out


# ============================================================
# Plotting: impact profiles
# ============================================================

def plot_impact_profiles(env_map: Dict[str, StationaryBatchedEnv]):
    """
    A compact profile figure. Keep only one such figure family.
    """
    for dataset in DATASET_ORDER:
        env = env_map[dataset]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(env.K)

        ax.plot(x, env.impacts_true, marker="o", lw=2.6, color="black", label="true impact")

        for alpha in ALPHAS:
            if alpha == 0:
                continue
            r_hat = build_misspecified_impacts(env.impacts_true, alpha, MISSPEC_MODE)
            ax.plot(x, r_hat, marker=".", lw=1.5, alpha=0.85, label=f"α={alpha:.2f}")

        ax.set_title(f"{dataset}: true vs misspecified impacts")
        ax.set_xlabel("Arm index")
        ax.set_ylabel("Impact scale")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8, frameon=True)
        plt.tight_layout()

        outpath = os.path.join(OUTDIR, f"{dataset}_impact_profiles.png")
        plt.savefig(outpath, dpi=260)
        plt.close()


# ============================================================
# Plotting: win-rate heatmap
# ============================================================

def plot_win_rate_heatmap(df_final: pd.DataFrame):
    mat = np.zeros((len(DATASET_ORDER), len(ALPHAS)), dtype=float)

    for i, ds in enumerate(DATASET_ORDER):
        for j, alpha in enumerate(ALPHAS):
            sub = df_final[(df_final["dataset"] == ds) & (df_final["alpha"] == alpha)]
            piv = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")
            mat[i, j] = float(np.mean(piv["KLUCB_OSUB"].values < piv["UCB"].values))

    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(ALPHAS), 1.8 + 0.9 * len(DATASET_ORDER)))
    im = ax.imshow(mat, aspect="auto", cmap="plasma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(ALPHAS)))
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS], rotation=25, ha="right")
    ax.set_yticklabels(DATASET_ORDER)
    ax.set_xlabel("Misspecification strength α")
    ax.set_title("KL-UCB win-rate vs UCB")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if val < 0.4 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Win rate")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "heatmap_winrate_vs_alpha.png"), dpi=260)
    plt.close()


# ============================================================
# Plotting: opt-arm shift heatmap
# ============================================================

def plot_opt_arm_shift_heatmap(df_diag: pd.DataFrame):
    """
    Color shows mis_opt_arm - true_opt_arm.
    This directly visualizes perceived-optimum drift under misspecification.
    """
    df_diag = _ensure_alpha_sorted(df_diag)
    mat = np.zeros((len(DATASET_ORDER), len(ALPHAS)), dtype=float)

    for i, ds in enumerate(DATASET_ORDER):
        for j, alpha in enumerate(ALPHAS):
            sub = df_diag[(df_diag["dataset"] == ds) & (df_diag["alpha"] == alpha)]
            if len(sub) == 0:
                mat[i, j] = np.nan
            else:
                row = sub.iloc[0]
                mat[i, j] = float(row["mis_opt_arm"] - row["true_opt_arm"])

    vmax = max(1.0, np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(ALPHAS), 1.8 + 0.9 * len(DATASET_ORDER)))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(ALPHAS)))
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS], rotation=25, ha="right")
    ax.set_yticklabels(DATASET_ORDER)
    ax.set_xlabel("Misspecification strength α")
    ax.set_title("Shift in perceived optimal arm")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{int(val):d}", ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("mis-opt arm − true-opt arm")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "heatmap_opt_arm_shift.png"), dpi=260)
    plt.close()


# ============================================================
# Plotting: regret-gap boxplot
# ============================================================

def plot_regret_gap_boxplot(df_final: pd.DataFrame, metric: str = "final_realized_regret"):
    """
    Plot gap = UCB regret - KLUCB_OSUB regret.
    Positive => KL-UCB better.
    """
    work = df_final.copy()
    piv = work.pivot_table(
        index=["dataset", "alpha", "seed"],
        columns="algorithm",
        values=metric
    ).reset_index()

    piv["gap_ucb_minus_klucb"] = piv["UCB"] - piv["KLUCB_OSUB"]
    piv = _ensure_alpha_sorted(piv)

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(6.2 * len(DATASET_ORDER), 4.8), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = piv[piv["dataset"] == dataset].copy()
        data_by_alpha = [sub[sub["alpha"] == a]["gap_ucb_minus_klucb"].values for a in ALPHAS]
        positions = np.arange(len(ALPHAS))

        bp = ax.boxplot(
            data_by_alpha,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False
        )

        for box in bp["boxes"]:
            box.set(facecolor="#f4a261", alpha=0.65, edgecolor="#7f5539")
        for med in bp["medians"]:
            med.set(color="#7f0000", linewidth=2.0)
        for whisk in bp["whiskers"]:
            whisk.set(color="#555555", linewidth=1.2)
        for cap in bp["caps"]:
            cap.set(color="#555555", linewidth=1.2)

        for i, alpha in enumerate(ALPHAS):
            y = sub[sub["alpha"] == alpha]["gap_ucb_minus_klucb"].values
            x = np.random.normal(loc=i, scale=0.05, size=len(y))
            ax.scatter(x, y, s=16, alpha=0.45, color="#264653")

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS], rotation=25, ha="right")
        ax.set_xlabel("Misspecification strength α")
        ax.set_title(dataset)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Regret gap: UCB − KL-UCB")
    plt.suptitle("Seed-level regret gap under impact misspecification", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"boxplot_regret_gap_{metric}.png"), dpi=260, bbox_inches="tight")
    plt.close()


# ============================================================
# Plotting: arm-frequency heatmap
# ============================================================

def plot_arm_frequency_heatmap(df_long: pd.DataFrame, tail_batches: int = 40):
    """
    For each dataset, show a 2-panel heatmap:
    rows = alpha
    cols = arm index
    color = selection frequency in the last `tail_batches` batches
    one panel per algorithm
    """
    max_batch = int(df_long["batch"].max())
    batch_cut = max(1, max_batch - tail_batches + 1)

    tail_df = df_long[df_long["batch"] >= batch_cut].copy()
    tail_df["alpha"] = tail_df["alpha"].astype(float)

    for dataset in DATASET_ORDER:
        fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(6.0 * len(ALGORITHMS), 5.2), sharey=True)
        if len(ALGORITHMS) == 1:
            axes = [axes]

        vmin, vmax = 0.0, 1.0

        ims = []
        for ax, algo in zip(axes, ALGORITHMS):
            sub = tail_df[(tail_df["dataset"] == dataset) & (tail_df["algorithm"] == algo)].copy()

            mat = np.zeros((len(ALPHAS), K), dtype=float)
            for i, alpha in enumerate(ALPHAS):
                sub_a = sub[sub["alpha"] == alpha]
                total = len(sub_a)
                if total == 0:
                    continue
                counts = sub_a["chosen_arm"].value_counts().to_dict()
                for arm in range(K):
                    mat[i, arm] = counts.get(arm, 0) / total

            im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
            ims.append(im)

            ax.set_title(f"{dataset} | {LABEL_MAP[algo]}")
            ax.set_xlabel("Arm index")
            ax.set_xticks(np.arange(K))
            ax.set_yticks(np.arange(len(ALPHAS)))
            ax.set_yticklabels([f"{a:.2f}" for a in ALPHAS])
            if ax is axes[0]:
                ax.set_ylabel("Misspecification strength α")

            for i in range(len(ALPHAS)):
                for j in range(K):
                    val = mat[i, j]
                    if val >= 0.15:
                        color = "white" if val > 0.45 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

        cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.92)
        cbar.set_label(f"Arm-selection frequency in last {tail_batches} batches")

        plt.suptitle(f"Arm concentration under misspecified impacts: {dataset}", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{dataset}_heatmap_arm_frequency_tail{tail_batches}.png"),
                    dpi=260, bbox_inches="tight")
        plt.close()


# ============================================================
# Optional statistics summary
# ============================================================

def build_stat_summary(df_final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASET_ORDER:
        for alpha in ALPHAS:
            sub = df_final[(df_final["dataset"] == ds) & (df_final["alpha"] == alpha)]
            piv = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")

            a = piv["UCB"].values.astype(float)
            b = piv["KLUCB_OSUB"].values.astype(float)

            gap = a - b
            t_stat, t_p = safe_paired_ttest(a, b)
            w_stat, w_p = safe_wilcoxon(a, b)

            lo, hi = t_ci_mean(gap)
            rows.append({
                "dataset": ds,
                "alpha": alpha,
                "mean_gap_ucb_minus_klucb": float(np.mean(gap)),
                "ci_low": lo,
                "ci_high": hi,
                "klucb_win_rate": float(np.mean(b < a)),
                "paired_t_stat": t_stat,
                "paired_t_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
            })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    prepared_map = load_all_prepared_datasets()

    env_map: Dict[str, StationaryBatchedEnv] = {}
    for name in DATASET_ORDER:
        env_map[name] = build_stationary_batched_env(
            prepared=prepared_map[name],
            K=K,
            tau=TAU,
            q_defense=Q_DEFENSE,
            n_base_attack_samples=N_BASE_ATTACK_SAMPLES,
            seed=123
        )

    final_rows = []
    long_rows = []
    diag_rows = []

    for dataset_name in DATASET_ORDER:
        env = env_map[dataset_name]
        print(f"\n[INFO] Running dataset={dataset_name}")

        for alpha in ALPHAS:
            impacts_hat = build_misspecified_impacts(env.impacts_true, alpha, MISSPEC_MODE)
            mets = impact_error_metrics(env.impacts_true, impacts_hat)

            mu_true = env.impacts_true * env.success_probs
            mu_mis = impacts_hat * env.success_probs

            true_opt = int(np.argmax(mu_true))
            mis_opt = int(np.argmax(mu_mis))

            diag_rows.append({
                "dataset": dataset_name,
                "alpha": alpha,
                "misspec_mode": MISSPEC_MODE,
                "true_opt_arm": true_opt,
                "mis_opt_arm": mis_opt,
                "opt_changed": bool(true_opt != mis_opt),
                "opt_shift": int(mis_opt - true_opt),
                "rel_l2_error": mets["rel_l2_error"],
                "max_abs_error": mets["max_abs_error"],
                "mean_abs_error": mets["mean_abs_error"],
                "rank_corr": mets["rank_corr"],
                "mu_true_top1": float(np.max(mu_true)),
                "mu_mis_top1": float(np.max(mu_mis)),
                "r_true": json.dumps([float(x) for x in env.impacts_true]),
                "r_hat": json.dumps([float(x) for x in impacts_hat]),
                "mu_true": json.dumps([float(x) for x in mu_true]),
                "mu_mis": json.dumps([float(x) for x in mu_mis]),
            })

            print(
                f"  [INFO] alpha={alpha:.2f}, mis_opt={mis_opt}, true_opt={true_opt}, "
                f"rel_l2={mets['rel_l2_error']:.4f}, rank_corr={mets['rank_corr']:.4f}"
            )
            print(f"         r_hat={np.round(impacts_hat, 4)}")
            print(f"         mu_mis={np.round(mu_mis, 4)}")

            for seed in SEEDS:
                for algo in ALGORITHMS:
                    res = run_single_seed_stationary_batched_misspecified(
                        env=env,
                        algo_name=algo,
                        impacts_for_learning=impacts_hat,
                        seed=seed,
                        T_batches=T_BATCHES
                    )

                    final_rows.append({
                        "dataset": dataset_name,
                        "alpha": alpha,
                        "misspec_mode": MISSPEC_MODE,
                        "seed": seed,
                        "algorithm": algo,
                        "rel_l2_error": mets["rel_l2_error"],
                        "rank_corr": mets["rank_corr"],
                        "mis_opt_arm": mis_opt,
                        "true_opt_arm": true_opt,
                        "final_realized_regret": res.final_realized_regret,
                        "final_pseudo_regret": res.final_pseudo_regret,
                    })

                    for b in range(T_BATCHES):
                        long_rows.append({
                            "dataset": dataset_name,
                            "alpha": alpha,
                            "misspec_mode": MISSPEC_MODE,
                            "seed": seed,
                            "algorithm": algo,
                            "batch": b + 1,
                            "cum_realized_regret": float(res.realized_regret_curve[b]),
                            "cum_pseudo_regret": float(res.pseudo_regret_curve[b]),
                            "batch_reward": float(res.reward_per_batch[b]),
                            "chosen_arm": int(res.chosen_arms[b]),
                        })

    df_final = pd.DataFrame(final_rows)
    df_long = pd.DataFrame(long_rows)
    df_diag = pd.DataFrame(diag_rows)
    df_stat = build_stat_summary(df_final)

    df_final.to_csv(os.path.join(OUTDIR, "impact_misspecification_final.csv"), index=False)
    df_long.to_csv(os.path.join(OUTDIR, "impact_misspecification_long.csv"), index=False)
    df_diag.to_csv(os.path.join(OUTDIR, "impact_misspecification_diagnostics.csv"), index=False)
    df_stat.to_csv(os.path.join(OUTDIR, "impact_misspecification_stat_summary.csv"), index=False)

    # Figures
    plot_impact_profiles(env_map)
    plot_win_rate_heatmap(df_final)
    plot_opt_arm_shift_heatmap(df_diag)
    plot_regret_gap_boxplot(df_final, metric="final_realized_regret")
    plot_regret_gap_boxplot(df_final, metric="final_pseudo_regret")
    plot_arm_frequency_heatmap(df_long, tail_batches=TAIL_BATCHES_FOR_FREQ)

    run_config = {
        "root": ROOT,
        "datasets": DATASET_ORDER,
        "seeds": SEEDS,
        "K": K,
        "tau": TAU,
        "T_batches": T_BATCHES,
        "T_pulls": T_PULLS,
        "q_defense": Q_DEFENSE,
        "n_base_attack_samples": N_BASE_ATTACK_SAMPLES,
        "algorithms": ALGORITHMS,
        "gamma_lead": GAMMA_LEAD,
        "alphas": ALPHAS,
        "misspec_mode": MISSPEC_MODE,
        "tail_batches_for_freq": TAIL_BATCHES_FOR_FREQ
    }
    with open(os.path.join(OUTDIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()