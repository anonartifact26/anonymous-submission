# -*- coding: utf-8 -*-
"""
Ablation experiment: Adult + Bank Marketing
Full visualization + 30 seeds + component-wise decomposition

What this script does:
- builds stationary unimodal environments from real tabular datasets
- runs 30 seeds for the following algorithms:
    * UCB              : baseline
    * KLUCB_GLOBAL     : KL-UCB without neighborhood restriction
    * UCB_OSUB         : UCB-style index with OSUB neighborhood restriction
    * KLUCB_OSUB       : full method
- saves long-form results
- generates multiple plots:
    1) cumulative realized regret curves
    2) final realized regret boxplots
    3) final realized regret violin plots
    4) mean final regret heatmap
    5) pairwise win-rate heatmap vs full method
    6) arm-pull heatmaps
    7) optimal-neighborhood concentration heatmap
    8) paired scatter plots (full vs component ablations)
    9) arm trajectory plots
   10) relative improvement bars
   11) environment profile plots
   12) component contribution bars

Important:
- This script does NOT fabricate outcomes.
- All figures are generated from actual simulation outputs.
- Designed to quantify the contribution of:
    (i) KL confidence bound
    (ii) OSUB neighborhood search / leader schedule
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
OUTDIR = os.path.join(ROOT, "outputs_ablation_stationary_batched_adult_bank_30seeds_fullviz")
os.makedirs(OUTDIR, exist_ok=True)

SEEDS = list(range(30))
DATASET_ORDER = ["Adult", "BankMarketing"]

K = 11
TAU = 20
T_BATCHES = 140
T_PULLS = T_BATCHES * TAU

Q_DEFENSE = 0.15
N_BASE_ATTACK_SAMPLES = 3000

ABLATION_ALGORITHMS = ["UCB", "KLUCB_GLOBAL", "UCB_OSUB", "KLUCB_OSUB"]
FULL_ALGO = "KLUCB_OSUB"
BASELINE_ALGO = "UCB"

GAMMA_LEAD = 2

COLOR_MAP = {
    "UCB": "#1f77b4",
    "KLUCB_GLOBAL": "#ff7f0e",
    "UCB_OSUB": "#2ca02c",
    "KLUCB_OSUB": "#d62728",
}
LABEL_MAP = {
    "UCB": "UCB",
    "KLUCB_GLOBAL": "KL-UCB (global)",
    "UCB_OSUB": "UCB-OSUB",
    "KLUCB_OSUB": "KL-UCB-OSUB",
}

BOOTSTRAP_B = 10000
BOOTSTRAP_RANDOM_SEED = 12345

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
    impacts: np.ndarray
    success_probs: np.ndarray
    mean_rewards: np.ndarray
    opt_arm: int
    opt_mean: float
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

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(os.path.join(OUTDIR, f"{prepared.name}_env_direction_diagnostics.csv"), index=False)

    dname, success_probs, means = best_pack
    opt_arm = int(np.argmax(means))
    opt_mean = float(means[opt_arm])

    env = StationaryBatchedEnv(
        name=prepared.name,
        K=K,
        tau=tau,
        q_defense=q_defense,
        arms=arms,
        impacts=impacts,
        success_probs=success_probs,
        mean_rewards=means,
        opt_arm=opt_arm,
        opt_mean=opt_mean,
        threshold=threshold,
        direction_name=dname,
        base_step=base_step
    )

    print(f"[INFO] Built env for {prepared.name}")
    print(f"       direction={env.direction_name}, opt_arm={env.opt_arm}, opt_mean={env.opt_mean:.4f}")
    print(f"       impacts={np.round(env.impacts, 4)}")
    print(f"       success_probs={np.round(env.success_probs, 4)}")
    print(f"       mean_rewards={np.round(env.mean_rewards, 4)}")
    print(f"       monotone impact={monotone_nondecreasing(env.impacts)}")
    print(f"       monotone survival={monotone_nonincreasing(env.success_probs)}")
    print(f"       unimodal utility={is_unimodal(env.mean_rewards)[0]}")
    return env


# ============================================================
# Algorithms for ablation
# ============================================================

class UCBTrimmedBatch:
    def __init__(self, impacts: np.ndarray):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.batch_t = 0

    def select_arm(self) -> int:
        self.batch_t += 1
        for k in range(self.K):
            if self.N[k] == 0:
                return k

        t = self.batch_t
        p_hat = self.S / np.maximum(self.N, 1)
        mu_hat = self.impacts * p_hat
        u = mu_hat + self.impacts * np.sqrt(2.0 * np.log(max(t, 2)) / np.maximum(self.N, 1))
        return int(np.argmax(u))

    def update_batch(self, arm: int, successes: int, tau: int):
        self.N[arm] += tau
        self.S[arm] += successes


class KLUCBGlobalTrimmedBatch:
    def __init__(self, impacts: np.ndarray):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.batch_t = 0

    def _kl_ucb_survival(self, p_hat: float, N_k: int, t_now: int) -> float:
        if N_k <= 0:
            return 1.0

        rhs = np.log(max(t_now, 2))
        if t_now >= 3:
            rhs += 3.0 * np.log(np.log(t_now))
        rhs /= N_k

        lo = min(max(p_hat, 0.0), 1.0)
        hi = 1.0
        for _ in range(40):
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

        t_now = self.batch_t
        p_hat = self.S / np.maximum(self.N, 1)

        indices = np.zeros(self.K, dtype=float)
        for k in range(self.K):
            q = self._kl_ucb_survival(float(p_hat[k]), int(self.N[k]), int(t_now))
            indices[k] = self.impacts[k] * q
        return int(np.argmax(indices))

    def update_batch(self, arm: int, successes: int, tau: int):
        self.N[arm] += tau
        self.S[arm] += successes


class UCBOSUBTrimmedBatch:
    def __init__(self, impacts: np.ndarray, gamma_lead: int = 2):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.gamma_lead = gamma_lead

        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.L = np.zeros(self.K, dtype=int)
        self.batch_t = 0

    def select_arm(self) -> int:
        self.batch_t += 1
        for k in range(self.K):
            if self.N[k] == 0:
                return k

        t_now = self.batch_t
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
            idx = mu_hat[k] + self.impacts[k] * np.sqrt(2.0 * np.log(max(t_now, 2)) / np.maximum(self.N[k], 1))
            if idx > best_idx:
                best_idx = idx
                best_arm = k
        return int(best_arm)

    def update_batch(self, arm: int, successes: int, tau: int):
        self.N[arm] += tau
        self.S[arm] += successes


class OSUBKLTrimmedBatch:
    def __init__(self, impacts: np.ndarray, gamma_lead: int = 2):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.gamma_lead = gamma_lead

        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.L = np.zeros(self.K, dtype=int)
        self.batch_t = 0

    def _kl_ucb_survival(self, p_hat: float, N_k: int, t_now: int) -> float:
        if N_k <= 0:
            return 1.0

        rhs = np.log(max(t_now, 2))
        if t_now >= 3:
            rhs += 3.0 * np.log(np.log(t_now))
        rhs /= N_k

        lo = min(max(p_hat, 0.0), 1.0)
        hi = 1.0
        for _ in range(40):
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

        t_now = self.batch_t
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
            q = self._kl_ucb_survival(float(p_hat[k]), int(self.N[k]), int(t_now))
            idx = self.impacts[k] * q
            if idx > best_idx:
                best_idx = idx
                best_arm = k

        return int(best_arm)

    def update_batch(self, arm: int, successes: int, tau: int):
        self.N[arm] += tau
        self.S[arm] += successes


def make_agent(algo_name: str, impacts: np.ndarray):
    if algo_name == "UCB":
        return UCBTrimmedBatch(impacts)
    elif algo_name == "KLUCB_GLOBAL":
        return KLUCBGlobalTrimmedBatch(impacts)
    elif algo_name == "UCB_OSUB":
        return UCBOSUBTrimmedBatch(impacts, gamma_lead=GAMMA_LEAD)
    elif algo_name == "KLUCB_OSUB":
        return OSUBKLTrimmedBatch(impacts, gamma_lead=GAMMA_LEAD)
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


def run_single_seed_stationary_batched(
    env: StationaryBatchedEnv,
    algo_name: str,
    seed: int,
    T_batches: int
) -> RunResult:
    rng = np.random.RandomState(seed)
    agent = make_agent(algo_name, env.impacts)

    chosen_arms = []
    rewards = []
    realized_regrets = []
    pseudo_regrets = []

    cum_realized_regret = 0.0
    cum_pseudo_regret = 0.0

    for _ in range(T_batches):
        arm = agent.select_arm()

        successes = int(rng.binomial(env.tau, env.success_probs[arm]))
        batch_reward = env.impacts[arm] * successes

        agent.update_batch(arm, successes, env.tau)

        batch_oracle = env.tau * env.opt_mean
        batch_mean = env.tau * env.mean_rewards[arm]

        cum_realized_regret += (batch_oracle - batch_reward)
        cum_pseudo_regret += (batch_oracle - batch_mean)

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
        final_pseudo_regret=float(pseudo_regrets[-1])
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
# Plot helpers
# ============================================================

def plot_matrix_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    outpath: str,
    fmt: str = ".2f",
    cmap: str = "viridis"
):
    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(col_labels), 1.8 + 0.9 * len(row_labels)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="white", fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


# ============================================================
# Plotting
# ============================================================

def plot_regret_curves_and_box(dataset_name: str, df_long: pd.DataFrame):
    sub = df_long[df_long["dataset"] == dataset_name].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9))

    ax = axes[0]
    for algo in ABLATION_ALGORITHMS:
        ss = sub[sub["algorithm"] == algo]
        pivot = ss.pivot(index="seed", columns="batch", values="cum_realized_regret").sort_index(axis=1)
        arr = pivot.values
        mean, ci = mean_ci95(arr, axis=0)
        x = pivot.columns.values.astype(int)

        ax.plot(x, mean, label=LABEL_MAP[algo], lw=2.0, color=COLOR_MAP[algo])
        ax.fill_between(x, mean - ci, mean + ci, color=COLOR_MAP[algo], alpha=0.18)

    ax.set_title(f"{dataset_name}: cumulative realized regret")
    ax.set_xlabel("Batch step")
    ax.set_ylabel("Empirical cumulative realized regret")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    data = []
    labels = []
    colors = []
    for algo in ABLATION_ALGORITHMS:
        vals = sub[sub["batch"] == sub["batch"].max()]
        vals = vals[vals["algorithm"] == algo]["cum_realized_regret"].values
        data.append(vals)
        labels.append(LABEL_MAP[algo])
        colors.append(COLOR_MAP[algo])

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)

    ax.set_title(f"{dataset_name}: final realized regret distribution")
    ax.set_ylabel("Final realized regret")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, f"{dataset_name}_ablation_regret_curve_box.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_regret_violin(dataset_name: str, df_final: pd.DataFrame):
    sub = df_final[df_final["dataset"] == dataset_name].copy()

    data = []
    labels = []
    colors = []
    for algo in ABLATION_ALGORITHMS:
        vals = sub[sub["algorithm"] == algo]["final_realized_regret"].values.astype(float)
        data.append(vals)
        labels.append(LABEL_MAP[algo])
        colors.append(COLOR_MAP[algo])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    vp = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.25)

    for partname in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
        if partname in vp:
            vp[partname].set_color("black")
            vp[partname].set_linewidth(1.0)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title(f"{dataset_name}: final realized regret violin plot")
    ax.set_ylabel("Final realized regret")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, f"{dataset_name}_ablation_final_regret_violin.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_environment_profile(env: StationaryBatchedEnv):
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8))
    x = np.arange(env.K)

    ax.plot(x, env.impacts, marker="o", lw=2, label="impact", color="#2ca02c")
    ax.plot(x, env.success_probs, marker="s", lw=2, label="survival prob", color="#9467bd")
    ax.plot(x, env.mean_rewards, marker="^", lw=2.4, label="expected utility", color="#ff7f0e")

    ax.axvline(env.opt_arm, ls="--", lw=1.4, color="black", alpha=0.7, label=f"opt arm = {env.opt_arm}")
    ax.set_title(f"{env.name}: environment profile")
    ax.set_xlabel("Arm index")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    outpath = os.path.join(OUTDIR, f"{env.name}_environment_profile.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_paired_scatter_full_vs_others(dataset_name: str, df_final: pd.DataFrame):
    sub = df_final[df_final["dataset"] == dataset_name].copy()
    piv = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")

    if FULL_ALGO not in piv.columns:
        return

    others = [a for a in ABLATION_ALGORITHMS if a != FULL_ALGO]
    fig, axes = plt.subplots(1, len(others), figsize=(5.3 * len(others), 4.8))
    if len(others) == 1:
        axes = [axes]

    y = piv[FULL_ALGO].values.astype(float)

    for ax, other in zip(axes, others):
        if other not in piv.columns:
            continue
        x = piv[other].values.astype(float)
        m = max(float(np.max(x)), float(np.max(y))) * 1.05

        ax.scatter(x, y, s=46, color="#8c564b", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.plot([0, m], [0, m], ls="--", color="gray", lw=1.3)

        better = int(np.sum(y < x))
        equal = int(np.sum(y == x))
        worse = int(np.sum(y > x))

        ax.set_title(f"{LABEL_MAP[FULL_ALGO]} vs {LABEL_MAP[other]}\nfull better on {better}/30 seeds", fontsize=10)
        ax.set_xlabel(f"{LABEL_MAP[other]}\nfinal realized regret")
        ax.set_ylabel(f"{LABEL_MAP[FULL_ALGO]}\nfinal realized regret")
        ax.grid(True, alpha=0.3)

        txt = f"better={better}\nequal={equal}\nworse={worse}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"), fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, f"{dataset_name}_ablation_paired_scatter_full_vs_others.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_arm_trajectory(dataset_name: str, df_long: pd.DataFrame, env: StationaryBatchedEnv):
    sub = df_long[df_long["dataset"] == dataset_name].copy()

    fig, ax = plt.subplots(figsize=(9.4, 5.0))

    for algo in ABLATION_ALGORITHMS:
        ss = sub[sub["algorithm"] == algo]
        pivot = ss.pivot(index="seed", columns="batch", values="chosen_arm").sort_index(axis=1)
        arr = pivot.values.astype(float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)

        x = pivot.columns.values.astype(int)
        ax.plot(x, mean, lw=2.0, color=COLOR_MAP[algo], label=LABEL_MAP[algo])
        ax.fill_between(x, mean - std, mean + std, color=COLOR_MAP[algo], alpha=0.13)

    ax.axhline(env.opt_arm, ls="--", lw=1.5, color="black", alpha=0.75, label=f"optimal arm={env.opt_arm}")
    ax.set_title(f"{dataset_name}: arm trajectory across batches")
    ax.set_xlabel("Batch step")
    ax.set_ylabel("Chosen arm index (mean ± 1 std over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    outpath = os.path.join(OUTDIR, f"{dataset_name}_ablation_arm_trajectory.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_arm_pull_heatmaps(dataset_name: str, df_arms: pd.DataFrame, env: StationaryBatchedEnv):
    for algo in ABLATION_ALGORITHMS:
        sub = df_arms[(df_arms["dataset"] == dataset_name) & (df_arms["algorithm"] == algo)].copy()

        mat = np.zeros((len(SEEDS), env.K), dtype=float)
        for _, row in sub.iterrows():
            s = int(row["seed"])
            a = int(row["arm"])
            mat[s, a] = float(row["n_batches"])

        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")

        ax.set_title(f"{dataset_name}: arm-pull heatmap ({LABEL_MAP[algo]})")
        ax.set_xlabel("Arm index")
        ax.set_ylabel("Seed")
        ax.set_xticks(np.arange(env.K))
        ax.set_yticks(np.arange(len(SEEDS)))
        ax.set_yticklabels(SEEDS)

        plt.colorbar(im, ax=ax, label="# batches")
        plt.tight_layout()

        outpath = os.path.join(OUTDIR, f"{dataset_name}_{algo}_ablation_arm_pull_heatmap.png")
        plt.savefig(outpath, dpi=240)
        plt.close()
        print(f"[INFO] Saved figure: {outpath}")


def plot_relative_improvement(df_final: pd.DataFrame):
    rows = []
    for dataset in DATASET_ORDER:
        sub = df_final[df_final["dataset"] == dataset]

        mu_base = sub[sub["algorithm"] == BASELINE_ALGO]["final_realized_regret"].mean()
        for algo in ABLATION_ALGORITHMS:
            mu_algo = sub[sub["algorithm"] == algo]["final_realized_regret"].mean()
            improve = 100.0 * (mu_base - mu_algo) / max(abs(mu_base), 1e-12)
            rows.append({
                "dataset": dataset,
                "algorithm": algo,
                "improvement_pct_vs_ucb": improve
            })

    sdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(6.2 * len(DATASET_ORDER), 4.8), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = sdf[sdf["dataset"] == dataset].copy()
        sub = sub.set_index("algorithm").loc[ABLATION_ALGORITHMS].reset_index()

        bars = ax.bar(
            [LABEL_MAP[a] for a in sub["algorithm"]],
            sub["improvement_pct_vs_ucb"].values,
            color=[COLOR_MAP[a] for a in sub["algorithm"]],
            alpha=0.8
        )

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"{dataset}: relative improvement vs UCB")
        ax.set_ylabel("Mean final regret reduction (%)")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "ablation_relative_improvement_bar.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_component_contribution(df_final: pd.DataFrame):
    rows = []
    for dataset in DATASET_ORDER:
        sub = df_final[df_final["dataset"] == dataset].copy()
        means = {
            algo: float(sub[sub["algorithm"] == algo]["final_realized_regret"].mean())
            for algo in ABLATION_ALGORITHMS
        }

        rows.append({
            "dataset": dataset,
            "component": "KL only gain\n(UCB - KLUCB_GLOBAL)",
            "value": means["UCB"] - means["KLUCB_GLOBAL"]
        })
        rows.append({
            "dataset": dataset,
            "component": "OSUB only gain\n(UCB - UCB_OSUB)",
            "value": means["UCB"] - means["UCB_OSUB"]
        })
        rows.append({
            "dataset": dataset,
            "component": "Full gain\n(UCB - KLUCB_OSUB)",
            "value": means["UCB"] - means["KLUCB_OSUB"]
        })
        rows.append({
            "dataset": dataset,
            "component": "Extra synergy\n(min ablation - full)",
            "value": min(means["KLUCB_GLOBAL"], means["UCB_OSUB"]) - means["KLUCB_OSUB"]
        })

    cdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(6.2 * len(DATASET_ORDER), 4.8), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]

    color_map_component = {
        "KL only gain\n(UCB - KLUCB_GLOBAL)": "#ff7f0e",
        "OSUB only gain\n(UCB - UCB_OSUB)": "#2ca02c",
        "Full gain\n(UCB - KLUCB_OSUB)": "#d62728",
        "Extra synergy\n(min ablation - full)": "#9467bd",
    }

    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = cdf[cdf["dataset"] == dataset].copy()
        bars = ax.bar(
            sub["component"],
            sub["value"],
            color=[color_map_component[x] for x in sub["component"]],
            alpha=0.82
        )

        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}", ha="center", va=va, fontsize=9)

        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"{dataset}: component contributions")
        ax.set_ylabel("Reduction in mean final realized regret")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "ablation_component_contribution_bar.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_global_heatmaps(df_final: pd.DataFrame, df_neigh: pd.DataFrame):
    # mean final regret heatmap
    mat = np.zeros((len(DATASET_ORDER), len(ABLATION_ALGORITHMS)), dtype=float)
    for i, ds in enumerate(DATASET_ORDER):
        for j, algo in enumerate(ABLATION_ALGORITHMS):
            val = df_final[(df_final["dataset"] == ds) & (df_final["algorithm"] == algo)]["final_realized_regret"].mean()
            mat[i, j] = float(val)
    plot_matrix_heatmap(
        mat=mat,
        row_labels=DATASET_ORDER,
        col_labels=[LABEL_MAP[a] for a in ABLATION_ALGORITHMS],
        title="Ablation: mean final realized regret",
        outpath=os.path.join(OUTDIR, "ablation_heatmap_mean_final_regret.png"),
        fmt=".2f",
        cmap="magma"
    )

    # neighborhood concentration heatmap
    mat2 = np.zeros((len(DATASET_ORDER), len(ABLATION_ALGORITHMS)), dtype=float)
    for i, ds in enumerate(DATASET_ORDER):
        for j, algo in enumerate(ABLATION_ALGORITHMS):
            val = df_neigh[(df_neigh["dataset"] == ds) & (df_neigh["algorithm"] == algo)]["neighbor_fraction"].mean()
            mat2[i, j] = float(val)
    plot_matrix_heatmap(
        mat=mat2,
        row_labels=DATASET_ORDER,
        col_labels=[LABEL_MAP[a] for a in ABLATION_ALGORITHMS],
        title="Ablation: mean optimal-neighborhood concentration",
        outpath=os.path.join(OUTDIR, "ablation_heatmap_optimal_neighborhood_concentration.png"),
        fmt=".3f",
        cmap="viridis"
    )

    # win-rate of full method against each ablation
    mat3 = np.zeros((len(DATASET_ORDER), len([a for a in ABLATION_ALGORITHMS if a != FULL_ALGO])), dtype=float)
    col_algos = [a for a in ABLATION_ALGORITHMS if a != FULL_ALGO]
    for i, ds in enumerate(DATASET_ORDER):
        piv = df_final[df_final["dataset"] == ds].pivot(index="seed", columns="algorithm", values="final_realized_regret")
        for j, algo in enumerate(col_algos):
            mat3[i, j] = float(np.mean(piv[FULL_ALGO].values < piv[algo].values))

    plot_matrix_heatmap(
        mat=mat3,
        row_labels=DATASET_ORDER,
        col_labels=[f"{LABEL_MAP[FULL_ALGO]} beats\n{LABEL_MAP[a]}" for a in col_algos],
        title="Ablation: win-rate heatmap",
        outpath=os.path.join(OUTDIR, "ablation_heatmap_win_rate_full_vs_ablations.png"),
        fmt=".3f",
        cmap="plasma"
    )


# ============================================================
# Summaries
# ============================================================

def build_summary_tables(
    df_long: pd.DataFrame,
    df_final: pd.DataFrame,
    df_arms: pd.DataFrame,
    env_map: Dict[str, StationaryBatchedEnv]
):
    rows = []
    detailed_rows = []

    for (dataset, algo), sub in df_final.groupby(["dataset", "algorithm"]):
        vals = sub["final_realized_regret"].values.astype(float)
        pvals = sub["final_pseudo_regret"].values.astype(float)

        mean_r = float(np.mean(vals))
        std_r = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        median_r = float(np.median(vals))
        ci_low_r, ci_high_r = t_ci_mean(vals, alpha=0.05)
        bci_low_r, bci_high_r = bootstrap_ci_mean(vals, B=BOOTSTRAP_B, alpha=0.05, seed=BOOTSTRAP_RANDOM_SEED)

        mean_p = float(np.mean(pvals))
        std_p = float(np.std(pvals, ddof=1)) if len(pvals) > 1 else 0.0
        median_p = float(np.median(pvals))
        ci_low_p, ci_high_p = t_ci_mean(pvals, alpha=0.05)
        bci_low_p, bci_high_p = bootstrap_ci_mean(pvals, B=BOOTSTRAP_B, alpha=0.05, seed=BOOTSTRAP_RANDOM_SEED)

        rows.append({
            "dataset": dataset,
            "algorithm": algo,
            "n_seeds": len(vals),
            "mean_final_realized_regret": mean_r,
            "std_final_realized_regret": std_r,
            "median_final_realized_regret": median_r,
            "mean_final_pseudo_regret": mean_p,
            "std_final_pseudo_regret": std_p
        })

        detailed_rows.append({
            "dataset": dataset,
            "algorithm": algo,
            "n_seeds": len(vals),

            "mean_final_realized_regret": mean_r,
            "std_final_realized_regret": std_r,
            "median_final_realized_regret": median_r,
            "t_ci95_low_final_realized_regret": ci_low_r,
            "t_ci95_high_final_realized_regret": ci_high_r,
            "bootstrap_ci95_low_final_realized_regret": bci_low_r,
            "bootstrap_ci95_high_final_realized_regret": bci_high_r,

            "mean_final_pseudo_regret": mean_p,
            "std_final_pseudo_regret": std_p,
            "median_final_pseudo_regret": median_p,
            "t_ci95_low_final_pseudo_regret": ci_low_p,
            "t_ci95_high_final_pseudo_regret": ci_high_p,
            "bootstrap_ci95_low_final_pseudo_regret": bci_low_p,
            "bootstrap_ci95_high_final_pseudo_regret": bci_high_p,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUTDIR, "ablation_summary_final_regret.csv"), index=False)

    summary_detailed_df = pd.DataFrame(detailed_rows)
    summary_detailed_df.to_csv(os.path.join(OUTDIR, "ablation_summary_final_regret_detailed.csv"), index=False)

    # pairwise significance against full model
    sig_rows = []
    compare_algos = [a for a in ABLATION_ALGORITHMS if a != FULL_ALGO]

    for dataset, sub in df_final.groupby("dataset"):
        piv_r = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")
        piv_p = sub.pivot(index="seed", columns="algorithm", values="final_pseudo_regret")

        for other in compare_algos:
            if FULL_ALGO in piv_r.columns and other in piv_r.columns:
                a = piv_r[other].values.astype(float)
                b = piv_r[FULL_ALGO].values.astype(float)
                diff = a - b

                t_stat_r, t_p_r = safe_paired_ttest(a, b)
                w_stat_r, w_p_r = safe_wilcoxon(a, b)
                ci_low, ci_high = t_ci_mean(diff)
                bci_low, bci_high = bootstrap_ci_mean(diff, B=BOOTSTRAP_B, alpha=0.05, seed=BOOTSTRAP_RANDOM_SEED)

                sig_rows.append({
                    "dataset": dataset,
                    "metric": "final_realized_regret",
                    "algo_A": other,
                    "algo_B": FULL_ALGO,
                    "n_pairs": len(diff),
                    "mean_A": float(np.mean(a)),
                    "mean_B": float(np.mean(b)),
                    "mean_diff_A_minus_B": float(np.mean(diff)),
                    "median_diff_A_minus_B": float(np.median(diff)),
                    "t_ci95_low_diff_A_minus_B": ci_low,
                    "t_ci95_high_diff_A_minus_B": ci_high,
                    "bootstrap_ci95_low_diff_A_minus_B": bci_low,
                    "bootstrap_ci95_high_diff_A_minus_B": bci_high,
                    "paired_t_stat": t_stat_r,
                    "paired_t_pvalue": t_p_r,
                    "wilcoxon_stat": w_stat_r,
                    "wilcoxon_pvalue": w_p_r,
                    "B_win_rate_over_A": float(np.mean(b < a))
                })

            if FULL_ALGO in piv_p.columns and other in piv_p.columns:
                a = piv_p[other].values.astype(float)
                b = piv_p[FULL_ALGO].values.astype(float)
                diff = a - b

                t_stat_p, t_p_p = safe_paired_ttest(a, b)
                w_stat_p, w_p_p = safe_wilcoxon(a, b)
                ci_low, ci_high = t_ci_mean(diff)
                bci_low, bci_high = bootstrap_ci_mean(diff, B=BOOTSTRAP_B, alpha=0.05, seed=BOOTSTRAP_RANDOM_SEED)

                sig_rows.append({
                    "dataset": dataset,
                    "metric": "final_pseudo_regret",
                    "algo_A": other,
                    "algo_B": FULL_ALGO,
                    "n_pairs": len(diff),
                    "mean_A": float(np.mean(a)),
                    "mean_B": float(np.mean(b)),
                    "mean_diff_A_minus_B": float(np.mean(diff)),
                    "median_diff_A_minus_B": float(np.median(diff)),
                    "t_ci95_low_diff_A_minus_B": ci_low,
                    "t_ci95_high_diff_A_minus_B": ci_high,
                    "bootstrap_ci95_low_diff_A_minus_B": bci_low,
                    "bootstrap_ci95_high_diff_A_minus_B": bci_high,
                    "paired_t_stat": t_stat_p,
                    "paired_t_pvalue": t_p_p,
                    "wilcoxon_stat": w_stat_p,
                    "wilcoxon_pvalue": w_p_p,
                    "B_win_rate_over_A": float(np.mean(b < a))
                })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(os.path.join(OUTDIR, "ablation_pairwise_significance.csv"), index=False)

    # neighborhood concentration
    neigh_rows = []
    for dataset in DATASET_ORDER:
        env = env_map[dataset]
        neigh = set([env.opt_arm])
        if env.opt_arm - 1 >= 0:
            neigh.add(env.opt_arm - 1)
        if env.opt_arm + 1 < env.K:
            neigh.add(env.opt_arm + 1)

        sub = df_long[df_long["dataset"] == dataset]
        for (seed, algo), g in sub.groupby(["seed", "algorithm"]):
            arms = g.sort_values("batch")["chosen_arm"].values.astype(int)
            frac = float(np.mean(np.isin(arms, list(neigh))))
            opt_frac = float(np.mean(arms == env.opt_arm))
            neigh_rows.append({
                "dataset": dataset,
                "seed": int(seed),
                "algorithm": algo,
                "opt_arm": int(env.opt_arm),
                "neighbor_fraction": frac,
                "opt_arm_fraction": opt_frac
            })
    df_neigh = pd.DataFrame(neigh_rows)
    df_neigh.to_csv(os.path.join(OUTDIR, "ablation_optimal_neighborhood_concentration.csv"), index=False)

    arm_summary = (
        df_arms.groupby(["dataset", "algorithm", "arm"], as_index=False)["n_batches"]
        .sum()
        .sort_values(["dataset", "algorithm", "arm"])
    )
    arm_summary.to_csv(os.path.join(OUTDIR, "ablation_arm_pull_counts_aggregated.csv"), index=False)

    print(f"[INFO] Saved ablation summaries to {OUTDIR}")
    return df_neigh


# ============================================================
# Main
# ============================================================

def main():
    prepared_map = load_all_prepared_datasets()

    env_map: Dict[str, StationaryBatchedEnv] = {}
    env_rows = []

    for name in DATASET_ORDER:
        env = build_stationary_batched_env(
            prepared=prepared_map[name],
            K=K,
            tau=TAU,
            q_defense=Q_DEFENSE,
            n_base_attack_samples=N_BASE_ATTACK_SAMPLES,
            seed=123
        )
        env_map[name] = env

        uni_ok, peak = is_unimodal(env.mean_rewards)
        env_rows.append({
            "dataset": name,
            "K": env.K,
            "tau": env.tau,
            "q_defense": env.q_defense,
            "direction_name": env.direction_name,
            "base_step": env.base_step,
            "threshold": env.threshold,
            "impact_monotone_nondecreasing": monotone_nondecreasing(env.impacts),
            "survival_monotone_nonincreasing": monotone_nonincreasing(env.success_probs),
            "utility_unimodal": uni_ok,
            "peak_arm": int(peak),
            "opt_arm": int(env.opt_arm),
            "opt_mean": float(env.opt_mean),
            "arms": json.dumps([float(x) for x in env.arms]),
            "impacts": json.dumps([float(x) for x in env.impacts]),
            "success_probs": json.dumps([float(x) for x in env.success_probs]),
            "mean_rewards": json.dumps([float(x) for x in env.mean_rewards]),
        })

    pd.DataFrame(env_rows).to_csv(os.path.join(OUTDIR, "ablation_environment_summary.csv"), index=False)

    long_rows = []
    final_rows = []
    arm_rows = []

    for dataset_name in DATASET_ORDER:
        env = env_map[dataset_name]
        print(f"\n[INFO] Running ablation dataset={dataset_name}")

        for seed in SEEDS:
            print(f"  [INFO] seed={seed}")
            for algo in ABLATION_ALGORITHMS:
                res = run_single_seed_stationary_batched(
                    env=env,
                    algo_name=algo,
                    seed=seed,
                    T_batches=T_BATCHES
                )

                final_rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "algorithm": algo,
                    "final_realized_regret": res.final_realized_regret,
                    "final_pseudo_regret": res.final_pseudo_regret
                })

                for b in range(T_BATCHES):
                    long_rows.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "algorithm": algo,
                        "batch": b + 1,
                        "cum_realized_regret": float(res.realized_regret_curve[b]),
                        "cum_pseudo_regret": float(res.pseudo_regret_curve[b]),
                        "batch_reward": float(res.reward_per_batch[b]),
                        "chosen_arm": int(res.chosen_arms[b]),
                    })

                uniq, cnt = np.unique(res.chosen_arms, return_counts=True)
                for a, c in zip(uniq, cnt):
                    arm_rows.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "algorithm": algo,
                        "arm": int(a),
                        "n_batches": int(c),
                        "n_pulls": int(c * TAU)
                    })

    df_long = pd.DataFrame(long_rows)
    df_final = pd.DataFrame(final_rows)
    df_arms = pd.DataFrame(arm_rows)

    df_long.to_csv(os.path.join(OUTDIR, "ablation_long.csv"), index=False)
    df_final.to_csv(os.path.join(OUTDIR, "ablation_final.csv"), index=False)
    df_arms.to_csv(os.path.join(OUTDIR, "ablation_arm_pull_counts.csv"), index=False)

    df_neigh = build_summary_tables(df_long, df_final, df_arms, env_map)

    # environment plots
    for dataset_name in DATASET_ORDER:
        plot_environment_profile(env_map[dataset_name])

    # per-dataset plots
    for dataset_name in DATASET_ORDER:
        plot_regret_curves_and_box(dataset_name, df_long)
        plot_regret_violin(dataset_name, df_final)
        plot_paired_scatter_full_vs_others(dataset_name, df_final)
        plot_arm_trajectory(dataset_name, df_long, env_map[dataset_name])
        plot_arm_pull_heatmaps(dataset_name, df_arms, env_map[dataset_name])

    # global plots
    plot_relative_improvement(df_final)
    plot_component_contribution(df_final)
    plot_global_heatmaps(df_final, df_neigh)

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
        "algorithms": ABLATION_ALGORITHMS,
        "full_algorithm": FULL_ALGO,
        "baseline_algorithm": BASELINE_ALGO,
        "gamma_lead": GAMMA_LEAD,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_random_seed": BOOTSTRAP_RANDOM_SEED
    }
    with open(os.path.join(OUTDIR, "ablation_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n[INFO] Ablation done.")
    print(f"[INFO] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()