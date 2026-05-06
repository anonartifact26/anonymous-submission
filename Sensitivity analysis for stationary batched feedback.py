# -*- coding: utf-8 -*-
"""
All-in-one sensitivity analysis for stationary arm-resolved homogeneous batching.

Includes:
1) K-sensitivity:
      K in {7, 11, 15, 21}
      fixed tau=20, q=0.15

2) Batch-size sensitivity:
      tau in {5, 10, 20, 50}
      fixed K=11, q=0.15

3) Defense-level sensitivity:
      q in {0.05, 0.10, 0.15, 0.20, 0.30}
      fixed K=11, tau=20

Datasets:
    - Adult
    - BankMarketing

Algorithms:
    - UCB
    - KLUCB_OSUB

Seeds:
    - 30 seeds

Outputs:
    - long CSVs
    - final CSVs
    - environment summaries
    - line plots
    - heatmaps
    - relative improvement plots
    - paired comparison tables

Important:
- This script runs actual simulations.
- It does not fabricate final results.
"""

import os
import re
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# Default config
# ============================================================

SEEDS = list(range(30))
DATASET_ORDER = ["Adult", "BankMarketing"]
ALGORITHMS = ["UCB", "KLUCB_OSUB"]

BASE_K = 11
BASE_TAU = 20
BASE_Q = 0.15
T_BATCHES = 140
N_BASE_ATTACK_SAMPLES = 3000
GAMMA_LEAD = 2

K_LIST = [7, 11, 15, 21]
TAU_LIST = [5, 10, 20, 50]
Q_LIST = [0.05, 0.10, 0.15, 0.20, 0.30]

COLOR_MAP = {
    "UCB": "#1f77b4",
    "KLUCB_OSUB": "#d62728",
}
LABEL_MAP = {
    "UCB": "UCB",
    "KLUCB_OSUB": "KL-UCB",
}

np.set_printoptions(precision=4, suppress=True)


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
        if "." not in os.path.basename(low):
            s += 2
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


def plot_matrix_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    outpath: str,
    fmt: str = ".2f",
    cmap: str = "viridis"
):
    fig, ax = plt.subplots(figsize=(2.0 + 1.2 * len(col_labels), 2.0 + 0.85 * len(row_labels)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="white", fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


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
# Environment
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
        obj += 20.0 if monotone_nonincreasing(ps) else -200.0
        obj += 10.0 if monotone_nondecreasing(impacts) else -100.0
        obj += 18.0 if uni_ok else -20.0
        if uni_ok and 0 < peak < K - 1:
            obj += 4.0
        obj += 6.0 * float(means.max() - means.min())
        obj += 2.0 * float(ps[0] - ps[-1])

        if obj > best_obj:
            best_obj = obj
            best_pack = (dname, ps, means)

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

    print(
        f"[INFO] Built env: dataset={prepared.name}, "
        f"K={K}, tau={tau}, q={q_defense}, opt_arm={opt_arm}, opt_mean={opt_mean:.4f}"
    )
    return env


# ============================================================
# Algorithms
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


class OSUBKLTrimmedBatch:
    def __init__(self, impacts: np.ndarray, gamma_lead: int = 2):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.gamma_lead = gamma_lead
        self.N = np.zeros(self.K, dtype=int)
        self.S = np.zeros(self.K, dtype=float)
        self.L = np.zeros(self.K, dtype=int)
        self.batch_t = 0

    def _kl_ucb_survival(self, p_hat: float, N_k: int, t: int) -> float:
        if N_k <= 0:
            return 1.0

        rhs = np.log(max(t, 2))
        if t >= 3:
            rhs += 3.0 * np.log(np.log(t))
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

        t = self.batch_t
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
            q = self._kl_ucb_survival(float(p_hat[k]), int(self.N[k]), int(t))
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
    if algo_name == "KLUCB_OSUB":
        return OSUBKLTrimmedBatch(impacts, gamma_lead=GAMMA_LEAD)
    raise ValueError(f"Unknown algorithm: {algo_name}")


# ============================================================
# Runner
# ============================================================

@dataclass
class RunResult:
    final_realized_regret: float
    final_pseudo_regret: float
    realized_regret_curve: np.ndarray


def run_single_seed_stationary_batched(
    env: StationaryBatchedEnv,
    algo_name: str,
    seed: int,
    T_batches: int
) -> RunResult:
    rng = np.random.RandomState(seed)
    agent = make_agent(algo_name, env.impacts)

    realized_regrets = []
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
        realized_regrets.append(cum_realized_regret)

    return RunResult(
        final_realized_regret=float(cum_realized_regret),
        final_pseudo_regret=float(cum_pseudo_regret),
        realized_regret_curve=np.asarray(realized_regrets, dtype=float)
    )


# ============================================================
# Load datasets
# ============================================================

def load_all_prepared_datasets(data_root: str) -> Dict[str, PreparedDataset]:
    print("[INFO] Files under data_root:")
    for x in safe_listdir(data_root):
        print("   ", x)

    adult_df = load_adult(data_root)
    bank_df = load_bank_marketing(data_root)

    adult = prepare_dataset(adult_df, "Adult", "income")
    bank = prepare_dataset(bank_df, "BankMarketing", "target")
    return {"Adult": adult, "BankMarketing": bank}


# ============================================================
# Generic sensitivity runner
# ============================================================

def run_sensitivity_block(
    block_name: str,
    prepared_map: Dict[str, PreparedDataset],
    varying_name: str,
    varying_values: List,
    fixed_K: int,
    fixed_tau: int,
    fixed_q: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    env_rows = []
    final_rows = []
    curve_rows = []

    for dataset_name in DATASET_ORDER:
        prepared = prepared_map[dataset_name]

        for val in varying_values:
            K = fixed_K
            tau = fixed_tau
            q = fixed_q

            if varying_name == "K":
                K = int(val)
            elif varying_name == "tau":
                tau = int(val)
            elif varying_name == "q":
                q = float(val)
            else:
                raise ValueError(f"Unknown varying_name={varying_name}")

            print(f"\n[INFO] [{block_name}] dataset={dataset_name}, {varying_name}={val}")
            env = build_stationary_batched_env(
                prepared=prepared,
                K=K,
                tau=tau,
                q_defense=q,
                n_base_attack_samples=N_BASE_ATTACK_SAMPLES,
                seed=123
            )

            uni_ok, peak = is_unimodal(env.mean_rewards)
            env_rows.append({
                "block": block_name,
                "dataset": dataset_name,
                "varying_name": varying_name,
                "varying_value": val,
                "K": K,
                "tau": tau,
                "q_defense": q,
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

            for seed in SEEDS:
                for algo in ALGORITHMS:
                    res = run_single_seed_stationary_batched(
                        env=env,
                        algo_name=algo,
                        seed=seed,
                        T_batches=T_BATCHES
                    )

                    final_rows.append({
                        "block": block_name,
                        "dataset": dataset_name,
                        "varying_name": varying_name,
                        "varying_value": val,
                        "K": K,
                        "tau": tau,
                        "q_defense": q,
                        "seed": seed,
                        "algorithm": algo,
                        "final_realized_regret": res.final_realized_regret,
                        "final_pseudo_regret": res.final_pseudo_regret
                    })

                    for b in range(T_BATCHES):
                        curve_rows.append({
                            "block": block_name,
                            "dataset": dataset_name,
                            "varying_name": varying_name,
                            "varying_value": val,
                            "K": K,
                            "tau": tau,
                            "q_defense": q,
                            "seed": seed,
                            "algorithm": algo,
                            "batch": b + 1,
                            "cum_realized_regret": float(res.realized_regret_curve[b]),
                        })

    return pd.DataFrame(env_rows), pd.DataFrame(final_rows), pd.DataFrame(curve_rows)


# ============================================================
# Plotting
# ============================================================

def plot_sensitivity_lines(df_final: pd.DataFrame, outdir: str, block_name: str, varying_name: str, varying_values: List):
    for dataset_name in DATASET_ORDER:
        sub = df_final[(df_final["block"] == block_name) & (df_final["dataset"] == dataset_name)].copy()

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for algo in ALGORITHMS:
            means = []
            cis = []
            for val in varying_values:
                vals = sub[(sub["algorithm"] == algo) & (sub["varying_value"] == val)]["final_realized_regret"].values
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci = 2.045 * s / np.sqrt(max(len(vals), 1))
                means.append(m)
                cis.append(ci)

            x = np.arange(len(varying_values))
            ax.plot(x, means, marker="o", lw=2.0, color=COLOR_MAP[algo], label=LABEL_MAP[algo])
            ax.fill_between(x, np.array(means) - np.array(cis), np.array(means) + np.array(cis),
                            color=COLOR_MAP[algo], alpha=0.20)

        ax.set_xticks(np.arange(len(varying_values)))
        ax.set_xticklabels([str(v) for v in varying_values])
        ax.set_title(f"{dataset_name}: {block_name} sensitivity")
        ax.set_xlabel(varying_name)
        ax.set_ylabel("Final realized regret")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        outpath = os.path.join(outdir, f"{block_name}_{dataset_name}_line.png")
        plt.savefig(outpath, dpi=240)
        plt.close()
        print(f"[INFO] Saved figure: {outpath}")


def plot_win_rate_lines(df_final: pd.DataFrame, outdir: str, block_name: str, varying_name: str, varying_values: List):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for dataset_name in DATASET_ORDER:
        ys = []
        for val in varying_values:
            sub = df_final[
                (df_final["block"] == block_name) &
                (df_final["dataset"] == dataset_name) &
                (df_final["varying_value"] == val)
            ]
            piv = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")
            win_rate = float(np.mean(piv["KLUCB_OSUB"].values < piv["UCB"].values))
            ys.append(win_rate)

        x = np.arange(len(varying_values))
        ax.plot(x, ys, marker="o", lw=2.0, label=dataset_name)

    ax.set_xticks(np.arange(len(varying_values)))
    ax.set_xticklabels([str(v) for v in varying_values])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{block_name}: KL-UCB win rate over UCB")
    ax.set_xlabel(varying_name)
    ax.set_ylabel("Win rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{block_name}_win_rate_line.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_relative_improvement_lines(df_final: pd.DataFrame, outdir: str, block_name: str, varying_name: str, varying_values: List):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for dataset_name in DATASET_ORDER:
        ys = []
        for val in varying_values:
            sub = df_final[
                (df_final["block"] == block_name) &
                (df_final["dataset"] == dataset_name) &
                (df_final["varying_value"] == val)
            ]
            mu_ucb = sub[sub["algorithm"] == "UCB"]["final_realized_regret"].mean()
            mu_kl = sub[sub["algorithm"] == "KLUCB_OSUB"]["final_realized_regret"].mean()
            improve = 100.0 * (mu_ucb - mu_kl) / max(abs(mu_ucb), 1e-12)
            ys.append(improve)

        x = np.arange(len(varying_values))
        ax.plot(x, ys, marker="o", lw=2.0, label=dataset_name)

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(len(varying_values)))
    ax.set_xticklabels([str(v) for v in varying_values])
    ax.set_title(f"{block_name}: relative improvement of KL-UCB over UCB")
    ax.set_xlabel(varying_name)
    ax.set_ylabel("Mean final regret reduction (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, f"{block_name}_relative_improvement.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[INFO] Saved figure: {outpath}")


def plot_block_heatmaps(df_final: pd.DataFrame, outdir: str, block_name: str, varying_name: str, varying_values: List):
    row_labels = DATASET_ORDER
    col_labels = [str(v) for v in varying_values]

    mat = np.zeros((len(DATASET_ORDER), len(varying_values)), dtype=float)
    for i, ds in enumerate(DATASET_ORDER):
        for j, v in enumerate(varying_values):
            sub = df_final[
                (df_final["block"] == block_name) &
                (df_final["dataset"] == ds) &
                (df_final["varying_value"] == v) &
                (df_final["algorithm"] == "KLUCB_OSUB")
            ]
            mat[i, j] = float(sub["final_realized_regret"].mean())

    plot_matrix_heatmap(
        mat=mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{block_name}: mean final regret of KL-UCB",
        outpath=os.path.join(outdir, f"{block_name}_heatmap_klucb_mean_regret.png"),
        fmt=".2f",
        cmap="magma"
    )

    mat2 = np.zeros((len(DATASET_ORDER), len(varying_values)), dtype=float)
    for i, ds in enumerate(DATASET_ORDER):
        for j, v in enumerate(varying_values):
            sub = df_final[
                (df_final["block"] == block_name) &
                (df_final["dataset"] == ds) &
                (df_final["varying_value"] == v)
            ]
            piv = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")
            mat2[i, j] = float(np.mean(piv["KLUCB_OSUB"].values < piv["UCB"].values))

    plot_matrix_heatmap(
        mat=mat2,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"{block_name}: KL-UCB win rate heatmap",
        outpath=os.path.join(outdir, f"{block_name}_heatmap_win_rate.png"),
        fmt=".3f",
        cmap="plasma"
    )


def plot_combined_heatmap(df_final: pd.DataFrame, outdir: str):
    summary = (
        df_final.groupby(["block", "dataset", "algorithm"], as_index=False)["final_realized_regret"]
        .mean()
    )

    blocks = ["K_sensitivity", "Tau_sensitivity", "Q_sensitivity"]
    rows = []
    mat = np.zeros((len(blocks) * len(DATASET_ORDER), len(ALGORITHMS)), dtype=float)

    r = 0
    for blk in blocks:
        for ds in DATASET_ORDER:
            rows.append(f"{blk}-{ds}")
            for j, algo in enumerate(ALGORITHMS):
                val = summary[
                    (summary["block"] == blk) &
                    (summary["dataset"] == ds) &
                    (summary["algorithm"] == algo)
                ]["final_realized_regret"].mean()
                mat[r, j] = float(val)
            r += 1

    plot_matrix_heatmap(
        mat=mat,
        row_labels=rows,
        col_labels=[LABEL_MAP[a] for a in ALGORITHMS],
        title="Combined mean final realized regret",
        outpath=os.path.join(outdir, "combined_heatmap_mean_final_regret.png"),
        fmt=".2f",
        cmap="viridis"
    )


# ============================================================
# Summaries
# ============================================================

def save_block_summaries(df_env: pd.DataFrame, df_final: pd.DataFrame, df_curve: pd.DataFrame, outdir: str, block_name: str):
    df_env.to_csv(os.path.join(outdir, f"{block_name}_environment_summary.csv"), index=False)
    df_final.to_csv(os.path.join(outdir, f"{block_name}_final.csv"), index=False)
    df_curve.to_csv(os.path.join(outdir, f"{block_name}_curves_long.csv"), index=False)

    summary_rows = []
    for (dataset, algorithm, varying_value), sub in df_final.groupby(["dataset", "algorithm", "varying_value"]):
        vals = sub["final_realized_regret"].values
        pvals = sub["final_pseudo_regret"].values
        summary_rows.append({
            "block": block_name,
            "dataset": dataset,
            "algorithm": algorithm,
            "varying_value": varying_value,
            "n_seeds": len(vals),
            "mean_final_realized_regret": float(np.mean(vals)),
            "std_final_realized_regret": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median_final_realized_regret": float(np.median(vals)),
            "mean_final_pseudo_regret": float(np.mean(pvals)),
            "std_final_pseudo_regret": float(np.std(pvals, ddof=1)) if len(pvals) > 1 else 0.0
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(outdir, f"{block_name}_summary.csv"), index=False)

    pair_rows = []
    for (dataset, varying_value), sub in df_final.groupby(["dataset", "varying_value"]):
        piv = sub.pivot(index="seed", columns="algorithm", values="final_realized_regret")
        if set(ALGORITHMS).issubset(piv.columns):
            u = piv["UCB"].values
            k = piv["KLUCB_OSUB"].values
            diff = u - k
            pair_rows.append({
                "block": block_name,
                "dataset": dataset,
                "varying_value": varying_value,
                "mean_diff_UCB_minus_KL": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "KL_win_rate": float(np.mean(k < u)),
                "UCB_win_rate": float(np.mean(u < k)),
                "ties": int(np.sum(u == k)),
            })

    pd.DataFrame(pair_rows).to_csv(os.path.join(outdir, f"{block_name}_paired_comparison.csv"), index=False)


# ============================================================
# CLI
# ============================================================

def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root_default = script_dir.parent if script_dir.name == "scripts" else script_dir
    data_root_default = project_root_default / "data"
    outdir_default = project_root_default / "outputs" / "sensitivity_stationary_batched"

    parser = argparse.ArgumentParser(description="Sensitivity analysis for stationary batched feedback.")
    parser.add_argument("--project_root", type=str, default=str(project_root_default))
    parser.add_argument("--data_root", type=str, default=str(data_root_default))
    parser.add_argument("--outdir", type=str, default=str(outdir_default))
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] PROJECT_ROOT={project_root}")
    print(f"[INFO] DATA_ROOT={data_root}")
    print(f"[INFO] OUTDIR={outdir}")

    prepared_map = load_all_prepared_datasets(str(data_root))

    df_env_k, df_final_k, df_curve_k = run_sensitivity_block(
        block_name="K_sensitivity",
        prepared_map=prepared_map,
        varying_name="K",
        varying_values=K_LIST,
        fixed_K=BASE_K,
        fixed_tau=BASE_TAU,
        fixed_q=BASE_Q
    )
    save_block_summaries(df_env_k, df_final_k, df_curve_k, str(outdir), "K_sensitivity")
    plot_sensitivity_lines(df_final_k, str(outdir), "K_sensitivity", "K", K_LIST)
    plot_win_rate_lines(df_final_k, str(outdir), "K_sensitivity", "K", K_LIST)
    plot_relative_improvement_lines(df_final_k, str(outdir), "K_sensitivity", "K", K_LIST)
    plot_block_heatmaps(df_final_k, str(outdir), "K_sensitivity", "K", K_LIST)

    df_env_tau, df_final_tau, df_curve_tau = run_sensitivity_block(
        block_name="Tau_sensitivity",
        prepared_map=prepared_map,
        varying_name="tau",
        varying_values=TAU_LIST,
        fixed_K=BASE_K,
        fixed_tau=BASE_TAU,
        fixed_q=BASE_Q
    )
    save_block_summaries(df_env_tau, df_final_tau, df_curve_tau, str(outdir), "Tau_sensitivity")
    plot_sensitivity_lines(df_final_tau, str(outdir), "Tau_sensitivity", "tau", TAU_LIST)
    plot_win_rate_lines(df_final_tau, str(outdir), "Tau_sensitivity", "tau", TAU_LIST)
    plot_relative_improvement_lines(df_final_tau, str(outdir), "Tau_sensitivity", "tau", TAU_LIST)
    plot_block_heatmaps(df_final_tau, str(outdir), "Tau_sensitivity", "tau", TAU_LIST)

    df_env_q, df_final_q, df_curve_q = run_sensitivity_block(
        block_name="Q_sensitivity",
        prepared_map=prepared_map,
        varying_name="q",
        varying_values=Q_LIST,
        fixed_K=BASE_K,
        fixed_tau=BASE_TAU,
        fixed_q=BASE_Q
    )
    save_block_summaries(df_env_q, df_final_q, df_curve_q, str(outdir), "Q_sensitivity")
    plot_sensitivity_lines(df_final_q, str(outdir), "Q_sensitivity", "q", Q_LIST)
    plot_win_rate_lines(df_final_q, str(outdir), "Q_sensitivity", "q", Q_LIST)
    plot_relative_improvement_lines(df_final_q, str(outdir), "Q_sensitivity", "q", Q_LIST)
    plot_block_heatmaps(df_final_q, str(outdir), "Q_sensitivity", "q", Q_LIST)

    df_all_env = pd.concat([df_env_k, df_env_tau, df_env_q], axis=0, ignore_index=True)
    df_all_final = pd.concat([df_final_k, df_final_tau, df_final_q], axis=0, ignore_index=True)
    df_all_curve = pd.concat([df_curve_k, df_curve_tau, df_curve_q], axis=0, ignore_index=True)

    df_all_env.to_csv(outdir / "all_environment_summary.csv", index=False)
    df_all_final.to_csv(outdir / "all_final_results.csv", index=False)
    df_all_curve.to_csv(outdir / "all_curves_long.csv", index=False)

    combined_summary = (
        df_all_final.groupby(["block", "dataset", "algorithm"], as_index=False)
        .agg(
            mean_final_realized_regret=("final_realized_regret", "mean"),
            std_final_realized_regret=("final_realized_regret", "std"),
            mean_final_pseudo_regret=("final_pseudo_regret", "mean"),
            n=("final_realized_regret", "size")
        )
    )
    combined_summary.to_csv(outdir / "combined_summary.csv", index=False)

    plot_combined_heatmap(df_all_final, str(outdir))

    run_config = {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "outdir": str(outdir),
        "datasets": DATASET_ORDER,
        "algorithms": ALGORITHMS,
        "seeds": SEEDS,
        "baseline": {
            "K": BASE_K,
            "tau": BASE_TAU,
            "q": BASE_Q,
            "T_batches": T_BATCHES
        },
        "K_list": K_LIST,
        "tau_list": TAU_LIST,
        "q_list": Q_LIST,
        "n_base_attack_samples": N_BASE_ATTACK_SAMPLES,
        "gamma_lead": GAMMA_LEAD
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n[INFO] All sensitivity experiments completed.")
    print(f"[INFO] Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
