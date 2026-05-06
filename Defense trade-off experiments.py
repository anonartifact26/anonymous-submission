# -*- coding: utf-8 -*-
"""
Paper-final experiment code with enhanced publication-quality plotting.

This version:
1) keeps dataset path / loading style close to the user's existing code
2) upgrades plotting into more paper-ready NeurIPS / ICML / ICLR-like style
3) supports:
   - stationary baselines
   - nonstationary baselines
   - defense trade-off sweeps
   - sensitivity-style visualizations
"""

import os
import re
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

SEEDS = list(range(30))
DATASET_ORDER = ["Adult", "BankMarketing"]

STATIONARY_ALGORITHMS = ["UCB", "KLUCB_GLOBAL", "TS_GLOBAL", "OSUB_KL"]

NONSTATIONARY_ALGORITHMS = [
    "UCB", "KLUCB_GLOBAL", "TS_GLOBAL",
    "SW_UCB", "D_UCB", "CUSUM_UCB", "SW_KLUCB",
    "UCB_PH", "KLUCB_PH", "OSUB_KL", "OSUB_KL_PH"
]

K = 11
DEFAULT_TAU = 20
Q_DEFENSE = 0.15
STATIONARY_T_BATCHES = 140
NONSTATIONARY_T_BATCHES = 180
N_BASE_ATTACK_SAMPLES = 3000
GAMMA_LEAD = 2

SW_WINDOW_BATCHES = 30
DISCOUNT_GAMMA = 0.985
CUSUM_EPS = 0.02
CUSUM_H = 8.0
CUSUM_MIN_SAMPLES = 50
PH_DELTA = 0.01
PH_LAMBDA = 8.0
PH_MIN_SAMPLES = 50

BATCHING_GRID = [1, 5, 10, 20, 50, 100]
LDP_EPS_GRID = [0.2, 0.5, 1.0, 2.0, 4.0, float("inf")]
DITHER_SIGMA_GRID = [0.0, 0.01, 0.03, 0.05, 0.1]
MOTION_AMPLITUDE_GRID = [0.0, 0.02, 0.05, 0.08, 0.12]
MOTION_PERIOD_GRID = [500, 1000, 2000, 5000]
COARSENING_GRID = ["exact", "binary", "3bin", "5bin"]
EQUALIZATION_GRID = [0.0, 0.1, 0.2, 0.35, 0.5]

COLOR_MAP = {
    "UCB": "#4C78A8",
    "KLUCB_GLOBAL": "#F58518",
    "TS_GLOBAL": "#54A24B",
    "OSUB_KL": "#E45756",
    "SW_UCB": "#72B7B2",
    "D_UCB": "#B279A2",
    "CUSUM_UCB": "#9D755D",
    "SW_KLUCB": "#FF9DA6",
    "UCB_PH": "#79706E",
    "KLUCB_PH": "#B6992D",
    "OSUB_KL_PH": "#000000",
}

LABEL_MAP = {
    "UCB": "UCB",
    "KLUCB_GLOBAL": "KL-UCB",
    "TS_GLOBAL": "TS",
    "OSUB_KL": "OSUB-KL",
    "SW_UCB": "SW-UCB",
    "D_UCB": "D-UCB",
    "CUSUM_UCB": "CUSUM-UCB",
    "SW_KLUCB": "SW-KL-UCB",
    "UCB_PH": "UCB+PH",
    "KLUCB_PH": "KL-UCB+PH",
    "OSUB_KL_PH": "OSUB-KL+PH",
}


# ============================================================
# Paper plotting style
# ============================================================

def setup_paper_style():
    rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "axes.grid": True,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig_paper(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[INFO] Saved figure: {path}")


setup_paper_style()


# ============================================================
# Utilities
# ============================================================

def safe_listdir(path: str):
    try:
        return os.listdir(path)
    except Exception:
        return []


def find_file_fuzzy(root: str, include_keywords):
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

    def score(path):
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


def preview_file(path: str, n_lines: int = 5):
    print(f"[INFO] Preview: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(n_lines):
                line = f.readline()
                if not line:
                    break
                print(f"  L{i+1:02d}: {repr(line[:200])}")
    except Exception as e:
        print(f"[WARN] preview failed: {e}")


def read_head_text(path: str, n_chars: int = 5000):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(n_chars)
    except Exception:
        return ""


def looks_like_arff(path: str):
    text = read_head_text(path, 5000).lower()
    return ("@relation" in text) and ("@attribute" in text) and ("@data" in text)


def split_arff_row(line: str):
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


def parse_arff_simple(path: str):
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


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def mean_ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    m = float(np.mean(arr))
    if n <= 1:
        return m, m, m
    s = float(np.std(arr, ddof=1))
    half = t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, m - half, m + half


def kl_bernoulli(p, q):
    eps = 1e-12
    p = min(max(float(p), eps), 1 - eps)
    q = min(max(float(q), eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_ucb_index(p_hat, n, t_round):
    if n <= 0:
        return 1.0
    rhs = np.log(max(t_round, 2))
    if t_round >= 3:
        rhs += 3.0 * np.log(np.log(t_round))
    rhs /= n
    lo = min(max(p_hat, 0.0), 1.0)
    hi = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(p_hat, mid) <= rhs:
            lo = mid
        else:
            hi = mid
    return lo


def monotone_nonincreasing(x, tol=1e-12):
    return bool(np.all(np.diff(np.asarray(x, dtype=float)) <= tol))


def monotone_nondecreasing(x, tol=1e-12):
    return bool(np.all(np.diff(np.asarray(x, dtype=float)) >= -tol))


def is_unimodal(x, tol=1e-12):
    x = np.asarray(x, dtype=float)
    peak = int(np.argmax(x))
    left_ok = np.all(np.diff(x[:peak + 1]) >= -tol)
    right_ok = np.all(np.diff(x[peak:]) <= tol)
    return bool(left_ok and right_ok), peak


# ============================================================
# Dataset loading
# ============================================================

def normalize_adult_columns(df):
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


def parse_adult_raw(path):
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


def load_adult(root):
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


def normalize_bank_columns(df):
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


def load_bank_marketing(root):
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


@dataclass
class PreparedDataset:
    name: str
    raw_df: pd.DataFrame
    X_proc: np.ndarray
    y: np.ndarray


def prepare_dataset(df, name, target_col):
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


def load_all_prepared_datasets(data_root: str):
    print("[INFO] Files under data_root:")
    for x in safe_listdir(data_root):
        print("   ", x)

    adult_df = load_adult(data_root)
    bank_df = load_bank_marketing(data_root)

    adult = prepare_dataset(adult_df, "Adult", "income")
    bank = prepare_dataset(bank_df, "BankMarketing", "target")
    return {"Adult": adult, "BankMarketing": bank}


# ============================================================
# Environment
# ============================================================

@dataclass
class BanditEnv:
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
    benign_accept_rate: float


def anomaly_score(X, center, scale):
    Z = (X - center) / np.maximum(scale, 1e-8)
    return np.sqrt(np.sum(Z ** 2, axis=1))


def impact_scale(arms):
    return 0.12 + 0.88 * np.power(arms, 0.8)


def candidate_directions(X, rng):
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


def build_stationary_env(prepared, K, tau, q_defense, n_base_attack_samples, seed=123):
    rng = np.random.RandomState(seed)
    X = prepared.X_proc
    n, d = X.shape

    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0) + 1e-8
    ref_scores = anomaly_score(X, center, scale)
    threshold = float(np.quantile(ref_scores, 1.0 - q_defense))
    benign_accept_rate = float(np.mean(ref_scores <= threshold))

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

    env = BanditEnv(
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
        base_step=base_step,
        benign_accept_rate=benign_accept_rate,
    )
    print(f"[INFO] Built env: dataset={prepared.name}, K={K}, tau={tau}, q={q_defense}, opt_arm={opt_arm}, opt_mean={opt_mean:.4f}")
    return env


# ============================================================
# Defenses
# ============================================================

def privatize_count_rr(success_count, tau, eps, rng):
    if np.isinf(eps):
        return int(success_count)
    p_true = np.exp(eps) / (np.exp(eps) + 1.0)
    z = 0
    for i in range(tau):
        bit = 1 if i < success_count else 0
        z += bit if (rng.rand() < p_true) else (1 - bit)
    return int(z)


def coarsen_count(success_count, tau, mode):
    if mode == "exact":
        return int(success_count)
    if mode == "binary":
        return int(success_count > 0) * tau
    if mode == "3bin":
        frac = success_count / max(tau, 1)
        if frac == 0:
            return 0
        elif frac < 0.5:
            return tau // 2
        else:
            return tau
    if mode == "5bin":
        bins = np.linspace(0, tau, 5)
        idx = np.digitize(success_count, bins, right=True)
        idx = min(max(idx, 0), 4)
        mids = [0, tau * 0.25, tau * 0.5, tau * 0.75, tau]
        return int(round(mids[idx]))
    raise ValueError(mode)


def apply_equalization(probs, alpha):
    probs = np.asarray(probs, dtype=float)
    mean_p = float(np.mean(probs))
    newp = (1 - alpha) * probs + alpha * mean_p
    return np.clip(newp, 1e-6, 1 - 1e-6)


def apply_dither_to_probs(probs, sigma, rng):
    probs = np.asarray(probs, dtype=float)
    if sigma <= 0:
        return probs.copy()
    noisy = probs + rng.normal(0.0, sigma, size=len(probs))
    return np.clip(noisy, 1e-6, 1 - 1e-6)


def apply_motion_probs(base_probs, t_pull, amplitude, period):
    if amplitude <= 0:
        return np.asarray(base_probs, dtype=float)
    base_probs = np.asarray(base_probs, dtype=float)
    phase = 2.0 * np.pi * (t_pull / max(period, 1))
    pattern = np.linspace(-1.0, 1.0, len(base_probs))
    drift = amplitude * np.sin(phase) * pattern
    moved = base_probs + drift
    return np.clip(moved, 1e-6, 1 - 1e-6)


# ============================================================
# Algorithms
# ============================================================

class BaseBandit:
    def __init__(self, impacts, tau=1):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.tau = tau
        self.reset()

    def reset(self):
        self.N = np.zeros(self.K, dtype=float)
        self.S = np.zeros(self.K, dtype=float)
        self.R = np.zeros(self.K, dtype=float)
        self.batch_t = 0

    def empirical_success(self):
        return self.S / np.maximum(self.N, 1)

    def empirical_reward(self):
        return self.R / np.maximum(self.N, 1)

    def select_arm(self):
        raise NotImplementedError

    def update_batch(self, arm, successes, reward):
        self.N[arm] += self.tau
        self.S[arm] += successes
        self.R[arm] += reward
        self.batch_t += 1


class UCB(BaseBandit):
    def select_arm(self):
        for k in range(self.K):
            if self.N[k] == 0:
                return k
        t_round = max(self.batch_t, 2)
        mu_hat = self.empirical_reward()
        idx = mu_hat + np.sqrt(2.0 * np.log(t_round) / np.maximum(self.N, 1))
        return int(np.argmax(idx))


class KLUCBGlobal(BaseBandit):
    def select_arm(self):
        for k in range(self.K):
            if self.N[k] == 0:
                return k
        t_round = max(self.batch_t, 2)
        p_hat = self.empirical_success()
        vals = np.zeros(self.K)
        for k in range(self.K):
            q = kl_ucb_index(float(p_hat[k]), int(self.N[k]), int(t_round))
            vals[k] = self.impacts[k] * q
        return int(np.argmax(vals))


class ThompsonSamplingGlobal(BaseBandit):
    def __init__(self, impacts, tau=1, seed=0):
        self.rng = np.random.RandomState(seed)
        super().__init__(impacts, tau)

    def select_arm(self):
        vals = np.zeros(self.K)
        for k in range(self.K):
            alpha = 1.0 + self.S[k]
            beta = 1.0 + max(self.N[k] - self.S[k], 0.0)
            theta = self.rng.beta(alpha, beta)
            vals[k] = self.impacts[k] * theta
        return int(np.argmax(vals))


class OSUBKL(BaseBandit):
    def __init__(self, impacts, tau=1, gamma_lead=2):
        self.gamma_lead = gamma_lead
        super().__init__(impacts, tau)
        self.L = np.zeros(self.K, dtype=int)

    def select_arm(self):
        for k in range(self.K):
            if self.N[k] == 0:
                return k

        t_round = max(self.batch_t, 2)
        p_hat = self.empirical_success()
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
            q = kl_ucb_index(float(p_hat[k]), int(self.N[k]), int(t_round))
            idx = self.impacts[k] * q
            if idx > best_idx:
                best_idx = idx
                best_arm = k
        return int(best_arm)


class SWUCB:
    def __init__(self, impacts, tau=1, window_batches=30):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.tau = tau
        self.window_batches = window_batches
        self.name = "SW_UCB"
        self.reset()

    def reset(self):
        self.hist = [deque() for _ in range(self.K)]
        self.batch_t = 0

    def select_arm(self):
        self.batch_t += 1
        for k in range(self.K):
            if len(self.hist[k]) == 0:
                return k
        vals = np.zeros(self.K)
        total = max(self.batch_t, 2)
        for k in range(self.K):
            rewards = [x[1] for x in self.hist[k]]
            mu = np.mean(rewards) / self.tau
            n = len(self.hist[k]) * self.tau
            vals[k] = mu + np.sqrt(2.0 * np.log(total) / max(n, 1))
        return int(np.argmax(vals))

    def update_batch(self, arm, successes, reward):
        self.hist[arm].append((successes, reward))
        while len(self.hist[arm]) > self.window_batches:
            self.hist[arm].popleft()


class DiscountedUCB:
    def __init__(self, impacts, tau=1, gamma=0.985):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.tau = tau
        self.gamma = gamma
        self.name = "D_UCB"
        self.reset()

    def reset(self):
        self.n = np.zeros(self.K, dtype=float)
        self.s = np.zeros(self.K, dtype=float)
        self.batch_t = 0

    def select_arm(self):
        self.batch_t += 1
        for k in range(self.K):
            if self.n[k] <= 0:
                return k
        total = max(np.sum(self.n), 2.0)
        vals = np.zeros(self.K)
        for k in range(self.K):
            mu = self.s[k] / max(self.n[k], 1e-12)
            vals[k] = mu + 2.0 * np.sqrt(np.log(total) / max(self.n[k], 1e-12))
        return int(np.argmax(vals))

    def update_batch(self, arm, successes, reward):
        self.n *= self.gamma
        self.s *= self.gamma
        self.n[arm] += self.tau
        self.s[arm] += reward / self.tau


class SWKLUCB:
    def __init__(self, impacts, tau=1, window_batches=30):
        self.impacts = np.asarray(impacts, dtype=float)
        self.K = len(self.impacts)
        self.tau = tau
        self.window_batches = window_batches
        self.name = "SW_KLUCB"
        self.reset()

    def reset(self):
        self.hist = [deque() for _ in range(self.K)]
        self.batch_t = 0

    def select_arm(self):
        self.batch_t += 1
        for k in range(self.K):
            if len(self.hist[k]) == 0:
                return k
        vals = np.zeros(self.K)
        for k in range(self.K):
            succ = sum(x[0] for x in self.hist[k])
            n = len(self.hist[k]) * self.tau
            p_hat = succ / max(n, 1)
            vals[k] = self.impacts[k] * kl_ucb_index(p_hat, n, max(self.batch_t, 2))
        return int(np.argmax(vals))

    def update_batch(self, arm, successes, reward):
        self.hist[arm].append((successes, reward))
        while len(self.hist[arm]) > self.window_batches:
            self.hist[arm].popleft()


class CUSUMUCB(UCB):
    def __init__(self, impacts, tau=1, eps=0.02, h=8.0, min_samples=50):
        self.eps = eps
        self.h = h
        self.min_samples = min_samples
        super().__init__(impacts, tau)
        self.name = "CUSUM_UCB"
        self.g_plus = np.zeros(self.K)
        self.g_minus = np.zeros(self.K)

    def update_batch(self, arm, successes, reward):
        prev_mean = self.R[arm] / max(self.N[arm], 1) if self.N[arm] > 0 else 0.0
        super().update_batch(arm, successes, reward)
        if self.N[arm] < self.min_samples:
            return
        x = reward / self.tau
        s_plus = x - prev_mean - self.eps
        s_minus = prev_mean - x - self.eps
        self.g_plus[arm] = max(0.0, self.g_plus[arm] + s_plus)
        self.g_minus[arm] = max(0.0, self.g_minus[arm] + s_minus)
        if self.g_plus[arm] > self.h or self.g_minus[arm] > self.h:
            self.N[arm] = 0.0
            self.S[arm] = 0.0
            self.R[arm] = 0.0
            self.g_plus[arm] = 0.0
            self.g_minus[arm] = 0.0


class PageHinkleyWrapper:
    def __init__(self, base_algo, delta=0.01, lamb=8.0, min_samples=50):
        self.base = base_algo
        self.delta = delta
        self.lamb = lamb
        self.min_samples = min_samples
        self.name = f"{getattr(base_algo, 'name', base_algo.__class__.__name__)}_PH"
        self.reset()

    def reset(self):
        self.base.reset()
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    @property
    def tau(self):
        return self.base.tau

    def select_arm(self):
        return self.base.select_arm()

    def update_batch(self, arm, successes, reward):
        x = reward / max(self.base.tau, 1)
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        self.base.update_batch(arm, successes, reward)
        if self.n >= self.min_samples and (self.cum - self.min_cum) > self.lamb:
            self.reset()


def make_agent(algo_name, impacts, tau, seed):
    if algo_name == "UCB":
        a = UCB(impacts, tau)
        a.name = "UCB"
        return a
    elif algo_name == "KLUCB_GLOBAL":
        a = KLUCBGlobal(impacts, tau)
        a.name = "KLUCB_GLOBAL"
        return a
    elif algo_name == "TS_GLOBAL":
        a = ThompsonSamplingGlobal(impacts, tau, seed=seed)
        a.name = "TS_GLOBAL"
        return a
    elif algo_name == "OSUB_KL":
        a = OSUBKL(impacts, tau, gamma_lead=GAMMA_LEAD)
        a.name = "OSUB_KL"
        return a
    elif algo_name == "SW_UCB":
        return SWUCB(impacts, tau, window_batches=SW_WINDOW_BATCHES)
    elif algo_name == "D_UCB":
        return DiscountedUCB(impacts, tau, gamma=DISCOUNT_GAMMA)
    elif algo_name == "SW_KLUCB":
        return SWKLUCB(impacts, tau, window_batches=SW_WINDOW_BATCHES)
    elif algo_name == "CUSUM_UCB":
        return CUSUMUCB(impacts, tau, eps=CUSUM_EPS, h=CUSUM_H, min_samples=CUSUM_MIN_SAMPLES)
    elif algo_name == "UCB_PH":
        base = UCB(impacts, tau)
        base.name = "UCB"
        return PageHinkleyWrapper(base, delta=PH_DELTA, lamb=PH_LAMBDA, min_samples=PH_MIN_SAMPLES)
    elif algo_name == "KLUCB_PH":
        base = KLUCBGlobal(impacts, tau)
        base.name = "KLUCB"
        return PageHinkleyWrapper(base, delta=PH_DELTA, lamb=PH_LAMBDA, min_samples=PH_MIN_SAMPLES)
    elif algo_name == "OSUB_KL_PH":
        base = OSUBKL(impacts, tau, gamma_lead=GAMMA_LEAD)
        base.name = "OSUB_KL"
        return PageHinkleyWrapper(base, delta=PH_DELTA, lamb=PH_LAMBDA, min_samples=PH_MIN_SAMPLES)
    else:
        raise ValueError(algo_name)


# ============================================================
# Simulation
# ============================================================

def run_stationary_single(env, algo_name, seed, tau, T_batches,
                          ldp_eps=np.inf, dither_sigma=0.0, coarsening="exact", equalization=0.0):
    rng = np.random.RandomState(seed)

    probs = apply_equalization(env.success_probs, equalization)
    probs = apply_dither_to_probs(probs, dither_sigma, rng)
    impacts = env.impacts.copy()
    means = impacts * probs
    opt_arm = int(np.argmax(means))
    opt_mean = float(means[opt_arm])

    agent = make_agent(algo_name, impacts, tau, seed)

    curve_rows = []
    cum_regret = 0.0
    cum_reward = 0.0

    for b in range(T_batches):
        arm = agent.select_arm()

        true_success = int(rng.binomial(tau, probs[arm]))
        obs_success = privatize_count_rr(true_success, tau, ldp_eps, rng)
        obs_success = coarsen_count(obs_success, tau, coarsening)

        reward = impacts[arm] * obs_success
        agent.update_batch(arm, obs_success, reward)

        oracle_batch = tau * opt_mean
        chosen_mean = tau * means[arm]
        inst_regret = oracle_batch - chosen_mean

        cum_reward += chosen_mean
        cum_regret += inst_regret

        curve_rows.append({
            "batch": b + 1,
            "arm": arm,
            "cumulative_regret": cum_regret,
            "cumulative_reward": cum_reward,
        })

    final_row = {
        "final_regret": cum_regret,
        "final_expected_reward": cum_reward,
        "optimal_arm": opt_arm,
        "optimal_mean": opt_mean,
        "benign_accept_rate": env.benign_accept_rate,
        "leakage_proxy_std": float(np.std(probs)),
        "leakage_proxy_range": float(np.max(probs) - np.min(probs)),
    }
    return pd.DataFrame(curve_rows), final_row


def run_nonstationary_single(env, algo_name, seed, tau, T_batches, motion_amplitude=0.05, motion_period=1000):
    rng = np.random.RandomState(seed)
    impacts = env.impacts.copy()

    agent = make_agent(algo_name, impacts, tau, seed)

    curve_rows = []
    cum_regret = 0.0
    cum_reward = 0.0

    for b in range(T_batches):
        t_pull = (b + 1) * tau
        probs_t = apply_motion_probs(env.success_probs, t_pull, motion_amplitude, motion_period)
        means_t = impacts * probs_t
        opt_arm_t = int(np.argmax(means_t))
        opt_mean_t = float(means_t[opt_arm_t])

        arm = agent.select_arm()
        success = int(rng.binomial(tau, probs_t[arm]))
        reward = impacts[arm] * success
        agent.update_batch(arm, success, reward)

        oracle_batch = tau * opt_mean_t
        chosen_mean = tau * means_t[arm]
        inst_regret = oracle_batch - chosen_mean

        cum_reward += chosen_mean
        cum_regret += inst_regret

        curve_rows.append({
            "batch": b + 1,
            "arm": arm,
            "cumulative_regret": cum_regret,
            "cumulative_reward": cum_reward,
        })

    final_row = {
        "final_regret": cum_regret,
        "final_expected_reward": cum_reward,
        "benign_accept_rate": env.benign_accept_rate,
        "motion_amplitude": motion_amplitude,
        "motion_period": motion_period,
        "leakage_proxy_std": float(np.std(env.success_probs)),
        "leakage_proxy_range": float(np.max(env.success_probs) - np.min(env.success_probs)),
    }
    return pd.DataFrame(curve_rows), final_row


# ============================================================
# Statistical analysis
# ============================================================

def pairwise_significance(final_df, group_cols, metric="final_regret"):
    rows = []
    for key, sub in final_df.groupby(group_cols):
        algos = sorted(sub["algorithm"].unique())
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a1, a2 = algos[i], algos[j]
                s1 = sub[sub["algorithm"] == a1][["seed", metric]].sort_values("seed")
                s2 = sub[sub["algorithm"] == a2][["seed", metric]].sort_values("seed")
                merged = pd.merge(s1, s2, on="seed", suffixes=(f"_{a1}", f"_{a2}"))
                if len(merged) == 0:
                    continue

                x = merged[f"{metric}_{a1}"].values
                y = merged[f"{metric}_{a2}"].values

                try:
                    p_t = ttest_rel(x, y).pvalue
                except Exception:
                    p_t = np.nan
                try:
                    p_w = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                except Exception:
                    p_w = np.nan

                base = {}
                if isinstance(key, tuple):
                    for c, v in zip(group_cols, key):
                        base[c] = v
                else:
                    base[group_cols[0]] = key

                rows.append({
                    **base,
                    "algo_1": a1,
                    "algo_2": a2,
                    "mean_1": float(np.mean(x)),
                    "mean_2": float(np.mean(y)),
                    "mean_diff_1_minus_2": float(np.mean(x - y)),
                    "win_rate_algo1_lower_better": float(np.mean(x < y)),
                    "paired_t_pvalue": p_t,
                    "wilcoxon_pvalue": p_w,
                })
    return pd.DataFrame(rows)


# ============================================================
# Enhanced plotting
# ============================================================

def plot_regret_curves_paper(long_df, title, outpath):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    algo_order = [a for a in COLOR_MAP.keys() if a in set(long_df["algorithm"].unique())]
    for algo in algo_order:
        sub = long_df[long_df["algorithm"] == algo]
        grouped = sub.groupby("batch")["cumulative_regret"].apply(list).reset_index()
        xs, means, lows, highs = [], [], [], []
        for _, row in grouped.iterrows():
            m, lo, hi = mean_ci95(row["cumulative_regret"])
            xs.append(row["batch"])
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        ax.plot(xs, means, color=COLOR_MAP[algo], label=LABEL_MAP.get(algo, algo), linewidth=2.1)
        ax.fill_between(xs, lows, highs, color=COLOR_MAP[algo], alpha=0.18)

    ax.set_xlabel("Batch")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.legend(ncol=2, loc="best")
    savefig_paper(outpath)


def plot_boxplot_paper(final_df, title, outpath):
    algos = [a for a in COLOR_MAP.keys() if a in set(final_df["algorithm"].unique())]
    data = [final_df[final_df["algorithm"] == a]["final_regret"].values for a in algos]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bp = ax.boxplot(
        data,
        labels=[LABEL_MAP.get(a, a) for a in algos],
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    for patch, a in zip(bp["boxes"], algos):
        patch.set_facecolor(COLOR_MAP[a])
        patch.set_alpha(0.6)

    ax.set_ylabel("Final regret")
    ax.set_title(title)
    plt.xticks(rotation=20)
    savefig_paper(outpath)


def plot_tradeoff_paper(summary_df, xcol, title_prefix, out_prefix):
    metrics = [
        ("attacker_regret_mean", "Attacker final regret"),
        ("attacker_reward_mean", "Attacker expected reward"),
        ("benign_accept_rate_mean", "Benign acceptance rate"),
        ("leakage_proxy_std_mean", "Leakage proxy (std)"),
    ]

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(5.8, 3.9))
        for ds, color in zip(DATASET_ORDER, ["#1f77b4", "#d62728"]):
            sub = summary_df[summary_df["dataset"] == ds].sort_values(xcol)
            ax.plot(sub[xcol], sub[metric], marker="o", linewidth=2.1, label=ds, color=color)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}: {ylabel}")
        ax.legend(loc="best")
        savefig_paper(f"{out_prefix}_{metric}.png")


def plot_heatmap_paper(mat, row_labels, col_labels, title, outpath, fmt=".2f", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(1.8 + 1.1 * len(col_labels), 1.8 + 0.65 * len(row_labels)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    vmin, vmax = np.min(mat), np.max(mat)
    thresh = (vmin + vmax) / 2.0 if vmax > vmin else vmax

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > thresh else "black"
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color=color, fontsize=8.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    savefig_paper(outpath)


def plot_relative_improvement_paper(df_final, xcol, xvalues, title, outpath):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for ds, color in zip(DATASET_ORDER, ["#1f77b4", "#d62728"]):
        ys = []
        subds = df_final[df_final["dataset"] == ds]
        for x in xvalues:
            sub = subds[subds[xcol] == x]
            mu_ucb = sub[sub["algorithm"] == "UCB"]["final_regret"].mean()
            mu_best = sub[sub["algorithm"] == "OSUB_KL"]["final_regret"].mean()
            improvement = 100.0 * (mu_ucb - mu_best) / max(abs(mu_ucb), 1e-12)
            ys.append(improvement)
        ax.plot(range(len(xvalues)), ys, marker="o", linewidth=2.1, label=ds, color=color)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(range(len(xvalues)))
    ax.set_xticklabels([str(v) for v in xvalues])
    ax.set_xlabel(xcol)
    ax.set_ylabel("Regret reduction (%)")
    ax.set_title(title)
    ax.legend(loc="best")
    savefig_paper(outpath)


# ============================================================
# Main experiments
# ============================================================

def run_stationary_baselines(prepared_map, outdir: str, figdir: str):
    env_rows, final_rows, curve_rows = [], [], []

    for dataset_name in DATASET_ORDER:
        prepared = prepared_map[dataset_name]
        env = build_stationary_env(prepared, K, DEFAULT_TAU, Q_DEFENSE, N_BASE_ATTACK_SAMPLES, seed=123)

        env_rows.append({
            "dataset": dataset_name,
            "K": env.K,
            "tau": env.tau,
            "q": env.q_defense,
            "opt_arm": env.opt_arm,
            "opt_mean": env.opt_mean,
            "direction_name": env.direction_name,
            "base_step": env.base_step,
            "threshold": env.threshold,
            "benign_accept_rate": env.benign_accept_rate,
            "utility_unimodal": is_unimodal(env.mean_rewards)[0],
        })

        for algo in STATIONARY_ALGORITHMS:
            print(f"[Stationary] dataset={dataset_name}, algo={algo}")
            for seed in SEEDS:
                curve_df, final = run_stationary_single(
                    env, algo, seed, DEFAULT_TAU, STATIONARY_T_BATCHES
                )
                curve_df["dataset"] = dataset_name
                curve_df["algorithm"] = algo
                curve_df["seed"] = seed
                curve_rows.append(curve_df)

                final_rows.append({
                    "dataset": dataset_name,
                    "algorithm": algo,
                    "seed": seed,
                    **final
                })

    df_env = pd.DataFrame(env_rows)
    df_final = pd.DataFrame(final_rows)
    df_curve = pd.concat(curve_rows, axis=0, ignore_index=True)

    df_env.to_csv(os.path.join(outdir, "stationary_environment_summary.csv"), index=False)
    df_final.to_csv(os.path.join(outdir, "stationary_baselines_final.csv"), index=False)
    df_curve.to_csv(os.path.join(outdir, "stationary_baselines_curves_long.csv"), index=False)

    sig = pairwise_significance(df_final, group_cols=["dataset"], metric="final_regret")
    sig.to_csv(os.path.join(outdir, "pairwise_significance_stationary.csv"), index=False)

    for ds in DATASET_ORDER:
        plot_regret_curves_paper(
            df_curve[df_curve["dataset"] == ds],
            title=f"Stationary baselines on {ds}",
            outpath=os.path.join(figdir, f"stationary_regret_{ds}.png")
        )
        plot_boxplot_paper(
            df_final[df_final["dataset"] == ds],
            title=f"Stationary final regret on {ds}",
            outpath=os.path.join(figdir, f"stationary_boxplot_{ds}.png")
        )

    return df_env, df_final, df_curve


def run_nonstationary_baselines(prepared_map, outdir: str, figdir: str):
    final_rows, curve_rows = [], []

    for dataset_name in DATASET_ORDER:
        prepared = prepared_map[dataset_name]
        env = build_stationary_env(prepared, K, DEFAULT_TAU, Q_DEFENSE, N_BASE_ATTACK_SAMPLES, seed=123)

        for algo in NONSTATIONARY_ALGORITHMS:
            print(f"[Nonstationary] dataset={dataset_name}, algo={algo}")
            for seed in SEEDS:
                curve_df, final = run_nonstationary_single(
                    env, algo, seed, DEFAULT_TAU, NONSTATIONARY_T_BATCHES,
                    motion_amplitude=0.05, motion_period=1000
                )
                curve_df["dataset"] = dataset_name
                curve_df["algorithm"] = algo
                curve_df["seed"] = seed
                curve_rows.append(curve_df)

                final_rows.append({
                    "dataset": dataset_name,
                    "algorithm": algo,
                    "seed": seed,
                    **final
                })

    df_final = pd.DataFrame(final_rows)
    df_curve = pd.concat(curve_rows, axis=0, ignore_index=True)

    df_final.to_csv(os.path.join(outdir, "nonstationary_baselines_final.csv"), index=False)
    df_curve.to_csv(os.path.join(outdir, "nonstationary_baselines_curves_long.csv"), index=False)

    sig = pairwise_significance(df_final, group_cols=["dataset"], metric="final_regret")
    sig.to_csv(os.path.join(outdir, "pairwise_significance_nonstationary.csv"), index=False)

    for ds in DATASET_ORDER:
        plot_regret_curves_paper(
            df_curve[df_curve["dataset"] == ds],
            title=f"Nonstationary baselines on {ds}",
            outpath=os.path.join(figdir, f"nonstationary_regret_{ds}.png")
        )
        plot_boxplot_paper(
            df_final[df_final["dataset"] == ds],
            title=f"Nonstationary final regret on {ds}",
            outpath=os.path.join(figdir, f"nonstationary_boxplot_{ds}.png")
        )

    return df_final, df_curve


def summarize_tradeoff(final_df, defense_name):
    rows = []
    for (dataset, strength), sub in final_df.groupby(["dataset", "strength"]):
        rows.append({
            "defense": defense_name,
            "dataset": dataset,
            "strength": strength,
            "attacker_regret_mean": sub["final_regret"].mean(),
            "attacker_reward_mean": sub["final_expected_reward"].mean(),
            "benign_accept_rate_mean": sub["benign_accept_rate"].mean(),
            "leakage_proxy_std_mean": sub["leakage_proxy_std"].mean(),
            "leakage_proxy_range_mean": sub["leakage_proxy_range"].mean(),
        })
    return pd.DataFrame(rows)


def run_defense_tradeoff(prepared_map, outdir: str, figdir: str):
    long_rows = []
    final_rows = []

    for dataset_name in DATASET_ORDER:
        prepared = prepared_map[dataset_name]
        env = build_stationary_env(prepared, K, DEFAULT_TAU, Q_DEFENSE, N_BASE_ATTACK_SAMPLES, seed=123)

        for tau in BATCHING_GRID:
            print(f"[Defense-batching] dataset={dataset_name}, tau={tau}")
            for seed in SEEDS:
                curve_df, final = run_stationary_single(env, "OSUB_KL", seed, tau, STATIONARY_T_BATCHES)
                curve_df["dataset"] = dataset_name
                curve_df["defense"] = "batching"
                curve_df["strength"] = tau
                curve_df["seed"] = seed
                long_rows.append(curve_df)
                final_rows.append({
                    "dataset": dataset_name, "defense": "batching", "strength": tau, "seed": seed, **final
                })

        for eps in LDP_EPS_GRID:
            print(f"[Defense-ldp] dataset={dataset_name}, eps={eps}")
            for seed in SEEDS:
                curve_df, final = run_stationary_single(env, "OSUB_KL", seed, DEFAULT_TAU, STATIONARY_T_BATCHES, ldp_eps=eps)
                val = 999999 if np.isinf(eps) else eps
                curve_df["dataset"] = dataset_name
                curve_df["defense"] = "ldp"
                curve_df["strength"] = val
                curve_df["seed"] = seed
                long_rows.append(curve_df)
                final_rows.append({
                    "dataset": dataset_name, "defense": "ldp", "strength": val, "seed": seed, **final
                })

        for sigma in DITHER_SIGMA_GRID:
            print(f"[Defense-dither] dataset={dataset_name}, sigma={sigma}")
            for seed in SEEDS:
                curve_df, final = run_stationary_single(env, "OSUB_KL", seed, DEFAULT_TAU, STATIONARY_T_BATCHES, dither_sigma=sigma)
                curve_df["dataset"] = dataset_name
                curve_df["defense"] = "dither"
                curve_df["strength"] = sigma
                curve_df["seed"] = seed
                long_rows.append(curve_df)
                final_rows.append({
                    "dataset": dataset_name, "defense": "dither", "strength": sigma, "seed": seed, **final
                })

        level_map = {"exact": 0, "binary": 1, "3bin": 3, "5bin": 5}
        for mode in COARSENING_GRID:
            print(f"[Defense-coarsening] dataset={dataset_name}, mode={mode}")
            for seed in SEEDS:
                curve_df, final = run_stationary_single(env, "OSUB_KL", seed, DEFAULT_TAU, STATIONARY_T_BATCHES, coarsening=mode)
                curve_df["dataset"] = dataset_name
                curve_df["defense"] = "coarsening"
                curve_df["strength"] = level_map[mode]
                curve_df["seed"] = seed
                long_rows.append(curve_df)
                final_rows.append({
                    "dataset": dataset_name, "defense": "coarsening", "strength": level_map[mode], "seed": seed, **final
                })

        for alpha in EQUALIZATION_GRID:
            print(f"[Defense-equalization] dataset={dataset_name}, alpha={alpha}")
            for seed in SEEDS:
                curve_df, final = run_stationary_single(env, "OSUB_KL", seed, DEFAULT_TAU, STATIONARY_T_BATCHES, equalization=alpha)
                curve_df["dataset"] = dataset_name
                curve_df["defense"] = "equalization"
                curve_df["strength"] = alpha
                curve_df["seed"] = seed
                long_rows.append(curve_df)
                final_rows.append({
                    "dataset": dataset_name, "defense": "equalization", "strength": alpha, "seed": seed, **final
                })

        for amp in MOTION_AMPLITUDE_GRID:
            for period in MOTION_PERIOD_GRID:
                print(f"[Defense-motion] dataset={dataset_name}, amp={amp}, period={period}")
                for seed in SEEDS:
                    curve_df, final = run_nonstationary_single(
                        env, "OSUB_KL_PH", seed, DEFAULT_TAU, NONSTATIONARY_T_BATCHES,
                        motion_amplitude=amp, motion_period=period
                    )
                    curve_df["dataset"] = dataset_name
                    curve_df["defense"] = "motion"
                    curve_df["strength"] = amp
                    curve_df["motion_period"] = period
                    curve_df["seed"] = seed
                    long_rows.append(curve_df)
                    final_rows.append({
                        "dataset": dataset_name,
                        "defense": "motion",
                        "strength": amp,
                        "motion_period": period,
                        "seed": seed,
                        **final
                    })

    df_long = pd.concat(long_rows, axis=0, ignore_index=True)
    df_final = pd.DataFrame(final_rows)

    df_long.to_csv(os.path.join(outdir, "defense_tradeoff_long.csv"), index=False)
    df_final.to_csv(os.path.join(outdir, "defense_tradeoff_final.csv"), index=False)

    all_summaries = []

    for defense in ["batching", "ldp", "dither", "coarsening", "equalization"]:
        sub = df_final[df_final["defense"] == defense].copy()
        summary = summarize_tradeoff(sub, defense)
        all_summaries.append(summary)
        plot_tradeoff_paper(
            summary,
            xcol="strength",
            title_prefix=f"{defense.capitalize()} trade-off",
            out_prefix=os.path.join(figdir, f"tradeoff_{defense}")
        )

    motion = df_final[df_final["defense"] == "motion"].copy()
    motion_summary_rows = []
    for (ds, amp, period), sub in motion.groupby(["dataset", "strength", "motion_period"]):
        motion_summary_rows.append({
            "dataset": ds,
            "strength": amp,
            "motion_period": period,
            "attacker_regret_mean": sub["final_regret"].mean(),
            "attacker_reward_mean": sub["final_expected_reward"].mean(),
            "benign_accept_rate_mean": sub["benign_accept_rate"].mean(),
        })
    motion_summary = pd.DataFrame(motion_summary_rows)
    all_summaries.append(motion_summary.assign(defense="motion"))

    for ds in DATASET_ORDER:
        for metric, ylabel in [
            ("attacker_regret_mean", "Attacker final regret"),
            ("attacker_reward_mean", "Attacker expected reward"),
            ("benign_accept_rate_mean", "Benign acceptance rate"),
        ]:
            fig, ax = plt.subplots(figsize=(5.8, 3.9))
            sub = motion_summary[motion_summary["dataset"] == ds]
            for period in sorted(sub["motion_period"].unique()):
                tmp = sub[sub["motion_period"] == period].sort_values("strength")
                ax.plot(tmp["strength"], tmp[metric], marker="o", linewidth=2.1, label=f"P={period}")
            ax.set_xlabel("Motion amplitude")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Quantile motion on {ds}: {ylabel}")
            ax.legend(ncol=2, loc="best")
            savefig_paper(os.path.join(figdir, f"tradeoff_motion_{ds}_{metric}.png"))

    summary_df = pd.concat(all_summaries, axis=0, ignore_index=True)
    summary_df.to_csv(os.path.join(outdir, "defense_tradeoff_summary.csv"), index=False)

    return df_long, df_final, summary_df


# ============================================================
# Extra summary visualizations
# ============================================================

def make_combined_heatmaps(stationary_final, nonstationary_final, figdir: str):
    mat = np.zeros((len(DATASET_ORDER), len(STATIONARY_ALGORITHMS)), dtype=float)
    for i, ds in enumerate(DATASET_ORDER):
        for j, algo in enumerate(STATIONARY_ALGORITHMS):
            sub = stationary_final[(stationary_final["dataset"] == ds) & (stationary_final["algorithm"] == algo)]
            mat[i, j] = sub["final_regret"].mean()

    plot_heatmap_paper(
        mat,
        row_labels=DATASET_ORDER,
        col_labels=[LABEL_MAP[a] for a in STATIONARY_ALGORITHMS],
        title="Stationary mean final regret",
        outpath=os.path.join(figdir, "heatmap_stationary_mean_final_regret.png"),
        fmt=".1f",
        cmap="magma"
    )

    mat2 = np.zeros((len(DATASET_ORDER), len(NONSTATIONARY_ALGORITHMS)), dtype=float)
    for i, ds in enumerate(DATASET_ORDER):
        for j, algo in enumerate(NONSTATIONARY_ALGORITHMS):
            sub = nonstationary_final[(nonstationary_final["dataset"] == ds) & (nonstationary_final["algorithm"] == algo)]
            mat2[i, j] = sub["final_regret"].mean()

    plot_heatmap_paper(
        mat2,
        row_labels=DATASET_ORDER,
        col_labels=[LABEL_MAP.get(a, a) for a in NONSTATIONARY_ALGORITHMS],
        title="Nonstationary mean final regret",
        outpath=os.path.join(figdir, "heatmap_nonstationary_mean_final_regret.png"),
        fmt=".1f",
        cmap="viridis"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root_default = script_dir.parent if script_dir.name == "scripts" else script_dir
    data_root_default = project_root_default / "data"
    outdir_default = project_root_default / "outputs" / "paper_final_tradeoff_and_baselines"

    parser = argparse.ArgumentParser(description="Defense trade-off and baseline experiments.")
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
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] PROJECT_ROOT={project_root}")
    print(f"[INFO] DATA_ROOT={data_root}")
    print(f"[INFO] OUTDIR={outdir}")
    print(f"[INFO] FIGDIR={figdir}")

    prepared_map = load_all_prepared_datasets(str(data_root))

    run_config = {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "outdir": str(outdir),
        "figdir": str(figdir),
        "datasets": DATASET_ORDER,
        "seeds": SEEDS,
        "K": K,
        "DEFAULT_TAU": DEFAULT_TAU,
        "Q_DEFENSE": Q_DEFENSE,
        "STATIONARY_T_BATCHES": STATIONARY_T_BATCHES,
        "NONSTATIONARY_T_BATCHES": NONSTATIONARY_T_BATCHES,
        "N_BASE_ATTACK_SAMPLES": N_BASE_ATTACK_SAMPLES,
        "GAMMA_LEAD": GAMMA_LEAD,
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n[INFO] Running stationary baselines...")
    stationary_env, stationary_final, stationary_curve = run_stationary_baselines(prepared_map, str(outdir), str(figdir))

    print("\n[INFO] Running nonstationary baselines...")
    nonstationary_final, nonstationary_curve = run_nonstationary_baselines(prepared_map, str(outdir), str(figdir))

    print("\n[INFO] Running defense trade-off...")
    defense_long, defense_final, defense_summary = run_defense_tradeoff(prepared_map, str(outdir), str(figdir))

    print("\n[INFO] Making combined heatmaps...")
    make_combined_heatmaps(stationary_final, nonstationary_final, str(figdir))

    print("\n[INFO] All experiments completed.")
    print(f"[INFO] Outputs saved to: {outdir}")
    print(f"[INFO] Figures saved to: {figdir}")


if __name__ == "__main__":
    main()
