# -*- coding: utf-8 -*-
"""
Assumption validation for dataset-built tabular trimming simulators.

Datasets:
- Adult
- Bank Marketing

Robust to:
- extensionless files, e.g. "Adult", "Bank Marketing"
- ARFF content without .arff suffix
- raw Adult-style comma-separated text if needed
"""

import os
import re
import json
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(2026)


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

    matches = []
    for fname in os.listdir(root):
        fpath = os.path.join(root, fname)
        if not os.path.isfile(fpath):
            continue
        low = fname.lower()
        if all(k.lower() in low for k in include_keywords):
            matches.append(fpath)

    if not matches:
        return None

    def score(path: str):
        low = os.path.basename(path).lower()
        s = 0
        if ".arff" in low:
            s += 5
        if ".data" in low:
            s += 5
        if ".csv" in low:
            s += 4
        if ".txt" in low:
            s += 3
        if "." not in os.path.basename(low):
            s += 2
        return -s

    matches = sorted(matches, key=score)
    return matches[0]


def preview_file(path: str, n_lines: int = 15):
    print(f"\n[INFO] Preview of file: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i in range(n_lines):
                line = f.readline()
                if not line:
                    break
                print(f"  L{i+1:02d}: {repr(line[:220])}")
    except Exception as e:
        print(f"[WARN] Failed to preview file: {e}")


def read_head_text(path: str, n_chars: int = 5000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(n_chars)
    except Exception:
        return ""


def looks_like_arff(path: str) -> bool:
    text = read_head_text(path, 5000).lower()
    return ("@relation" in text) and ("@attribute" in text) and ("@data" in text)


def is_unimodal(seq: np.ndarray, tol: float = 1e-10) -> Tuple[bool, int]:
    seq = np.asarray(seq, dtype=float)
    peak = int(np.argmax(seq))
    left = seq[:peak + 1]
    right = seq[peak:]
    left_ok = np.all(np.diff(left) >= -tol)
    right_ok = np.all(np.diff(right) <= tol)
    return bool(left_ok and right_ok), peak


def monotone_nonincreasing(seq: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    dif = np.diff(np.asarray(seq, dtype=float))
    viol = float(np.sum(np.maximum(dif, 0.0)))
    return bool(np.all(dif <= tol)), viol


def monotone_nondecreasing(seq: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    dif = np.diff(np.asarray(seq, dtype=float))
    viol = float(np.sum(np.maximum(-dif, 0.0)))
    return bool(np.all(dif >= -tol)), viol


# ============================================================
# ARFF parser
# ============================================================

def split_arff_row(line: str) -> List[str]:
    parts = []
    cur = []
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
    attributes = []
    data_rows = []
    in_data = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            low = line.lower()
            if low.startswith("@attribute"):
                m = re.match(r'@attribute\s+(".*?"|\'.*?\'|\S+)\s+(.*)', line, flags=re.I)
                if m:
                    name = m.group(1).strip().strip('"').strip("'")
                    attributes.append(name)
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                parts = split_arff_row(line)
                if len(parts) == len(attributes):
                    cleaned = []
                    for x in parts:
                        x = x.strip().strip('"').strip("'")
                        if x == "?":
                            x = np.nan
                        cleaned.append(x)
                    data_rows.append(cleaned)

    if not attributes or not data_rows:
        raise ValueError(f"Failed to parse ARFF file: {path}")

    df = pd.DataFrame(data_rows, columns=attributes)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


# ============================================================
# Adult helpers
# ============================================================

def normalize_adult_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    for c in df.columns:
        low = str(c).strip().lower().replace("_", "-")

        if low == "education-num":
            rename_map[c] = "education_num"
        elif low == "capital-gain":
            rename_map[c] = "capital_gain"
        elif low == "capital-loss":
            rename_map[c] = "capital_loss"
        elif low == "hours-per-week":
            rename_map[c] = "hours_per_week"
        elif low == "marital-status":
            rename_map[c] = "marital_status"
        elif low == "native-country":
            rename_map[c] = "native_country"
        elif low in ["class", "income", "salary", "target"]:
            rename_map[c] = "income"
        else:
            rename_map[c] = str(c).strip().replace("-", "_")

    df = df.rename(columns=rename_map)

    if "income" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "income"})

    return df


def parse_adult_robust(path: str) -> pd.DataFrame:
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]

    valid_rows = []
    bad_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            low = line.lower()
            if (
                low.startswith("%")
                or low.startswith("|")
                or low.startswith("@")
                or "attribute" in low
                or "relation" in low
                or low == "adult"
            ):
                continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 15:
                valid_rows.append(parts)
            else:
                bad_rows.append((line_no, line[:200], len(parts)))

    print(f"[INFO] Adult valid rows parsed: {len(valid_rows)}")
    print(f"[INFO] Adult skipped malformed rows: {len(bad_rows)}")

    if len(valid_rows) == 0:
        raise ValueError(f"Adult parsing failed: no valid 15-column rows found in {path}")

    df = pd.DataFrame(valid_rows, columns=cols)

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace("?", np.nan)

    numeric_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip().str.lower()
    income_map = {"<=50k": 0, ">50k": 1, "0": 0, "1": 1}
    df["income"] = df["income"].map(income_map)

    df = df.dropna().reset_index(drop=True)
    df["income"] = df["income"].astype(int)
    return df


def load_adult(root: str) -> pd.DataFrame:
    path = find_file_fuzzy(root, ["adult"])
    if path is None:
        raise FileNotFoundError(f"Adult dataset not found in {root}. Files present: {safe_listdir(root)}")

    print(f"[INFO] Adult file found: {path}")
    preview_file(path, n_lines=12)

    if looks_like_arff(path):
        print("[INFO] Adult file detected as ARFF.")
        df = parse_arff_simple(path)
        df = normalize_adult_columns(df)

        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip().replace("?", np.nan)

        for c in ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip().str.lower()
        income_map = {
            "<=50k": 0, ">50k": 1,
            "<=50k.": 0, ">50k.": 1,
            "0": 0, "1": 1
        }

        mapped = df["income"].map(income_map)
        if mapped.notna().sum() == 0:
            codes, uniques = pd.factorize(df["income"])
            if len(uniques) != 2:
                raise ValueError(f"Adult target is not binary. Unique labels: {list(uniques)}")
            df["income"] = codes.astype(int)
        else:
            df["income"] = mapped

        df = df.dropna().reset_index(drop=True)
        df["income"] = df["income"].astype(int)
    else:
        print("[INFO] Adult file not detected as ARFF. Using robust raw-data parser.")
        df = parse_adult_robust(path)

    print(f"[INFO] Adult loaded successfully: shape={df.shape}")
    return df


# ============================================================
# Bank Marketing helpers
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
        raise FileNotFoundError(f"Bank Marketing dataset not found in {root}. Files present: {safe_listdir(root)}")

    print(f"[INFO] Bank Marketing file found: {path}")
    preview_file(path, n_lines=12)

    if looks_like_arff(path):
        print("[INFO] Bank Marketing file detected as ARFF.")
        df = parse_arff_simple(path)
    else:
        print("[INFO] Bank Marketing file not detected as ARFF. Trying CSV/semicolon parsing.")
        try:
            df = pd.read_csv(path, sep=";")
            if df.shape[1] < 2:
                raise ValueError("Too few columns with semicolon parse.")
        except Exception:
            try:
                df = pd.read_csv(path)
                if df.shape[1] < 2:
                    raise ValueError("Too few columns with csv parse.")
            except Exception:
                raise ValueError(f"Could not parse Bank Marketing file: {path}")

    df = normalize_bank_columns(df)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace("?", np.nan)

    df["target"] = df["target"].astype(str).str.strip().str.lower()

    target_map = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    }
    mapped = df["target"].map(target_map)

    if mapped.notna().sum() == 0:
        codes, uniques = pd.factorize(df["target"])
        if len(uniques) != 2:
            raise ValueError(f"Bank Marketing target is not binary. Unique labels: {list(uniques)}")
        df["target"] = codes.astype(int)
    else:
        df["target"] = mapped

    df = df.dropna().reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    print(f"[INFO] Bank Marketing loaded successfully: shape={df.shape}")
    return df


# ============================================================
# Preprocessing
# ============================================================

@dataclass
class PreparedDataset:
    name: str
    raw_df: pd.DataFrame
    X_proc: np.ndarray
    y: np.ndarray


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_dataset(df: pd.DataFrame, name: str, target_col: str) -> PreparedDataset:
    if df.shape[0] == 0:
        raise ValueError(f"{name} dataframe is empty after parsing.")

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
        ("onehot", make_onehot_encoder())
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_proc = pre.fit_transform(X).astype(float)

    print(f"[INFO] Prepared {name}: n={X_proc.shape[0]}, d={X_proc.shape[1]}")
    return PreparedDataset(name=name, raw_df=df, X_proc=X_proc, y=y)


# ============================================================
# Simulator
# ============================================================

@dataclass
class ValidationOracle:
    arms: np.ndarray
    impacts: np.ndarray
    ref_scores: np.ndarray
    attack_scores_by_arm: List[np.ndarray]
    direction_name: str


class AssumptionValidatorSimulator:
    def __init__(self, prepared: PreparedDataset, K: int = 11, n_base_samples: int = 2000, random_state: int = 0):
        self.prepared = prepared
        self.X = prepared.X_proc
        self.n, self.d = self.X.shape
        self.K = K
        self.n_base_samples = n_base_samples
        self.rng = np.random.RandomState(random_state)

    def anomaly_score(self, X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        Z = (X - center) / np.maximum(scale, 1e-8)
        return np.sqrt(np.sum(Z ** 2, axis=1))

    def arm_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.K)

    def impact_scale(self, arms: np.ndarray) -> np.ndarray:
        return 0.15 + 0.95 * np.power(arms, 0.75)

    def candidate_directions(self) -> Dict[str, np.ndarray]:
        dirs = {}
        n_comp = min(3, self.d, max(1, min(self.n - 1, 3)))
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(self.X)

        for i, vec in enumerate(pca.components_):
            v = vec / (np.linalg.norm(vec) + 1e-12)
            dirs[f"pca_{i+1}"] = v

        for j in range(3):
            v = self.rng.normal(size=self.d)
            v = v / (np.linalg.norm(v) + 1e-12)
            dirs[f"rand_{j+1}"] = v

        v = np.mean(np.abs(self.X), axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        dirs["absmean"] = v
        return dirs

    def simulate_attack_scores(self, direction: np.ndarray, arms: np.ndarray, center: np.ndarray, scale: np.ndarray) -> List[np.ndarray]:
        idx = self.rng.choice(
            self.n,
            size=min(self.n_base_samples, self.n),
            replace=(self.n_base_samples > self.n)
        )
        X_base = self.X[idx]
        feature_std = float(np.std(self.X))
        base_step = 0.55 * feature_std + 0.25

        scores_by_arm = []
        for a in arms:
            step = base_step * (0.15 + 1.45 * a)
            X_att = X_base + step * direction[None, :]
            s = self.anomaly_score(X_att, center, scale)
            scores_by_arm.append(s)
        return scores_by_arm

    def survival_probs(self, ref_scores: np.ndarray, attack_scores_by_arm: List[np.ndarray], q: float) -> np.ndarray:
        tau = np.quantile(ref_scores, 1.0 - q)
        return np.array([(s <= tau).mean() for s in attack_scores_by_arm], dtype=float)

    def choose_direction(self, q_select: float = 0.15) -> ValidationOracle:
        center = np.mean(self.X, axis=0)
        scale = np.std(self.X, axis=0) + 1e-8
        ref_scores = self.anomaly_score(self.X, center, scale)
        arms = self.arm_grid()
        impacts = self.impact_scale(arms)

        best = None
        best_score = -1e18

        for name, direction in self.candidate_directions().items():
            attack_scores = self.simulate_attack_scores(direction, arms, center, scale)
            p = self.survival_probs(ref_scores, attack_scores, q_select)
            p_proj = np.minimum.accumulate(p)
            mu = impacts * p_proj

            uni, peak = is_unimodal(mu)
            mono, viol = monotone_nonincreasing(p_proj)

            score = 0.0
            score += 10.0 if mono else -50.0 * viol
            score += 10.0 if uni else -20.0
            score += 2.0 * (mu.max() - mu.min())
            if 0 < peak < len(mu) - 1:
                score += 3.0

            if score > best_score:
                best_score = score
                best = (name, attack_scores)

        return ValidationOracle(
            arms=arms,
            impacts=impacts,
            ref_scores=ref_scores,
            attack_scores_by_arm=best[1],
            direction_name=best[0]
        )


# ============================================================
# Validation
# ============================================================

def validate_dataset(prepared: PreparedDataset, q_values: List[float], K: int, random_state: int):
    sim = AssumptionValidatorSimulator(
        prepared=prepared,
        K=K,
        n_base_samples=2500,
        random_state=random_state
    )
    oracle = sim.choose_direction(q_select=q_values[0])

    impact_ok, impact_viol = monotone_nondecreasing(oracle.impacts)

    records = []
    survival_raw_dict = {}
    survival_proj_dict = {}
    utility_dict = {}

    for q in q_values:
        p_raw = sim.survival_probs(oracle.ref_scores, oracle.attack_scores_by_arm, q)
        p_proj = np.minimum.accumulate(p_raw)
        mu = oracle.impacts * p_proj

        s_ok_raw, s_viol_raw = monotone_nonincreasing(p_raw)
        s_ok_proj, s_viol_proj = monotone_nonincreasing(p_proj)
        u_ok, peak = is_unimodal(mu)

        survival_raw_dict[q] = p_raw
        survival_proj_dict[q] = p_proj
        utility_dict[q] = mu

        records.append({
            "dataset": prepared.name,
            "K": K,
            "quantile_q": q,
            "direction": oracle.direction_name,
            "impact_monotone": impact_ok,
            "impact_violation": impact_viol,
            "survival_monotone_raw": s_ok_raw,
            "survival_violation_raw": s_viol_raw,
            "survival_monotone_projected": s_ok_proj,
            "survival_violation_projected": s_viol_proj,
            "utility_unimodal": u_ok,
            "peak_arm_index_0based": int(peak),
            "peak_arm_value": float(oracle.arms[peak]),
            "max_utility": float(np.max(mu)),
            "min_utility": float(np.min(mu))
        })

    return oracle, pd.DataFrame(records), survival_raw_dict, survival_proj_dict, utility_dict


# ============================================================
# Plotting
# ============================================================

def plot_validation_figure(dataset_name, oracle, q_values, survival_raw_dict, survival_proj_dict, utility_dict, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    axes[0].plot(oracle.arms, oracle.impacts, marker="o", linewidth=2, color="black")
    axes[0].set_title("Known / calibrated impact scale")
    axes[0].set_xlabel("Attack arm")
    axes[0].set_ylabel("r_k")
    axes[0].grid(True, alpha=0.3)

    for i, q in enumerate(q_values):
        c = colors[i % len(colors)]
        axes[1].plot(
            oracle.arms, survival_raw_dict[q],
            marker="o", linestyle="--", linewidth=1.5, color=c, alpha=0.5,
            label=f"raw q={q:.2f}"
        )
        axes[1].plot(
            oracle.arms, survival_proj_dict[q],
            marker="o", linestyle="-", linewidth=2, color=c,
            label=f"proj q={q:.2f}"
        )
    axes[1].set_title("Survival curves")
    axes[1].set_xlabel("Attack arm")
    axes[1].set_ylabel("p_k(q)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for i, q in enumerate(q_values):
        c = colors[i % len(colors)]
        mu = utility_dict[q]
        peak = int(np.argmax(mu))
        axes[2].plot(oracle.arms, mu, marker="o", linewidth=2, color=c, label=f"q={q:.2f}")
        axes[2].scatter([oracle.arms[peak]], [mu[peak]], color=c, s=70, zorder=5)

    axes[2].set_title("Expected utility profiles")
    axes[2].set_xlabel("Attack arm")
    axes[2].set_ylabel("mu_k(q) = r_k p_k(q)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{dataset_name}: assumption validation", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# Summary
# ============================================================

def write_summary_txt(df_all: pd.DataFrame, outpath: str):
    lines = []
    lines.append("Assumption Validation Summary\n")
    lines.append("=" * 70 + "\n\n")

    for dataset in df_all["dataset"].unique():
        sub = df_all[df_all["dataset"] == dataset].copy()

        lines.append(f"Dataset: {dataset}\n")
        lines.append("-" * 70 + "\n")
        lines.append(f"Impact monotone for all q: {bool(sub['impact_monotone'].all())}\n")
        lines.append(f"Raw survival monotone for all q: {bool(sub['survival_monotone_raw'].all())}\n")
        lines.append(f"Projected survival monotone for all q: {bool(sub['survival_monotone_projected'].all())}\n")
        lines.append(f"Utility unimodal for all q: {bool(sub['utility_unimodal'].all())}\n\n")

        for _, row in sub.iterrows():
            lines.append(
                f"q={row['quantile_q']:.2f} | "
                f"direction={row['direction']} | "
                f"raw_survival_monotone={row['survival_monotone_raw']} | "
                f"proj_survival_monotone={row['survival_monotone_projected']} | "
                f"utility_unimodal={row['utility_unimodal']} | "
                f"peak_arm_idx={int(row['peak_arm_index_0based'])} | "
                f"peak_arm={row['peak_arm_value']:.3f}\n"
            )
        lines.append("\n")

    with open(outpath, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ============================================================
# CLI
# ============================================================

def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root_default = script_dir.parent if script_dir.name == "scripts" else script_dir
    data_root_default = project_root_default / "data"
    outdir_default = project_root_default / "outputs" / "assumption_validation"

    parser = argparse.ArgumentParser(description="Assumption validation for dataset-built simulators.")
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

    print("[INFO] Directory listing under data_root:")
    for name in safe_listdir(str(data_root)):
        print("   ", name)

    q_values = [0.10, 0.15, 0.25]
    K = 11

    adult_df = load_adult(str(data_root))
    bank_df = load_bank_marketing(str(data_root))

    adult = prepare_dataset(adult_df, "Adult", "income")
    bank = prepare_dataset(bank_df, "BankMarketing", "target")

    adult_oracle, adult_table, adult_surv_raw, adult_surv_proj, adult_util = validate_dataset(
        adult, q_values=q_values, K=K, random_state=11
    )
    bank_oracle, bank_table, bank_surv_raw, bank_surv_proj, bank_util = validate_dataset(
        bank, q_values=q_values, K=K, random_state=17
    )

    all_table = pd.concat([adult_table, bank_table], axis=0, ignore_index=True)
    csv_path = outdir / "assumption_validation_table.csv"
    all_table.to_csv(csv_path, index=False)

    adult_fig = outdir / "adult_assumption_validation.png"
    bank_fig = outdir / "bank_marketing_assumption_validation.png"

    plot_validation_figure("Adult", adult_oracle, q_values, adult_surv_raw, adult_surv_proj, adult_util, str(adult_fig))
    plot_validation_figure("Bank Marketing", bank_oracle, q_values, bank_surv_raw, bank_surv_proj, bank_util, str(bank_fig))

    meta = {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "outdir": str(outdir),
        "Adult": {
            "direction": adult_oracle.direction_name,
            "arms": adult_oracle.arms.tolist(),
            "impacts": adult_oracle.impacts.tolist()
        },
        "BankMarketing": {
            "direction": bank_oracle.direction_name,
            "arms": bank_oracle.arms.tolist(),
            "impacts": bank_oracle.impacts.tolist()
        }
    }

    meta_path = outdir / "assumption_validation_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary_path = outdir / "assumption_validation_summary.txt"
    write_summary_txt(all_table, str(summary_path))

    print("\n[INFO] Done.")
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] Adult figure saved to: {adult_fig}")
    print(f"[INFO] Bank figure saved to: {bank_fig}")
    print(f"[INFO] Meta saved to: {meta_path}")
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
