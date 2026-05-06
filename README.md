# Batched Structured Bandits for Adversarial Trade-offs on Real Tabular Data

This repository contains experiment code for batched bandit studies on real tabular datasets, using the Adult and Bank Marketing datasets. The experiments focus on structured arm spaces, stationary and nonstationary settings, defense–utility trade-offs, robustness under impact-scale misspecification, and ablation analyses.

## Overview

The repository currently includes the following experiment scripts:

- Defense trade-off experiments.py
- Impact-scale misspecification under fixed ordered neighborhoods.py
- Ablation under stationary arm-resolved batched feedback.py

These scripts build batched structured bandit environments from real datasets and generate outputs such as:

- CSV summaries
- long-form simulation logs
- significance tables
- regret curves
- boxplots
- violin plots
- heatmaps
- arm-frequency figures
- arm trajectory figures

## Datasets

Place the datasets inside the data directory.

Supported datasets:

- Adult
- Bank Marketing

The data loaders use fuzzy filename matching, so common filenames are supported, for example:

- adult.data
- adult.csv
- adult.arff
- bank.csv
- bank-full.csv
- bank_marketing.csv
- bank-additional.arff

If multiple files match, the scripts generally prefer formats in the following order:

1. .arff
2. .csv
3. .data
4. .txt

Only one valid file for Adult and one valid file for Bank Marketing are needed.

## Suggested Repository Structure

.
├── data/
│   ├── <Adult dataset file>
│   └── <Bank Marketing dataset file>
├── scripts/
│   ├── Defense trade-off experiments.py
│   ├── Impact-scale misspecification under fixed ordered neighborhoods.py
│   └── Ablation under stationary arm-resolved batched feedback.py
├── outputs/
│   └── ...
└── README.md

## Environment Setup

Recommended Python version:

Python 3.10+

The scripts were written for a standard scientific Python environment.

## Installation

Install the required packages with:

pip install numpy pandas matplotlib scikit-learn scipy

If you prefer a virtual environment, you can create one with:

python -m venv .venv

Activate it on Linux or macOS with:

source .venv/bin/activate

Activate it on Windows PowerShell with:

.venv\Scripts\Activate.ps1

Then install dependencies:

pip install numpy pandas matplotlib scikit-learn scipy

## How to Run

Because your script filenames currently contain spaces, it is safest to run them using quotation marks.

Run the defense trade-off and baseline experiments:

python "scripts/Defense trade-off experiments.py"

Run the impact-scale misspecification robustness experiments:

python "scripts/Impact-scale misspecification under fixed ordered neighborhoods.py"

Run the ablation experiments:

python "scripts/Ablation under stationary arm-resolved batched feedback.py"

If you later rename the scripts to space-free filenames, the commands can be simplified.

## Script Descriptions

Defense trade-off experiments.py

This script runs the main paper-style experiments, including:

- stationary baselines
- nonstationary baselines
- defense trade-off sweeps

Defense settings include:

- batching
- local differential privacy
- dithering
- coarsening
- equalization
- motion-based nonstationarity

Typical outputs include:

- stationary environment summaries
- baseline comparison CSV files
- long-form regret logs
- pairwise significance results
- trade-off summary tables
- figures
- heatmaps

Impact-scale misspecification under fixed ordered neighborhoods.py

This script studies robustness when the learner uses a misspecified impact scale while performance is still evaluated under the true environment.

Typical outputs include:

- impact misspecification diagnostics
- final regret summaries
- long-form simulation logs
- statistical summaries
- impact profile plots
- win-rate heatmaps
- optimal-arm shift heatmaps
- regret-gap boxplots
- arm-frequency heatmaps

Ablation under stationary arm-resolved batched feedback.py

This script analyzes the contribution of different components of the structured batched bandit method.

Algorithms compared include:

- UCB
- KLUCB_GLOBAL
- UCB_OSUB
- KLUCB_OSUB

The ablation is designed to isolate the contribution of:

- KL-based confidence bounds
- OSUB-style neighborhood restriction and leader scheduling

Typical outputs include:

- environment summaries
- long-form logs
- final regret summaries
- pairwise significance tables
- regret curves
- boxplots
- violin plots
- paired scatter plots
- arm trajectory plots
- arm-pull heatmaps
- relative improvement figures
- component contribution figures
- global heatmaps

## Outputs

The scripts save outputs to directories defined inside the code. Since your current code still contains its own output-path settings unless you have already modified them, please make sure the output paths in the scripts match your local environment.

Depending on the script, outputs may include:

- .csv result files
- .json configuration files
- .png figures

## Experimental Notes

Across the scripts, the experiments use batched bandit environments with real tabular data and structured arm-dependent rewards.

Common properties include:

- preprocessing with imputation, scaling, and one-hot encoding
- arm-dependent impact scales
- anomaly-based attack survival probabilities
- batched feedback
- repeated runs over multiple random seeds
- evaluation under stationary and nonstationary conditions

Typical default settings appearing in the scripts include:

- K = 11 arms
- tau = 20 in standard batched settings
- 30 random seeds
- datasets: Adult and BankMarketing

Some scripts additionally use grids or robustness parameters such as:

- misspecification strength values
- batching grids
- privacy levels
- noise strengths
- coarsening modes
- motion amplitudes
- motion periods

## Notes

If you plan to upload this repository publicly, it is strongly recommended to eventually rename the script files to names without spaces, for example:

- defense_tradeoff_experiments.py
- impact_scale_misspecification_fixed_ordered_neighborhoods.py
- ablation_stationary_arm_resolved_batched_feedback.py

This is not required for local use, but it makes command-line execution and repository maintenance easier.
