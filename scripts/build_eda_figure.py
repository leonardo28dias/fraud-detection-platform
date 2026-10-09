"""Generate a compact EDA overview figure for the README.

Reads the raw Kaggle dataset and produces a single 2x2 panel showing the
four findings that shaped the modelling decisions:
  * class imbalance
  * transaction amount distribution by class
  * fraud frequency over time (hour of day)
  * top |correlation| features with the target

Output: reports/eda_overview.png
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/creditcard.csv")
OUT_PATH = Path("reports/eda_overview.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %s rows from %s", len(df), DATA_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) Class imbalance
    counts = df["Class"].value_counts().sort_index()
    axes[0, 0].bar(
        ["Legitimate", "Fraud"],
        counts.values,
        color=["#1f77b4", "#d62728"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(
        f"Class imbalance — fraud is {counts[1] / len(df) * 100:.2f}% of rows"
    )
    axes[0, 0].set_ylabel("Count (log scale)")

    # (2) Amount distribution by class
    bins = np.linspace(0, 2500, 50)
    axes[0, 1].hist(
        df[df["Class"] == 0]["Amount"],
        bins=bins,
        alpha=0.55,
        label="Legitimate",
        density=True,
        color="#1f77b4",
    )
    axes[0, 1].hist(
        df[df["Class"] == 1]["Amount"],
        bins=bins,
        alpha=0.55,
        label="Fraud",
        density=True,
        color="#d62728",
    )
    axes[0, 1].set_title("Transaction amount by class (0-2500 range)")
    axes[0, 1].set_xlabel("Amount")
    axes[0, 1].legend()

    # (3) Fraud over time (hour of day)
    hours = (df["Time"] % 86400) / 3600
    hourly = df.assign(Hour=hours).groupby("Hour")["Class"].agg(["sum", "count"])
    hourly["rate"] = hourly["sum"] / hourly["count"] * 100
    axes[1, 0].plot(hourly.index, hourly["rate"], color="#d62728", lw=1.4)
    axes[1, 0].set_title("Fraud rate by hour of day")
    axes[1, 0].set_xlabel("Hour of day")
    axes[1, 0].set_ylabel("Fraud rate (%)")

    # (4) Top |correlation| with target
    corrs = df.corr()["Class"].abs().sort_values(ascending=False).drop("Class")
    top = corrs.head(8)
    axes[1, 1].barh(top.index[::-1], top.values[::-1], color="#2ca02c")
    axes[1, 1].set_title("Top |correlation| with Class")
    axes[1, 1].set_xlabel("|Pearson correlation|")

    fig.suptitle(
        "Credit Card Fraud Detection — EDA overview", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    logger.info("Saved EDA overview to %s", OUT_PATH)


if __name__ == "__main__":
    main()
