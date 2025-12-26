from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    income_quintile: int
    n_coastal: int
    n_inland: int
    mean_coastal: float
    mean_inland: float
    welch_t: float
    welch_p: float
    mannwhitney_u: float
    mannwhitney_p: float
    cohens_d: float


def cohens_d_welch(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d using pooled SD (classic), robust enough for reporting effect size.

    Note: Welch's t-test is used for inference; Cohen's d is for magnitude.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return float("nan")

    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0

    return (x.mean() - y.mean()) / pooled


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction.

    Returns adjusted p-values in original order.
    """

    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    prev = 0.0
    for rank, idx in enumerate(order):
        p = float(p_values[idx])
        adj = (m - rank) * p
        adj = min(1.0, adj)
        adj = max(prev, adj)  # enforce monotonicity
        adjusted[idx] = adj
        prev = adj

    return adjusted.tolist()


def load_samples(csv_path: Path) -> pd.DataFrame:
    """Load sample rows (income quintiles 3 & 4) via DuckDB SQL."""

    con = duckdb.connect(database=":memory:")

    # DuckDB reads CSVs directly; using SQL keeps this consistent with the assignment constraints.
    query = """
    WITH housing_train AS (
      SELECT *
      FROM read_csv_auto(?, header = TRUE)
    ),
    labeled AS (
      SELECT
        longitude,
        latitude,
        median_income,
        median_house_value,
        CASE
          WHEN longitude <= -121.0 THEN 'coastal_like'
          ELSE 'inland_like'
        END AS geo_group
      FROM housing_train
      WHERE median_income IS NOT NULL
        AND median_house_value IS NOT NULL
    ),
    banded AS (
      SELECT
        *,
        NTILE(5) OVER (ORDER BY median_income) AS income_quintile
      FROM labeled
    )
    SELECT
      income_quintile,
      geo_group,
      median_house_value
    FROM banded
    WHERE income_quintile IN (3, 4)
    ;
    """

    df = con.execute(query, [str(csv_path)]).df()
    df["income_quintile"] = df["income_quintile"].astype(int)
    return df


def run_tests(df: pd.DataFrame) -> tuple[list[TestResult], list[float]]:
    results: list[TestResult] = []
    pvals: list[float] = []

    for quintile in sorted(df["income_quintile"].unique().tolist()):
        sub = df[df["income_quintile"] == quintile]
        coastal = sub[sub["geo_group"] == "coastal_like"]["median_house_value"].to_numpy(dtype=float)
        inland = sub[sub["geo_group"] == "inland_like"]["median_house_value"].to_numpy(dtype=float)

        tstat, p = stats.ttest_ind(coastal, inland, equal_var=False)
        # Robustness check (non-parametric). Two-sided by default.
        ustat, p_u = stats.mannwhitneyu(coastal, inland, alternative="two-sided")

        d = cohens_d_welch(coastal, inland)

        results.append(
            TestResult(
                income_quintile=int(quintile),
                n_coastal=int(coastal.size),
                n_inland=int(inland.size),
                mean_coastal=float(np.mean(coastal)),
                mean_inland=float(np.mean(inland)),
                welch_t=float(tstat),
                welch_p=float(p),
                mannwhitney_u=float(ustat),
                mannwhitney_p=float(p_u),
                cohens_d=float(d),
            )
        )
        pvals.append(float(p))

    holm = holm_adjust(pvals)
    return results, holm


def plot(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    ax = sns.boxenplot(
        data=df,
        x="income_quintile",
        y="median_house_value",
        hue="geo_group",
    )
    ax.set_title("Jan 2021 snapshot: median_house_value by coastal vs inland (income-controlled)")
    ax.set_xlabel("Income quintile (computed on train dataset)")
    ax.set_ylabel("Median house value")
    ax.legend(title="Group", loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / "california_housing_train.csv"

    df = load_samples(csv_path)
    results, holm = run_tests(df)

    print("Problem 3 — Coastal premium within income quintiles (3 & 4)")
    print("Rule: coastal_like if longitude <= -121.0")
    print()

    for r, holm_p in zip(results, holm):
        print(
            f"Q{r.income_quintile}: n_coastal={r.n_coastal}, n_inland={r.n_inland} | "
            f"mean_coastal={r.mean_coastal:,.0f}, mean_inland={r.mean_inland:,.0f} | "
            f"Welch t={r.welch_t:.3f}, p={r.welch_p:.4g}, Holm p={holm_p:.4g} | "
            f"Mann-Whitney U p={r.mannwhitney_p:.4g} | "
            f"Cohen's d={r.cohens_d:.3f}"
        )

    out_path = here / "problem3_coastal_premium_boxen.png"
    plot(df, out_path)
    print()
    print(f"Saved dataviz: {out_path}")


if __name__ == "__main__":
    main()
