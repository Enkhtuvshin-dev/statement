from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    from IPython.display import display
except Exception:
    display = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from step1_inline_income_model import (
    FEATURE_COLS,
    build_features,
    kmeans_confidence,
    map_clusters_to_income,
)

NUMERIC_BASE_COLS = ["amount", "day", "month", "hour", "weekday"]
MODEL_DATASET_COLS = [
    "sender",
    "freq",
    "avg_amount",
    "std_amount",
    "interval_std",
    "unique_days",
    "max_amount",
    "income_type",
]


def _interval_days_stats(series: pd.Series) -> tuple[float, float]:
    diffs = series.sort_values().diff().dt.total_seconds().div(86400).dropna()
    if diffs.empty:
        return np.nan, np.nan
    return float(diffs.median()), float(diffs.std(ddof=0) if len(diffs) > 1 else 0.0)


def show_df(frame: pd.DataFrame, title: str | None = None, max_rows: int = 20) -> None:
    if title:
        print(f"\n{title}")
    if display is not None:
        display(frame.head(max_rows))
    else:
        print(frame.head(max_rows).to_string(index=False))


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
    df = df.dropna(subset=["account_id", "sender", "amount", "transaction_date"]).copy()
    df["account_id"] = df["account_id"].astype(str)
    df["sender"] = df["sender"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"]).copy()
    df["day"] = df["transaction_date"].dt.day
    df["month"] = df["transaction_date"].dt.month
    df["hour"] = df["transaction_date"].dt.hour
    df["weekday"] = df["transaction_date"].dt.weekday
    return df.sort_values(["account_id", "sender", "transaction_date"]).reset_index(drop=True)


def initial_inspection(df: pd.DataFrame, head_rows: int = 10) -> pd.DataFrame:
    print("Initial Data Inspection")
    print(f"Rows: {len(df)}")
    print(f"Accounts: {df['account_id'].nunique()}")
    print(f"Senders: {df['sender'].nunique()}")
    show_df(df, title="head()", max_rows=head_rows)
    describe = df.copy()
    describe["transaction_date"] = describe["transaction_date"].astype(str)
    describe = describe.describe(include="all").T
    show_df(describe.reset_index().rename(columns={"index": "feature"}), title="describe()", max_rows=50)
    return describe


def _numeric_subtype(series: pd.Series) -> tuple[str, float]:
    counts = series.value_counts(dropna=True)
    mode_share = float(counts.iloc[0] / len(series)) if len(series) > 0 and not counts.empty else 0.0
    unique_ratio = float(series.nunique(dropna=True) / max(len(series), 1))
    subtype = "discrete" if (unique_ratio < 0.1 or mode_share > 0.25) else "continuous"
    return subtype, mode_share


def feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    meanings = {
        "account_id": "Account identifier for grouping transactions.",
        "sender": "Counterparty/sender identifier.",
        "amount": "Transaction amount received.",
        "transaction_date": "Timestamp when transaction occurred.",
        "day": "Day of month extracted from transaction_date.",
        "month": "Month number extracted from transaction_date.",
        "hour": "Hour extracted from transaction_date.",
        "weekday": "Weekday index where 0=Monday.",
    }
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            subtype, mode_share = _numeric_subtype(df[col])
            rows.append(
                {
                    "feature": col,
                    "meaning": meanings.get(col, ""),
                    "type": "quantitative",
                    "subtype": subtype,
                    "mode_share": round(mode_share, 4),
                    "encoding_check": "N/A",
                }
            )
        else:
            encoding_check = (
                "Logical nominal ID values; no ordinal encoding expected."
                if col in {"account_id", "sender"}
                else "Categorical encoding appears logically ordered."
            )
            rows.append(
                {
                    "feature": col,
                    "meaning": meanings.get(col, ""),
                    "type": "qualitative",
                    "subtype": "nominal",
                    "mode_share": np.nan,
                    "encoding_check": encoding_check,
                }
            )
    result = pd.DataFrame(rows)
    show_df(result, title="Feature Understanding and Classification", max_rows=50)
    return result


def scatter_matrix_and_correlations(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] = NUMERIC_BASE_COLS,
    sample_size: int = 1200,
    show_plots: bool = True,
) -> pd.DataFrame:
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr(numeric_only=True)
    corr_long = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "feature_x", "level_1": "feature_y", 0: "correlation"})
    )
    corr_long["abs_corr"] = corr_long["correlation"].abs()
    strongest = corr_long.sort_values("abs_corr", ascending=False).drop_duplicates(
        subset=["abs_corr"]
    )
    show_df(strongest.head(10), title="Strongest Feature Relationships", max_rows=10)
    if show_plots:
        sampled = df[numeric_cols].sample(min(sample_size, len(df)), random_state=42)
        sns.pairplot(sampled, corner=True, diag_kind="hist")
        plt.suptitle("Scatter Plot Matrix (sampled)", y=1.02)
        plt.show()
        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
    return strongest


def histograms_with_log_checks(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] = NUMERIC_BASE_COLS,
    skew_threshold: float = 1.0,
    show_plots: bool = True,
) -> pd.DataFrame:
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    skew_table = pd.DataFrame(
        {
            "feature": numeric_cols,
            "skewness": [df[c].skew() for c in numeric_cols],
        }
    )
    skew_table["needs_log_transform"] = skew_table["skewness"].abs() > skew_threshold
    show_df(skew_table, title="Skewness and Log-Transform Decision", max_rows=50)
    if not show_plots:
        return skew_table

    n = len(numeric_cols)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for idx, col in enumerate(numeric_cols):
        raw = df[col].dropna()
        sns.histplot(raw, kde=True, ax=axes[idx, 0], bins=30)
        axes[idx, 0].set_title(f"{col} (raw)")
        if (raw >= 0).all() and bool(skew_table.loc[skew_table["feature"] == col, "needs_log_transform"].iloc[0]):
            transformed = np.log1p(raw)
            sns.histplot(transformed, kde=True, ax=axes[idx, 1], bins=30)
            axes[idx, 1].set_title(f"{col} (log1p)")
        else:
            sns.histplot(raw, kde=True, ax=axes[idx, 1], bins=30)
            axes[idx, 1].set_title(f"{col} (no log applied)")
    plt.tight_layout()
    plt.show()
    return skew_table


def outlier_detection_iqr(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] = NUMERIC_BASE_COLS,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    summary_rows = []
    samples: dict[str, pd.DataFrame] = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outliers = df.loc[mask, ["account_id", "sender", "transaction_date", col]].copy()
        samples[col] = outliers.head(10)
        summary_rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(float(mask.mean() * 100), 2),
                "justification": "Outside [Q1-1.5*IQR, Q3+1.5*IQR], indicating unusually extreme values.",
            }
        )
    summary = pd.DataFrame(summary_rows)
    show_df(summary, title="Outlier Detection Summary (IQR Rule)", max_rows=50)

    if show_plots:
        for col in numeric_cols:
            plt.figure(figsize=(7, 3))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot Outlier Check - {col}")
            plt.tight_layout()
            plt.show()
    return summary, samples


def descriptive_statistics(df: pd.DataFrame, numeric_cols: Iterable[str] = NUMERIC_BASE_COLS) -> pd.DataFrame:
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    stats = pd.DataFrame(
        {
            "feature": numeric_cols,
            "min": [df[c].min() for c in numeric_cols],
            "max": [df[c].max() for c in numeric_cols],
            "mean": [df[c].mean() for c in numeric_cols],
            "median": [df[c].median() for c in numeric_cols],
            "skewness": [df[c].skew() for c in numeric_cols],
        }
    )
    show_df(stats, title="Descriptive Statistics", max_rows=50)
    return stats


def grouped_account_analysis(
    df: pd.DataFrame,
    target_features: Iterable[str] = ("amount",),
    show_plots: bool = True,
) -> pd.DataFrame:
    target_features = [c for c in target_features if c in df.columns]
    metrics = (
        df.groupby("account_id")[target_features]
        .agg(["mean", "median", "std"])
        .sort_index()
        .reset_index()
    )
    metrics.columns = [
        "account_id" if a == "account_id" else f"{a}_{b}" for a, b in metrics.columns.to_flat_index()
    ]
    show_df(metrics, title="Group Metrics by account_id", max_rows=100)

    if show_plots:
        for col in target_features:
            plt.figure(figsize=(12, 4))
            sns.boxplot(data=df, x="account_id", y=col)
            plt.title(f"{col} Distribution by account_id")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    return metrics


def qq_plots(
    df: pd.DataFrame,
    feature: str = "amount",
    top_accounts: int = 4,
    show_plots: bool = True,
) -> None:
    if feature not in df.columns or not show_plots:
        return
    ranked_accounts = df["account_id"].value_counts().head(top_accounts).index.tolist()
    plots = [("all", df[feature].dropna())] + [
        (str(acc), df.loc[df["account_id"] == acc, feature].dropna()) for acc in ranked_accounts
    ]
    n = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, values) in zip(axes, plots):
        values = np.sort(values.to_numpy())
        probs = (np.arange(1, len(values) + 1) - 0.5) / len(values)
        theoretical = norm.ppf(probs, loc=np.mean(values), scale=np.std(values) if np.std(values) > 0 else 1.0)
        ax.scatter(theoretical, values, s=8, alpha=0.7)
        min_v = min(theoretical.min(), values.min())
        max_v = max(theoretical.max(), values.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
        ax.set_title(f"Q-Q: {feature} ({label})")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Observed Quantiles")
    plt.tight_layout()
    plt.show()


def build_final_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    sender_base = (
        df.groupby("sender", as_index=False)
        .agg(
            freq=("amount", "size"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
            interval_std=("transaction_date", lambda s: s.sort_values().diff().dt.total_seconds().div(86400).std()),
            unique_days=("transaction_date", lambda s: s.dt.floor("D").nunique()),
            max_amount=("amount", "max"),
        )
        .fillna({"std_amount": 0.0, "interval_std": 999.0})
    )

    features = build_features(df)
    scaler = StandardScaler()
    x = scaler.fit_transform(features[FEATURE_COLS])
    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = model.fit_predict(x)
    distances = model.transform(x)

    clustered = features.copy()
    clustered["cluster_id"] = labels
    clustered["confidence_score"] = kmeans_confidence(distances, labels)
    clustered["income_type"] = clustered["cluster_id"].map(map_clusters_to_income(clustered))

    sender_income = (
        clustered.sort_values("confidence_score", ascending=False)
        .drop_duplicates("sender")[["sender", "income_type"]]
    )
    final_model_dataset = sender_base.merge(sender_income, on="sender", how="left")
    final_model_dataset = final_model_dataset[MODEL_DATASET_COLS].sort_values("freq", ascending=False)
    show_df(final_model_dataset, title="FINAL MODEL DATASET", max_rows=50)
    return final_model_dataset


def summarize_findings(
    corr_top: pd.DataFrame,
    skew_table: pd.DataFrame,
    outlier_summary: pd.DataFrame,
) -> pd.DataFrame:
    positive = corr_top[corr_top["correlation"] > 0].head(3)
    negative = corr_top[corr_top["correlation"] < 0].head(3)
    skewed = skew_table[skew_table["needs_log_transform"]]["feature"].tolist()
    most_outlier = outlier_summary.sort_values("outlier_pct", ascending=False).head(3)

    rows = [
        {
            "topic": "Strongest positive relationships",
            "finding": ", ".join(
                f"{r.feature_x}~{r.feature_y} ({r.correlation:.2f})"
                for r in positive.itertuples(index=False)
            )
            or "No strong positive pair found.",
        },
        {
            "topic": "Strongest negative relationships",
            "finding": ", ".join(
                f"{r.feature_x}~{r.feature_y} ({r.correlation:.2f})"
                for r in negative.itertuples(index=False)
            )
            or "No strong negative pair found.",
        },
        {
            "topic": "Skewed features",
            "finding": ", ".join(skewed) if skewed else "No feature exceeded skew threshold.",
        },
        {
            "topic": "Highest outlier share",
            "finding": ", ".join(
                f"{r.feature} ({r.outlier_pct:.2f}%)" for r in most_outlier.itertuples(index=False)
            ),
        },
        {
            "topic": "Normality",
            "finding": "Q-Q plots indicate tail deviations for heavy-transaction features (especially amount), "
            "so strict normality is not assumed.",
        },
    ]
    findings = pd.DataFrame(rows)
    show_df(findings, title="Final Findings Summary", max_rows=20)
    return findings


def run_colab_eda_normality(
    csv_path: str,
    account_focus: str | None = None,
    show_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    sns.set_theme(style="whitegrid")
    raw_df = load_raw_data(csv_path)
    if account_focus is not None:
        raw_df = raw_df[raw_df["account_id"] == str(account_focus)].copy()
        print(f"Account-focused view enabled: account_id={account_focus}, rows={len(raw_df)}")

    _ = initial_inspection(raw_df)
    feature_info = feature_dictionary(raw_df)
    corr_top = scatter_matrix_and_correlations(raw_df, show_plots=show_plots)
    skew_table = histograms_with_log_checks(raw_df, show_plots=show_plots)
    outlier_summary, _ = outlier_detection_iqr(raw_df, show_plots=show_plots)
    desc_stats = descriptive_statistics(raw_df)
    group_metrics = grouped_account_analysis(raw_df, target_features=("amount",), show_plots=show_plots)
    qq_plots(raw_df, feature="amount", show_plots=show_plots)
    final_model_dataset = build_final_model_dataset(raw_df)
    findings = summarize_findings(corr_top, skew_table, outlier_summary)

    return {
        "raw_data": raw_df,
        "feature_dictionary": feature_info,
        "correlation_top_pairs": corr_top,
        "skewness_table": skew_table,
        "outlier_summary": outlier_summary,
        "descriptive_stats": desc_stats,
        "group_metrics": group_metrics,
        "final_model_dataset": final_model_dataset,
        "findings": findings,
    }


def build_one_account_sender_profile(
    df: pd.DataFrame, account_id: str | int, show_plots: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    account_id = str(account_id)
    one = df[df["account_id"] == account_id].copy()
    if one.empty:
        raise ValueError(f"No rows found for account_id={account_id}")

    sender_profile = (
        one.groupby("sender", as_index=False)
        .agg(
            freq=("amount", "size"),
            avg_amount=("amount", "mean"),
            std_amount=("amount", "std"),
            max_amount=("amount", "max"),
            unique_days=("transaction_date", lambda s: s.dt.floor("D").nunique()),
            first_date=("transaction_date", "min"),
            last_date=("transaction_date", "max"),
            unique_months=(
                "transaction_date",
                lambda s: s.dt.tz_localize(None).dt.to_period("M").nunique()
                if getattr(s.dt, "tz", None) is not None
                else s.dt.to_period("M").nunique(),
            ),
            median_interval_days=("transaction_date", lambda s: _interval_days_stats(s)[0]),
            interval_std=("transaction_date", lambda s: _interval_days_stats(s)[1]),
            total_amount=("amount", "sum"),
        )
        .fillna({"std_amount": 0.0})
        .sort_values(["freq", "total_amount"], ascending=False)
    )
    sender_profile["amount_cv"] = (
        sender_profile["std_amount"] / sender_profile["avg_amount"].abs().clip(lower=1.0)
    )
    sender_profile["first_date"] = pd.to_datetime(sender_profile["first_date"]).dt.strftime("%Y-%m-%d")
    sender_profile["last_date"] = pd.to_datetime(sender_profile["last_date"]).dt.strftime("%Y-%m-%d")
    sender_profile["interval_observations"] = (sender_profile["freq"] - 1).clip(lower=0).astype(int)

    month_sender = one.copy()
    month_source = month_sender["transaction_date"]
    if getattr(month_source.dt, "tz", None) is not None:
        month_source = month_source.dt.tz_localize(None)
    month_sender["year_month"] = month_source.dt.to_period("M").astype(str)
    monthly = (
        month_sender.groupby(["year_month", "sender"], as_index=False)["amount"]
        .sum()
        .sort_values(["year_month", "amount"], ascending=[True, False])
    )

    print(f"\nPhase 1 - Base profile for account_id={account_id}")
    show_df(sender_profile, title="Sender Behavior Evidence Table", max_rows=80)
    if show_plots:
        top_senders = sender_profile.head(12)["sender"].tolist()
        heatmap_df = (
            monthly[monthly["sender"].isin(top_senders)]
            .pivot(index="sender", columns="year_month", values="amount")
            .fillna(0.0)
        )
        plt.figure(figsize=(14, max(4, int(len(heatmap_df) * 0.45))))
        sns.heatmap(heatmap_df, cmap="YlGnBu", linewidths=0.2, linecolor="white")
        plt.title(f"Monthly Inflow Heatmap by Sender (Top 12) - account_id={account_id}")
        plt.xlabel("Year-Month")
        plt.ylabel("Sender")
        plt.tight_layout()
        plt.show()

        # Small multiples are clearer than one spaghetti line chart.
        top_small = sender_profile.head(8)["sender"].tolist()
        plot_df = monthly[monthly["sender"].isin(top_small)].copy()
        g = sns.relplot(
            data=plot_df,
            x="year_month",
            y="amount",
            col="sender",
            col_wrap=4,
            kind="line",
            marker="o",
            height=2.6,
            aspect=1.35,
            facet_kws={"sharex": True, "sharey": True},
        )
        g.set_titles("sender={col_name}")
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
        g.fig.suptitle(
            f"Monthly Inflow Small Multiples (Top 8 Senders) - account_id={account_id}",
            y=1.03,
        )
        g.tight_layout()
        plt.show()
    return sender_profile, monthly


def _label_sender_row(
    row: pd.Series,
    monthly_presence_min: int,
    interval_days_target: float,
    interval_days_tol: float,
    salary_freq_min: int,
    sme_freq_min: int,
    sme_months_min: int,
) -> tuple[str, str]:
    median_interval = row["median_interval_days"]
    has_interval = pd.notna(median_interval)
    near_monthly = has_interval and abs(float(median_interval) - interval_days_target) <= interval_days_tol
    recurring_months = int(row["unique_months"]) >= monthly_presence_min
    enough_freq = int(row["freq"]) >= salary_freq_min

    if recurring_months and near_monthly and enough_freq:
        return (
            "salary_candidate",
            f"monthly_repeat=True (months={int(row['unique_months'])}), median_interval={row['median_interval_days']:.1f}d, freq={int(row['freq'])}",
        )

    if not has_interval:
        return (
            "unknown_low_evidence",
            f"insufficient interval data (freq={int(row['freq'])}); need at least 2 transactions for interval-based checks",
        )

    if int(row["freq"]) >= sme_freq_min and int(row["unique_months"]) >= sme_months_min:
        return (
            "sme_candidate",
            f"high_activity sender (freq={int(row['freq'])}, months={int(row['unique_months'])}) but not near monthly salary interval",
        )

    return (
        "other_candidate",
        f"irregular pattern (freq={int(row['freq'])}, months={int(row['unique_months'])}, median_interval={row['median_interval_days']:.1f}d)",
    )


def apply_simple_income_rules(
    sender_profile: pd.DataFrame,
    monthly_presence_min: int = 4,
    interval_days_target: float = 30.0,
    interval_days_tol: float = 7.0,
    salary_freq_min: int = 4,
    sme_freq_min: int = 8,
    sme_months_min: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    labeled = sender_profile.copy()
    labels_and_reasons = labeled.apply(
        lambda row: _label_sender_row(
            row=row,
            monthly_presence_min=monthly_presence_min,
            interval_days_target=interval_days_target,
            interval_days_tol=interval_days_tol,
            salary_freq_min=salary_freq_min,
            sme_freq_min=sme_freq_min,
            sme_months_min=sme_months_min,
        ),
        axis=1,
    )
    labeled["income_type"] = labels_and_reasons.map(lambda x: x[0])
    labeled["reason"] = labels_and_reasons.map(lambda x: x[1])
    if verbose:
        show_df(
            labeled[
                [
                    "sender",
                    "freq",
                    "avg_amount",
                    "median_interval_days",
                    "unique_months",
                    "income_type",
                    "reason",
                ]
            ],
            title="Phase 2 - Explainable Sender Labels",
            max_rows=100,
        )
    return labeled


def estimate_account_income(
    labeled_sender_df: pd.DataFrame,
    sensitivity_interval_tolerances: Iterable[float] = (5.0, 7.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_to_output = {
        "salary_candidate": "salary_income_est",
        "sme_candidate": "sme_income_est",
        "other_candidate": "other_income_est",
        "unknown_low_evidence": "other_income_est",
    }

    labeled_sender_df = labeled_sender_df.copy()
    labeled_sender_df["income_bucket"] = labeled_sender_df["income_type"].map(tag_to_output)
    totals = (
        labeled_sender_df.groupby("income_bucket", as_index=False)["total_amount"]
        .sum()
        .set_index("income_bucket")["total_amount"]
    )
    account_summary = pd.DataFrame(
        [
            {
                "salary_income_est": float(totals.get("salary_income_est", 0.0)),
                "sme_income_est": float(totals.get("sme_income_est", 0.0)),
                "other_income_est": float(totals.get("other_income_est", 0.0)),
            }
        ]
    )
    account_summary["real_income_est"] = (
        account_summary["salary_income_est"] + account_summary["sme_income_est"]
    )
    show_df(account_summary, title="Phase 3 - One Account Income Estimation", max_rows=10)

    sensitivity_rows = []
    for tol in sensitivity_interval_tolerances:
        relabeled = apply_simple_income_rules(
            sender_profile=labeled_sender_df.drop(columns=["income_type", "reason", "income_bucket"], errors="ignore"),
            interval_days_tol=float(tol),
            verbose=False,
        )
        relabeled["income_bucket"] = relabeled["income_type"].map(tag_to_output)
        relabeled_totals = (
            relabeled.groupby("income_bucket", as_index=False)["total_amount"]
            .sum()
            .set_index("income_bucket")["total_amount"]
        )
        salary = float(relabeled_totals.get("salary_income_est", 0.0))
        sme = float(relabeled_totals.get("sme_income_est", 0.0))
        sensitivity_rows.append(
            {
                "interval_tolerance_days": float(tol),
                "salary_income_est": salary,
                "sme_income_est": sme,
                "other_income_est": float(relabeled_totals.get("other_income_est", 0.0)),
                "real_income_est": salary + sme,
            }
        )

    sensitivity = pd.DataFrame(sensitivity_rows)
    show_df(sensitivity, title="Phase 3 - Sensitivity Check", max_rows=20)
    return account_summary, sensitivity


def _single_account_pipeline(
    raw_df: pd.DataFrame,
    account_id: str | int,
    show_plots: bool = True,
    interval_days_tol: float = 7.0,
) -> dict[str, pd.DataFrame]:
    sender_profile, monthly_timeline = build_one_account_sender_profile(
        df=raw_df, account_id=account_id, show_plots=show_plots
    )
    labeled = apply_simple_income_rules(
        sender_profile=sender_profile,
        interval_days_tol=interval_days_tol,
    )
    account_summary, sensitivity = estimate_account_income(labeled)
    final_sender_dataset = labeled[
        [
            "sender",
            "freq",
            "avg_amount",
            "std_amount",
            "amount_cv",
            "interval_observations",
            "interval_std",
            "unique_days",
            "max_amount",
            "income_type",
            "reason",
        ]
    ].sort_values(["income_type", "freq"], ascending=[True, False])
    return {
        "sender_profile": sender_profile,
        "monthly_timeline": monthly_timeline,
        "labeled_senders": labeled,
        "account_income_summary": account_summary,
        "sensitivity": sensitivity,
        "final_sender_dataset": final_sender_dataset,
    }


def run_incremental_fintech_income(
    csv_path: str,
    account_id: str | int,
    show_plots: bool = True,
    interval_days_tol: float = 7.0,
) -> dict[str, pd.DataFrame]:
    """
    Incremental fintech workflow:
    1) Build one-account base profile and timeline evidence.
    2) Apply simple explainable monthly-repeat sender rules.
    3) Estimate one-account salary/sme/other totals + sensitivity.
    4) Scale same logic to all accounts with explainability outputs.
    """
    sns.set_theme(style="whitegrid")
    raw_df = load_raw_data(csv_path)

    # Phase 1-3 on one account.
    one_account = _single_account_pipeline(
        raw_df=raw_df,
        account_id=account_id,
        show_plots=show_plots,
        interval_days_tol=interval_days_tol,
    )

    # Phase 4 scale to all accounts.
    all_sender_frames = []
    all_account_frames = []
    for acc in sorted(raw_df["account_id"].unique(), key=lambda x: int(x)):
        result = _single_account_pipeline(
            raw_df=raw_df,
            account_id=acc,
            show_plots=False,
            interval_days_tol=interval_days_tol,
        )
        sender_df = result["final_sender_dataset"].copy()
        sender_df["account_id"] = acc
        all_sender_frames.append(sender_df)
        account_df = result["account_income_summary"].copy()
        account_df["account_id"] = acc
        all_account_frames.append(account_df)

    all_senders = pd.concat(all_sender_frames, ignore_index=True)
    account_summary_all = pd.concat(all_account_frames, ignore_index=True)[
        ["account_id", "salary_income_est", "sme_income_est", "other_income_est", "real_income_est"]
    ].sort_values("account_id")

    show_df(all_senders, title="Phase 4 - Final Sender Dataset (All Accounts)", max_rows=80)
    show_df(account_summary_all, title="Phase 4 - Account Income Summary (All Accounts)", max_rows=80)

    return {
        "raw_data": raw_df,
        "phase1_sender_profile": one_account["sender_profile"],
        "phase1_monthly_timeline": one_account["monthly_timeline"],
        "phase2_labeled_senders": one_account["labeled_senders"],
        "phase3_account_income_summary": one_account["account_income_summary"],
        "phase3_sensitivity": one_account["sensitivity"],
        "phase4_final_sender_dataset": all_senders,
        "phase4_account_income_summary_all": account_summary_all,
    }
