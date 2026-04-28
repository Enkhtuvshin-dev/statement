from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    # Force inline backend when running inside notebook kernels (Colab/Jupyter).
    import matplotlib

    matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass

try:
    from IPython.display import display
except Exception:
    display = None


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def mode_share(day_series: pd.Series) -> float:
    if day_series.empty:
        return 0.0
    counts = day_series.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / counts.sum())


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["account_id", "sender", "amount", "transaction_date"])
    df = df.sort_values(["account_id", "sender", "transaction_date"])

    df["txn_day"] = df["transaction_date"].dt.floor("D")
    month_source = df["transaction_date"]
    if getattr(month_source.dt, "tz", None) is not None:
        month_source = month_source.dt.tz_localize(None)
    df["month"] = month_source.dt.to_period("M").astype(str)
    df["day_of_month"] = df["transaction_date"].dt.day
    df["interval_days"] = (
        df.groupby(["account_id", "sender"])["transaction_date"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    account_totals = (
        df.groupby("account_id", as_index=False)
        .agg(account_total_txn_count=("amount", "size"))
        .astype({"account_total_txn_count": float})
    )

    features = (
        df.groupby(["account_id", "sender"], as_index=False)
        .agg(
            transaction_count=("amount", "size"),
            average_amount=("amount", "mean"),
            amount_std=("amount", "std"),
            max_amount=("amount", "max"),
            unique_transaction_days=("txn_day", "nunique"),
            unique_months=("month", "nunique"),
            average_interval_days=("interval_days", "mean"),
            interval_std_days=("interval_days", "std"),
            median_interval_days=("interval_days", "median"),
            day_of_month_mode_share=("day_of_month", mode_share),
        )
    )

    features = features.merge(account_totals, on="account_id", how="left")
    features["sender_txn_share_in_account"] = (
        features["transaction_count"] / features["account_total_txn_count"].clip(lower=1)
    )
    features["periodicity_30d_score"] = np.exp(
        -np.abs(features["average_interval_days"] - 30.0) / 10.0
    ) / (1.0 + features["interval_std_days"].fillna(15.0) / 30.0)
    features["amount_cv"] = (
        features["amount_std"].fillna(0)
        / features["average_amount"].abs().clip(lower=1.0)
    )

    return features.fillna(
        {
            "amount_std": 0.0,
            "average_interval_days": 0.0,
            "interval_std_days": 999.0,
            "median_interval_days": 0.0,
            "sender_txn_share_in_account": 0.0,
            "periodicity_30d_score": 0.0,
            "amount_cv": 0.0,
        }
    )


FEATURE_COLS = [
    "transaction_count",
    "average_amount",
    "amount_std",
    "max_amount",
    "unique_transaction_days",
    "unique_months",
    "average_interval_days",
    "interval_std_days",
    "median_interval_days",
    "day_of_month_mode_share",
    "sender_txn_share_in_account",
    "periodicity_30d_score",
    "amount_cv",
]


def kmeans_confidence(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.clip(distances, 1e-9, None)
    probs = inv / inv.sum(axis=1, keepdims=True)
    return probs[np.arange(len(labels)), labels]


def map_clusters_to_income(out: pd.DataFrame) -> dict[int, str]:
    profile = (
        out.groupby("cluster_id", as_index=False)
        .agg(
            periodicity_30d_score=("periodicity_30d_score", "mean"),
            interval_std_days=("interval_std_days", "mean"),
        )
    )
    salary_cid = int(
        profile.sort_values("periodicity_30d_score", ascending=False).iloc[0]["cluster_id"]
    )
    other_cid = int(
        profile.sort_values("interval_std_days", ascending=False).iloc[0]["cluster_id"]
    )

    mapping = {}
    for cid in profile["cluster_id"].astype(int):
        if cid == salary_cid:
            mapping[cid] = "salary_income"
        elif cid == other_cid:
            mapping[cid] = "other_irregular_income"
        else:
            mapping[cid] = "sme_business_income"
    return mapping


def show_table(frame: pd.DataFrame) -> None:
    if display is not None:
        display(frame)
    else:
        print(frame.to_string(index=False))


def render_plot(fig: plt.Figure, show_plots: bool = True) -> None:
    if show_plots:
        plt.show()
    plt.close(fig)


def show_account_view(out: pd.DataFrame, account_id: int | str, limit_rows: int = 50) -> None:
    account_df = out[out["account_id"] == account_id].copy()
    if account_df.empty:
        print(f"No sender profiles found for account_id={account_id}")
        return

    print(f"\nAccount Analysis: {account_id}")
    print(f"Total sender profiles: {len(account_df)}")
    print("Income type distribution:")
    print(account_df["predicted_income_type"].value_counts())

    detail_cols = [
        "account_id",
        "sender",
        "predicted_income_type",
        "confidence_score",
        "transaction_count",
        "average_amount",
        "periodicity_30d_score",
        "interval_std_days",
        "amount_cv",
    ]
    account_df = account_df.sort_values(
        ["predicted_income_type", "confidence_score", "transaction_count"],
        ascending=[True, False, False],
    )
    show_table(account_df[detail_cols].head(limit_rows))


def run_from_dataframe(
    transactions: pd.DataFrame,
    limit_rows: int = 20,
    show_plots: bool | None = None,
    account_id: int | str | None = None,
) -> pd.DataFrame:
    if show_plots is None:
        show_plots = in_notebook()

    sns.set_theme(style="whitegrid")
    features = build_features(transactions)

    scaler = StandardScaler()
    X = scaler.fit_transform(features[FEATURE_COLS])

    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    distances = model.transform(X)

    out = features.copy()
    out["cluster_id"] = labels
    out["confidence_score"] = kmeans_confidence(distances, labels).clip(0, 0.99)
    cluster_map = map_clusters_to_income(out)
    out["predicted_income_type"] = out["cluster_id"].map(cluster_map)

    print("Model Evaluation Metrics:")
    print(f"Rows: {len(out)}")
    print(f"Accounts: {out['account_id'].nunique()}")
    print(f"Silhouette: {silhouette_score(X, labels):.4f}")
    print(f"Davies-Bouldin: {davies_bouldin_score(X, labels):.4f}")
    print(f"Avg confidence: {out['confidence_score'].mean():.4f}")

    print("\nClass Distribution:")
    print(out["predicted_income_type"].value_counts())

    print("\nSample Predictions:")
    sample_cols = [
        "account_id",
        "sender",
        "predicted_income_type",
        "confidence_score",
        "transaction_count",
        "average_amount",
        "periodicity_30d_score",
        "interval_std_days",
        "amount_cv",
    ]
    if account_id is None:
        show_table(out[sample_cols].head(limit_rows))
    else:
        show_account_view(out=out, account_id=account_id, limit_rows=limit_rows)

    fig1, _ = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=out,
        x="predicted_income_type",
        hue="predicted_income_type",
        dodge=False,
        legend=False,
    )
    plt.title("Income Type Distribution")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    render_plot(fig1, show_plots=show_plots)

    fig2, _ = plt.subplots(figsize=(8, 4))
    sns.histplot(
        out,
        x="confidence_score",
        hue="predicted_income_type",
        bins=20,
        element="step",
        common_norm=False,
    )
    plt.title("Confidence Distribution by Income Type")
    plt.tight_layout()
    render_plot(fig2, show_plots=show_plots)

    fig3, _ = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=out,
        x="periodicity_30d_score",
        y="amount_cv",
        hue="predicted_income_type",
        size="transaction_count",
        alpha=0.75,
    )
    plt.axhline(y=1.0, color="r", linestyle="--")
    plt.title("Periodicity vs Amount Variability")
    plt.xlabel("Periodicity (~30 day score)")
    plt.ylabel("Amount CV")
    plt.tight_layout()
    render_plot(fig3, show_plots=show_plots)

    return out


def run_from_csv(
    csv_path: str,
    limit_rows: int = 20,
    show_plots: bool | None = None,
    account_id: int | str | None = None,
) -> pd.DataFrame:
    transactions = pd.read_csv(csv_path)
    return run_from_dataframe(
        transactions=transactions,
        limit_rows=limit_rows,
        show_plots=show_plots,
        account_id=account_id,
    )


def run(
    csv_path: str,
    limit_rows: int = 20,
    show_plots: bool | None = None,
    account_id: int | str | None = None,
) -> pd.DataFrame:
    # Backward-compatible alias for existing notebook cells.
    return run_from_csv(
        csv_path=csv_path,
        limit_rows=limit_rows,
        show_plots=show_plots,
        account_id=account_id,
    )
