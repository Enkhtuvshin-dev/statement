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


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    required_cols = {"account_id", "sender", "amount", "transaction_date"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(
            "Missing required transaction columns: "
            + ", ".join(missing)
            + ". Expected columns: account_id, sender, amount, transaction_date."
        )
    transaction_date_text = (
        df["transaction_date"].astype(str).str.replace(",", " ", regex=False).str.strip()
    )
    df["transaction_date"] = pd.to_datetime(transaction_date_text, errors="coerce")
    amount_text = df["amount"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["amount"] = pd.to_numeric(amount_text, errors="coerce")
    df = df.dropna(subset=["account_id", "sender", "amount", "transaction_date"])
    if df.empty:
        return pd.DataFrame(
            columns=[
                "account_id",
                "sender",
                "total_txns",
                "average_monthly_amount",
                "active_months",
                "monthly_amount_std",
                "average_txns_per_month",
                "account_total_txn_count",
                "sender_txn_share_in_account",
                "monthly_amount_cv",
            ]
        )

    month_source = df["transaction_date"]
    if getattr(month_source.dt, "tz", None) is not None:
        month_source = month_source.dt.tz_localize(None)
    df["month"] = month_source.dt.to_period("M").astype(str)

    account_totals = (
        df.groupby("account_id", as_index=False)
        .agg(account_total_txn_count=("amount", "size"))
        .astype({"account_total_txn_count": float})
    )

    monthly_df = (
        df.groupby(["account_id", "sender", "month"], as_index=False)
        .agg(
            monthly_total_amount=("amount", "sum"),
            txns_in_month=("amount", "size"),
        )
    )
    medians = monthly_df.groupby(["account_id", "sender"])[
        "monthly_total_amount"
    ].transform("median")
    monthly_df["monthly_total_amount_capped"] = np.where(
        monthly_df["monthly_total_amount"] > medians * 2,
        medians,
        monthly_df["monthly_total_amount"],
    )

    features = (
        monthly_df.groupby(["account_id", "sender"], as_index=False)
        .agg(
            total_txns=("txns_in_month", "sum"),
            average_monthly_amount=("monthly_total_amount", "mean"),
            active_months=("month", "nunique"),
            monthly_amount_std=("monthly_total_amount_capped", "std"),
            average_txns_per_month=("txns_in_month", "mean"),
        )
    )

    features = features.merge(account_totals, on="account_id", how="left")
    features["sender_txn_share_in_account"] = (
        features["total_txns"] / features["account_total_txn_count"].clip(lower=1)
    )
    features["monthly_amount_cv"] = features["monthly_amount_std"].fillna(
        0
    ) / features["average_monthly_amount"].abs().clip(lower=1.0)

    return features.fillna(
        {
            "monthly_amount_std": 0.0,
            "sender_txn_share_in_account": 0.0,
            "monthly_amount_cv": 0.0,
        }
    )


FEATURE_COLS = [
    "total_txns",
    "average_monthly_amount",
    "active_months",
    "average_txns_per_month",
    "sender_txn_share_in_account",
    "monthly_amount_cv",
]

MIN_SALARY_ACTIVE_MONTHS = 2
MIN_SALARY_TOTAL_TXNS = 2
SALARY_PROMOTION_MIN_ACTIVE_MONTHS = 4
SALARY_PROMOTION_CV_MAX = 0.25
SALARY_PROMOTION_MIN_SCORE = 0.62
SME_PROTECTION_TXNS_PER_MONTH = 4.0
SME_PROTECTION_TXN_SHARE = 0.05


def kmeans_confidence(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.clip(distances, 1e-9, None)
    probs = inv / inv.sum(axis=1, keepdims=True)
    return probs[np.arange(len(labels)), labels]


def map_clusters_to_income(out: pd.DataFrame) -> dict[int, str]:
    profile = (
        out.groupby("cluster_id", as_index=False)
        .agg(
            monthly_amount_cv=("monthly_amount_cv", "mean"),
            average_txns_per_month=("average_txns_per_month", "mean"),
            sender_txn_share_in_account=("sender_txn_share_in_account", "mean"),
            active_months=("active_months", "mean"),
        )
    )
    profile["txns_per_month_distance"] = (
        profile["average_txns_per_month"] - 2.0
    ).abs()
    profile["cv_rank"] = profile["monthly_amount_cv"].rank(method="average", pct=True)
    profile["dist_rank"] = profile["txns_per_month_distance"].rank(
        method="average", pct=True
    )
    profile["txns_rank"] = profile["average_txns_per_month"].rank(
        method="average", pct=True
    )
    profile["share_rank"] = profile["sender_txn_share_in_account"].rank(
        method="average", pct=True
    )
    profile["months_rank"] = profile["active_months"].rank(method="average", pct=True)

    # Composite scores make mapping less brittle than single-column sorting.
    profile["salary_score"] = (
        (1 - profile["cv_rank"]) * 0.40
        + (1 - profile["dist_rank"]) * 0.25
        + profile["share_rank"] * 0.20
        + profile["months_rank"] * 0.15
    )
    profile["sme_score"] = (
        profile["txns_rank"] * 0.45
        + profile["share_rank"] * 0.35
        + profile["months_rank"] * 0.20
    )
    profile["irregular_score"] = (
        profile["cv_rank"] * 0.55
        + profile["dist_rank"] * 0.30
        + (1 - profile["months_rank"]) * 0.15
    )

    salary_cid = int(profile.sort_values("salary_score", ascending=False).iloc[0]["cluster_id"])
    remaining = profile[profile["cluster_id"] != salary_cid]
    sme_cid = int(remaining.sort_values("sme_score", ascending=False).iloc[0]["cluster_id"])

    mapping: dict[int, str] = {}
    for cid in profile["cluster_id"].astype(int):
        if cid == salary_cid:
            mapping[cid] = "salary_income"
        elif cid == sme_cid:
            mapping[cid] = "sme_business_income"
        else:
            mapping[cid] = "other_irregular_income"
    return mapping


def _salary_evidence_features(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["salary_txns_per_month_distance"] = (out["average_txns_per_month"] - 2.0).abs()
    recurrence_score = ((out["active_months"] - 1) / 5.0).clip(lower=0.0, upper=1.0)
    stability_score = (1.0 - (out["monthly_amount_cv"] / 0.60)).clip(lower=0.0, upper=1.0)
    schedule_score = (1.0 - (out["salary_txns_per_month_distance"] / 2.0)).clip(
        lower=0.0, upper=1.0
    )
    concentration_score = (out["sender_txn_share_in_account"] / 0.08).clip(
        lower=0.0, upper=1.0
    )
    out["salary_evidence_score"] = (
        recurrence_score * 0.35
        + stability_score * 0.30
        + schedule_score * 0.20
        + concentration_score * 0.15
    ).round(4)
    recurrence_flag = out["active_months"] >= SALARY_PROMOTION_MIN_ACTIVE_MONTHS
    stability_flag = out["monthly_amount_cv"] <= SALARY_PROMOTION_CV_MAX
    schedule_flag = out["average_txns_per_month"].between(0.8, 2.5, inclusive="both")
    concentration_flag = out["sender_txn_share_in_account"] >= 0.01
    out["salary_evidence_flags"] = (
        "months="
        + recurrence_flag.map({True: "1", False: "0"})
        + "|cv="
        + stability_flag.map({True: "1", False: "0"})
        + "|schedule="
        + schedule_flag.map({True: "1", False: "0"})
        + "|share="
        + concentration_flag.map({True: "1", False: "0"})
    )
    out["salary_promotion_candidate"] = (
        recurrence_flag
        & stability_flag
        & schedule_flag
        & (out["salary_evidence_score"] >= SALARY_PROMOTION_MIN_SCORE)
    )
    out["sme_protection_flag"] = (
        (out["average_txns_per_month"] >= SME_PROTECTION_TXNS_PER_MONTH)
        & (out["sender_txn_share_in_account"] >= SME_PROTECTION_TXN_SHARE)
        & (out["active_months"] >= 4)
    )
    return out


def apply_income_evidence_rules(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["cluster_income_type"] = out["predicted_income_type"]
    out["income_label_rule"] = "cluster_model"
    out = _salary_evidence_features(out)

    low_recurrence_salary = (
        out["predicted_income_type"].eq("salary_income")
        & (
            (out["active_months"] < MIN_SALARY_ACTIVE_MONTHS)
            | (out["total_txns"] < MIN_SALARY_TOTAL_TXNS)
        )
    )
    out.loc[low_recurrence_salary, "predicted_income_type"] = "other_irregular_income"
    out.loc[low_recurrence_salary, "income_label_rule"] = "low_recurrence_override"

    high_evidence_salary = (
        ~out["predicted_income_type"].eq("salary_income")
        & out["salary_promotion_candidate"]
        & ~out["sme_protection_flag"]
    )
    out.loc[high_evidence_salary, "predicted_income_type"] = "salary_income"
    out.loc[
        high_evidence_salary, "income_label_rule"
    ] = "high_recurrence_salary_override"

    if "confidence_score" in out.columns:
        out.loc[low_recurrence_salary, "confidence_score"] = out.loc[
            low_recurrence_salary, "confidence_score"
        ].clip(upper=0.50)
        out.loc[high_evidence_salary, "confidence_score"] = out.loc[
            high_evidence_salary, "confidence_score"
        ].clip(lower=0.60)
    return out


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
        "cluster_income_type",
        "predicted_income_type",
        "income_label_rule",
        "salary_evidence_score",
        "salary_evidence_flags",
        "confidence_score",
        "total_txns",
        "average_monthly_amount",
        "active_months",
        "average_txns_per_month",
        "sender_txn_share_in_account",
        "monthly_amount_cv",
    ]
    account_df = account_df.sort_values(
        ["predicted_income_type", "confidence_score", "total_txns"],
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
    if features.empty:
        raise ValueError(
            "No sender features could be built after cleaning input data. "
            "Check transaction_date/amount formats and null-heavy rows."
        )

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
    out = apply_income_evidence_rules(out)

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
        "cluster_income_type",
        "predicted_income_type",
        "income_label_rule",
        "salary_evidence_score",
        "salary_evidence_flags",
        "confidence_score",
        "total_txns",
        "average_monthly_amount",
        "active_months",
        "average_txns_per_month",
        "sender_txn_share_in_account",
        "monthly_amount_cv",
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
        x="average_txns_per_month",
        y="monthly_amount_cv",
        hue="predicted_income_type",
        size="total_txns",
        alpha=0.75,
    )
    plt.axhline(y=1.0, color="r", linestyle="--")
    plt.axvline(x=2.0, color="g", linestyle="--")
    plt.title("Monthly Frequency vs Amount Variability")
    plt.xlabel("Average Transactions per Month")
    plt.ylabel("Monthly Amount CV")
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
