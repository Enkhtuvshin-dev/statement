from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
REQUIRED_COLUMNS = ["account_id", "sender", "amount", "transaction_date"]
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _connection_params_from_env() -> dict[str, str]:
    """
    Reads DB connection settings from env vars:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", ""),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def _build_query(
    source_table: str,
    date_column: str,
    tx_type_column: str,
    tx_type_value: str,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, dict[str, str]]:
    for value, name in (
        (source_table, "source_table"),
        (date_column, "date_column"),
        (tx_type_column, "tx_type_column"),
    ):
        if not IDENTIFIER_RE.match(value):
            raise ValueError(f"Invalid SQL identifier for {name}: {value}")

    params: dict[str, str] = {"tx_type": tx_type_value}
    where_parts = [f"{tx_type_column} = %(tx_type)s"]

    if start_date:
        where_parts.append(f"{date_column} >= %(start_date)s")
        params["start_date"] = start_date
    if end_date:
        where_parts.append(f"{date_column} <= %(end_date)s")
        params["end_date"] = end_date

    query = f"""
        SELECT account_id, sender, amount, {date_column} AS transaction_date
        FROM {source_table}
        WHERE {" AND ".join(where_parts)}
        ORDER BY account_id, sender, {date_column}
    """
    return query, params


def export_credit_transactions(
    output_csv: str,
    source_table: str = "transactions",
    date_column: str = "transaction_date",
    tx_type_column: str = "transaction_type",
    tx_type_value: str = "CREDIT",
    start_date: str | None = None,
    end_date: str | None = None,
    sample_accounts: int | None = None,
) -> pd.DataFrame:
    conn_params = _connection_params_from_env()
    missing = [k for k in ("dbname", "user", "password") if not conn_params[k]]
    if missing:
        names = ", ".join(
            {"dbname": "DB_NAME", "user": "DB_USER", "password": "DB_PASSWORD"}[k] for k in missing
        )
        raise ValueError(f"Missing required env vars: {names}")

    query, params = _build_query(
        source_table=source_table,
        date_column=date_column,
        tx_type_column=tx_type_column,
        tx_type_value=tx_type_value,
        start_date=start_date,
        end_date=end_date,
    )

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    if sample_accounts is not None and sample_accounts > 0 and not df.empty:
        keep = df["account_id"].drop_duplicates().head(sample_accounts)
        df = df[df["account_id"].isin(keep)].copy()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Exported rows: {len(df)}")
    print(f"Exported accounts: {df['account_id'].nunique() if not df.empty else 0}")
    print(f"Saved CSV: {output_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CREDIT transactions from local PostgreSQL to CSV for Colab."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/colab/credit_transactions.csv",
        help="Target CSV path.",
    )
    parser.add_argument("--source-table", type=str, default="transactions")
    parser.add_argument("--date-column", type=str, default="transaction_date")
    parser.add_argument("--tx-type-column", type=str, default="transaction_type")
    parser.add_argument("--tx-type-value", type=str, default="CREDIT")
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Optional YYYY-MM-DD")
    parser.add_argument(
        "--sample-accounts",
        type=int,
        default=None,
        help="Optional: export only first N accounts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_credit_transactions(
        output_csv=args.output_csv,
        source_table=args.source_table,
        date_column=args.date_column,
        tx_type_column=args.tx_type_column,
        tx_type_value=args.tx_type_value,
        start_date=args.start_date,
        end_date=args.end_date,
        sample_accounts=args.sample_accounts,
    )
