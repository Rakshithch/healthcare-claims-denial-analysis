"""
Load the synthetic CSVs into Snowflake RAW tables.

Prereqs:
  1. Run sql/snowflake_setup.sql once (creates DB, schemas, tables, stage, views).
  2. pip install "snowflake-connector-python[pandas]"
  3. Export credentials as env vars (never hardcode):
       SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
       SNOWFLAKE_WAREHOUSE (default COMPUTE_WH), SNOWFLAKE_ROLE (optional)

Usage:
  python scripts/load_to_snowflake.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
DATABASE = "HEALTHCARE_CLAIMS_DENIAL"

# (csv file, target table) pairs loaded into the RAW schema
LOADS = [
    ("claims_50k.csv", "CLAIMS"),
    ("providers.csv", "PROVIDERS"),
    ("payers.csv", "PAYERS"),
]


def get_connection():
    """Build a Snowflake connection from environment variables."""
    try:
        import snowflake.connector
    except ImportError:
        sys.exit("Install the connector first: pip install 'snowflake-connector-python[pandas]'")

    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        sys.exit(f"Missing required env vars: {', '.join(missing)}")

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        database=DATABASE,
        schema="RAW",
    )


def main() -> None:
    from snowflake.connector.pandas_tools import write_pandas

    conn = get_connection()
    try:
        for csv_name, table in LOADS:
            path = DATA_DIR / csv_name
            if not path.exists():
                sys.exit(f"{path} not found — run scripts/generate_synthetic_data.py first")

            df = pd.read_csv(path)
            # Snowflake convention: uppercase column names
            df.columns = [c.upper() for c in df.columns]

            print(f"Loading {len(df):,} rows into RAW.{table} ...")
            ok, _, nrows, _ = write_pandas(
                conn, df, table_name=table, database=DATABASE, schema="RAW",
                overwrite=True, auto_create_table=False,
            )
            print(f"  {'OK' if ok else 'FAILED'} — {nrows:,} rows written")
    finally:
        conn.close()
    print("Done. Run dbt next: cd dbt_project && dbt run && dbt test")


if __name__ == "__main__":
    main()
