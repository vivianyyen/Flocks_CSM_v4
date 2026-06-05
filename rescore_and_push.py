"""
rescore_and_push.py
────────────────────────────────────────────────────────────────────────────────
Rescore BOTH global_news and incidents (Malaysia) tables in Supabase.

This script:
  1. Pulls every row from global_news AND incidents (paginated)
  2. Normalises column names so the scorer works on both tables
  3. Applies the custom risk formula from utils/risk_scorer.py
  4. Upserts risk_score + sub-scores + severity back to Supabase

Usage
-----
    # Score both tables (default):
    python rescore_and_push.py

    # Score only global table:
    python rescore_and_push.py --table global_news

    # Score only Malaysia table:
    python rescore_and_push.py --table incidents

    # Dry-run (print results, no DB write):
    python rescore_and_push.py --dry-run

Environment / secrets
---------------------
Set these in a .env file (or your shell) OR in .streamlit/secrets.toml.

    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=your-service-role-key     ← use service_role, NOT anon

────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import time
import pandas as pd
from supabase import create_client

# ── Try loading from .env (optional) ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Try loading from .streamlit/secrets.toml (optional) ──────────────────────
def _load_streamlit_secrets():
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return None, None

    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return None, None
    with open(secrets_path, "rb") as f:
        data = tomllib.load(f)
    sb = data.get("supabase", {})
    return sb.get("url"), sb.get("key")


def get_credentials():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        url, key = _load_streamlit_secrets()
    if not url or not key:
        print("❌  Could not find Supabase credentials.")
        print("    Set SUPABASE_URL and SUPABASE_KEY in .env or .streamlit/secrets.toml")
        sys.exit(1)
    return url, key


# ── Paginated fetch ───────────────────────────────────────────────────────────
def fetch_all(client, table: str, page_size: int = 1000) -> pd.DataFrame:
    all_rows, page = [], 0
    while True:
        start = page * page_size
        end   = start + page_size - 1
        try:
            resp  = client.table(table).select("*").range(start, end).execute()
            batch = resp.data or []
            all_rows.extend(batch)
            print(f"  Fetched page {page+1}: {len(batch)} rows")
            if len(batch) < page_size:
                break
            page += 1
        except Exception as e:
            print(f"  ⚠️  Error fetching page {page+1}: {e}")
            break
    return pd.DataFrame(all_rows)


# ── Column normalisation (mirrors application.py get_data logic) ──────────────
def normalise_columns(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """
    Ensure the scorer-required columns exist regardless of how the source
    table names them.  Applies to BOTH 'incidents' and 'global_news'.
    """
    if df.empty:
        return df

    df = df.copy()

    # incident_date  ─────────────────────────────────────────────────────────
    if "incident_date" not in df.columns:
        for alt in ("publication_date", "date", "created_at"):
            if alt in df.columns:
                df["incident_date"] = df[alt]
                break
        else:
            df["incident_date"] = pd.NaT

    # category  ──────────────────────────────────────────────────────────────
    if "category" not in df.columns:
        for alt in ("type", "threat_type", "incident_category"):
            if alt in df.columns:
                df["category"] = df[alt]
                break
        else:
            df["category"] = "Uncategorised"

    # incident_type  ─────────────────────────────────────────────────────────
    if "incident_type" not in df.columns:
        for alt in ("type", "attack_type", "threat_type"):
            if alt in df.columns:
                df["incident_type"] = df[alt]
                break
        else:
            df["incident_type"] = df.get("category", "Unknown")

    # impact  ────────────────────────────────────────────────────────────────
    if "impact" not in df.columns:
        for alt in ("criticality", "priority", "severity"):
            if alt in df.columns:
                df["impact"] = df[alt]
                break
        else:
            df["impact"] = "Unknown"

    # country  ───────────────────────────────────────────────────────────────
    if "country" not in df.columns:
        for alt in ("nation", "location", "geo", "region"):
            if alt in df.columns:
                df["country"] = df[alt]
                break
        else:
            df["country"] = "Unknown"

    # sector  ────────────────────────────────────────────────────────────────
    # global_news often stores the sector in category; fall back gracefully
    if "sector" not in df.columns:
        for alt in ("industry", "vertical", "category"):
            if alt in df.columns:
                df["sector"] = df[alt]
                break
        else:
            df["sector"] = "Unknown"

    # entity_affected  ───────────────────────────────────────────────────────
    if "entity_affected" not in df.columns:
        for alt in ("entity", "target", "victim", "organization"):
            if alt in df.columns:
                df["entity_affected"] = df[alt]
                break

    return df


# ── Build the upsert payload for one scored row ───────────────────────────────
def _build_payload(row: pd.Series) -> dict:
    """
    Extract only the columns we want to write back to Supabase.
    Falls back gracefully if the scorer didn't produce a sub-score column.
    """
    payload: dict = {}

    # Core score
    if "risk_score" in row.index and pd.notna(row["risk_score"]):
        payload["risk_score"] = float(row["risk_score"])

    # Sub-scores (may not exist for every scorer version)
    for col in ("sector_score", "country_score", "attack_type_score", "data_exposure_score"):
        if col in row.index and pd.notna(row[col]):
            payload[col] = float(row[col])

    # Derived text fields
    for col in ("severity", "attack_class"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip() not in ("", "nan", "None"):
            payload[col] = str(row[col])

    # impact — write back only if scorer produced one and table already has the col
    if "impact" in row.index and pd.notna(row["impact"]) and str(row["impact"]) not in ("", "nan", "None"):
        payload["impact"] = str(row["impact"])

    return payload


# ── Process a single table ────────────────────────────────────────────────────
def rescore_table(
    client,
    table_name: str,
    id_col: str,
    batch_size: int,
    dry_run: bool,
    score_dataframe,
    build_update_payload=None,          # optional — we use _build_payload instead
) -> dict:
    """Fetch → normalise → score → upsert one Supabase table."""

    print(f"\n{'─'*70}")
    print(f"  Processing table : {table_name}")
    print(f"{'─'*70}")

    # ── 1. Fetch ───────────────────────────────────────────────────────────────
    print(f"[1/4] Fetching rows from '{table_name}'…")
    df = fetch_all(client, table_name)
    if df.empty:
        print(f"  ⚠️  No rows found in '{table_name}'. Skipping.\n")
        return {"table": table_name, "success": 0, "errors": 0, "skipped": True}
    print(f"  ✓  {len(df)} total rows loaded\n")

    # ── 2. Normalise columns ───────────────────────────────────────────────────
    print(f"[2/4] Normalising columns for '{table_name}'…")
    df_norm = normalise_columns(df, table_name)
    print(f"  ✓  Columns ready: {[c for c in ('category','incident_type','impact','country','sector') if c in df_norm.columns]}\n")

    # ── 3. Score ───────────────────────────────────────────────────────────────
    print(f"[3/4] Applying risk formula to '{table_name}'…")
    t0 = time.time()
    try:
        df_scored = score_dataframe(df_norm)
    except Exception as e:
        print(f"  ❌  score_dataframe raised an error: {e}")
        return {"table": table_name, "success": 0, "errors": len(df), "score_error": True}
    elapsed = time.time() - t0
    print(f"  ✓  Scored {len(df_scored)} rows in {elapsed:.2f}s\n")

    # Severity distribution summary
    if "severity" in df_scored.columns:
        dist = df_scored["severity"].value_counts()
        print(f"  Severity distribution ({table_name}):")
        for lvl in ["Critical", "High", "Medium", "Low"]:
            n   = dist.get(lvl, 0)
            pct = n / len(df_scored) * 100 if len(df_scored) > 0 else 0
            bar = "█" * int(pct / 3)
            print(f"    {lvl:<10} {bar:<35} {n:>5} ({pct:5.1f}%)")
    elif "risk_score" in df_scored.columns:
        # Fallback: derive severity from risk_score for display only
        def _sev(s):
            try:
                v = float(s)
                if v > 0.8: return "Critical"
                if v > 0.6: return "High"
                if v > 0.4: return "Medium"
                return "Low"
            except: return "Unknown"
        dist = df_scored["risk_score"].apply(_sev).value_counts()
        print(f"  Risk score distribution ({table_name}):")
        for lvl in ["Critical", "High", "Medium", "Low"]:
            n   = dist.get(lvl, 0)
            pct = n / len(df_scored) * 100 if len(df_scored) > 0 else 0
            bar = "█" * int(pct / 3)
            print(f"    {lvl:<10} {bar:<35} {n:>5} ({pct:5.1f}%)")
    else:
        print(f"  ⚠️  No risk_score or severity column produced by scorer — check utils/risk_scorer.py")
    print()

    # Dry-run exit
    if dry_run:
        preview_cols = [id_col, "severity", "risk_score", "sector_score",
                        "country_score", "attack_type_score", "data_exposure_score", "attack_class"]
        show = [c for c in preview_cols if c in df_scored.columns]
        print(f"[DRY RUN] Sample of scored rows from '{table_name}':")
        print(df_scored[show].head(10).to_string(index=False))
        print(f"\n  Dry run complete for '{table_name}' — no changes written.\n")
        return {"table": table_name, "success": 0, "errors": 0, "dry_run": True}

    # ── 4. Upsert ──────────────────────────────────────────────────────────────
    if id_col not in df_scored.columns:
        print(f"  ❌  ID column '{id_col}' not found in '{table_name}'.")
        print(f"      Available columns: {list(df_scored.columns)}")
        print(f"  Skipping '{table_name}'.\n")
        return {"table": table_name, "success": 0, "errors": len(df_scored), "id_col_missing": True}

    # Use provided build_update_payload if available, else fall back to ours
    _payload_fn = build_update_payload if callable(build_update_payload) else _build_payload

    print(f"[4/4] Upserting scored rows to '{table_name}' (batch={batch_size})…")
    records = []
    for _, row in df_scored.iterrows():
        payload = _payload_fn(row)
        if not payload:
            # Nothing to write for this row (scorer produced no score columns)
            continue
        payload[id_col] = row[id_col]   # include PK for upsert
        records.append(payload)

    if not records:
        print(f"  ⚠️  No payloads built — scorer may not be producing score columns.\n")
        return {"table": table_name, "success": 0, "errors": 0, "no_payload": True}

    success, errors = 0, 0
    total_batches = (len(records) + batch_size - 1) // batch_size

    for i in range(0, len(records), batch_size):
        batch_num = i // batch_size + 1
        batch     = records[i : i + batch_size]
        try:
            client.table(table_name).upsert(batch, on_conflict=id_col).execute()
            success += len(batch)
            print(f"  Batch {batch_num}/{total_batches}: ✓  {len(batch)} rows upserted")
        except Exception as e:
            errors += len(batch)
            print(f"  Batch {batch_num}/{total_batches}: ✗  Error — {e}")

    print()
    return {"table": table_name, "success": success, "errors": errors}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Rescore cyber incidents across Supabase tables (global_news + incidents)"
    )
    parser.add_argument(
        "--table",
        help="Specific table to score. Choices: global_news, incidents. Default: both.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print scores without writing to DB",
    )
    parser.add_argument(
        "--id-col", default="id",
        help="Primary key column name (default: id)",
    )
    parser.add_argument(
        "--batch", type=int, default=50,
        help="Upsert batch size (default: 50)",
    )
    args = parser.parse_args()

    # ── Import scorer ──────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from utils.risk_scorer import score_dataframe
    except ImportError as e:
        print(f"❌  Could not import utils.risk_scorer: {e}")
        print("    Make sure utils/risk_scorer.py exists in your project directory.")
        sys.exit(1)

    # build_update_payload is optional — older versions of risk_scorer may not export it
    try:
        from utils.risk_scorer import build_update_payload as _ext_payload
    except ImportError:
        _ext_payload = None
        print("  ℹ️  build_update_payload not found in risk_scorer — using built-in payload builder.\n")

    # ── Banner ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Cyber Threat Intelligence — Multi-Table Risk Rescoring Engine")
    print(f"{'='*70}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  ID col   : {args.id_col}")
    print(f"  Batch sz : {args.batch}")

    url, key = get_credentials()
    client   = create_client(url, key)

    # ── Determine which tables to process ─────────────────────────────────────
    TABLES = ["global_news", "incidents"]

    if args.table:
        if args.table not in TABLES:
            print(f"\n⚠️  Unknown table '{args.table}'. Valid options: {TABLES}")
            print("   Processing it anyway — make sure the table exists in Supabase.\n")
        tables_to_process = [args.table]
        print(f"  Tables   : {args.table} (single)\n")
    else:
        tables_to_process = TABLES
        print(f"  Tables   : {', '.join(TABLES)} (all)\n")

    # ── Process each table ────────────────────────────────────────────────────
    results = []
    for table_name in tables_to_process:
        result = rescore_table(
            client=client,
            table_name=table_name,
            id_col=args.id_col,
            batch_size=args.batch,
            dry_run=args.dry_run,
            score_dataframe=score_dataframe,
            build_update_payload=_ext_payload,
        )
        results.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")

    total_success = 0
    total_errors  = 0

    for res in results:
        table   = res["table"]
        success = res.get("success", 0)
        errors  = res.get("errors",  0)
        total_success += success
        total_errors  += errors

        if res.get("dry_run"):
            print(f"  ○ {table:<22} DRY RUN — no changes made")
        elif res.get("skipped"):
            print(f"  ⊘ {table:<22} SKIPPED — no data found")
        elif res.get("id_col_missing"):
            print(f"  ✗ {table:<22} ERROR — ID column '{args.id_col}' not found")
        elif res.get("score_error"):
            print(f"  ✗ {table:<22} ERROR — scorer raised an exception")
        elif res.get("no_payload"):
            print(f"  ⚠ {table:<22} WARNING — scorer produced no score columns")
        else:
            icon = "✓" if errors == 0 else "⚠"
            print(f"  {icon} {table:<22} {success:>6} updated,  {errors:>4} errors")

    print(f"{'─'*70}")
    print(f"  TOTAL : {total_success} rows updated,  {total_errors} errors")
    print(f"{'='*70}\n")

    # ── Migration hint on errors ───────────────────────────────────────────────
    if total_errors > 0 and not args.dry_run:
        print("  ⚠️  Some rows failed. Common causes:")
        print("     1. Score columns don't exist yet in the table.")
        print("     2. You used the anon key instead of the service_role key.")
        print()
        print("  Run this SQL ONCE in Supabase → SQL Editor for each affected table:\n")
        for table in tables_to_process:
            print(_migration_sql(table))
            print()


def _migration_sql(table: str) -> str:
    return f"""-- Migration: add score columns to {table}
-- Run this ONCE in Supabase → SQL Editor before pushing scores:
ALTER TABLE public.{table}
  ADD COLUMN IF NOT EXISTS risk_score          FLOAT,
  ADD COLUMN IF NOT EXISTS sector_score        FLOAT,
  ADD COLUMN IF NOT EXISTS country_score       FLOAT,
  ADD COLUMN IF NOT EXISTS attack_type_score   FLOAT,
  ADD COLUMN IF NOT EXISTS data_exposure_score FLOAT,
  ADD COLUMN IF NOT EXISTS attack_class        TEXT,
  ADD COLUMN IF NOT EXISTS severity            TEXT,
  ADD COLUMN IF NOT EXISTS impact              TEXT;
"""


if __name__ == "__main__":
    main()