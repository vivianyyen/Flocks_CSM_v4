"""
utils/supabase_client.py
────────────────────────
Handles all Supabase connectivity for the Incident Intel dashboard.

Add to .streamlit/secrets.toml:
    [supabase]
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]

Note: Do NOT add a `table` key — get_data() accepts the table name
      as an argument so each page can query its own table independently.
"""

import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo

TZ_MY = ZoneInfo("Asia/Kuala_Lumpur")  # GMT+8

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_client() -> tuple:
    """
    Create and cache the Supabase client (one connection per app session).
    Returns (client, status_string).
    status is "ok" on success, or a short error code string on failure.
    """
    if not SUPABASE_AVAILABLE:
        st.error(
            "❌ `supabase` package not installed. Run:\n"
            "```\npip install supabase\n```"
        )
        return None, "package_missing"

    # ── Validate secrets exist ────────────────────────────────────────────────
    if "supabase" not in st.secrets:
        st.error(
            "❌ **No `[supabase]` section found in secrets.**\n\n"
            "In Streamlit Community Cloud go to **App settings → Secrets** and add:\n"
            "```toml\n[supabase]\nurl = \"https://bjm...\"\nkey = \" sb_publishable_...\"\n```"
        )
        return None, "no_section"

    missing = [f for f in ("url", "key") if f not in st.secrets["supabase"]]
    if missing:
        st.error(f"❌ Missing secret fields: {missing}. Check your Secrets configuration.")
        return None, "missing_fields"

    url = st.secrets["supabase"]["url"].strip()
    key = st.secrets["supabase"]["key"].strip()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not url or "xxxxxxxxxxxx" in url:
        st.error("❌ Supabase URL looks like a placeholder. Paste your real project URL.")
        return None, "placeholder_url"
    if not url.startswith("https://"):
        st.error(f"❌ Supabase URL must start with `https://`. Got: `{url[:40]}`")
        return None, "bad_url"

    try:
        client = create_client(url, key)
        return client, "ok"
    except Exception as e:
        st.error(f"❌ Supabase client creation failed: {e}")
        return None, str(e)


# ── Data fetching ─────────────────────────────────────────────────────────────

def get_data(table: str, filters: dict | None = None) -> pd.DataFrame:
    """
    Fetch all rows from a Supabase table and return as a DataFrame.

    Parameters
    ----------
    table   : exact Supabase table name, e.g. "incidents" or "ransomware_victims"
    filters : optional dict of {column: value} equality filters

    Falls back to synthetic demo data ONLY for the "incidents" table when
    Supabase is genuinely unreachable, so the dashboard still renders.
    """
    # BUG FIX: removed the secrets table-name override that was forcing every
    # call (incidents AND ransomware_victims) to use the same table name.
    client, status = _get_client()   # BUG FIX: unpack tuple correctly

    if client is None:
        if table == "incidents":
            st.warning("⚠️ Using **demo data** (Supabase not connected). See error above.")
            return _demo_data()
        else:
            st.warning(f"⚠️ Could not connect to Supabase — no demo data for `{table}`.")
            return pd.DataFrame()

    try:
        query = client.table(table).select("*")
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        response = query.execute()

        if not response.data:
            st.info(
                f"ℹ️ Table `{table}` returned 0 rows. "
                "If you expect data, check that **Row Level Security (RLS)** allows SELECT "
                "for the `anon` role in Supabase → Authentication → Policies."
            )
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        df = _localise_timestamps(df)
        return df

    except Exception as e:
        err = str(e)
        if "permission denied" in err.lower() or "not found" in err.lower():
            st.error(
                f"❌ Supabase query failed on `{table}`: `{err}`\n\n"
                "**Check:** table name spelling, and that RLS policy allows anon SELECT."
            )
        else:
            st.error(f"❌ Supabase query error on `{table}`: {err}")

        if table == "incidents":
            st.warning("⚠️ Falling back to demo data.")
            return _demo_data()
        return pd.DataFrame()


# ── Write helpers ─────────────────────────────────────────────────────────────

def insert_row(table: str, row: dict) -> bool:
    """Insert a single row into a Supabase table."""
    client, status = _get_client()   # BUG FIX: was not unpacking the tuple
    if client is None:
        st.error(f"Cannot insert — Supabase not connected (status: {status})")
        return False
    try:
        client.table(table).insert(row).execute()
        return True
    except Exception as e:
        st.error(f"Insert failed on `{table}`: {e}")
        return False


def upsert_row(table: str, row: dict, on_conflict: str = "id") -> bool:
    """Upsert a single row into a Supabase table."""
    client, status = _get_client()
    if client is None:
        st.error(f"Cannot upsert — Supabase not connected (status: {status})")
        return False
    try:
        client.table(table).upsert(row, on_conflict=on_conflict).execute()
        return True
    except Exception as e:
        st.error(f"Upsert failed on `{table}`: {e}")
        return False


# ── Timezone helper ───────────────────────────────────────────────────────────

def _localise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all known datetime columns to Asia/Kuala_Lumpur (GMT+8)."""
    DATE_COLS = [
        "incident_date", "publication_date",
        "created_at", "updated_at",
        "date",          # ransomware_victims table
    ]
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col], errors="coerce", utc=True)
        df[col] = series.dt.tz_convert(TZ_MY)
    return df


# ── Demo / fallback data ──────────────────────────────────────────────────────

def _demo_data() -> pd.DataFrame:
    """Synthetic dataset used when Supabase is not reachable (incidents table only)."""
    import numpy as np
    from datetime import datetime, timedelta

    rng = np.random.default_rng(42)
    n   = 200

    categories     = ["Cybersecurity", "Financial Fraud", "Data Breach", "Misinformation", "Physical Security"]
    incident_types = ["Ransomware", "Phishing", "DDoS", "Insider Threat", "Supply Chain", "Zero-Day", "Social Engineering"]
    countries      = ["Malaysia", "United States", "China", "Germany", "United Kingdom", "Singapore", "Australia", "India"]
    impacts        = ["Critical", "High", "Medium", "Low"]
    sources        = ["Reuters", "BBC", "BleepingComputer", "Krebs on Security", "The Hacker News", "TechCrunch", "Wired"]
    entities       = ["Government", "Financial Institution", "Healthcare", "Technology Company", "Educational Institution", "Critical Infrastructure"]

    base_date = datetime.now() - timedelta(days=180)
    dates     = [base_date + timedelta(days=int(rng.integers(0, 180))) for _ in range(n)]

    summaries = [
        "A major ransomware attack disrupted hospital operations across multiple regions.",
        "State-sponsored hackers breached defence contractor networks stealing sensitive data.",
        "Phishing campaign targeted banking customers resulting in significant financial losses.",
        "Critical infrastructure vulnerability discovered in industrial control systems.",
        "Data leak exposed personal information of millions of users from social platform.",
    ]
    keywords_pool = [
        "ransomware, malware, encryption, bitcoin, hospital",
        "APT, espionage, zero-day, nation-state, defence",
        "phishing, credential, banking, social engineering",
        "ICS, SCADA, vulnerability, critical infrastructure, exploit",
        "data breach, PII, GDPR, social media, leak",
    ]

    return pd.DataFrame({
        "id":                range(1, n + 1),
        "title":             [f"Incident Report #{i}" for i in range(1, n + 1)],
        "publication_date":  dates,
        "source":            rng.choice(sources, n),
        "url":               [f"https://example.com/incident-{i}" for i in range(1, n + 1)],
        "summary":           rng.choice(summaries, n),
        "relevant_keywords": rng.choice(keywords_pool, n),
        "category":          rng.choice(categories, n),
        "country":           rng.choice(countries, n),
        "impact":            rng.choice(impacts, n, p=[0.1, 0.3, 0.4, 0.2]),
        "incident_type":     rng.choice(incident_types, n),
        "entity_affected":   rng.choice(entities, n),
        "incident_date":     dates,
    })