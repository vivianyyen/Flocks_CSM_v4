"""
crawlers/ransomware_crawler.py
─────────────────────────────
Fetches recent ransomware victims from ransomware.live and upserts them
into the Supabase `ransomware_victims` table.

Usage
─────
    python crawlers/ransomware_crawler.py

Environment variables required (set in .env or your task scheduler):
    SUPABASE_URL   = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY   = st.secrets["SUPABASE_KEY"]

Schedule recommendation: every 4–6 hours
    • Flocks: set cron expression  0 */4 * * *
    • GitHub Actions: schedule: cron: '0 */4 * * *'
"""

import os
import sys
import logging
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
TABLE        = "ransomware_victims"

# ransomware.live public API endpoints
API_RECENT = "https://api.ransomware.live/recentvictims"
API_ALL    = "https://api.ransomware.live/victims"

HEADERS      = {"User-Agent": "IncidentIntelBot/1.0 (+https://github.com/your-org/incident-intel)"}
TIMEOUT      = 30   # seconds


# ── Supabase helpers ──────────────────────────────────────────────────────────

def _sb_headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates",   # upsert behaviour
    }


def upsert_rows(rows: list[dict]) -> bool:
    """Upsert a list of rows into Supabase via the REST API."""
    if not rows:
        log.info("No rows to upsert.")
        return True

    url = f"{SUPABASE_URL}/rest/v1/{TABLE}"
    resp = requests.post(url, json=rows, headers=_sb_headers(), timeout=TIMEOUT)

    if resp.status_code in (200, 201):
        log.info(f"✅ Upserted {len(rows)} rows.")
        return True
    else:
        log.error(f"❌ Upsert failed {resp.status_code}: {resp.text[:300]}")
        return False


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_recent_victims() -> list[dict]:
    """Fetch recent victims from ransomware.live API."""
    log.info(f"Fetching from {API_RECENT} …")
    resp = requests.get(API_RECENT, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    log.info(f"Fetched {len(data)} raw records.")
    return data


def normalise(raw: dict) -> dict:
    """
    Map ransomware.live API fields → Supabase table columns
    """

        # ── Severity Logic ─────────────────────────────
    sector = (
        raw.get("activity")
        or raw.get("sector")
        or "Unknown"
    )

    high_risk_sectors = [
        "Government",
        "Healthcare",
        "Energy",
        "Finance",
        "Transportation/Logistics",
        "Telecommunications"
    ]

    if sector in high_risk_sectors:
        severity = "High"
    elif sector == "Unknown":
        severity = "Low"
    else:
        severity = "Medium"

    def _ts(val):
        """Return ISO-8601 string or None."""
        if not val:
            return None

        try:
            dt = datetime.fromisoformat(
                val.replace("Z", "+00:00")
            )

            return dt.astimezone(
                timezone.utc
            ).isoformat()

        except Exception:
            return None
        
         # ── Match your Supabase table columns ───────────────────────────────
    return {

        "organization": (
            raw.get("post_title")
            or raw.get("victim")
            or "Unknown Victim"
        )[:500],

        "threat_actor": (
            raw.get("threat_actor")
            or "Unknown"
        )[:100],

        "country": (
            raw.get("country")
            or "Unknown"
        )[:100],

        "sector": (
            raw.get("activity")
            or raw.get("sector")
            or "Unknown"
        )[:200],

         "severity": severity,

        "date": _ts(
            raw.get("discovered")
            or raw.get("published")
        )
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Validate config
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set as environment variables.")
        sys.exit(1)

    # Fetch
    try:
        raw_list = fetch_recent_victims()
    except requests.RequestException as e:
        log.error(f"API request failed: {e}")
        sys.exit(1)

    # Normalise
    rows = [normalise(r) for r in raw_list]
    # Drop rows with no post_url (can't dedup)
    rows = [r for r in rows if r.get("post_url")]
    log.info(f"Normalised {len(rows)} rows (with post_url).")

    # Upsert
    ok = upsert_rows(rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
