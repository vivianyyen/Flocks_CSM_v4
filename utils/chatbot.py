"""
utils/chatbot.py
─────────────────────────────────────────────────────────────────────────────
Hybrid AI Analyst: Groq (LLaMA 3.3-70b) + ThreatBook CTI enrichment.

Flow
────
1. User sends a message.
2. IOC extractor scans the message for IPs, domains, file hashes, URLs, CVEs.
3. For each IOC found → silently call the matching ThreatBook API endpoint.
4. Inject enriched CTI data into Groq's system prompt.
5. Groq generates a clear, analyst-style response backed by real intel.

Secrets required (.streamlit/secrets.toml)
───────────────────────────────────────────
[groq]
api_key = "gsk_..."

[threatbook]
api_key = "your-threatbook-api-key"
"""

import re
import json
import hashlib
import requests
import streamlit as st
import pandas as pd

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  ThreatBook API helpers
# ─────────────────────────────────────────────────────────────

TB_BASE = "https://api.threatbook.io"
TB_TIMEOUT = 10  # seconds per request


def _tb_key() -> str | None:
    """Return ThreatBook API key from secrets, or None."""
    try:
        return st.secrets["threatbook"]["api_key"]
    except Exception:
        return None


def _tb_ip(ip: str, api_key: str) -> dict:
    """Call ThreatBook IP Report endpoint, with community fallback."""
    # Try the paid endpoint first, then fall back to community
    endpoints = [
        ("GET", f"{TB_BASE}/v1/ip/query"),
        ("GET", f"{TB_BASE}/v1/community/ip"),
    ]
    last_err = {}
    for method, url in endpoints:
        try:
            r = requests.request(
                method, url,
                params={"apikey": api_key, "resource": ip},
                timeout=TB_TIMEOUT,
            )
            data = r.json() if r.status_code == 200 else {}
            if not data:
                last_err = {"error": r.text[:200]}
                continue
            rc = data.get("response_code", 0)
            if rc == 401:
                last_err = {"error": f"API key has no access to {url} (401)"}
                continue
            if rc == 405:
                last_err = {"error": f"Invalid API method for {url} (405)"}
                continue
            return data
        except Exception as e:
            last_err = {"error": str(e)}
    return last_err


def _tb_domain(domain: str, api_key: str) -> dict:
    """Call ThreatBook Domain Intelligence endpoint."""
    # Try POST first (required by ThreatBook v2), then GET as fallback
    endpoints = [
        ("POST", f"{TB_BASE}/v1/domain/query"),
        ("GET",  f"{TB_BASE}/v2/domain/report"),
        ("POST", f"{TB_BASE}/v2/domain/report"),
    ]
    last_err = {}
    for method, url in endpoints:
        try:
            r = requests.request(
                method, url,
                params={"apikey": api_key, "resource": domain},
                timeout=TB_TIMEOUT,
            )
            data = r.json() if r.status_code == 200 else {}
            if not data:
                last_err = {"error": r.text[:200]}
                continue
            rc = data.get("response_code", 0)
            if rc in (401, 405):
                last_err = {"error": f"response_code:{rc} from {url}"}
                continue
            return data
        except Exception as e:
            last_err = {"error": str(e)}
    return last_err


def _tb_file(hash_val: str, api_key: str) -> dict:
    """Call ThreatBook File Intelligence endpoint."""
    try:
        r = requests.post(
            f"{TB_BASE}/v2/file/report",
            params={"apikey": api_key, "resource": hash_val},
            timeout=TB_TIMEOUT,
        )
        return r.json() if r.status_code == 200 else {"error": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def _tb_url(url_val: str, api_key: str) -> dict:
    """Call ThreatBook URL Intelligence endpoint."""
    try:
        r = requests.post(
            f"{TB_BASE}/v2/url/report",
            params={"apikey": api_key, "resource": url_val},
            timeout=TB_TIMEOUT,
        )
        return r.json() if r.status_code == 200 else {"error": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def _tb_vuln(cve: str, api_key: str) -> dict:
    """Call ThreatBook Vulnerability Intelligence endpoint."""
    # ThreatBook uses different param names and paths depending on API tier.
    # Try multiple combinations of endpoint + method + param key.
    attempts = [
        ("GET",  f"{TB_BASE}/v1/vuln/info",   {"apikey": api_key, "cve_id": cve}),
        ("POST", f"{TB_BASE}/v1/vuln/info",   {"apikey": api_key, "cve_id": cve}),
        ("GET",  f"{TB_BASE}/v1/vuln/info",   {"apikey": api_key, "resource": cve}),
        ("GET",  f"{TB_BASE}/v2/vuln/report", {"apikey": api_key, "cve_id": cve}),
        ("POST", f"{TB_BASE}/v2/vuln/report", {"apikey": api_key, "cve_id": cve}),
        ("GET",  f"{TB_BASE}/v2/vuln/report", {"apikey": api_key, "resource": cve}),
    ]
    last_err = {}
    for method, url, params in attempts:
        try:
            r = requests.request(method, url, params=params, timeout=TB_TIMEOUT)
            if r.status_code != 200:
                last_err = {"error": f"HTTP {r.status_code} from {url}: {r.text[:150]}"}
                continue
            data = r.json()
            rc = data.get("response_code", 0)
            if rc in (401, 405):
                last_err = {"error": f"response_code:{rc} ({data.get('msg', '')}) from {url}"}
                continue
            if rc == 0 or data.get("data"):
                return data
            last_err = {"error": f"response_code:{rc} ({data.get('msg', '')}) from {url}"}
        except Exception as e:
            last_err = {"error": str(e)}
    return last_err


# ─────────────────────────────────────────────────────────────
#  IOC extractor
# ─────────────────────────────────────────────────────────────

_RE_IPV4    = re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b')
_RE_IPV6    = re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b')
_RE_DOMAIN  = re.compile(r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|gov|edu|mil|int|co|uk|de|fr|jp|cn|ru|br|in|au|ca|my|sg|id|th|vn|ph|info|biz|xyz|online|site|tech|app|dev|cloud)\b', re.IGNORECASE)
_RE_MD5     = re.compile(r'\b[0-9a-fA-F]{32}\b')
_RE_SHA1    = re.compile(r'\b[0-9a-fA-F]{40}\b')
_RE_SHA256  = re.compile(r'\b[0-9a-fA-F]{64}\b')
_RE_URL     = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)
_RE_CVE     = re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE)

# Private/reserved IP ranges to skip enriching
_PRIVATE_CIDRS = [
    re.compile(r'^10\.'), re.compile(r'^192\.168\.'),
    re.compile(r'^172\.(1[6-9]|2\d|3[01])\.'),
    re.compile(r'^127\.'), re.compile(r'^0\.'), re.compile(r'^255\.'),
    re.compile(r'^169\.254\.'),
]

def _is_private_ip(ip: str) -> bool:
    return any(p.match(ip) for p in _PRIVATE_CIDRS)


def extract_iocs(text: str) -> dict:
    """
    Extract all IOC types from a text string.
    Returns dict: {type: [list of unique values]}
    """
    urls     = list(dict.fromkeys(_RE_URL.findall(text)))
    # Extract IPs but skip private ranges; deduplicate
    ips_raw  = _RE_IPV4.findall(text) + _RE_IPV6.findall(text)
    ips      = list(dict.fromkeys(ip for ip in ips_raw if not _is_private_ip(ip)))
    # Extract domains but skip those that look like part of a URL already captured
    url_hosts = set()
    for u in urls:
        m = re.match(r'https?://([^/]+)', u)
        if m:
            url_hosts.add(m.group(1).lower().split(':')[0])
    domains  = [d for d in list(dict.fromkeys(_RE_DOMAIN.findall(text)))
                if d.lower() not in url_hosts]
    sha256   = list(dict.fromkeys(_RE_SHA256.findall(text)))
    sha1     = list(dict.fromkeys(h for h in _RE_SHA1.findall(text) if h not in sha256))
    md5      = list(dict.fromkeys(h for h in _RE_MD5.findall(text) if h not in sha256 and h not in sha1))
    hashes   = sha256 + sha1 + md5
    cves     = list(dict.fromkeys(_RE_CVE.findall(text)))

    return {
        "ips":     ips[:3],      # cap to 3 per type to respect rate limits
        "domains": domains[:3],
        "hashes":  hashes[:3],
        "urls":    urls[:3],
        "cves":    cves[:3],
    }


# ─────────────────────────────────────────────────────────────
#  ThreatBook enrichment orchestrator
# ─────────────────────────────────────────────────────────────

def _summarise_ip_report(data: dict) -> str:
    """Convert raw ThreatBook IP report JSON into a compact analyst summary."""
    if "error" in data:
        return f"(lookup failed: {data['error']})"
    rc = data.get("response_code")
    if rc is not None and rc != 0:
        msg = data.get("msg", "unknown error")
        return f"(ThreatBook error {rc}: {msg})"
    d = data.get("data", {})
    summary = d.get("summary", {})
    basic   = d.get("basic", {})
    loc     = basic.get("location", {})
    asn     = d.get("asn", {})
    judgments = summary.get("judgments", [])
    whitelist = summary.get("whitelist", False)
    parts = []
    if judgments:
        parts.append(f"Threat labels: {', '.join(judgments)}")
    elif whitelist:
        parts.append("Verdict: Whitelisted / benign")
    else:
        parts.append("Verdict: No known malicious labels")
    if loc.get("country"):
        parts.append(f"Location: {loc['country']}{', ' + loc['city'] if loc.get('city') else ''}")
    if asn.get("info"):
        parts.append(f"ASN: {asn['info']}")
    if summary.get("first_seen"):
        parts.append(f"First seen: {summary['first_seen']} | Last seen: {summary.get('last_seen','?')}")
    ports = [str(p.get("port","")) for p in d.get("ports", [])[:5]]
    if ports:
        parts.append(f"Open ports: {', '.join(ports)}")
    return " | ".join(parts)


def _summarise_generic(data: dict, ioc_type: str) -> str:
    """Generic summariser for domain/file/URL/vuln responses."""
    if "error" in data:
        return f"(lookup failed: {data['error']})"
    rc = data.get("response_code")
    if rc is not None and rc != 0:
        msg = data.get("msg", "unknown error")
        return f"(ThreatBook error {rc}: {msg})"
    # Just return a compact JSON excerpt of the most useful fields
    d = data.get("data", data)
    # Pull out any top-level verdict/label/summary fields
    keys_of_interest = ["judgments", "verdict", "severity", "whitelist",
                        "tags", "malware_families", "threat_level",
                        "score", "summary", "description", "cvss",
                        "first_seen", "last_seen", "location"]
    snippet = {k: d[k] for k in keys_of_interest if k in d and d[k]}
    if not snippet and isinstance(d, dict):
        # Fallback: first 5 keys
        snippet = dict(list(d.items())[:5])
    return json.dumps(snippet, ensure_ascii=False)[:600]


def enrich_with_threatbook(iocs: dict) -> str:
    """
    Call ThreatBook for each extracted IOC.
    Returns a formatted string block to inject into the system prompt.
    """
    api_key = _tb_key()
    if not api_key:
        return ""
    if not any(iocs.values()):
        return ""

    lines = ["=== THREATBOOK CTI ENRICHMENT ==="]

    for ip in iocs.get("ips", []):
        raw = _tb_ip(ip, api_key)
        lines.append(f"IP {ip}: {_summarise_ip_report(raw)}")

    for domain in iocs.get("domains", []):
        raw = _tb_domain(domain, api_key)
        lines.append(f"Domain {domain}: {_summarise_generic(raw, 'domain')}")

    for h in iocs.get("hashes", []):
        raw = _tb_file(h, api_key)
        lines.append(f"Hash {h[:16]}…: {_summarise_generic(raw, 'file')}")

    for url in iocs.get("urls", []):
        raw = _tb_url(url, api_key)
        lines.append(f"URL {url[:60]}…: {_summarise_generic(raw, 'url')}")

    for cve in iocs.get("cves", []):
        raw = _tb_vuln(cve, api_key)
        lines.append(f"Vuln {cve}: {_summarise_generic(raw, 'vuln')}")

    lines.append("=== END CTI ENRICHMENT ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────

def _build_system_prompt(df: pd.DataFrame, cti_block: str = "") -> str:
    lines = [
        "You are a senior cybersecurity and threat intelligence analyst embedded in a live SOC dashboard.",
        "You have access to ThreatBook CTI — a commercial-grade threat intelligence platform.",
        "",
        "RESPONSE FORMAT (follow exactly):",
        "- Write in clear, professional English suitable for a security operations briefing.",
        "- Begin every response with one sentence summarising the key finding.",
        "- Use proper markdown: **bold** for key terms, ### headers for 3+ topic responses.",
        "- Use numbered lists for ranked items; bullet lists for non-ordered items.",
        "- Separate sections with a blank line for readability.",
        "- Keep responses under 400 words unless more detail is explicitly requested.",
        "- End with a **Recommendation:** line when actionable advice is relevant.",
        "",
        "CONTENT RULES:",
        "- Answer accurately using the dashboard data context provided below.",
        "- If ThreatBook enrichment data is present, use it for specific, accurate verdicts.",
        "- For IPs/domains/hashes, always state the threat verdict, location, and key labels.",
        "- For CVEs, always mention the CVSS score and affected systems if available.",
        "- When analysing trends, reason step by step before stating conclusions.",
        "",
        "=== DASHBOARD DATA SUMMARY ===",
        f"Total incidents: {len(df)}",
    ]

    def top5(col):
        if col in df.columns:
            return df[col].value_counts().head(5).to_dict()
        return {}

    for label, col in [
        ("Top categories",        "category"),
        ("Top incident types",    "incident_type"),
        ("Top countries",         "country"),
        ("Impact breakdown",      "impact"),
        ("Top sources",           "source"),
        ("Top entities affected", "entity_affected"),
    ]:
        d = top5(col)
        if d:
            lines.append(f"{label}: {json.dumps(d)}")

    if "incident_date" in df.columns:
        dated = df.dropna(subset=["incident_date"])
        if not dated.empty:
            lines.append(
                f"Date range: {dated['incident_date'].min().date()} "
                f"to {dated['incident_date'].max().date()} (GMT+8)"
            )

    lines.append("=== END DASHBOARD DATA ===")

    if cti_block:
        lines += ["", cti_block]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Groq client (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_groq_client(_bust=None):
    if not GROQ_AVAILABLE:
        return None, "package_missing"
    if "groq" not in st.secrets or "api_key" not in st.secrets["groq"]:
        return None, "missing_key"
    try:
        return Groq(api_key=st.secrets["groq"]["api_key"]), "ok"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────────────────────

def chatbot_ui(df: pd.DataFrame):
    # ── Session state ──────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False

    tb_key_present = bool(_tb_key())

    # ── Global CSS ─────────────────────────────────────────────
    st.markdown("""
    <style>
    #groq-fab-anchor {
        position: fixed; bottom: 28px; right: 28px; z-index: 9999;
    }
    #groq-chat-panel {
        position: fixed; bottom: 100px; right: 28px;
        width: 420px; max-height: 640px;
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 16px; box-shadow: 0 12px 48px rgba(0,0,0,0.65);
        z-index: 9998; display: flex; flex-direction: column;
        overflow: hidden; font-family: 'IBM Plex Sans', sans-serif;
    }
    #groq-chat-panel .gcp-header {
        background: linear-gradient(135deg,#161b22,#1c2128);
        border-bottom: 1px solid #21262d;
        padding: 14px 16px; display: flex; align-items: center;
        gap: 10px; flex-shrink: 0;
    }
    #groq-chat-panel .gcp-avatar {
        width:34px; height:34px;
        background: linear-gradient(135deg,#cc0000,#f55036);
        border-radius:50%; display:flex; align-items:center;
        justify-content:center; font-size:16px; flex-shrink:0;
    }
    #groq-chat-panel .gcp-title  { font-weight:600; font-size:14px; color:#f0f6fc; }
    #groq-chat-panel .gcp-sub    { font-size:11px; color:#8b949e; margin-top:1px; }
    .gcp-tb-badge {
        margin-left:auto; font-size:10px; font-weight:600;
        padding:3px 8px; border-radius:20px;
        background: #0d2a1a; color:#3fb950; border:1px solid #238636;
    }
    .gcp-tb-badge.off {
        background:#1c1c1c; color:#484f58; border-color:#30363d;
    }
    #groq-chat-panel .gcp-body {
        flex:1; overflow-y:auto; padding:14px;
        display:flex; flex-direction:column; gap:10px;
        scrollbar-width:thin; scrollbar-color:#30363d transparent;
    }
    #groq-chat-panel .gcp-body::-webkit-scrollbar { width:4px; }
    #groq-chat-panel .gcp-body::-webkit-scrollbar-thumb { background:#30363d; border-radius:4px; }
    .gc-msg { display:flex; flex-direction:column; max-width:90%; }
    .gc-msg.user { align-self:flex-end; align-items:flex-end; }
    .gc-msg.bot  { align-self:flex-start; align-items:flex-start; }
    .gc-role { font-size:10px; font-weight:600; letter-spacing:.07em;
               text-transform:uppercase; margin-bottom:3px; }
    .gc-msg.user .gc-role { color:#388bfd; }
    .gc-msg.bot  .gc-role { color:#cc0000; }
    .gc-bubble { padding:10px 13px; border-radius:14px;
                 font-size:13px; line-height:1.6; color:#c9d1d9; }
    .gc-msg.user .gc-bubble { background:#1f3349; border:1px solid #2d4a6e; border-bottom-right-radius:4px; }
    .gc-msg.bot  .gc-bubble { background:#161b22; border:1px solid #21262d; border-bottom-left-radius:4px; }
    /* Native markdown rendered bot responses */
    .gc-bot-md + div, .gc-bot-md ~ div > div {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-bottom-left-radius: 4px !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
        font-size: 13px !important;
        line-height: 1.7 !important;
        color: #c9d1d9 !important;
        max-width: 90% !important;
        margin: 0 0 8px 0 !important;
    }
    /* Style markdown elements inside bot bubble */
    #groq-chat-panel .stMarkdown p { margin: 0 0 6px 0; color: #c9d1d9; font-size: 13px; line-height: 1.7; }
    #groq-chat-panel .stMarkdown h3 { font-size: 12px; font-weight: 700; color: #f0f6fc;
        text-transform: uppercase; letter-spacing: .08em; margin: 12px 0 4px;
        border-bottom: 1px solid #21262d; padding-bottom: 4px; }
    #groq-chat-panel .stMarkdown ul, #groq-chat-panel .stMarkdown ol
        { padding-left: 16px; margin: 4px 0 8px; color: #c9d1d9; font-size: 13px; }
    #groq-chat-panel .stMarkdown li { margin-bottom: 3px; line-height: 1.6; }
    #groq-chat-panel .stMarkdown strong { color: #f0f6fc; font-weight: 600; }
    #groq-chat-panel .stMarkdown code
        { background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
          padding: 1px 5px; font-size: 11px; color: #79c0ff; font-family: "IBM Plex Mono", monospace; }
    /* IOC enrichment notice badge inside bubble */
    .gc-cti-tag {
        display:inline-block; font-size:10px; font-weight:600;
        background:#0d2a1a; color:#3fb950; border:1px solid #238636;
        border-radius:4px; padding:1px 6px; margin-bottom:6px;
    }
    .gc-chips {
        padding:8px 14px 6px; display:flex; flex-wrap:wrap; gap:6px;
        border-top:1px solid #161b22; flex-shrink:0;
    }
    .gc-input-wrap {
        border-top: 1px solid #21262d; flex-shrink: 0;
        padding: 6px 10px 8px; background: #0d1117;
    }
    #groq-chat-panel .stChatInput { border:none !important; background:transparent !important; }
    #groq-chat-panel .stChatInput > div {
        background:#161b22 !important; border:1px solid #30363d !important;
        border-radius:20px !important;
    }
    #groq-chat-panel .stChatInput textarea { color:#c9d1d9 !important; font-size:13px !important; }
    #groq-chat-panel .stChatInput button {
        background: linear-gradient(135deg,#cc0000,#f55036) !important;
        border-radius:50% !important;
    }
    .gc-empty { text-align:center; padding:24px 16px;
                color:#484f58; font-size:13px; line-height:1.8; }
    .gc-empty-icon { font-size:34px; margin-bottom:8px; }
    .gc-dot { width:6px; height:6px; background:#cc0000; border-radius:50%;
              animation:gc-bounce 1.2s infinite; display:inline-block; }
    .gc-dot:nth-child(2){ animation-delay:.2s; }
    .gc-dot:nth-child(3){ animation-delay:.4s; }
    @keyframes gc-bounce {
        0%,60%,100%{ transform:translateY(0); }
        30%{ transform:translateY(-6px); }
    }
    </style>
    """, unsafe_allow_html=True)

    # ── FAB button ─────────────────────────────────────────────
    st.markdown('<div id="groq-fab-anchor">', unsafe_allow_html=True)
    msg_count = len(st.session_state.chat_history)
    badge = f" ({msg_count})" if msg_count > 0 and not st.session_state.chat_open else ""
    fab_label = "✕ Close" if st.session_state.chat_open else f"🛡️ Chatbot Analyst{badge}"

    if st.button(fab_label, key="groq_fab_btn", help="Toggle Chatbot"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    #groq-fab-anchor button {
        position:fixed !important; bottom:28px !important; right:28px !important;
        z-index:9999 !important;
        background:linear-gradient(135deg,#cc0000,#f55036) !important;
        color:white !important; border:none !important;
        border-radius:28px !important; padding:12px 20px !important;
        font-size:14px !important; font-weight:600 !important;
        box-shadow:0 4px 20px rgba(204,0,0,0.45) !important;
        cursor:pointer !important; transition:transform 0.2s,box-shadow 0.2s !important;
        white-space:nowrap !important; min-width:unset !important;
        width:auto !important; height:auto !important;
    }
    #groq-fab-anchor button:hover {
        transform:scale(1.05) !important;
        box-shadow:0 6px 28px rgba(204,0,0,0.6) !important;
    }
    #groq-fab-anchor > div { margin:0 !important; }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_open:
        return

    # ── Chat panel ─────────────────────────────────────────────
    st.markdown('<div id="groq-chat-panel">', unsafe_allow_html=True)

    tb_badge_cls  = "gcp-tb-badge" if tb_key_present else "gcp-tb-badge off"
    tb_badge_text = "🟢 ThreatBook Live" if tb_key_present else "⚪ ThreatBook Off"
    st.markdown(f"""
    <div class="gcp-header">
        <div class="gcp-avatar">🛡️</div>
        <div>
            <div class="gcp-title">CTI Analyst</div>
            <div class="gcp-sub">LLaMA 3.3-70b · ThreatBook Intelligence</div>
        </div>
        <span class="{tb_badge_cls}">{tb_badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Messages ───────────────────────────────────────────────
    st.markdown('<div class="gcp-body">', unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div class="gc-empty">
            <div class="gc-empty-icon">🛡️</div>
            <strong style="color:#f0f6fc;font-size:14px;">Analyst Ready</strong><br>
            <span style="color:#8b949e;font-size:12px;">Submit a query about incidents, threat trends,<br>
            or paste an indicator for live enrichment.</span><br><br>
            <span style="font-size:11px;font-family:'IBM Plex Mono',monospace;color:#3fb950;">
              IP &nbsp;·&nbsp; Domain &nbsp;·&nbsp; Hash &nbsp;·&nbsp; CVE &nbsp;·&nbsp; URL
            </span>
            {"" if tb_key_present else
             "<br><br><span style='color:#f85149;font-size:11px;'>⚠ ThreatBook key not configured —<br>add [threatbook] api_key to secrets.toml</span>"}
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            role  = "user" if msg["role"] == "user" else "bot"
            label = "You" if msg["role"] == "user" else "CTI Analyst"
            cti_tag = ""
            if msg.get("cti_enriched"):
                cti_tag = '<span class="gc-cti-tag">⚡ ThreatBook enriched</span>'
            if role == "user":
                # User messages: simple escaped text
                safe = (msg["content"]
                        .replace("&", "&amp;")
                        .replace("<", "&lt;").replace(">", "&gt;")
                        .replace("\n", "<br>"))
                st.markdown(f"""
                <div class="gc-msg user">
                    <div class="gc-role">You</div>
                    <div class="gc-bubble">{safe}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Bot messages: render via st.markdown inside a container for proper formatting
                st.markdown(f'''
                <div class="gc-msg bot">
                    <div class="gc-role">CTI Analyst{" &nbsp;" + cti_tag if cti_tag else ""}</div>
                </div>''', unsafe_allow_html=True)
                with st.container():
                    st.markdown(
                        "<div class='gc-bot-md'></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Suggestion chips ───────────────────────────────────────
    suggestions = [
        "Top critical incidents",
        "Which country has most incidents?",
        "Main threat categories",
        "Any emerging trends?",
    ]
    st.markdown('<div class="gc-chips">', unsafe_allow_html=True)
    chip_cols = st.columns(len(suggestions))
    for i, (col, tip) in enumerate(zip(chip_cols, suggestions)):
        with col:
            if st.button(tip, key=f"chip_{i}", help=tip):
                st.session_state["_chip"] = tip
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Input ──────────────────────────────────────────────────
    st.markdown('<div class="gc-input-wrap">', unsafe_allow_html=True)
    user_input = st.chat_input("Ask analyst or paste IP/domain/hash/CVE…", key="groq_chat_input")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # close panel

    # ── Handle chip ────────────────────────────────────────────
    question = user_input
    if not question and "_chip" in st.session_state:
        question = st.session_state.pop("_chip")

    # ── Process question ───────────────────────────────────────
    if not question or not question.strip():
        return

    question = question.strip()

    # Deduplicate
    if (st.session_state.chat_history
            and st.session_state.chat_history[-1]["role"] == "user"
            and st.session_state.chat_history[-1]["content"] == question):
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    # ── Step 1: Extract IOCs ───────────────────────────────────
    iocs = extract_iocs(question)
    has_iocs = any(iocs.values())

    # ── Step 2: Enrich via ThreatBook ─────────────────────────
    cti_block = ""
    if has_iocs and tb_key_present:
        with st.spinner("🔍 Enriching with ThreatBook CTI…"):
            cti_block = enrich_with_threatbook(iocs)

    # ── Step 3: Call Groq ──────────────────────────────────────
    client, status = _get_groq_client()
    if status == "package_missing":
        answer = "❌ `groq` package not installed. Run: `pip install groq`"
    elif status == "missing_key":
        answer = "❌ Groq API key missing. Add `[groq] api_key` to `.streamlit/secrets.toml`."
    elif client is None:
        answer = f"❌ Groq init error: {status}"
    else:
        try:
            system_prompt = _build_system_prompt(df, cti_block)
            messages = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.chat_history:
                r = "assistant" if m["role"] == "assistant" else "user"
                messages.append({"role": r, "content": m["content"]})

            with st.spinner("🧠 Analysing…"):
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"❌ Groq error: {e}"

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "cti_enriched": bool(cti_block),
    })
    st.rerun()