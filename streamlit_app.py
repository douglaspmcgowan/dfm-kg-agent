"""
DFM Knowledge Graph Agent -- Streamlit Interface

Visual interface for the DFM RAG agent. Retrieves knowledge frames from
Qdrant, traverses the provenance graph, and synthesizes structured DFM
reports via GPT with full interactive frame inspection.

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import io
import json
import uuid
import pickle
import hashlib
import datetime
import smtplib
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from pathlib import Path

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
try:
    import gspread
except ImportError:
    gspread = None

# Import agent module first -- it handles the cp1252 encoding fix at load time.
# Do NOT duplicate the fix here (double TextIOWrapper wrapping causes I/O errors).
import dfm_agent as agent
from dfm_agent import (
    query_dfm,
    collect_cited_frame_ids,
    get_frame_details,
    gather_provenance_context,
    summarize_context,
    GRAPH_PICKLE,
    QDRANT_PATH,
    SEMANTIC_EDGE_TYPES,
    MODEL,
)
import dfm_agent as _agent_module
from openai import OpenAI
from dfm_report_schema import (
    DFMReportV2, BackfillOutput, QueryDecomposition,
    DECOMPOSE_PROMPT, SYNTHESIS_PROMPT_V2, BACKFILL_PROMPT,
)


# ─────────────────────────────────────────────
# ANALYSIS MODE CONFIGS
# ─────────────────────────────────────────────

FLASH_MODE = {
    "model": "gpt-4.1-mini",
    "reasoning_effort": None,
    "icon": "\u26a1",
    "time_range": "~20 sec",
    "token_range": "~7K tokens",
    "cost_estimate": "~$0.01",
    "description": "Fastest results. Best for quick checks and simple questions.",
    "top_k": 5, "max_hops": 1, "max_frames": 10,
}

ANALYSIS_MODES = {
    "Quick Search": {
        "model": "gpt-4.1",
        "reasoning_effort": None,
        "icon": "\U0001f50d",
        "time_range": "1-2 min",
        "token_range": "~11K tokens",
        "cost_estimate": "~$0.05",
        "description": "Fast, concise reports. Best for interactive use.",
    },
    "Efficient Thinking": {
        "model": "gpt-5",
        "reasoning_effort": "low",
        "icon": "\U0001f9e0",
        "time_range": "3-4 min",
        "token_range": "~15K tokens",
        "cost_estimate": "~$0.39",
        "description": "Deeper analysis with more design rules and disciplines.",
    },
    "Deep Reasoning": {
        "model": "gpt-5",
        "reasoning_effort": None,  # default = medium
        "icon": "\U0001f52c",
        "time_range": "7-10 min",
        "token_range": "~26K tokens",
        "cost_estimate": "~$0.92",
        "description": "Most thorough analysis. Richest rules and knowledge gaps.",
    },
}


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DFM Knowledge Graph Agent",
    layout="wide",
    page_icon="\U0001f5ff",  # moyai
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CUSTOM CSS + JS
# ─────────────────────────────────────────────

_CUSTOM_STYLES = """
<style>
/* ── Frame popover buttons: keep icon + type label on one line, hide arrow ── */

/* The trigger button (path: stPopover > div > button) */
[data-testid="stPopoverButton"] {
    white-space: nowrap !important;
    padding: 0.2rem 0.5rem !important;
    min-height: unset !important;
    font-size: 0.85rem !important;
    min-width: unset !important;
    gap: 0 !important;
    border: 1px solid rgba(250, 250, 250, 0.2) !important;
    border-radius: 6px !important;
    background-color: rgba(250, 250, 250, 0.06) !important;
}

/* Hide the expand_more dropdown arrow (last child of the button container) */
[data-testid="stPopoverButton"] > div > div:last-child:not(:first-child) {
    display: none !important;
}

/* Fallback: hide any material icon inside the popover button */
[data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* Label text div: center content, cancel the negative margin from hidden arrow */
[data-testid="stPopoverButton"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    margin: 0 !important;
}

/* Vertically center the popover column with the text column */
[data-testid="stPopover"] {
    display: flex;
    align-items: center;
    justify-content: flex-end;
}

/* Ensure the column holding the popover has a usable minimum width */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child:has([data-testid="stPopover"]) {
    min-width: 60px !important;
    flex-shrink: 0 !important;
}

/* ── Pipeline animation (bouncing dots) ── */
.dfm-anim { display: flex; justify-content: center; gap: 8px; padding: 16px 0; }
.dfm-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: #3498db; animation: dfm-bounce 1.4s ease-in-out infinite;
}
.dfm-dot:nth-child(2) { animation-delay: 0.2s; }
.dfm-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes dfm-bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1.0); opacity: 1.0; }
}

/* ── Mobile responsiveness ── */
@media (max-width: 768px) {
    /* Stack columns on narrow screens */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    /* Give narrow columns more room */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 80px !important;
    }
    /* Larger touch targets for popover buttons */
    [data-testid="stPopoverButton"] {
        padding: 0.4rem 0.6rem !important;
        font-size: 0.9rem !important;
        min-height: 44px !important;
    }
    /* Constrain popover body */
    [data-testid="stPopoverBody"] {
        max-width: 90vw !important;
        max-height: 70vh !important;
        overflow-y: auto !important;
    }
    /* Readable metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    /* Bigger touch targets for buttons and inputs */
    [data-testid="stDownloadButton"] button,
    [data-testid="stTextInput"] input {
        min-height: 44px !important;
        font-size: 1rem !important;
    }
    /* Suggestion buttons: stack vertically */
    .dfm-suggestions [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
}
</style>
"""
st.markdown(_CUSTOM_STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────

@st.cache_resource
def load_graph():
    """Load the provenance graph once per server lifetime."""
    if not GRAPH_PICKLE.exists():
        return None
    with open(GRAPH_PICKLE, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_openai_client():
    """Create OpenAI client once per server lifetime.

    Checks Streamlit secrets first (for Cloud deployment), then env var.
    """
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    # Also inject into os.environ so dfm_agent.py can find it
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key, timeout=300.0)


@st.cache_data
def get_graph_stats(_G):
    """Count frames and semantic edges for display."""
    n_frames = sum(1 for _, d in _G.nodes(data=True) if d.get("node_type") == "Frame")
    n_sem = sum(
        1 for _, _, d in _G.edges(data=True)
        if d.get("edge_type") in SEMANTIC_EDGE_TYPES
    )
    n_authors = sum(1 for _, d in _G.nodes(data=True) if d.get("node_type") == "Author")
    n_threads = sum(1 for _, d in _G.nodes(data=True) if d.get("node_type") == "Thread")
    return {
        "frames": n_frames,
        "semantic_edges": n_sem,
        "authors": n_authors,
        "threads": n_threads,
        "total_nodes": _G.number_of_nodes(),
        "total_edges": _G.number_of_edges(),
    }


_EXAMPLE_CACHE_FILE = Path(__file__).parent / "example_cache.json"


@st.cache_resource
def load_example_cache():
    """Load pre-computed example answers (once per server lifetime)."""
    if not _EXAMPLE_CACHE_FILE.exists():
        return {}
    try:
        with open(_EXAMPLE_CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_session_state():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "active_idx" not in st.session_state:
        st.session_state.active_idx = -1
    if "context_cache" not in st.session_state:
        st.session_state.context_cache = {}
    if "is_querying" not in st.session_state:
        st.session_state.is_querying = False
    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = {
            "queries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embedding_tokens": 0,
            "total_tokens": 0,
        }

init_session_state()


# ─────────────────────────────────────────────
# GOOGLE SHEETS PERSISTENCE
# ─────────────────────────────────────────────

@st.cache_resource
def _get_gsheet_client():
    """Get an authenticated gspread client using Streamlit secrets.

    Returns None if credentials are not configured (falls back to local files).
    """
    try:
        creds = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(dict(creds))
        return gc
    except Exception:
        return None


@st.cache_resource
def _get_gsheet_spreadsheet():
    """Open the logging spreadsheet. Returns None if not configured."""
    gc = _get_gsheet_client()
    if gc is None:
        return None
    try:
        sheet_key = st.secrets.get("GSHEET_LOG_KEY", "")
        if sheet_key:
            return gc.open_by_key(sheet_key)
        return None
    except Exception:
        return None


def _gsheet_append_row(worksheet_name, row_dict, columns):
    """Append a row to a Google Sheet worksheet. Creates the sheet if needed.

    Args:
        worksheet_name: Tab/worksheet name (e.g. "visitors", "queries")
        row_dict: Dict of values to write
        columns: Ordered list of column names (used for header row and ordering)
    """
    ss = _get_gsheet_spreadsheet()
    if ss is None:
        return False
    try:
        try:
            ws = ss.worksheet(worksheet_name)
        except Exception:  # WorksheetNotFound or gspread not installed
            ws = ss.add_worksheet(title=worksheet_name, rows=1000, cols=len(columns))
            ws.append_row(columns, value_input_option="RAW")
        row = [str(row_dict.get(c, "")) for c in columns]
        ws.append_row(row, value_input_option="RAW")
        return True
    except Exception:
        return False


def _gsheet_read_all(worksheet_name):
    """Read all rows from a worksheet as a list of dicts. Returns [] on failure."""
    ss = _get_gsheet_spreadsheet()
    if ss is None:
        return []
    try:
        ws = ss.worksheet(worksheet_name)
        return ws.get_all_records()
    except Exception:
        return []


def _gsheet_read_cell(worksheet_name, cell):
    """Read a single cell value. Returns None on failure."""
    ss = _get_gsheet_spreadsheet()
    if ss is None:
        return None
    try:
        ws = ss.worksheet(worksheet_name)
        return ws.acell(cell).value
    except Exception:
        return None


def _gsheet_update_tally(tally_dict):
    """Write the full token tally as a single-row key-value store in the 'tally' sheet."""
    ss = _get_gsheet_spreadsheet()
    if ss is None:
        return False
    try:
        try:
            ws = ss.worksheet("tally")
        except Exception:  # WorksheetNotFound or gspread not installed
            ws = ss.add_worksheet(title="tally", rows=2, cols=20)
        keys = list(tally_dict.keys())
        vals = []
        for k in keys:
            v = tally_dict[k]
            if isinstance(v, list):
                v = json.dumps(v)
            vals.append(str(v))
        ws.clear()
        ws.append_row(keys, value_input_option="RAW")
        ws.append_row(vals, value_input_option="RAW")
        return True
    except Exception:
        return False


def _gsheet_read_tally():
    """Read the token tally from the 'tally' sheet. Returns None if not available."""
    ss = _get_gsheet_spreadsheet()
    if ss is None:
        return None
    try:
        ws = ss.worksheet("tally")
        records = ws.get_all_records()
        if not records:
            return None
        tally = records[0]
        # Parse unique_ips back from JSON string
        if "unique_ips" in tally and isinstance(tally["unique_ips"], str):
            try:
                tally["unique_ips"] = json.loads(tally["unique_ips"])
            except (json.JSONDecodeError, ValueError):
                tally["unique_ips"] = []
        # Parse numeric fields
        for k in ["prompt_tokens", "completion_tokens", "embedding_tokens", "total_tokens", "queries", "page_views"]:
            if k in tally:
                try:
                    tally[k] = int(tally[k])
                except (ValueError, TypeError):
                    tally[k] = 0
        return tally
    except Exception:
        return None


# Columns for each log type
_VISITOR_COLUMNS = ["timestamp", "session_id", "ip", "ip_hash", "user_agent", "first_name", "last_name", "contact_info"]
_QUERY_COLUMNS = ["timestamp", "session_id", "ip", "is_admin", "visitor_name", "contact_info", "query",
                   "model", "flash_mode", "from_cache", "elapsed_s", "token_usage",
                   "num_rules", "num_disciplines", "num_gaps", "summary_preview"]
_FEEDBACK_COLUMNS = ["timestamp", "name", "email", "message", "ip_hash"]


# ─────────────────────────────────────────────
# PERSISTENT TOKEN TALLY
# ─────────────────────────────────────────────

_TOKEN_TALLY_FILE = Path(__file__).parent / "token_tally.json"


def _load_token_tally():
    """Load cumulative token tally from Google Sheets (primary) or local disk (fallback)."""
    # Try Google Sheets first
    tally = _gsheet_read_tally()
    if tally is not None:
        return tally
    # Fallback to local file
    if _TOKEN_TALLY_FILE.exists():
        try:
            with open(_TOKEN_TALLY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    defaults = {
        "since": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embedding_tokens": 0,
        "total_tokens": 0,
        "queries": 0,
        "unique_ips": [],
        "page_views": 0,
    }
    return defaults


def _update_token_tally(token_usage):
    """Add a query's token usage to the persistent tally."""
    tally = _load_token_tally()
    tally["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
    tally["completion_tokens"] += token_usage.get("completion_tokens", 0)
    tally["embedding_tokens"] += token_usage.get("embedding_tokens", 0)
    tally["total_tokens"] += token_usage.get("total_tokens", 0)
    tally["queries"] += 1
    tally.setdefault("unique_ips", [])
    tally.setdefault("page_views", 0)
    # Write to Google Sheets (primary) and local file (fallback)
    _gsheet_update_tally(tally)
    with open(_TOKEN_TALLY_FILE, "w") as f:
        json.dump(tally, f, indent=2)


def _reset_token_tally():
    """Reset the persistent token tally."""
    tally = {
        "since": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embedding_tokens": 0,
        "total_tokens": 0,
        "queries": 0,
        "unique_ips": [],
        "page_views": 0,
    }
    _gsheet_update_tally(tally)
    with open(_TOKEN_TALLY_FILE, "w") as f:
        json.dump(tally, f, indent=2)
    return tally


def _is_private_ip(ip):
    """Check if an IP address is private/internal (RFC 1918, loopback, etc.)."""
    if not ip:
        return True
    try:
        import ipaddress
        return ipaddress.ip_address(ip).is_private
    except (ValueError, TypeError):
        return False


def _is_admin_visitor():
    """Check if the current visitor is the site owner (exclude from tallies)."""
    # Check Streamlit secret for owner IPs
    try:
        owner_ips = st.secrets.get("OWNER_IPS", "")
        if owner_ips:
            ip = _get_visitor_ip()
            for owner_ip in owner_ips.split(","):
                if ip and ip.strip() == owner_ip.strip():
                    return True
    except Exception:
        pass
    # Also check if admin is authenticated this session
    return bool(st.session_state.get("_admin_authenticated"))


def _get_visitor_ip_hash():
    """Get a privacy-safe hash of the visitor's real public IP."""
    try:
        ip = _get_visitor_ip()
        if not ip:
            return None
        return hashlib.sha256(ip.encode()).hexdigest()[:12]
    except Exception:
        return None


def _inject_client_js():
    """Inject client-side JS to resolve public IP and welcome form state.

    Combines both into a single redirect so the server-side render already
    has all the info it needs (no flash of welcome dialog).

    Streamlit Cloud strips real IPs from headers, so we fetch client-side
    from ipify.org. The welcome-done flag is stored in localStorage so
    returning visitors skip the form entirely.
    """
    # Already resolved IP this session -- welcome form is handled by the dialog,
    # not by JS, so we only need to gate on IP resolution to avoid redirect loops.
    if st.session_state.get("_public_ip"):
        return

    # Read query params set by JS redirect
    ip_from_param = st.query_params.get("_cip")
    vf_from_param = st.query_params.get("_vf")

    if ip_from_param:
        st.session_state["_public_ip"] = ip_from_param
        del st.query_params["_cip"]
    if vf_from_param == "1":
        st.session_state["_visitor_form_shown"] = True
        del st.query_params["_vf"]

    # If IP resolved from params, nothing more to do
    if st.session_state.get("_public_ip"):
        return

    # Inject single JS snippet that handles both IP + welcome check
    import streamlit.components.v1 as components
    components.html(
        """<script>
        (function() {
            try {
                var url = new URL(window.parent.location.href);
                var changed = false;

                // -- Welcome form check --
                if (!url.searchParams.has('_vf')) {
                    if (localStorage.getItem('_dfm_welcome_done') === '1') {
                        url.searchParams.set('_vf', '1');
                        changed = true;
                    }
                }

                // -- IP resolution --
                if (!url.searchParams.has('_cip')) {
                    var cachedIp = localStorage.getItem('_dfm_public_ip');
                    if (cachedIp) {
                        url.searchParams.set('_cip', cachedIp);
                        changed = true;
                        if (changed) {
                            window.parent.location.replace(url.toString());
                        }
                        return;
                    }
                    // Fetch from ipify
                    fetch('https://api.ipify.org?format=text')
                        .then(function(r) { return r.text(); })
                        .then(function(ip) {
                            localStorage.setItem('_dfm_public_ip', ip);
                            url.searchParams.set('_cip', ip);
                            window.parent.location.replace(url.toString());
                        })
                        .catch(function() {
                            // IP fetch failed -- still redirect with welcome flag if needed
                            if (changed) {
                                window.parent.location.replace(url.toString());
                            }
                        });
                    return;
                }

                // Only welcome flag changed, no IP needed
                if (changed) {
                    window.parent.location.replace(url.toString());
                }
            } catch(e) {}
        })();
        </script>""",
        height=0,
    )


def _get_visitor_ip():
    """Get the real public IP (from JS fetch) or fall back to headers."""
    try:
        # 1. Public IP from client-side JS fetch
        public_ip = st.session_state.get("_public_ip")
        if public_ip:
            return public_ip
        # 2. st.context.ip_address (Streamlit 1.45+)
        ip = getattr(st.context, "ip_address", None)
        if ip:
            return ip
        # 3. Proxy headers (usually private IPs on Streamlit Cloud)
        for header in ("X-Forwarded-For", "CF-Connecting-IP", "X-Real-IP"):
            val = st.context.headers.get(header, "")
            if val:
                return val.split(",")[0].strip()
        return None
    except Exception:
        return None


def _get_user_agent():
    """Get the visitor's User-Agent string from headers."""
    try:
        return st.context.headers.get("User-Agent", "") or None
    except Exception:
        return None


# ─────────────────────────────────────────────
# ANALYTICS LOG FILES
# ─────────────────────────────────────────────

_PST = ZoneInfo("America/Los_Angeles")

def _now_pst():
    """Get current time in Pacific time."""
    return datetime.datetime.now(_PST)


def _relative_time(dt):
    """Return human-readable relative time like 'Analyzed 2 min ago'."""
    if dt is None:
        return ""
    delta = _now_pst() - dt
    seconds = int(delta.total_seconds())
    if seconds < 10:
        return "Analyzed just now"
    if seconds < 60:
        return f"Analyzed {seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"Analyzed {minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"Analyzed {hours}h ago"
    return f"Analyzed {dt.strftime('%I:%M %p')}"


_VISITOR_LOG_FILE = Path(__file__).parent / "visitor_log.jsonl"
_QUERY_LOG_FILE = Path(__file__).parent / "query_log.jsonl"


def _append_log(filepath, entry):
    """Append a JSON entry to a JSONL file and to Google Sheets."""
    # Write to local file (fallback)
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Non-critical -- don't break the app
    # Write to Google Sheets (primary / persistent)
    try:
        if filepath == _VISITOR_LOG_FILE:
            _gsheet_append_row("visitors", entry, _VISITOR_COLUMNS)
        elif filepath == _QUERY_LOG_FILE:
            # Serialize nested dicts for Sheets
            sheets_entry = dict(entry)
            if "token_usage" in sheets_entry and isinstance(sheets_entry["token_usage"], dict):
                sheets_entry["token_usage"] = json.dumps(sheets_entry["token_usage"])
            _gsheet_append_row("queries", sheets_entry, _QUERY_COLUMNS)
    except Exception:
        pass


def _read_log(filepath):
    """Read log entries from Google Sheets (primary) or local JSONL file (fallback)."""
    # Try Google Sheets first
    try:
        if filepath == _VISITOR_LOG_FILE:
            records = _gsheet_read_all("visitors")
        elif filepath == _QUERY_LOG_FILE:
            records = _gsheet_read_all("queries")
        else:
            records = []
        if records:
            return records
    except Exception:
        pass
    # Fallback to local file
    entries = []
    if not filepath.exists():
        return entries
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except IOError:
        pass
    return entries


def _get_session_id():
    """Get or create a unique session ID for linking visitor + query logs."""
    if "_session_id" not in st.session_state:
        st.session_state["_session_id"] = str(uuid.uuid4())[:12]
    return st.session_state["_session_id"]


def _log_query(result):
    """Log a query and its summary stats to query_log.jsonl."""
    report = result.get("report")
    visitor_name = None
    fn = st.session_state.get("_visitor_first_name", "")
    ln = st.session_state.get("_visitor_last_name", "")
    if fn or ln:
        visitor_name = f"{fn} {ln}".strip()
    contact = st.session_state.get("_visitor_contact", "")

    summary_preview = ""
    if report and hasattr(report, "summary"):
        summary_preview = report.summary[:200]

    entry = {
        "timestamp": _now_pst().strftime("%Y-%m-%d %I:%M:%S %p PST"),
        "session_id": _get_session_id(),
        "ip": _get_visitor_ip(),
        "is_admin": _is_admin_visitor(),
        "visitor_name": visitor_name,
        "contact_info": contact or None,
        "query": result.get("query", ""),
        "model": "gpt-4.1-mini" if result.get("flash_mode") else (os.environ.get("OPENAI_MODEL", "gpt-5")),
        "flash_mode": bool(result.get("flash_mode")),
        "from_cache": bool(result.get("from_cache")),
        "elapsed_s": result.get("elapsed_s", 0),
        "token_usage": result.get("token_usage", {}),
        "num_rules": len(report.design_rules) if report and hasattr(report, "design_rules") else 0,
        "num_disciplines": len(report.disciplines) if report and hasattr(report, "disciplines") else 0,
        "num_gaps": len(report.knowledge_gaps) if report and hasattr(report, "knowledge_gaps") else 0,
        "summary_preview": summary_preview,
    }
    _append_log(_QUERY_LOG_FILE, entry)


def _record_visitor():
    """Record a unique visitor (once per session).

    Waits for the real public IP (from JS fetch) before logging.
    Skips tally increment for admin/owner visits.
    """
    if st.session_state.get("_visitor_recorded"):
        return

    ip = _get_visitor_ip()

    # Wait for the JS IP fetch to complete -- don't log private IPs
    if not ip or _is_private_ip(ip):
        return  # Will be called again on next rerun after JS redirect

    st.session_state["_visitor_recorded"] = True
    ip_hash = _get_visitor_ip_hash()

    # Skip admin/owner visits entirely -- don't log or count
    if _is_admin_visitor():
        return

    # Append detailed entry to visitor log
    entry = {
        "timestamp": _now_pst().strftime("%Y-%m-%d %I:%M:%S %p PST"),
        "session_id": _get_session_id(),
        "ip": ip,
        "ip_hash": ip_hash,
        "user_agent": _get_user_agent(),
        "first_name": st.session_state.get("_visitor_first_name"),
        "last_name": st.session_state.get("_visitor_last_name"),
        "contact_info": st.session_state.get("_visitor_contact_info"),
    }
    _append_log(_VISITOR_LOG_FILE, entry)

    # Update token tally
    tally = _load_token_tally()
    tally.setdefault("unique_ips", [])
    tally.setdefault("page_views", 0)
    tally["page_views"] += 1
    if ip_hash and ip_hash not in tally["unique_ips"]:
        tally["unique_ips"].append(ip_hash)
    _gsheet_update_tally(tally)
    with open(_TOKEN_TALLY_FILE, "w") as f:
        json.dump(tally, f, indent=2)


# ─────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────

_FEEDBACK_FILE = Path(__file__).parent / "feedback.json"
_FEEDBACK_EMAIL = "dpm5970@berkeley.edu"


def _save_feedback(name, email, message):
    """Save feedback to local JSON and send email if SMTP is configured."""
    entry = {
        "timestamp": _now_pst().strftime("%Y-%m-%d %I:%M:%S %p PST"),
        "name": name,
        "email": email,
        "message": message,
        "ip_hash": _get_visitor_ip_hash(),
    }

    # Save to Google Sheets (persistent)
    _gsheet_append_row("feedback", entry, _FEEDBACK_COLUMNS)
    # Save locally (fallback)
    feedback = []
    if _FEEDBACK_FILE.exists():
        try:
            with open(_FEEDBACK_FILE, "r") as f:
                feedback = json.load(f)
        except (json.JSONDecodeError, IOError):
            feedback = []
    feedback.append(entry)
    with open(_FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)

    # Send email via Formsubmit.co (free, no API key needed)
    try:
        import urllib.request
        payload = json.dumps({
            "name": name or "Anonymous",
            "email": email or "Not provided",
            "message": message,
            "_subject": f"DFM KG Agent Feedback from {name or 'Anonymous'}",
            "_captcha": "false",
            "_template": "box",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"https://formsubmit.co/ajax/{_FEEDBACK_EMAIL}",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; DFM-KG-Agent/1.0)",
                "Origin": "https://dfm-kg-agent.streamlit.app",
                "Referer": "https://dfm-kg-agent.streamlit.app/",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                result = json.loads(resp.read())
                if result.get("success") == "true":
                    return True
    except Exception:
        pass  # Email failed silently -- feedback is still saved locally

    # Fallback: SMTP if configured
    try:
        smtp_user = st.secrets.get("SMTP_USER")
        smtp_pass = st.secrets.get("SMTP_PASSWORD")
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT", 587))

        if smtp_user and smtp_pass:
            subject = f"DFM KG Agent Feedback from {name or 'Anonymous'}"
            body = (
                f"Name: {name or 'Anonymous'}\n"
                f"Email: {email or 'Not provided'}\n"
                f"Time: {entry['timestamp']}\n\n"
                f"Message:\n{message}"
            )
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = _FEEDBACK_EMAIL

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, _FEEDBACK_EMAIL, msg.as_string())
            return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────
# SCROLL-DISMISS JS
# ─────────────────────────────────────────────

def inject_scroll_dismiss(pin_popovers=False):
    """Inject JS that closes open popovers on scroll (unless pinned).

    Uses st.components.v1.html so the <script> actually executes, then
    reaches into window.parent.document (same-origin Streamlit frame).
    """
    import streamlit.components.v1 as components

    pinned_val = "true" if pin_popovers else "false"
    components.html(f"""
    <script>
    (function() {{
        var doc = window.parent.document;

        // Update pin flag every rerun
        var pin = doc.getElementById('dfm-pin-popovers');
        if (!pin) {{
            pin = doc.createElement('div');
            pin.id = 'dfm-pin-popovers';
            pin.style.display = 'none';
            doc.body.appendChild(pin);
        }}
        pin.dataset.pinned = '{pinned_val}';

        // Attach scroll listener only once
        if (window.parent._dfmScrollDismissAttached) return;
        window.parent._dfmScrollDismissAttached = true;

        var observer = new MutationObserver(function() {{
            var main = doc.querySelector('[data-testid="stMain"]');
            if (!main || main._dfmScrollListener) return;
            main._dfmScrollListener = true;
            var scrollTimer = null;
            main.addEventListener('scroll', function() {{
                var p = doc.getElementById('dfm-pin-popovers');
                if (p && p.dataset.pinned === 'true') return;
                if (scrollTimer) clearTimeout(scrollTimer);
                scrollTimer = setTimeout(function() {{
                    var open = doc.querySelectorAll('[data-testid="stPopoverBody"]');
                    if (open.length > 0) {{
                        // Close by clicking the main content area (acts as outside-click)
                        var mc = doc.querySelector('[data-testid="stMainBlockContainer"]');
                        if (mc) {{
                            mc.dispatchEvent(new MouseEvent('mousedown', {{bubbles:true}}));
                            mc.dispatchEvent(new MouseEvent('mouseup', {{bubbles:true}}));
                            mc.dispatchEvent(new MouseEvent('click', {{bubbles:true}}));
                        }}
                    }}
                }}, 150);
            }});
        }});
        observer.observe(doc.body, {{childList: true, subtree: true}});
    }})();
    </script>
    """, height=0)


# ─────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────

# Map edge types to colors for graph viz
EDGE_COLORS = {
    "CAUSAL_LINK": "#e74c3c",    # red
    "MITIGATES": "#2ecc71",      # green
    "SPECIFIES": "#3498db",      # blue
    "CONTRADICTS": "#e67e22",    # orange
}

# Map frame types to colors for graph viz
FRAME_TYPE_COLORS = {
    "risk": "#e74c3c",
    "heuristic": "#3498db",
    "principle": "#9b59b6",
    "case": "#f39c12",
    "workaround": "#2ecc71",
    "observation": "#1abc9c",
    "comparison": "#e67e22",
    "other": "#95a5a6",
}

# Map chain roles to visual indicators
ROLE_INDICATORS = {
    "root_cause": ("\U0001f534", "Root Cause"),       # red circle
    "effect": ("\U0001f7e0", "Effect"),               # orange circle
    "mitigation": ("\U0001f7e2", "Mitigation"),       # green circle
    "condition": ("\U0001f535", "Condition"),          # blue circle
    "origin": ("\u26aa", "Origin"),                    # white circle
}


def render_frame_popover(frame_id, frame_lookup, short_label=None):
    """Render a frame_id as a clickable popover with full details."""
    detail = frame_lookup.get(frame_id)
    # Default label: frame_type from lookup, else frame_id
    if short_label is None:
        label = detail["frame_type"] if detail else frame_id
    else:
        label = short_label

    if not detail:
        st.caption(f"`{frame_id}`")
        return

    with st.popover(f"\U0001f4cb {label}"):
        # Header -- larger type/scope/applicability line + frame ID
        ft = detail["frame_type"]
        color = FRAME_TYPE_COLORS.get(ft, "#95a5a6")
        st.markdown(
            f"### {ft.upper()} &nbsp; | &nbsp; "
            f"`{detail['scope']}` &nbsp; | &nbsp; "
            f"`{detail['applicability']}`"
        )
        st.caption(f"`{frame_id}`")

        st.markdown(f"**Subject:** {detail['subject']}")
        st.markdown(f"**Main Point:** {detail['main_point']}")

        if detail.get("source_quote"):
            # Clean bullet points from source quotes to avoid formatting issues
            quote = detail["source_quote"].replace("\n- ", "\n  - ").replace("\n* ", "\n  * ")
            st.markdown("**Source Quote**")
            st.markdown(f"> {quote}")

        st.markdown(f"**Epistemic stance:** `{detail['epistemic_stance']}` "
                    f"&nbsp; | &nbsp; Post role: `{detail['post_role']}`")

        if detail.get("example_value"):
            st.markdown(f"**Example value:** {detail['example_value']}")
        if detail.get("example_context"):
            st.markdown(f"**Example context:** {detail['example_context']}")

        st.divider()

        # Author & credibility -- smaller/compact
        author = detail.get("author", "?")
        cred = detail.get("credibility", "?")
        rep = detail.get("reputation", -1)
        rep_str = f"{rep:,}" if rep != -1 else "N/A"
        score = detail.get("post_score", 0)
        accepted = "\u2713 Accepted" if detail.get("is_accepted") else ""
        st.caption(
            f"Author: **{author}** &nbsp; | &nbsp; "
            f"Credibility: **{cred}** &nbsp; | &nbsp; "
            f"Reputation: **{rep_str}** &nbsp; | &nbsp; "
            f"Post score: **{score}** "
            f"{'&nbsp; | &nbsp; ' + accepted if accepted else ''}"
        )

        if detail.get("source_url"):
            st.markdown(f"[\U0001f517 View Original Thread]({detail['source_url']})")
        if detail.get("se_tags"):
            st.caption(f"Tags: {detail['se_tags']}")


def render_diagnostic(diagnostic):
    """Show retrieval diagnostic as a metrics row."""
    if not diagnostic:
        return

    relevant = diagnostic.get("relevant", 0)
    total = diagnostic.get("total", 0)
    avg_sim = diagnostic.get("avg_sim", 0)

    # Reinterpret grade based on avg similarity (more useful than the 0.7 threshold)
    if avg_sim >= 0.6:
        match_label = "High"
    elif avg_sim >= 0.45:
        match_label = "Moderate"
    elif avg_sim >= 0.3:
        match_label = "Indirect"
    else:
        match_label = "Low"

    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval Match", match_label,
              help="How directly the query matched knowledge graph content. "
                   "'Indirect' or 'Moderate' can still produce strong reports "
                   "because graph traversal pulls in related frames via semantic edges.")
    c2.metric("Seed Frames", f"{total}",
              help="Number of unique frames retrieved from vector search before graph traversal.")
    c3.metric("Avg Similarity", f"{avg_sim:.3f}",
              help="Mean cosine similarity of the top seed frames to the query embedding. "
                   "Values above 0.4 typically produce useful results.")


def render_design_rules(report, frame_lookup, concise=False):
    """Render actionable design rules with frame popovers."""
    rules = report.design_rules
    if not rules:
        st.caption("No design rules extracted.")
        return

    show_rules = rules[:5] if concise else rules
    for i, rule in enumerate(show_rules, 1):
        st.markdown(f"**{i}.** {rule.claim}")
        render_frame_popover(rule.frame_id, frame_lookup)

    if concise and len(rules) > 5:
        with st.expander(f"Show all {len(rules)} rules"):
            for i, rule in enumerate(rules[5:], 6):
                st.markdown(f"**{i}.** {rule.claim}")
                render_frame_popover(rule.frame_id, frame_lookup)


def render_evidence(evidence_list, frame_lookup):
    """Render a list of SourcedClaim evidence items."""
    if not evidence_list:
        st.caption("No evidence in this discipline.")
        return

    for ev in evidence_list:
        col_text, col_ref = st.columns([7, 1])
        with col_text:
            st.markdown(f"- {ev.claim}")
        with col_ref:
            render_frame_popover(ev.frame_id, frame_lookup)


def render_reasoning_chains(chains, frame_lookup):
    """Render reasoning chains as individual dropdowns."""
    if not chains:
        return

    for chain in chains:
        with st.expander(f"\U0001f517 {chain.chain_label}", expanded=False):
            for cf in chain.frames:
                indicator, role_label = ROLE_INDICATORS.get(
                    cf.role_in_chain, ("\u2b55", cf.role_in_chain)
                )
                edge_info = ""
                if cf.edge_type and cf.edge_from:
                    edge_info = f" &larr; `{cf.edge_type}` from `{cf.edge_from}`"

                col_text, col_ref = st.columns([7, 1])
                with col_text:
                    st.markdown(
                        f"{indicator} **{role_label}**: {cf.claim}{edge_info}"
                    )
                with col_ref:
                    render_frame_popover(cf.frame_id, frame_lookup)


def render_formulas(formulas):
    """Render formula entries."""
    if not formulas:
        return

    st.markdown("**Governing Equations**")
    for f in formulas:
        with st.expander(f"\U0001f4d0 {f.formula_name}"):
            st.code(f.expression, language=None)
            st.caption(f"**Variables:** {f.variables}")
            st.markdown(f"*{f.relevance}*")


def _split_rationale_to_bullets(text):
    """Split a long first-principles rationale into bullet points by sentence."""
    import re
    # Split on sentence boundaries (period/semicolon followed by space and uppercase or end)
    sentences = re.split(r'(?<=[.;])\s+(?=[A-Z])', text.strip())
    # Filter out very short fragments
    return [s.strip().rstrip(".") + "." for s in sentences if len(s.strip()) > 15]


def render_disciplines(report, frame_lookup):
    """Render discipline sections as tabs."""
    disciplines = report.disciplines
    if not disciplines:
        st.caption("No discipline sections.")
        return

    tab_names = [d.discipline.replace("_", " ").title() for d in disciplines]
    tabs = st.tabs(tab_names)

    for tab, disc in zip(tabs, disciplines):
        with tab:
            # First principles -- split into bullets for readability
            if disc.first_principles:
                bullets = _split_rationale_to_bullets(disc.first_principles)
                if len(bullets) > 1:
                    bullet_md = "\n".join(f"- {b}" for b in bullets)
                    st.info(f"**Engineering Rationale**\n\n{bullet_md}")
                else:
                    st.info(disc.first_principles)

            # Formulas
            render_formulas(disc.formulas)

            # Evidence
            if disc.evidence:
                st.markdown(f"**Evidence**")
                with st.expander(f"Show evidence ({len(disc.evidence)} claims)", expanded=False):
                    render_evidence(disc.evidence, frame_lookup)

            # Reasoning chains
            if disc.reasoning_chains:
                st.markdown(f"**Reasoning Chains** ({len(disc.reasoning_chains)})")
                render_reasoning_chains(disc.reasoning_chains, frame_lookup)


def render_contradictions(report, frame_lookup):
    """Render contradictions as warning blocks with source details."""
    contras = report.contradictions
    if not contras:
        return

    st.markdown("### \u26a0\ufe0f Contradictions")
    for c in contras:
        with st.container(border=True):
            st.warning(f"**{c.summary}**")
            c1, c2 = st.columns(2)
            with c1:
                pos_col, ref_col = st.columns([6, 1])
                with pos_col:
                    st.markdown(f"**Position A:**\n\n{c.position_a}")
                with ref_col:
                    render_frame_popover(c.frame_id_a, frame_lookup)
                fa = frame_lookup.get(c.frame_id_a)
                if fa:
                    if fa.get("source_quote"):
                        quote = fa["source_quote"][:200].replace("\n- ", " - ").replace("\n* ", " * ")
                        st.caption(f'> "{quote}..."' if len(fa.get("source_quote", "")) > 200 else f'> "{quote}"')
                    link = fa.get("thread_url") or fa.get("url") or fa.get("source_url")
                    author = fa.get("author", "")
                    if link:
                        st.caption(f"Source: [{author or c.frame_id_a}]({link})")
            with c2:
                pos_col, ref_col = st.columns([6, 1])
                with pos_col:
                    st.markdown(f"**Position B:**\n\n{c.position_b}")
                with ref_col:
                    render_frame_popover(c.frame_id_b, frame_lookup)
                fb = frame_lookup.get(c.frame_id_b)
                if fb:
                    if fb.get("source_quote"):
                        quote = fb["source_quote"][:200].replace("\n- ", " - ").replace("\n* ", " * ")
                        st.caption(f'> "{quote}..."' if len(fb.get("source_quote", "")) > 200 else f'> "{quote}"')
                    link = fb.get("thread_url") or fb.get("url") or fb.get("source_url")
                    author = fb.get("author", "")
                    if link:
                        st.caption(f"Source: [{author or c.frame_id_b}]({link})")
            if hasattr(c, "edge_reasoning") and c.edge_reasoning:
                st.caption(f"Edge reasoning: {c.edge_reasoning}")


def _render_backfill_results(backfill_result):
    """Render backfill answers in an expander."""
    n = len(backfill_result.backfills)
    with st.expander(
        f"\U0001f4da Textbook Backfill ({n} gaps filled)",
        expanded=False,
    ):
        st.caption(
            "General engineering knowledge -- NOT from the knowledge graph. "
            "Use as starting guidance only."
        )
        for bf in backfill_result.backfills:
            with st.container(border=True):
                st.markdown(f"**Gap:** {bf.gap}")
                st.markdown(bf.backfill_answer)
                footer_parts = [f"*{bf.confidence_note}*"]
                if bf.relevant_standards:
                    footer_parts.append(f"Standards: {bf.relevant_standards}")
                st.caption(" | ".join(footer_parts))


def render_knowledge_gaps(report, backfill_result, result_dict=None, concise=False):
    """Render knowledge gaps, optional backfill, and Fill the Gaps button."""
    gaps = report.knowledge_gaps
    if not gaps:
        return

    st.markdown(f"### \U0001f50d Knowledge Gaps & Textbook Backfill")

    with st.expander(f"\u2753 Knowledge Gaps ({len(gaps)})", expanded=False):
        st.caption("Areas where the knowledge graph lacks coverage")
        for gap in gaps:
            st.markdown(f"- {gap}")

    # Backfill results (already computed or filled on-demand)
    if backfill_result and backfill_result.backfills:
        _render_backfill_results(backfill_result)

    # "Fill the Gaps" button — shown when backfill wasn't run (e.g. Flash Mode)
    elif gaps and result_dict is not None:
        query_key = result_dict.get("query", "")[:20]
        if st.button("\U0001f4da Fill the Gaps", key=f"fill_gaps_{query_key}",
                     help="Generate textbook answers for knowledge gaps using GPT"):
            client = get_openai_client()
            if not client:
                st.error("OPENAI_API_KEY not set.")
            else:
                with st.spinner("Generating textbook backfill for knowledge gaps..."):
                    from dfm_agent import backfill_knowledge_gaps
                    bf = backfill_knowledge_gaps(client, report)
                    if bf and bf.backfills:
                        result_dict["backfill_result"] = bf
                        st.rerun()
                    else:
                        st.caption("No backfill generated.")


def build_graph_figure(report, G, frame_lookup):
    """Build a Plotly network graph of cited frames and their semantic edges."""
    cited = collect_cited_frame_ids(report)
    if not cited or len(cited) < 2:
        return None

    # Build subgraph of cited frames
    cited_in_graph = [fid for fid in cited if G.has_node(fid)]
    if len(cited_in_graph) < 2:
        return None

    subG = G.subgraph(cited_in_graph).copy()

    # Keep only semantic edges
    edges_to_remove = [
        (u, v, k) for u, v, k, d in subG.edges(data=True, keys=True)
        if d.get("edge_type") not in SEMANTIC_EDGE_TYPES
    ]
    for u, v, k in edges_to_remove:
        subG.remove_edge(u, v, key=k)

    if subG.number_of_edges() == 0:
        return None

    # Layout -- higher k = more separation between nodes
    pos = nx.spring_layout(subG, seed=42, k=3.0, iterations=80)

    # Build edge traces (one per edge type for legend)
    edge_traces = {}
    for u, v, d in subG.edges(data=True):
        etype = d.get("edge_type", "UNKNOWN")
        if etype not in edge_traces:
            edge_traces[etype] = {"x": [], "y": []}
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces[etype]["x"].extend([x0, x1, None])
        edge_traces[etype]["y"].extend([y0, y1, None])

    fig = go.Figure()

    # Add edge traces
    for etype, coords in edge_traces.items():
        color = EDGE_COLORS.get(etype, "#bdc3c7")
        fig.add_trace(go.Scatter(
            x=coords["x"], y=coords["y"],
            mode="lines",
            line=dict(width=1.5, color=color),
            name=etype,
            hoverinfo="none",
        ))

    # Build node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_hover = []

    node_ids = []
    node_tags = []  # short numeric tags for display
    for idx, fid in enumerate(cited_in_graph, 1):
        if fid not in pos:
            continue
        x, y = pos[fid]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(fid)
        node_tags.append(f"#{idx}")

        detail = frame_lookup.get(fid, {})
        ft = detail.get("frame_type", "other")
        node_colors.append(FRAME_TYPE_COLORS.get(ft, "#95a5a6"))

        # Label: short numeric tag on the node
        node_text.append(f"#{idx}")

        # Hover text
        subject = detail.get("subject", "?")
        author = detail.get("author", "?")
        node_hover.append(
            f"<b>#{idx}</b> {ft.upper()}<br>"
            f"{fid}<br>"
            f"Subject: {subject}<br>"
            f"Author: {author}"
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=16, color=node_colors, line=dict(width=1.5, color="white")),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=node_hover,
        hoverinfo="text",
        name="Frames",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Cited Frames & Semantic Edges", y=0.98, x=0.01,
                   xanchor="left", yanchor="top"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig, node_ids, node_tags


# ─────────────────────────────────────────────
# SUMMARY FORMATTING
# ─────────────────────────────────────────────

def _render_summary(text):
    """Render summary text with improved readability.

    Splits long text into paragraphs on sentence boundaries and
    bolds the first sentence of each paragraph as a topic sentence.
    """
    import re
    text = text.strip()
    if not text:
        return

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 3:
        # Short summary: just bold the first sentence
        sentences[0] = f"**{sentences[0]}**"
        st.markdown(" ".join(sentences))
        return

    # Group sentences into paragraphs of 2-3 sentences
    paragraphs = []
    i = 0
    while i < len(sentences):
        chunk_size = 3 if len(sentences) - i > 4 else 2
        chunk = sentences[i:i + chunk_size]
        # Bold the first sentence of each paragraph (topic sentence)
        chunk[0] = f"**{chunk[0]}**"
        paragraphs.append(" ".join(chunk))
        i += chunk_size

    st.markdown("\n\n".join(paragraphs))


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────

def _sanitize_for_pdf(text):
    """Replace Unicode characters that aren't in Latin-1 with ASCII equivalents."""
    if not text:
        return ""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "-",    # bullet
        "\u2026": "...",  # ellipsis
        "\u2264": "<=",   # less than or equal
        "\u2265": ">=",   # greater than or equal
        "\u00b2": "^2",   # superscript 2
        "\u00b3": "^3",   # superscript 3
        "\u00b0": " deg", # degree
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b4": "delta",
        "\u03bc": "mu",
        "\u03bd": "nu",
        "\u03c1": "rho",
        "\u03c3": "sigma",
        "\u0394": "Delta",
    }
    for uni, ascii_rep in replacements.items():
        text = text.replace(uni, ascii_rep)
    # Strip remaining non-Latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _md_to_html(text):
    """Convert markdown bold (**text**) to HTML <b>text</b> and sanitize for PDF."""
    import re
    text = _sanitize_for_pdf(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return text


def _strip_md_bold(text):
    """Remove markdown **bold** markers for plain text rendering."""
    import re
    return re.sub(r'\*\*(.+?)\*\*', r'\1', _sanitize_for_pdf(text))


def generate_pdf_report(result):
    """Generate a polished PDF document from a DFM report result dict.

    Returns bytes (BytesIO) ready for st.download_button.
    """
    from fpdf import FPDF

    # Color palette
    C_PRIMARY = (41, 128, 185)      # blue
    C_ACCENT = (44, 62, 80)         # dark navy
    C_LIGHT_BG = (236, 240, 241)    # light gray
    C_RULE_GREEN = (39, 174, 96)    # green
    C_WARN_ORANGE = (230, 126, 34)  # orange
    C_CONTRA_RED = (192, 57, 43)    # red
    C_GAP_PURPLE = (142, 68, 173)   # purple
    C_BODY = (44, 62, 80)           # dark text
    C_MUTED = (127, 140, 141)       # gray text
    C_WHITE = (255, 255, 255)

    # Discipline colors (cycle)
    DISC_COLORS = [
        (41, 128, 185), (39, 174, 96), (142, 68, 173),
        (230, 126, 34), (192, 57, 43), (22, 160, 133),
        (52, 73, 94), (243, 156, 18),
    ]

    report = result["report"]
    backfill_result = result.get("backfill_result")
    query = result.get("query", "")
    elapsed = result.get("elapsed_s", 0)
    timestamp = result.get("timestamp", "")
    token_usage = result.get("token_usage", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    def _reset():
        pdf.set_x(pdf.l_margin)

    def _safe_text(h, text, **kw):
        _reset()
        pdf.multi_cell(0, h, _sanitize_for_pdf(text),
                       new_x="LMARGIN", new_y="NEXT", **kw)

    def _safe_html(html):
        _reset()
        pdf.write_html(html)
        pdf.ln(1)
        _reset()

    def _colored_banner(text, bg_color, text_color=C_WHITE, font_size=16):
        """Full-width colored banner with text."""
        _reset()
        y = pdf.get_y()
        pdf.set_fill_color(*bg_color)
        pdf.set_text_color(*text_color)
        pdf.set_font("Helvetica", "B", font_size)
        pdf.cell(usable_w, 10, _sanitize_for_pdf(f"  {text}"),
                 new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(3)

    def _section_header(title, color=C_PRIMARY, font_size=13):
        """Colored section header with accent bar."""
        _reset()
        pdf.ln(2)
        y = pdf.get_y()
        # Accent bar
        pdf.set_fill_color(*color)
        pdf.rect(pdf.l_margin, y, 3, 8, "F")
        pdf.set_x(pdf.l_margin + 6)
        pdf.set_font("Helvetica", "B", font_size)
        pdf.set_text_color(*color)
        pdf.cell(0, 8, _sanitize_for_pdf(title), new_x="LMARGIN", new_y="NEXT")
        # Thin line
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)
        pdf.set_text_color(*C_BODY)

    def _sub_header(title, color=C_ACCENT):
        """Smaller sub-section header."""
        _reset()
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*color)
        pdf.cell(0, 6, _sanitize_for_pdf(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*C_BODY)
        pdf.ln(1)

    def _quote_block(text, left_color=C_MUTED):
        """Indented quote block with left border."""
        _reset()
        y = pdf.get_y()
        pdf.set_fill_color(*left_color)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        indent = 8
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(usable_w - indent, 4.5, _sanitize_for_pdf(text),
                       new_x="LMARGIN", new_y="NEXT")
        y2 = pdf.get_y()
        # Left accent bar
        pdf.rect(pdf.l_margin + 3, y, 2, y2 - y, "F")
        pdf.set_text_color(*C_BODY)
        pdf.ln(2)

    def _bullet(text, bullet_color=C_PRIMARY):
        """Colored bullet point."""
        _reset()
        y = pdf.get_y()
        # Bullet dot
        pdf.set_fill_color(*bullet_color)
        pdf.ellipse(pdf.l_margin + 2, y + 1.5, 2.5, 2.5, "F")
        pdf.set_x(pdf.l_margin + 8)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*C_BODY)
        pdf.multi_cell(usable_w - 8, 5, _strip_md_bold(text),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def _numbered_item(num, text, num_color=C_RULE_GREEN):
        """Numbered item with colored number circle."""
        _reset()
        y = pdf.get_y()
        # Number circle
        pdf.set_fill_color(*num_color)
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 8)
        cx = pdf.l_margin + 4
        cy = y + 2.5
        pdf.ellipse(cx - 3.5, cy - 3.5, 7, 7, "F")
        pdf.set_xy(cx - 3.5, cy - 2.5)
        pdf.cell(7, 5, str(num), align="C")
        # Text (with bold support via write_html)
        pdf.set_x(pdf.l_margin + 12)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*C_BODY)
        pdf.multi_cell(usable_w - 12, 5, _strip_md_bold(text),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def _info_box(text, bg_color=C_LIGHT_BG, text_color=C_MUTED):
        """Subtle background box for metadata/notes."""
        _reset()
        y = pdf.get_y()
        pdf.set_font("Helvetica", "I", 8)
        # Measure height
        lines = pdf.multi_cell(usable_w - 8, 4, _sanitize_for_pdf(text),
                               dry_run=True, output="LINES")
        h = len(lines) * 4 + 4
        pdf.set_fill_color(*bg_color)
        pdf.rect(pdf.l_margin, y, usable_w, h, "F")
        pdf.set_xy(pdf.l_margin + 4, y + 2)
        pdf.set_text_color(*text_color)
        pdf.multi_cell(usable_w - 8, 4, _sanitize_for_pdf(text),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*C_BODY)
        pdf.ln(3)

    # ══════════════════════════════════════════════
    # TITLE BANNER
    # ══════════════════════════════════════════════
    _reset()
    pdf.set_fill_color(*C_ACCENT)
    pdf.rect(pdf.l_margin, pdf.get_y(), usable_w, 22, "F")
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(pdf.l_margin + 6, pdf.get_y() + 3)
    pdf.cell(0, 8, "DFM Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(pdf.l_margin + 6, pdf.get_y() - 1)
    pdf.cell(0, 5, "Design-for-Manufacturability Knowledge Graph Agent",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(pdf.get_y() + 4)

    # ── Metadata bar ──
    meta_parts = []
    if timestamp:
        meta_parts.append(f"Generated: {timestamp}")
    if elapsed:
        meta_parts.append(f"Analysis time: {elapsed}s")
    total_tok = token_usage.get("total_tokens", 0)
    if total_tok:
        meta_parts.append(f"{total_tok:,} tokens")
    if meta_parts:
        _info_box("  |  ".join(meta_parts))

    # ── Query ──
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*C_PRIMARY)
    _reset()
    pdf.cell(0, 6, "QUERY", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*C_BODY)
    pdf.set_font("Helvetica", "", 11)
    _safe_text(6, query)
    pdf.ln(3)

    # ══════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════
    _section_header("Summary", C_PRIMARY)
    pdf.set_font("Helvetica", "", 10)
    _safe_html(_md_to_html(report.summary))
    pdf.ln(4)

    # ══════════════════════════════════════════════
    # DESIGN RULES
    # ══════════════════════════════════════════════
    rules = report.design_rules
    if rules:
        _section_header(f"Design Rules ({len(rules)})", C_RULE_GREEN)
        for i, rule in enumerate(rules, 1):
            _numbered_item(i, rule.claim, C_RULE_GREEN)
        pdf.ln(2)

    # ══════════════════════════════════════════════
    # DISCIPLINES
    # ══════════════════════════════════════════════
    for d_idx, disc in enumerate(report.disciplines):
        disc_name = disc.discipline.replace("_", " ").title()
        disc_color = DISC_COLORS[d_idx % len(DISC_COLORS)]
        _colored_banner(disc_name, disc_color)

        # First principles
        if disc.first_principles:
            _quote_block(disc.first_principles, disc_color)

        # Formulas
        if disc.formulas:
            _sub_header("Governing Equations", disc_color)
            for f in disc.formulas:
                try:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(*disc_color)
                    _safe_text(5, f.formula_name)
                    pdf.set_text_color(*C_BODY)
                    pdf.set_font("Courier", "", 9)
                    _safe_text(5, f"  {f.expression}")
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(*C_MUTED)
                    _safe_text(4, f"  Variables: {f.variables}")
                    _safe_text(4, f"  {f.relevance}")
                    pdf.set_text_color(*C_BODY)
                    pdf.ln(2)
                except Exception:
                    pdf.set_text_color(*C_BODY)
                    pdf.set_font("Helvetica", "", 8)
                    _safe_text(4, f"[Formula: {f.formula_name}]")

        # Evidence
        if disc.evidence:
            _sub_header(f"Evidence ({len(disc.evidence)} claims)", disc_color)
            for ev in disc.evidence:
                _bullet(ev.claim, disc_color)
            pdf.ln(1)

        # Reasoning chains
        if disc.reasoning_chains:
            _sub_header(f"Reasoning Chains ({len(disc.reasoning_chains)})", disc_color)
            for chain in disc.reasoning_chains:
                pdf.set_font("Helvetica", "BI", 9)
                pdf.set_text_color(*disc_color)
                _safe_text(5, chain.chain_label)
                pdf.set_text_color(*C_BODY)
                pdf.set_font("Helvetica", "", 8.5)
                for cf in chain.frames:
                    role = cf.role_in_chain.replace("_", " ").title()
                    edge = f" [{cf.edge_type}]" if cf.edge_type else ""
                    _reset()
                    pdf.set_x(pdf.l_margin + 6)
                    # Role badge
                    role_colors = {
                        "Root Cause": (231, 76, 60),
                        "Effect": (230, 126, 34),
                        "Mitigation": (39, 174, 96),
                    }
                    rc = role_colors.get(role, disc_color)
                    pdf.set_text_color(*rc)
                    pdf.set_font("Helvetica", "B", 8)
                    pdf.cell(pdf.get_string_width(role) + 2, 4, role)
                    pdf.set_text_color(*C_MUTED)
                    pdf.set_font("Helvetica", "", 8)
                    if edge:
                        pdf.cell(pdf.get_string_width(edge) + 1, 4, edge)
                    pdf.set_text_color(*C_BODY)
                    pdf.set_font("Helvetica", "", 8.5)
                    pdf.ln(4)
                    pdf.set_x(pdf.l_margin + 10)
                    pdf.multi_cell(usable_w - 10, 4.5, _sanitize_for_pdf(cf.claim),
                                   new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                pdf.ln(2)

        pdf.ln(2)

    # ══════════════════════════════════════════════
    # CONTRADICTIONS
    # ══════════════════════════════════════════════
    if report.contradictions:
        _section_header("Contradictions", C_CONTRA_RED)
        for c in report.contradictions:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*C_CONTRA_RED)
            _safe_text(5, c.summary)
            pdf.set_text_color(*C_BODY)
            pdf.ln(1)

            # Position A
            _reset()
            pdf.set_fill_color(214, 234, 248)  # light blue
            y = pdf.get_y()
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C_PRIMARY)
            pdf.set_x(pdf.l_margin + 3)
            pdf.cell(0, 4, "POSITION A", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*C_BODY)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(usable_w - 6, 4.5, _sanitize_for_pdf(c.position_a),
                           new_x="LMARGIN", new_y="NEXT")
            y2 = pdf.get_y()
            pdf.rect(pdf.l_margin, y, usable_w, y2 - y + 2, "D")
            pdf.ln(3)

            # Position B
            _reset()
            pdf.set_fill_color(253, 235, 208)  # light orange
            y = pdf.get_y()
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C_WARN_ORANGE)
            pdf.set_x(pdf.l_margin + 3)
            pdf.cell(0, 4, "POSITION B", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*C_BODY)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(usable_w - 6, 4.5, _sanitize_for_pdf(c.position_b),
                           new_x="LMARGIN", new_y="NEXT")
            y2 = pdf.get_y()
            pdf.rect(pdf.l_margin, y, usable_w, y2 - y + 2, "D")
            pdf.ln(2)

            if hasattr(c, "edge_reasoning") and c.edge_reasoning:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(*C_MUTED)
                _safe_text(4, f"Edge reasoning: {c.edge_reasoning}")
                pdf.set_text_color(*C_BODY)
            pdf.ln(4)
        pdf.ln(2)

    # ══════════════════════════════════════════════
    # KNOWLEDGE GAPS
    # ══════════════════════════════════════════════
    gaps = report.knowledge_gaps
    if gaps:
        _section_header(f"Knowledge Gaps ({len(gaps)})", C_GAP_PURPLE)
        for gap in gaps:
            _bullet(gap, C_GAP_PURPLE)
        pdf.ln(2)

    # ══════════════════════════════════════════════
    # BACKFILL
    # ══════════════════════════════════════════════
    if backfill_result and hasattr(backfill_result, "backfills") and backfill_result.backfills:
        _section_header("Textbook Backfill", C_GAP_PURPLE)
        _info_box("General engineering knowledge -- NOT from the knowledge graph. "
                  "Provided for completeness.")

        for bf in backfill_result.backfills:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*C_GAP_PURPLE)
            _safe_text(5, f"Gap: {bf.gap}")
            pdf.set_text_color(*C_BODY)
            pdf.set_font("Helvetica", "", 9.5)
            _safe_text(5, bf.backfill_answer)
            footer = bf.confidence_note or ""
            if bf.relevant_standards:
                footer += f"  |  {bf.relevant_standards}"
            if footer:
                _info_box(footer)
            pdf.ln(3)

    # ══════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════
    pdf.ln(6)
    _reset()
    pdf.set_draw_color(*C_MUTED)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    _reset()
    pdf.cell(0, 4, "Generated by DFM Knowledge Graph Agent  |  "
             "UC Berkeley Co-Design Lab  |  Douglas McGowan",
             new_x="LMARGIN", new_y="NEXT")

    # Return as bytes
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────
# RENDER FULL REPORT
# ─────────────────────────────────────────────

def render_report(result, G, concise=False):
    """Render a complete DFM report from a result dict."""
    report = result["report"]
    enrichment = result["enrichment"]
    diagnostic = result["diagnostic"]
    sub_aspects = result["sub_aspects"]
    backfill_result = result["backfill_result"]
    frame_lookup = {d["frame_id"]: d for d in result["frame_details"]}
    query = result["query"]

    # Diagnostic
    render_diagnostic(diagnostic)

    st.divider()

    # Sub-aspects
    if sub_aspects:
        with st.expander("Query Decomposition", expanded=False):
            for asp in sub_aspects:
                st.markdown(f"- {asp}")

    # Summary -- improve readability by splitting into paragraphs
    st.markdown("### Summary")
    _render_summary(report.summary)

    st.divider()

    # Design Rules
    n_rules = len(report.design_rules)
    st.markdown(f"### Design Rules ({n_rules})")
    render_design_rules(report, frame_lookup, concise=concise)

    st.divider()

    # Disciplines
    st.markdown("### Analysis by Discipline")
    render_disciplines(report, frame_lookup)

    # Contradictions
    render_contradictions(report, frame_lookup)

    st.divider()

    # Knowledge Gaps + Backfill
    render_knowledge_gaps(report, backfill_result, result_dict=result, concise=concise)

    st.divider()

    # Context Summary & Graph
    st.markdown("### \U0001f50e Source Context & Knowledge Graph")
    cited_ids = sorted(collect_cited_frame_ids(report))
    cache_key = tuple(cited_ids) + (query,)

    if cache_key in st.session_state.context_cache:
        with st.expander("Source Context & Credibility Summary", expanded=False):
            st.markdown(st.session_state.context_cache[cache_key])
    else:
        if st.button("\U0001f50e Get Source Context & Credibility Summary", key=f"ctx_{query[:20]}"):
            client = get_openai_client()
            if not client:
                st.error("OPENAI_API_KEY not set. Cannot generate context summary.")
            else:
                with st.spinner("Walking provenance graph and generating summary..."):
                    provenance = gather_provenance_context(G, cited_ids)
                    summary = summarize_context(client, provenance, query)
                    st.session_state.context_cache[cache_key] = summary
                    st.rerun()

    # Graph visualization
    with st.expander("\U0001f578\ufe0f Knowledge Graph View", expanded=False):
        graph_result = build_graph_figure(report, G, frame_lookup)
        if graph_result:
            fig, graph_node_ids, graph_node_tags = graph_result
            st.plotly_chart(fig, use_container_width=True)
            # Render clickable frame links below the graph with matching tags
            st.caption("Click a frame to see full details:")
            cols = st.columns(min(len(graph_node_ids), 5))
            for i, fid in enumerate(graph_node_ids):
                tag = graph_node_tags[i]
                detail = frame_lookup.get(fid, {})
                ft = detail.get("frame_type", "other") if detail else "?"
                with cols[i % min(len(graph_node_ids), 5)]:
                    render_frame_popover(fid, frame_lookup, short_label=f"{tag} {ft}")
        else:
            st.caption("Not enough connected frames to visualize.")

    # Stats footer
    n_cited = len(collect_cited_frame_ids(report))
    n_rules = len(report.design_rules)
    n_chains = sum(len(d.reasoning_chains) for d in report.disciplines)
    n_gaps = len(report.knowledge_gaps)

    st.divider()
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Cited Frames", n_cited)
    fc2.metric("Design Rules", n_rules)
    fc3.metric("Reasoning Chains", n_chains)
    fc4.metric("Knowledge Gaps", n_gaps)

    # ── Token usage & timing for this query ──
    token_usage = result.get("token_usage", {})
    elapsed_s = result.get("elapsed_s")
    if token_usage or elapsed_s:
        with st.expander("\u23f1\ufe0f Pipeline stats", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            if elapsed_s:
                mins, secs = divmod(int(elapsed_s), 60)
                sc1.metric("Elapsed", f"{mins}m {secs}s" if mins else f"{secs}s")
            if token_usage:
                sc2.metric("API calls", token_usage.get("api_calls", 0))
                sc3.metric("Total tokens", f"{token_usage.get('total_tokens', 0):,}")

            if token_usage:
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Prompt tokens", f"{token_usage.get('prompt_tokens', 0):,}",
                           help="Tokens sent to the model (system + user messages)")
                tc2.metric("Completion tokens", f"{token_usage.get('completion_tokens', 0):,}",
                           help="Tokens generated by the model")
                tc3.metric("Embedding tokens", f"{token_usage.get('embedding_tokens', 0):,}",
                           help="Tokens used for query/sub-aspect embedding")


# ─────────────────────────────────────────────
# RESEARCH BACKGROUND
# ─────────────────────────────────────────────

def _render_research_background(stats):
    """Render the Research Background page."""
    st.title("Research Background")
    st.caption(
        "How this system works, why it exists, and how it connects to "
        "ongoing research in design, manufacturing, and AI."
    )

    # ── The Problem ──
    st.markdown("## The Problem: Late Manufacturing Feedback")
    st.markdown(
        "In mechanical product design, 70-80% of a product's lifecycle cost "
        "is locked in during the early design stage. Despite this, most "
        "manufacturability feedback arrives late -- after drawings reach the "
        "shop floor. A machinist might flag a tolerance that can't be held, a "
        "wall thickness that will chatter, or a pocket geometry that requires "
        "special fixturing. By then, redesign is expensive.\n\n"
        "Existing DFM tools (Siemens NX Checkmate, DFMPro, CoLab, etc.) "
        "check geometric rules -- thin walls, deep pockets, small fillets -- "
        "but these rules capture only a narrow slice of the manufacturing "
        "knowledge that experienced practitioners carry. The harder problems -- "
        "context-dependent process constraints, capability-specific limitations, "
        "assembly-level conflicts -- live in the heads of machinists and "
        "manufacturing engineers, not in rule databases."
    )

    # ── Tacit Knowledge ──
    st.markdown("## Tacit Knowledge: Theory and the Manufacturing Gap")
    st.markdown(
        "The concept of tacit knowledge has deep roots. The philosopher "
        "**Michael Polanyi** introduced the idea in *The Tacit Dimension* "
        "(1966), arguing that *\"we can know more than we can tell.\"* "
        "Polanyi observed that much of human expertise -- recognizing a face, "
        "riding a bicycle, diagnosing a fault by sound -- resists explicit "
        "articulation. The expert *knows* but cannot fully explain the "
        "rules they follow, because the knowledge is embedded in perception, "
        "motor skill, and pattern recognition built through practice.\n\n"
        "Polanyi distinguished between **focal awareness** (what we attend to) "
        "and **subsidiary awareness** (the background knowledge that shapes "
        "how we interpret what we see). A machinist hearing chatter doesn't "
        "consciously run through vibration theory -- they *hear* the problem "
        "and adjust. This subsidiary knowledge is precisely what's missing "
        "from rule-based DFM systems.\n\n"
        "In the 1990s, **Nonaka and Takeuchi** built on Polanyi's work in "
        "*The Knowledge-Creating Company* (1995), proposing the **SECI model** "
        "of organizational knowledge creation. Knowledge moves through four "
        "modes: **Socialization** (tacit to tacit, through shared experience), "
        "**Externalization** (tacit to explicit, through dialogue and metaphor), "
        "**Combination** (explicit to explicit, through systematic organization), "
        "and **Internalization** (explicit to tacit, through practice). "
        "Most existing DFM tools operate only in the Combination quadrant -- "
        "reorganizing knowledge that's already explicit. The harder step, "
        "Externalization, is what happens when a machinist explains *why* "
        "something fails in a forum post.\n\n"
        "More recent work has confirmed that this gap persists. Sun et al. "
        "(2024) surveyed tacit knowledge management in product design and "
        "found that while the problem is widely acknowledged, few scalable "
        "solutions exist. Fenoglio et al. (2022) proposed tacit knowledge "
        "elicitation processes for Industry 4.0, and Kernan Freire et al. "
        "(2023) explored using intelligent assistants to elicit shop-floor "
        "knowledge -- both pointing to the same conclusion: the knowledge "
        "exists in practitioners' heads, and extracting it remains a "
        "bottleneck.\n\n"
        "There's also an urgency to this. The manufacturing workforce is "
        "aging, and with retirements comes the risk of permanent knowledge "
        "loss. If a machinist with 30 years of experience retires and their "
        "know-how was never documented, it's gone. Online forums represent "
        "one of the few places where this externalization happens naturally -- "
        "practitioners voluntarily articulating their expertise to help "
        "others. The challenge is capturing and structuring that externalized "
        "knowledge at scale."
    )

    # ── Knowledge Frames ──
    st.markdown("## Knowledge Frames: From AI Theory to Practitioner Expertise")
    st.markdown(
        "The term **\"frame\"** in this project is a deliberate nod to a "
        "concept with a long history in artificial intelligence.\n\n"
        "In 1974, **Marvin Minsky** published *\"A Framework for Representing "
        "Knowledge,\"* proposing that humans understand new situations by "
        "retrieving and adapting mental **frames** -- stereotyped structures "
        "that carry default assumptions, slots for variable details, and "
        "connections to related frames. When you walk into a restaurant, you "
        "don't reason from first principles about chairs, menus, and waiters; "
        "you activate a *restaurant frame* and fill in the specifics. "
        "Minsky argued that AI systems would need similar structures to "
        "reason effectively about the real world.\n\n"
        "Around the same time, **Roger Schank and Robert Abelson** (1977) "
        "extended this idea with *scripts* -- frame-like structures for "
        "event sequences -- in *Scripts, Plans, Goals, and Understanding.* "
        "Their insight was that much of human reasoning depends on "
        "stereotyped knowledge structures that package expectations, "
        "exceptions, and contextual defaults together. A *machining script* "
        "might include default feeds and speeds, expected failure modes, "
        "and known workarounds -- exactly the kind of knowledge that "
        "experienced practitioners carry.\n\n"
        "Later knowledge representation work, including **ontologies** "
        "(Gruber, 1993) and **description logics** (Baader et al., 2003), "
        "formalized these ideas for computational use. Ontologies define "
        "shared vocabularies and relationships for a domain; description "
        "logics provide the mathematical basis for reasoning over them. "
        "Modern knowledge graphs descend from this tradition -- they "
        "structure knowledge as entities and relationships that machines "
        "can traverse and query.\n\n"
        "In this project, a **knowledge frame** adapts Minsky's original "
        "concept for manufacturing expertise. Each frame is one distinct "
        "unit of design or manufacturing knowledge, extracted from a forum "
        "post by a large language model. Like Minsky's frames, each one "
        "carries typed slots (the metadata schema), default assumptions "
        "(scope, applicability), and connections to related frames "
        "(the semantic edges in the knowledge graph). A single post often "
        "contains multiple frames -- a machinist recounting a failed job "
        "might share a universal lesson, a case-specific observation, and "
        "a clever workaround all in the same paragraph.\n\n"
        "Each frame is classified by type:"
    )

    frame_data = {
        "Frame Type": [
            "Risk", "Heuristic", "Principle", "Workaround",
            "Observation", "Comparison", "Case"
        ],
        "What It Captures": [
            "A condition that leads to failure or limitation",
            "A practical rule of thumb or directed advice",
            "A causal explanation of why something happens",
            "A non-obvious fix for a known limitation",
            "An empirical or factual claim",
            "Two or more approaches weighed against each other",
            "A specific real-world job or incident"
        ],
        "Example": [
            "Thin walls under 2mm deflect under cutting forces, causing chatter",
            "Keep pocket depth under 4x the tool diameter for rigidity",
            "Aluminum's thermal expansion causes drift during long cuts",
            "Fill thin-walled pockets with wax for support during machining",
            "6061-T6 machines significantly better than 7075 at high speeds",
            "Climb milling gives better finish; conventional is safer on older machines",
            "Ran 500 parts with 0.5mm walls using vacuum fixture, zero scrap"
        ]
    }
    st.table(frame_data)

    st.markdown(
        "Beyond the type classification, each frame carries structured "
        "metadata: the **verbatim source quote**, the engineering discipline "
        "it applies to (scope), how confident the author sounds (epistemic "
        "stance), specific numbers or thresholds mentioned, and conditions "
        "under which the knowledge applies. This metadata enables precise "
        "retrieval and allows downstream systems to reason about the "
        "knowledge rather than just pattern-match against it."
    )

    # ── Pipeline ──
    st.markdown("## How the Knowledge Base Is Built")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Forum Posts", f"{stats.get('posts', 1936):,}")
    col2.metric("Unique Authors", f"{stats.get('authors', 1138):,}")
    col3.metric("Knowledge Frames", f"{stats.get('frames', 3879):,}")
    col4.metric("Semantic Edges", f"{stats.get('semantic_edges', 4272):,}")

    st.markdown(
        "The pipeline has four stages:\n\n"
        "**1. Data Collection & Filtering.** "
        "Forum threads are collected from five Stack Exchange communities "
        "(Engineering, Mechanics, Woodworking, Electronics, Robotics) via "
        "the public API. A keyword strategy spanning both design decisions and "
        "machining failure modes captures posts where manufacturing knowledge "
        "is likely to surface. An LLM filters out off-topic threads, reducing "
        "~12,800 raw threads to ~1,400 relevant posts.\n\n"
        "**2. Knowledge Frame Extraction.** "
        "Each post is processed by a reasoning-capable LLM using a structured "
        "schema. The model extracts multiple frames per post, each grounded in "
        "a verbatim quote. A multi-step filtering pipeline checks epistemic "
        "stance classification, deduplicates overlapping frames, and validates "
        "source quotes against the original text.\n\n"
        "**3. Multi-Vector Embedding.** "
        "Each frame is embedded three ways: a *knowledge vector* (subject, "
        "main point, context), a *language vector* (the raw source quote), "
        "and a *metadata vector* (frame type, scope, stance). This enables "
        "retrieval by semantic meaning, by linguistic similarity to practitioner "
        "jargon, or by knowledge properties.\n\n"
        "**4. Knowledge Graph Construction.** "
        "A provenance graph links frames to their source posts, threads, and "
        "authors. Then, for each frame, nearest neighbors in the embedding "
        "space are evaluated and the LLM infers semantic relationships: one "
        "frame might **specify** a general principle, **mitigate** a risk, "
        "establish a **causal link**, or **contradict** another claim. This "
        "produces a graph where retrieval can follow reasoning chains, not just "
        "keyword matches."
    )

    # ── Why Both: KGs + Embeddings ──
    st.markdown("## Why Both a Knowledge Graph and an Embedding Space?")
    st.markdown(
        "A natural question is: why not just use one or the other? The answer "
        "lies in the different types of reasoning each structure supports, "
        "and why LLM-based agents need both to work well in expert domains.\n\n"
        "**The limits of LLMs alone.** "
        "Large language models are powerful general reasoners, but they have "
        "well-documented weaknesses in specialized domains: hallucination "
        "(generating plausible-sounding but incorrect claims), inability to "
        "cite specific sources, and a tendency to flatten nuance into generic "
        "advice. When a designer asks about thin-wall machining, an LLM might "
        "produce a reasonable-sounding answer, but it can't tell you *who* said "
        "it, *when* it was true, or *what conditions* it depends on. This is "
        "the core problem that retrieval-augmented generation (RAG) was "
        "designed to address (Lewis et al., 2020).\n\n"
        "**What embeddings provide: semantic recall.** "
        "Vector embeddings map knowledge into a continuous space where "
        "semantically similar content is geometrically close. This solves the "
        "*vocabulary mismatch problem* -- a designer asking about \"part "
        "deflection during cutting\" can find a machinist's post about "
        "\"workpiece flex when hogging\" even though no keywords overlap. "
        "Embeddings excel at **finding relevant knowledge** across different "
        "vocabularies and levels of formality. But they have a fundamental "
        "limitation: they treat each item independently. An embedding search "
        "can find a risk, but it can't tell you that another frame two hops "
        "away describes the workaround for that risk.\n\n"
        "**What knowledge graphs provide: structured reasoning.** "
        "Knowledge graphs capture *relationships* between entities -- "
        "relationships that embeddings flatten away. In the AI literature, "
        "this traces back to **semantic networks** (Quillian, 1968), one of "
        "the earliest knowledge representation formalisms, where concepts "
        "were nodes and relationships were typed edges. Modern knowledge "
        "graphs (Hogan et al., 2021) inherit this structure: they enable "
        "**multi-hop reasoning** (following a chain of connections), "
        "**constraint propagation** (if A contradicts B and B specifies C, "
        "then A may conflict with C), and **provenance tracking** (tracing "
        "a claim back through the discussion that produced it). For an "
        "engineering domain, this means the system can retrieve not just "
        "a risk, but the workaround that mitigates it, the principle that "
        "explains it, and the case study where someone applied it.\n\n"
        "**Why agents need both.** "
        "The emerging paradigm of **agentic AI** -- systems that plan, "
        "retrieve, reason, and act in multi-step workflows -- exposes the "
        "limits of either approach alone. An agent working with only "
        "embeddings can find relevant documents but can't follow reasoning "
        "chains. An agent working with only a knowledge graph can traverse "
        "relationships but can't handle queries that don't match its "
        "vocabulary exactly. The combination enables what neither can do "
        "alone: the embedding space casts a wide, vocabulary-agnostic net "
        "to find relevant starting points, and the knowledge graph provides "
        "structured paths to follow from those starting points to related "
        "insights. This is analogous to how a human expert might first "
        "*recall* relevant experiences (associative/embedding-like) and "
        "then *reason through* the connections between them "
        "(structured/graph-like).\n\n"
        "This dual-structure approach also provides a natural defense "
        "against hallucination. Because every claim in the synthesized "
        "report must trace back to a specific frame with a verbatim source "
        "quote, the agent can't fabricate expertise -- it can only report "
        "and synthesize what practitioners actually said."
    )

    # ── Validation ──
    st.markdown("## Validation: Human Experts + Automated Review")
    st.markdown(
        "For a knowledge base like this to be useful in engineering, it "
        "needs to be validated. The approach combines human expert review "
        "with calibrated LLM-as-a-judge evaluation.\n\n"
        "Professional machinists and designers reviewed a random sample of "
        "frames across multiple quality dimensions -- extraction accuracy, "
        "clarity, main point quality, and metadata correctness. The core "
        "quality metrics showed strong inter-rater agreement "
        "(Gwet's AC1 > 0.84), with 0 out of 26 sampled frames recommended "
        "for removal.\n\n"
        "An automated review pipeline was then calibrated against the human "
        "ratings using in-context learning from the expert gold standard. "
        "An interesting finding from this calibration: LLMs tend to be harsher "
        "critics of tacit and experiential knowledge, defaulting to textbook "
        "standards. The human expert ratings were needed to teach the model "
        "that anecdotal, context-specific insights have genuine engineering "
        "value.\n\n"
        "This connects to a broader challenge in LLM-as-a-judge research -- "
        "in expert domains, automated evaluation agreement with humans drops "
        "to 60-68%, and more reasoning doesn't always help. The calibration "
        "step is essential."
    )

    # ── How the Agent Works ──
    st.markdown("## How the Agent Uses This Knowledge")
    st.markdown(
        "The DFM Agent on this site is a retrieval-augmented generation (RAG) "
        "system built on top of the knowledge graph and embedding space. "
        "When you ask a question, it:\n\n"
        "1. **Decomposes** the query into sub-aspects (e.g., a question about "
        "thin-walled aluminum parts might decompose into geometry constraints, "
        "fixturing, thermal effects, and tool selection).\n"
        "2. **Retrieves** relevant frames using multi-vector search across all "
        "three embedding spaces.\n"
        "3. **Traverses** the knowledge graph to find connected frames -- a "
        "risk that has a known mitigation, a heuristic that contradicts another, "
        "a principle that explains a workaround.\n"
        "4. **Ranks** frames by a composite score combining embedding similarity, "
        "author credibility, graph distance, and edge confidence.\n"
        "5. **Synthesizes** a structured report with design rules, each citing "
        "the practitioner knowledge it's based on.\n"
        "6. **Identifies gaps** where the knowledge base lacks coverage and "
        "fills them with general engineering guidance, clearly marked as such.\n\n"
        "The key difference from a standard LLM query is grounding: every claim "
        "traces back to a specific practitioner's statement in a real "
        "discussion, not to the model's training data."
    )

    # ── Connections to Research ──
    st.markdown("## Connections to Ongoing Research")
    st.markdown(
        "This work sits at the intersection of several active research areas:\n\n"
        "**Knowledge graphs for manufacturing.** "
        "Recent work has used ontological structures and knowledge graphs to "
        "map design and manufacturing knowledge from structured documents "
        "(Sanya & Shehab, 2023; Jia et al., 2024). This system extends that "
        "approach to *unstructured* forum discourse, capturing knowledge that "
        "doesn't appear in formal documentation.\n\n"
        "**Experiential knowledge capture.** "
        "Goridkov et al. (2023) demonstrated embedding experiential design "
        "knowledge in interactive knowledge graphs through direct expert "
        "elicitation. Bolanos et al. (2025) built multimodal datasets of "
        "product failures for design risk analysis. This work complements "
        "both by mining existing practitioner discussions at scale rather "
        "than requiring direct expert participation.\n\n"
        "**LLM-as-a-judge for expert domains.** "
        "While LLM evaluation methods have shown strong results in general "
        "NLP tasks (Zheng et al., 2023), agreement drops in expert domains "
        "(Szymanski et al., 2025). This system's calibration approach -- using "
        "human expert ratings as gold standards for in-context learning -- "
        "offers one path forward for validating domain-specific knowledge.\n\n"
        "**AI as design collaborator.** "
        "Research on AI in design has shown that the tool's framing matters: "
        "AI that provides information while preserving designer agency leads to "
        "better outcomes than AI that generates solutions directly "
        "(Chong et al., 2023; Zhang et al., 2021). This system is designed as "
        "a knowledge retrieval tool, not a design generator -- it surfaces "
        "practitioner expertise to inform decisions, not to make them."
    )

    # ── Generalizability ──
    st.markdown("## Beyond Machining: A Generalizable Method")
    st.markdown(
        "While this prototype focuses on CNC machining knowledge from Stack "
        "Exchange, the underlying method is domain-agnostic. The same pipeline "
        "-- scrape practitioner discourse, extract structured knowledge frames, "
        "embed and graph them -- could be applied to:\n\n"
        "- **Other manufacturing processes** (welding, casting, additive "
        "manufacturing, sheet metal)\n"
        "- **Other engineering domains** (electrical, civil, software)\n"
        "- **Other knowledge sources** (internal engineering wikis, "
        "manufacturing reports, maintenance logs, supplier communications)\n\n"
        "The frame schema would need domain-specific adaptation, but the "
        "extraction, embedding, and graph construction stages are general. "
        "The validation approach -- human expert calibration followed by "
        "automated LLM review -- is also transferable to any domain where "
        "expert knowledge needs to be captured and verified at scale.\n\n"
        "Future directions include integrating visual data (CAD files, "
        "engineering drawings) through vision-language models, expanding "
        "the knowledge base with new data sources, and studying the impact "
        "of this type of AI-assisted feedback on real design outcomes."
    )

    st.divider()
    st.markdown(
        "*This research is being conducted at UC Berkeley's "
        "[Co-Design Lab](https://codesign.berkeley.edu/) by "
        "[Douglas McGowan](https://www.linkedin.com/in/douglas-mcgowan/). "
        "If you're interested in collaborating, have questions about the "
        "methodology, or want to apply this approach to your domain, "
        "please reach out via the feedback form in the sidebar or by email.*"
    )


# ─────────────────────────────────────────────
# KG EXPLORER
# ─────────────────────────────────────────────

_TOPIC_MODEL_FILE = Path(__file__).parent / "topic_model_output.json"
_SUBJECT_CLUSTERS_FILE = Path(__file__).parent / "subject_clusters.json"
_SEMANTIC_EDGES_FILE = Path(__file__).parent / "semantic_edges.json"


@st.cache_resource
def _ensure_semantic_edges(_G):
    """Load semantic edges from JSON into the graph if not already present."""
    # Check if semantic edges already exist
    has_sem = any(
        d.get("edge_type") in SEMANTIC_EDGE_TYPES
        for _, _, d in _G.edges(data=True)
    )
    if has_sem:
        return _G

    if not _SEMANTIC_EDGES_FILE.exists():
        return _G

    try:
        with open(_SEMANTIC_EDGES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return _G

    added = 0
    for edge in data.get("edges", []):
        fa = edge.get("frame_a", "")
        fb = edge.get("frame_b", "")
        label = edge.get("label", "")
        direction = edge.get("direction", "A_to_B")
        confidence = edge.get("confidence", "medium")
        reasoning = edge.get("reasoning", "")

        if not (_G.has_node(fa) and _G.has_node(fb)):
            continue
        if label not in SEMANTIC_EDGE_TYPES:
            continue

        if label == "CONTRADICTS" or direction == "bidirectional":
            _G.add_edge(fa, fb, edge_type=label, confidence=confidence, reasoning=reasoning)
            _G.add_edge(fb, fa, edge_type=label, confidence=confidence, reasoning=reasoning)
            added += 2
        elif direction == "B_to_A":
            _G.add_edge(fb, fa, edge_type=label, confidence=confidence, reasoning=reasoning)
            added += 1
        else:
            _G.add_edge(fa, fb, edge_type=label, confidence=confidence, reasoning=reasoning)
            added += 1

    return _G


@st.cache_data
def _load_topic_model():
    """Load BERTopic model output for topic filtering."""
    if not _TOPIC_MODEL_FILE.exists():
        return None
    try:
        with open(_TOPIC_MODEL_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


@st.cache_data
def _build_frame_index(_G):
    """Build a searchable index of all frames from the graph."""
    frames = []
    for nid, data in _G.nodes(data=True):
        if data.get("node_type") != "Frame":
            continue
        # Get parent post and author via graph edges
        author = "unknown"
        source_url = ""
        post_score = 0
        reputation = -1
        credibility = "?"
        is_accepted = False
        se_tags = ""
        thread_title = ""
        se_site = ""

        # Walk graph: Post -> Frame (so Frame's predecessor is Post)
        # Post -> Author (successor), Thread -> Post (so Post's predecessor is Thread)
        for pred in _G.predecessors(nid):
            pred_data = _G.nodes.get(pred, {})
            if pred_data.get("node_type") == "Post":
                post_score = pred_data.get("post_score", 0) or 0
                is_accepted = pred_data.get("is_accepted_answer", False)
                se_tags = pred_data.get("se_tags", "")
                se_site = pred_data.get("se_site", "")
                # Get author from post (Post -> Author via successor)
                for _, succ in _G.out_edges(pred):
                    succ_data = _G.nodes.get(succ, {})
                    if succ_data.get("node_type") == "Author":
                        author = succ_data.get("author_username", "unknown")
                        reputation = succ_data.get("reputation", -1)
                        credibility = succ_data.get("credibility_tier", "?")
                # Get thread from post (Thread -> Post, so Post's predecessor is Thread)
                for pp in _G.predecessors(pred):
                    pp_data = _G.nodes.get(pp, {})
                    if pp_data.get("node_type") == "Thread":
                        thread_title = pp_data.get("thread_title", "")
                        source_url = pp
                        break

        frames.append({
            "frame_id": nid,
            "frame_type": data.get("frame_type", "?"),
            "scope": data.get("scope", "?"),
            "subject": data.get("subject", "?"),
            "main_point": data.get("main_point", "?"),
            "source_quote": data.get("source_quote", ""),
            "applicability": data.get("applicability", "?"),
            "epistemic_stance": data.get("epistemic_stance", "?"),
            "post_role": data.get("post_role", "?"),
            "example_value": data.get("example_value", ""),
            "example_context": data.get("example_context", ""),
            "author": author,
            "credibility": credibility,
            "reputation": reputation,
            "post_score": post_score,
            "is_accepted": is_accepted,
            "source_url": source_url,
            "se_tags": se_tags,
            "se_site": se_site,
            "thread_title": thread_title,
        })
    return frames


def _build_neighborhood_figure(G, center_frame_id, max_hops=2):
    """Build a Plotly network graph showing a frame's neighborhood."""
    if not G.has_node(center_frame_id):
        return None

    # Collect neighbors via semantic edges (BFS up to max_hops)
    visited = {center_frame_id}
    frontier = {center_frame_id}
    for _ in range(max_hops):
        next_frontier = set()
        for fid in frontier:
            for _, nbr, d in G.edges(fid, data=True):
                if d.get("edge_type") in SEMANTIC_EDGE_TYPES and nbr not in visited:
                    if G.nodes.get(nbr, {}).get("node_type") == "Frame":
                        next_frontier.add(nbr)
            for pred, _, d in G.in_edges(fid, data=True):
                if d.get("edge_type") in SEMANTIC_EDGE_TYPES and pred not in visited:
                    if G.nodes.get(pred, {}).get("node_type") == "Frame":
                        next_frontier.add(pred)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    if len(visited) < 2:
        return None

    subG = G.subgraph(visited).copy()

    # Remove non-semantic edges
    edges_to_remove = [
        (u, v, k) for u, v, k, d in subG.edges(data=True, keys=True)
        if d.get("edge_type") not in SEMANTIC_EDGE_TYPES
    ]
    for u, v, k in edges_to_remove:
        subG.remove_edge(u, v, key=k)

    if subG.number_of_edges() == 0:
        return None

    pos = nx.spring_layout(subG, seed=42, k=3.0, iterations=80)

    # Edge traces
    edge_traces = {}
    for u, v, d in subG.edges(data=True):
        etype = d.get("edge_type", "UNKNOWN")
        if etype not in edge_traces:
            edge_traces[etype] = {"x": [], "y": []}
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces[etype]["x"].extend([x0, x1, None])
        edge_traces[etype]["y"].extend([y0, y1, None])

    fig = go.Figure()
    for etype, coords in edge_traces.items():
        color = EDGE_COLORS.get(etype, "#bdc3c7")
        fig.add_trace(go.Scatter(
            x=coords["x"], y=coords["y"],
            mode="lines",
            line=dict(width=1.5, color=color),
            name=etype,
            hoverinfo="none",
        ))

    # Node traces
    node_x, node_y, node_colors, node_text, node_hover = [], [], [], [], []
    for fid in visited:
        if fid not in pos:
            continue
        x, y = pos[fid]
        node_x.append(x)
        node_y.append(y)
        nd = G.nodes.get(fid, {})
        ft = nd.get("frame_type", "other")
        node_colors.append(FRAME_TYPE_COLORS.get(ft, "#95a5a6"))
        subject = nd.get("subject", "?")
        # Bold the center node label
        if fid == center_frame_id:
            node_text.append(f"** {subject[:25]} **")
        else:
            node_text.append(subject[:25])
        node_hover.append(
            f"<b>{ft.upper()}</b><br>"
            f"{fid}<br>"
            f"Subject: {subject}<br>"
            f"Main point: {nd.get('main_point', '?')[:80]}"
        )

    # Center node gets larger marker
    node_sizes = [
        24 if list(visited)[i] == center_frame_id else 14
        for i in range(len(node_x))
    ]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=1.5, color="white")),
        text=node_text,
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_hover,
        hoverinfo="text",
        name="Frames",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=f"Neighborhood of {center_frame_id}", y=0.98, x=0.01,
                   xanchor="left", yanchor="top"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        height=550,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def _render_kg_explorer(G, stats):
    """Render the Knowledge Graph Explorer page."""
    # Ensure semantic edges are loaded into the graph
    G = _ensure_semantic_edges(G)
    frames = _build_frame_index(G)
    topic_data = _load_topic_model()

    # Build frame_id -> topic mapping
    frame_to_topic = {}
    topic_labels = {}
    if topic_data:
        for t in topic_data.get("topics", []):
            tid = t["topic_id"]
            label = t.get("label", f"Topic {tid}")
            topic_labels[tid] = label
            for tf in t.get("top_frames", []):
                frame_to_topic[tf["frame_id"]] = tid

    # ── Sidebar controls ──
    with st.sidebar:
        st.markdown("**Filters**")

        search_text = st.text_input(
            "Search subjects & main points",
            placeholder="e.g. thread tapping, wall thickness",
            key="kg_search",
        )

        # Frame type filter
        all_types = sorted(set(f["frame_type"] for f in frames if f["frame_type"] != "?"))
        selected_types = st.multiselect("Frame type", all_types, default=[], key="kg_types")

        # Scope filter
        all_scopes = sorted(set(f["scope"] for f in frames if f["scope"] != "?"))
        selected_scopes = st.multiselect("Scope", all_scopes, default=[], key="kg_scopes")

        # SE site filter
        all_sites = sorted(set(f["se_site"] for f in frames if f["se_site"]))
        selected_sites = st.multiselect("SE site", all_sites, default=[], key="kg_sites")

        # Credibility filter
        selected_cred = st.multiselect(
            "Author credibility", ["high", "medium", "low"], default=[], key="kg_cred"
        )

        # Applicability filter
        selected_app = st.multiselect(
            "Applicability", ["universal", "context_dependent", "limited"],
            default=[], key="kg_app"
        )

        st.divider()
        st.markdown("**Graph settings**")
        explore_hops = st.slider("Neighborhood hops", 1, 3, 2, key="kg_hops",
                                 help="How many semantic edge hops to show around a selected frame")

    # ── Apply filters ──
    filtered = frames
    if search_text:
        terms = search_text.lower().split()
        filtered = [
            f for f in filtered
            if all(
                t in f["subject"].lower() or t in f["main_point"].lower()
                for t in terms
            )
        ]
    if selected_types:
        filtered = [f for f in filtered if f["frame_type"] in selected_types]
    if selected_scopes:
        filtered = [f for f in filtered if f["scope"] in selected_scopes]
    if selected_sites:
        filtered = [f for f in filtered if f["se_site"] in selected_sites]
    if selected_cred:
        filtered = [f for f in filtered if f["credibility"] in selected_cred]
    if selected_app:
        filtered = [f for f in filtered if f["applicability"] in selected_app]

    # ── Main area ──
    st.title("Knowledge Graph Explorer")
    st.caption(
        f"Browse and explore **{stats['frames']:,}** knowledge frames extracted from "
        f"practitioner discussions across **{stats['threads']:,}** threads."
    )

    # ── Overview metrics ──
    tab_browse, tab_stats, tab_diag = st.tabs(["Browse Frames", "Graph Statistics", "Data Diagnostics"])

    with tab_stats:
        st.markdown("### Frame Distributions")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", f"{stats['frames']:,}")
        col2.metric("Semantic Edges", f"{stats['semantic_edges']:,}")
        col3.metric("Authors", f"{stats['authors']:,}")
        col4.metric("Threads", f"{stats['threads']:,}")

        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### By Frame Type")
            type_counts = {}
            for f in frames:
                ft = f["frame_type"]
                type_counts[ft] = type_counts.get(ft, 0) + 1
            sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
            type_labels = [t[0] for t in sorted_types]
            type_values = [t[1] for t in sorted_types]
            type_colors = [FRAME_TYPE_COLORS.get(t, "#95a5a6") for t in type_labels]

            fig_type = go.Figure(go.Bar(
                x=type_labels, y=type_values,
                marker_color=type_colors,
                text=type_values, textposition="outside",
            ))
            fig_type.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="", yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_type, use_container_width=True)

        with chart_col2:
            st.markdown("#### By Scope")
            scope_counts = {}
            for f in frames:
                sc = f["scope"]
                scope_counts[sc] = scope_counts.get(sc, 0) + 1
            sorted_scopes = sorted(scope_counts.items(), key=lambda x: -x[1])
            scope_labels = [s[0] for s in sorted_scopes]
            scope_values = [s[1] for s in sorted_scopes]

            fig_scope = go.Figure(go.Bar(
                x=scope_labels, y=scope_values,
                marker_color="#3498db",
                text=scope_values, textposition="outside",
            ))
            fig_scope.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="", yaxis_title="Count",
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_scope, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("#### By Epistemic Stance")
            stance_counts = {}
            for f in frames:
                es = f["epistemic_stance"]
                stance_counts[es] = stance_counts.get(es, 0) + 1
            sorted_stances = sorted(stance_counts.items(), key=lambda x: -x[1])
            fig_stance = go.Figure(go.Bar(
                x=[s[0] for s in sorted_stances],
                y=[s[1] for s in sorted_stances],
                marker_color="#9b59b6",
                text=[s[1] for s in sorted_stances], textposition="outside",
            ))
            fig_stance.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="", yaxis_title="Count",
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_stance, use_container_width=True)

        with chart_col4:
            st.markdown("#### By Author Credibility")
            cred_counts = {}
            for f in frames:
                cr = f["credibility"]
                cred_counts[cr] = cred_counts.get(cr, 0) + 1
            cred_order = ["high", "medium", "low"]
            cred_colors_map = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
            fig_cred = go.Figure(go.Bar(
                x=[c for c in cred_order if c in cred_counts],
                y=[cred_counts.get(c, 0) for c in cred_order if c in cred_counts],
                marker_color=[cred_colors_map.get(c, "#95a5a6") for c in cred_order if c in cred_counts],
                text=[cred_counts.get(c, 0) for c in cred_order if c in cred_counts],
                textposition="outside",
            ))
            fig_cred.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="", yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cred, use_container_width=True)

        # Semantic edge breakdown
        st.markdown("#### Semantic Edge Types")
        edge_type_counts = {}
        for _, _, d in G.edges(data=True):
            et = d.get("edge_type")
            if et in SEMANTIC_EDGE_TYPES:
                edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        if edge_type_counts:
            sorted_edges = sorted(edge_type_counts.items(), key=lambda x: -x[1])
            ec1, ec2, ec3, ec4 = st.columns(len(sorted_edges))
            for col, (etype, count) in zip([ec1, ec2, ec3, ec4][:len(sorted_edges)], sorted_edges):
                col.metric(etype, f"{count:,}")

        # Topic model summary
        if topic_data:
            st.markdown("---")
            st.markdown("#### Topics (BERTopic)")
            meta = topic_data.get("metadata", {})
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Topics", meta.get("total_topics", "?"))
            tc2.metric("Outlier frames", f"{meta.get('outlier_count', '?')} ({meta.get('outlier_pct', '?')}%)")
            tc3.metric("Total frames", meta.get("total_frames", "?"))

            topics = topic_data.get("topics", [])
            top_topics = sorted(topics, key=lambda t: -t.get("n_frames", 0))[:15]
            fig_topics = go.Figure(go.Bar(
                x=[t["label"] for t in top_topics],
                y=[t["n_frames"] for t in top_topics],
                marker_color="#1abc9c",
                text=[t["n_frames"] for t in top_topics],
                textposition="outside",
            ))
            fig_topics.update_layout(
                height=400, margin=dict(l=20, r=20, t=20, b=100),
                xaxis_title="", yaxis_title="Frames",
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_topics, use_container_width=True)

    with tab_diag:
        # ── Pipeline Overview ──
        st.markdown("### Pipeline Overview")
        st.caption("End-to-end data flow from raw scrape to knowledge graph.")

        po_cols = st.columns(4)
        po_cols[0].metric("Raw Threads Scraped", "12,805")
        po_cols[1].metric("After GPT Filtering", "639 questions")
        po_cols[2].metric("Posts with Answers", "1,399")
        po_cols[3].metric("Posts Sent to LLM", "1,936")

        po_cols2 = st.columns(4)
        po_cols2[0].metric("Threads Processed", "911")
        po_cols2[1].metric("Frames Extracted", "3,879")
        po_cols2[2].metric("Embedding Vectors", "11,637")
        po_cols2[3].metric("Graph Nodes", "7,864")

        po_cols3 = st.columns(3)
        po_cols3[0].metric("Structural Edges", "8,776")
        po_cols3[1].metric("Semantic Edges", "4,372")
        po_cols3[2].metric("Total Edges", "13,148")

        st.markdown("---")

        # ── Post Role Distribution ──
        st.markdown("### Post Role Distribution")
        role_counts = {}
        for f in frames:
            pr = f.get("post_role", "?")
            if pr and pr != "?":
                role_counts[pr] = role_counts.get(pr, 0) + 1
        if role_counts:
            sorted_roles = sorted(role_counts.items(), key=lambda x: -x[1])
            fig_roles = go.Figure(go.Bar(
                x=[r[0] for r in sorted_roles],
                y=[r[1] for r in sorted_roles],
                marker_color="#e67e22",
                text=[r[1] for r in sorted_roles],
                textposition="outside",
            ))
            fig_roles.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="", yaxis_title="Count",
                xaxis_tickangle=-30,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_roles, use_container_width=True)

        st.markdown("---")

        diag_col1, diag_col2 = st.columns(2)

        # ── Applicability Distribution ──
        with diag_col1:
            st.markdown("### Applicability Distribution")
            app_data = [
                ("context_dependent", 2724),
                ("universal", 1115),
                ("situational", 40),
            ]
            fig_app = go.Figure(go.Bar(
                x=[a[0] for a in app_data],
                y=[a[1] for a in app_data],
                marker_color="#2ecc71",
                text=[a[1] for a in app_data],
                textposition="outside",
            ))
            fig_app.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="", yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_app, use_container_width=True)

        # ── SE Site Distribution ──
        with diag_col2:
            st.markdown("### SE Site Distribution")
            site_counts = {}
            for f in frames:
                site = f.get("se_site", "")
                if site:
                    site_counts[site] = site_counts.get(site, 0) + 1
            if site_counts:
                sorted_sites = sorted(site_counts.items(), key=lambda x: -x[1])
                fig_sites = go.Figure(go.Bar(
                    x=[s[0] for s in sorted_sites],
                    y=[s[1] for s in sorted_sites],
                    marker_color="#3498db",
                    text=[s[1] for s in sorted_sites],
                    textposition="outside",
                ))
                fig_sites.update_layout(
                    height=350, margin=dict(l=20, r=20, t=20, b=40),
                    xaxis_title="", yaxis_title="Frames",
                    xaxis_tickangle=-30,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sites, use_container_width=True)

        st.markdown("---")

        # ── Edge Type Breakdown ──
        st.markdown("### Edge Type Breakdown")
        st.caption("Structural edges (blue) and semantic edges (orange).")

        edge_labels = [
            "POST_CONTAINS_FRAME", "THREAD_CONTAINS_POST",
            "POST_AUTHORED_BY", "SE_ANSWERS",
            "SPECIFIES", "MITIGATES",
            "CAUSAL_LINK", "CONTRADICTS",
        ]
        edge_vals = [3879, 1936, 1936, 1025, 2159, 1218, 637, 358]
        edge_colors = [
            "#3498db", "#3498db", "#3498db", "#3498db",
            "#e67e22", "#e67e22", "#e67e22", "#e67e22",
        ]

        fig_edges = go.Figure(go.Bar(
            y=edge_labels[::-1],
            x=edge_vals[::-1],
            orientation="h",
            marker_color=edge_colors[::-1],
            text=edge_vals[::-1],
            textposition="outside",
        ))
        fig_edges.update_layout(
            height=350, margin=dict(l=180, r=40, t=20, b=40),
            xaxis_title="Count", yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_edges, use_container_width=True)

        st.markdown("---")

        # ── Frames per Post ──
        st.markdown("### Frames per Post")

        # Count frames per post from the graph
        post_frame_counts = []
        for nid, data in G.nodes(data=True):
            if data.get("node_type") == "Post":
                count = 0
                for _, succ, edata in G.out_edges(nid, data=True):
                    succ_data = G.nodes.get(succ, {})
                    if succ_data.get("node_type") == "Frame":
                        count += 1
                if count > 0:
                    post_frame_counts.append(count)

        if post_frame_counts:
            avg_fpp = sum(post_frame_counts) / len(post_frame_counts)
            sorted_counts = sorted(post_frame_counts)
            n = len(sorted_counts)
            median_fpp = (sorted_counts[n // 2] if n % 2 == 1
                          else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2)
            max_fpp = max(post_frame_counts)

            fpp_cols = st.columns(4)
            fpp_cols[0].metric("Posts with Frames", f"{len(post_frame_counts):,}")
            fpp_cols[1].metric("Average", f"{avg_fpp:.2f}")
            fpp_cols[2].metric("Median", f"{median_fpp:.0f}")
            fpp_cols[3].metric("Max", f"{max_fpp}")

            # Build histogram bins
            from collections import Counter
            fpp_counter = Counter(post_frame_counts)
            fpp_x = sorted(fpp_counter.keys())
            fpp_y = [fpp_counter[k] for k in fpp_x]

            fig_fpp = go.Figure(go.Bar(
                x=fpp_x,
                y=fpp_y,
                marker_color="#9b59b6",
                text=fpp_y,
                textposition="outside",
            ))
            fig_fpp.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="Frames per Post",
                yaxis_title="Number of Posts",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_fpp, use_container_width=True)
        else:
            st.info("No post-frame relationships found in graph.")

    with tab_browse:
        st.markdown(f"**{len(filtered):,}** frames match your filters")

        if not filtered:
            st.info("No frames match the current filters. Try broadening your search.")
            return

        # Pagination
        page_size = 25
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        if "kg_page" not in st.session_state:
            st.session_state.kg_page = 0
        # Reset page when filters change
        filter_hash = hash((search_text, tuple(selected_types), tuple(selected_scopes),
                           tuple(selected_sites), tuple(selected_cred), tuple(selected_app)))
        if st.session_state.get("_kg_filter_hash") != filter_hash:
            st.session_state.kg_page = 0
            st.session_state["_kg_filter_hash"] = filter_hash

        current_page = st.session_state.kg_page
        start = current_page * page_size
        end = min(start + page_size, len(filtered))
        page_frames = filtered[start:end]

        # Page navigation
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("< Prev", disabled=(current_page == 0), key="kg_prev"):
                st.session_state.kg_page = max(0, current_page - 1)
                st.rerun()
        with nav_col2:
            st.markdown(
                f"<div style='text-align:center'>Page {current_page + 1} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with nav_col3:
            if st.button("Next >", disabled=(current_page >= total_pages - 1), key="kg_next"):
                st.session_state.kg_page = min(total_pages - 1, current_page + 1)
                st.rerun()

        st.markdown("---")

        # Frame list
        for frame in page_frames:
            ft = frame["frame_type"]
            color = FRAME_TYPE_COLORS.get(ft, "#95a5a6")

            with st.expander(
                f"**{ft.upper()}** | {frame['scope']} | {frame['subject'][:60]}",
                expanded=False,
            ):
                # Header
                st.markdown(
                    f"### {ft.upper()} &nbsp; | &nbsp; "
                    f"`{frame['scope']}` &nbsp; | &nbsp; "
                    f"`{frame['applicability']}`"
                )
                st.caption(f"`{frame['frame_id']}`")

                st.markdown(f"**Subject:** {frame['subject']}")
                st.markdown(f"**Main Point:** {frame['main_point']}")

                if frame.get("source_quote"):
                    quote = frame["source_quote"].replace("\n- ", "\n  - ").replace("\n* ", "\n  * ")
                    st.markdown("**Source Quote**")
                    st.markdown(f"> {quote}")

                st.markdown(
                    f"**Epistemic stance:** `{frame['epistemic_stance']}` "
                    f"&nbsp; | &nbsp; Post role: `{frame['post_role']}`"
                )

                if frame.get("example_value"):
                    st.markdown(f"**Example value:** {frame['example_value']}")
                if frame.get("example_context"):
                    st.markdown(f"**Example context:** {frame['example_context']}")

                st.divider()

                # Author info
                rep = frame.get("reputation", -1)
                rep_str = f"{rep:,}" if rep != -1 else "N/A"
                accepted = " | Accepted" if frame.get("is_accepted") else ""
                st.caption(
                    f"Author: **{frame['author']}** &nbsp; | &nbsp; "
                    f"Credibility: **{frame['credibility']}** &nbsp; | &nbsp; "
                    f"Reputation: **{rep_str}** &nbsp; | &nbsp; "
                    f"Post score: **{frame['post_score']}**"
                    f"{accepted}"
                )

                if frame.get("source_url"):
                    st.markdown(f"[\U0001f517 View Original Thread]({frame['source_url']})")
                if frame.get("se_tags"):
                    st.caption(f"Tags: {frame['se_tags']}")
                if frame.get("thread_title"):
                    st.caption(f"Thread: {frame['thread_title']}")

                # Neighborhood graph button
                st.divider()
                if st.button(
                    "\U0001f578\ufe0f Show Neighborhood Graph",
                    key=f"nbr_{frame['frame_id']}",
                ):
                    st.session_state["_kg_selected_frame"] = frame["frame_id"]

                if st.session_state.get("_kg_selected_frame") == frame["frame_id"]:
                    fig = _build_neighborhood_figure(G, frame["frame_id"], max_hops=explore_hops)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                        # List connected frames
                        st.markdown("**Connected frames:**")
                        for _, nbr, d in G.edges(frame["frame_id"], data=True):
                            if d.get("edge_type") in SEMANTIC_EDGE_TYPES:
                                nbr_data = G.nodes.get(nbr, {})
                                if nbr_data.get("node_type") == "Frame":
                                    st.caption(
                                        f"`{d['edge_type']}` -> "
                                        f"**{nbr_data.get('frame_type', '?').upper()}** "
                                        f"{nbr_data.get('subject', '?')} "
                                        f"(`{nbr}`)"
                                    )
                        for pred, _, d in G.in_edges(frame["frame_id"], data=True):
                            if d.get("edge_type") in SEMANTIC_EDGE_TYPES:
                                pred_data = G.nodes.get(pred, {})
                                if pred_data.get("node_type") == "Frame":
                                    st.caption(
                                        f"**{pred_data.get('frame_type', '?').upper()}** "
                                        f"{pred_data.get('subject', '?')} "
                                        f"(`{pred}`) -> `{d['edge_type']}`"
                                    )
                    else:
                        st.info("No semantic edges found for this frame.")


# ─────────────────────────────────────────────
# VISITOR WELCOME FORM
# ─────────────────────────────────────────────

@st.dialog("Welcome to the DFM Knowledge Graph Agent!")
def _visitor_welcome_dialog():
    """First-time visitor popup to optionally collect name and contact."""
    st.markdown("Help us understand who's using this tool. All fields are **optional**.")
    first_name = st.text_input("First name", key="_dlg_first")
    last_name = st.text_input("Last name", key="_dlg_last")
    contact = st.text_input("Contact info / website (optional)", key="_dlg_contact")

    def _save_and_close():
        st.session_state["_visitor_form_shown"] = True

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit", use_container_width=True, type="primary"):
            st.session_state["_visitor_first_name"] = first_name.strip() or None
            st.session_state["_visitor_last_name"] = last_name.strip() or None
            st.session_state["_visitor_contact_info"] = contact.strip() or None
            _save_and_close()
            st.rerun()
    with col2:
        if st.button("Skip", use_container_width=True):
            _save_and_close()
            st.rerun()


def _inject_welcome_done():
    """Save to localStorage that welcome form was shown (persists across sessions)."""
    import streamlit.components.v1 as components
    components.html(
        """<script>
        try { localStorage.setItem('_dfm_welcome_done', '1'); } catch(e) {}
        </script>""",
        height=0,
    )


# ─────────────────────────────────────────────
# ADMIN PAGE
# ─────────────────────────────────────────────

def _make_admin_token():
    """Create a signed admin token using ADMIN_PASSWORD as the secret."""
    admin_pw = st.secrets.get("ADMIN_PASSWORD", "")
    if not admin_pw:
        return ""
    return hashlib.sha256(f"dfm_admin_{admin_pw}".encode()).hexdigest()[:24]


def _check_admin_token():
    """Check if a valid admin token was passed via query param (from localStorage).

    This allows persistent login across sessions without re-entering the password.
    """
    if st.session_state.get("_admin_authenticated"):
        return True
    token = st.query_params.get("_atok")
    if token and token == _make_admin_token():
        st.session_state["_admin_authenticated"] = True
        # Clean the URL
        del st.query_params["_atok"]
        return True
    return False


def _inject_admin_persistence(logged_in=False):
    """Inject JS to persist/restore admin login via localStorage."""
    import streamlit.components.v1 as components
    token = _make_admin_token()
    if not token:
        return
    if logged_in:
        # Just logged in -- save token to localStorage
        components.html(
            f"""<script>
            localStorage.setItem('_dfm_admin_token', '{token}');
            </script>""",
            height=0,
        )
    else:
        # Not logged in -- check localStorage and redirect with token if found
        components.html(
            f"""<script>
            (function() {{
                var t = localStorage.getItem('_dfm_admin_token');
                if (t && !window.parent.location.search.includes('_atok=')) {{
                    var url = new URL(window.parent.location.href);
                    url.searchParams.set('_atok', t);
                    window.parent.location.replace(url.toString());
                }}
            }})();
            </script>""",
            height=0,
        )


def _render_admin_page():
    """Password-protected admin page showing visitor and query logs."""
    st.title("Admin Dashboard")

    # Password gate
    admin_pw = st.secrets.get("ADMIN_PASSWORD", "")
    if not admin_pw:
        st.warning("ADMIN_PASSWORD not configured in Streamlit secrets.")
        return

    # Check for persistent login token from localStorage
    _check_admin_token()

    if not st.session_state.get("_admin_authenticated"):
        # Try to restore from localStorage
        _inject_admin_persistence(logged_in=False)
        pw = st.text_input("Enter admin password", type="password", key="admin_pw_input")
        if pw:
            if pw == admin_pw:
                st.session_state["_admin_authenticated"] = True
                _inject_admin_persistence(logged_in=True)
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    # Save token on successful login (in case it's a fresh password entry)
    _inject_admin_persistence(logged_in=True)

    # Logout button
    if st.button("Logout", key="admin_logout"):
        st.session_state["_admin_authenticated"] = False
        # Clear localStorage token via JS
        import streamlit.components.v1 as components
        components.html(
            """<script>
            localStorage.removeItem('_dfm_admin_token');
            </script>""",
            height=0,
        )
        st.rerun()

    # ── Token Tally ──
    st.subheader("Token Tally")
    tally = _load_token_tally()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", tally.get("queries", 0))
    col2.metric("Page Views", tally.get("page_views", 0))
    col3.metric("Unique IPs", len(tally.get("unique_ips", [])))
    st.caption(f"Tracking since: {tally.get('since', 'unknown')}")
    st.divider()

    # ── Visitor Log ──
    st.subheader("Visitor Log")
    visitors = _read_log(_VISITOR_LOG_FILE)
    if visitors:
        import pandas as pd
        df_v = pd.DataFrame(visitors)
        # Reorder columns for readability
        desired_cols = ["timestamp", "first_name", "last_name", "contact_info", "ip", "user_agent", "session_id", "ip_hash"]
        cols = [c for c in desired_cols if c in df_v.columns] + [c for c in df_v.columns if c not in desired_cols]
        df_v = df_v[cols].iloc[::-1].reset_index(drop=True)
        st.dataframe(df_v, use_container_width=True, hide_index=True)
        csv_v = df_v.to_csv(index=False)
        st.download_button("Download Visitor Log CSV", csv_v, "visitor_log.csv", "text/csv", key="dl_visitor_csv")
    else:
        st.caption("No visitor records yet.")
    st.divider()

    # ── Query Log ──
    st.subheader("Query Log")
    queries = _read_log(_QUERY_LOG_FILE)
    if queries:
        import pandas as pd
        df_q = pd.DataFrame(queries)
        desired_cols = ["timestamp", "visitor_name", "query", "model", "flash_mode", "from_cache",
                        "elapsed_s", "num_rules", "num_disciplines", "num_gaps", "ip", "session_id", "summary_preview"]
        cols = [c for c in desired_cols if c in df_q.columns] + [c for c in df_q.columns if c not in desired_cols]
        df_q = df_q[cols].iloc[::-1].reset_index(drop=True)
        st.dataframe(df_q, use_container_width=True, hide_index=True)
        csv_q = df_q.to_csv(index=False)
        st.download_button("Download Query Log CSV", csv_q, "query_log.csv", "text/csv", key="dl_query_csv")
    else:
        st.caption("No query records yet.")

    # Debug tools
    st.divider()
    with st.expander("Debug: Google Sheets Connection"):
        if st.button("Test Google Sheets", key="test_gsheet"):
            # Step 1: Check gspread import
            if gspread is None:
                st.error("gspread not installed")
            else:
                st.success(f"gspread {gspread.__version__} installed")

            # Step 2: Check credentials
            try:
                creds = st.secrets["gcp_service_account"]
                st.success(f"Service account found: {dict(creds).get('client_email', 'unknown')}")
            except Exception as e:
                st.error(f"No gcp_service_account in secrets: {e}")
                creds = None

            # Step 3: Check sheet key
            sheet_key = st.secrets.get("GSHEET_LOG_KEY", "")
            if sheet_key:
                st.success(f"GSHEET_LOG_KEY: {sheet_key[:10]}...")
            else:
                st.error("GSHEET_LOG_KEY not set in secrets")

            # Step 4: Try to connect
            if creds and sheet_key:
                try:
                    gc = gspread.service_account_from_dict(dict(creds))
                    st.success("Authenticated with Google API")
                except Exception as e:
                    st.error(f"Auth failed: {e}")
                    gc = None

                if gc:
                    try:
                        ss = gc.open_by_key(sheet_key)
                        st.success(f"Opened spreadsheet: {ss.title}")
                        worksheets = [ws.title for ws in ss.worksheets()]
                        st.info(f"Worksheets: {worksheets}")
                    except Exception as e:
                        st.error(f"Failed to open spreadsheet: {e}")

                    # Step 5: Try a test write
                    try:
                        test_ws = ss.worksheet("_test")
                    except Exception:
                        test_ws = ss.add_worksheet(title="_test", rows=10, cols=3)
                        test_ws.append_row(["timestamp", "status", "message"])
                    test_ws.append_row([
                        _now_pst().strftime("%Y-%m-%d %I:%M:%S %p PST"),
                        "OK",
                        "Connection test successful"
                    ])
                    st.success("Test row written to '_test' worksheet!")

    with st.expander("Debug: Request Headers"):
        try:
            headers = dict(st.context.headers)
            st.json(headers)
            ip_addr = getattr(st.context, "ip_address", "not available")
            st.caption(f"st.context.ip_address = `{ip_addr}`")
        except Exception as e:
            st.caption(f"Could not read headers: {e}")


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Load resources
    G = load_graph()
    client = get_openai_client()
    example_cache = load_example_cache()

    # Client-side JS: resolve public IP + check welcome form localStorage
    # This runs as a single redirect so server-side render has all info
    _inject_client_js()

    # Show welcome form for first-time visitors
    # Skip if: already shown, admin/owner, or IP not yet resolved (JS redirect pending)
    if not st.session_state.get("_visitor_form_shown"):
        ip = _get_visitor_ip()
        ip_resolved = ip and not _is_private_ip(ip)
        if ip_resolved and _is_admin_visitor():
            # Owner -- never show the form
            st.session_state["_visitor_form_shown"] = True
        elif ip_resolved:
            # Regular visitor with resolved IP -- show the form
            _visitor_welcome_dialog()
        # else: IP not resolved yet -- JS redirect is about to fire, skip for now
    if st.session_state.get("_visitor_form_shown"):
        # Persist the "shown" flag to localStorage for future sessions
        _inject_welcome_done()

    # Record visitor (once per session) -- after form so we capture name
    _record_visitor()

    if G is None:
        st.sidebar.error("Graph not found. Run `build_graph.py` first.")
        st.stop()

    G = _ensure_semantic_edges(G)
    stats = get_graph_stats(G)

    # ── SIDEBAR (common) ──
    with st.sidebar:
        st.title("\U0001f5ff DFM KG Agent")
        st.caption(
            f"**{stats['frames']:,}** frames &nbsp;|&nbsp; "
            f"**{stats['semantic_edges']:,}** semantic edges\n\n"
            f"**{stats['authors']:,}** authors &nbsp;|&nbsp; "
            f"**{stats['threads']:,}** threads"
        )
        st.divider()
        page = st.radio(
            "Navigate",
            ["\U0001f50d DFM Agent", "\U0001f4d6 Research Background", "\U0001f578\ufe0f KG Explorer", "\U0001f4ca Admin"],
            label_visibility="collapsed",
            key="page_nav",
        )
        st.divider()

    # ── Research Background page ──
    if page == "\U0001f4d6 Research Background":
        _render_research_background(stats)
        return

    # ── KG Explorer page ──
    if page == "\U0001f578\ufe0f KG Explorer":
        _render_kg_explorer(G, stats)
        return

    # ── Admin page ──
    if page == "\U0001f4ca Admin":
        _render_admin_page()
        return

    # ── DFM Agent page (original) ──
    with st.sidebar:
        if st.session_state.get("active_idx", -1) >= 0:
            if st.button("\u2190 Home", use_container_width=True):
                st.session_state.active_idx = -1
                st.session_state.flash_on = False
                st.rerun()

        st.markdown("**Controls**")

        # Flash Mode toggle — visually separate, above standard modes
        flash_on = st.toggle(
            "\u26a1 Flash Mode",
            value=False,
            key="flash_on",
            help="Ultra-fast ~20s analysis with gpt-4.1-mini. Disables advanced settings.",
        )
        if flash_on:
            st.caption(
                f"{FLASH_MODE['description']}  \n"
                f"**{FLASH_MODE['time_range']}** &nbsp;|&nbsp; "
                f"**{FLASH_MODE['token_range']}** &nbsp;|&nbsp; "
                f"**{FLASH_MODE['cost_estimate']}**/query"
            )

        st.divider()

        # Standard analysis modes — disabled when Flash is on
        mode_names = list(ANALYSIS_MODES.keys())
        selected_mode = st.radio(
            "Analysis mode",
            mode_names,
            format_func=lambda m: f"{ANALYSIS_MODES[m]['icon']} {m}",
            index=0,
            disabled=flash_on,
            help="Controls the LLM used for synthesis. Affects speed, depth, and cost.",
        )
        mode_cfg = ANALYSIS_MODES[selected_mode]
        if not flash_on:
            st.caption(
                f"{mode_cfg['description']}  \n"
                f"**{mode_cfg['time_range']}** &nbsp;|&nbsp; "
                f"**{mode_cfg['token_range']}** &nbsp;|&nbsp; "
                f"**{mode_cfg['cost_estimate']}**/query"
            )

        # Retrieval settings — disabled when Flash is on
        if flash_on:
            top_k = FLASH_MODE["top_k"]
            max_hops = FLASH_MODE["max_hops"]
            max_frames = FLASH_MODE["max_frames"]
            st.slider("Top K (frames per query)", min_value=3, max_value=25,
                      value=FLASH_MODE["top_k"], step=1, disabled=True)
            st.slider("Max Hops (graph traversal)", min_value=1, max_value=4,
                      value=FLASH_MODE["max_hops"], step=1, disabled=True)
            st.slider("Max Frames (for synthesis)", min_value=10, max_value=40,
                      value=FLASH_MODE["max_frames"], step=5, disabled=True)
            decompose = False
            backfill = False
            st.checkbox("Decompose query", value=False, disabled=True,
                        help="Disabled in Flash Mode")
            st.checkbox("Backfill knowledge gaps", value=False, disabled=True,
                        help="Disabled in Flash Mode")
        else:
            top_k = st.slider("Top K (frames per query)", min_value=3, max_value=25, value=10, step=1,
                              help="Number of frames to retrieve from the vector index per sub-aspect")
            max_hops = st.slider("Max Hops (graph traversal)", min_value=1, max_value=4, value=2, step=1,
                                 help="How many edge hops to follow in the semantic graph")
            max_frames = st.slider("Max Frames (for synthesis)", min_value=10, max_value=40, value=20, step=5,
                                   help="Maximum frames passed to the LLM for report synthesis")
            decompose = st.checkbox("Decompose query", value=True,
                                    help="Break question into engineering sub-aspects for broader retrieval")
            backfill = st.checkbox("Backfill knowledge gaps", value=True,
                                   help="Generate textbook answers for areas not covered by the knowledge graph")
        concise = st.checkbox("Concise output", value=False, disabled=flash_on,
                              help="Shorter report: max 5 design rules, collapsed evidence, compact knowledge gaps")
        pin_popovers = st.checkbox("Pin popovers", value=False,
                                   help="Keep frame detail popovers open while scrolling (default: auto-dismiss on scroll)")

        # Query history
        if st.session_state.query_history:
            st.divider()
            st.markdown("**Query History**")
            for i, res in enumerate(st.session_state.query_history):
                q_short = res["query"][:40] + ("..." if len(res["query"]) > 40 else "")
                ts = res.get("timestamp", "")
                is_active = (i == st.session_state.active_idx)
                btn_type = "primary" if is_active else "secondary"
                ts_rel = _relative_time(res.get("timestamp_dt"))
                btn_label = f"{ts_rel}  {q_short}" if ts_rel else (f"{ts}  {q_short}" if ts else q_short)
                if st.button(
                    btn_label,
                    key=f"hist_{i}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state.active_idx = i
                    st.session_state._scroll_to_report = True
                    st.rerun()

        # ── Session token usage ──
        st.divider()
        s_tok = st.session_state.session_tokens
        st.markdown("**Your Session**")
        tc1, tc2 = st.columns(2)
        tc1.metric("Queries", s_tok["queries"])
        tc2.metric("Total tokens", f"{s_tok['total_tokens']:,}")
        tc3, tc4 = st.columns(2)
        tc3.metric("Input tokens", f"{s_tok['prompt_tokens']:,}",
                   help="Tokens sent to the model (system + user messages)")
        tc4.metric("Output tokens", f"{s_tok['completion_tokens']:,}",
                   help="Tokens generated by the model")

        # ── About the Developer ──
        st.divider()
        st.markdown("**About the Developer**")
        st.markdown(
            "**Douglas McGowan**  \n"
            "Ph.D. Researcher, Mechanical Engineering  \n"
            "[Co-Design Lab](https://codesign.berkeley.edu/), UC Berkeley  \n"
            "Advisor: Dr. Kosa Goucher-Lambert"
        )
        st.caption(
            "Research focus: computational design, AI/ML in product "
            "development, and design for manufacturability."
        )
        st.markdown(
            "\U0001f4e7 [dpm5970@berkeley.edu](mailto:dpm5970@berkeley.edu)  \n"
            "\U0001f517 [LinkedIn](https://www.linkedin.com/in/douglas-mcgowan/) "
            " | [Co-Design Lab](https://codesign.berkeley.edu/team/douglas-mcgowan/)"
        )

        # ── Feedback ──
        st.divider()
        with st.expander("\U0001f4ac Send Feedback", expanded=False):
            st.caption("Loved it? Found a bug? Have an idea? All feedback welcome!")
            with st.form("feedback_form", clear_on_submit=True):
                fb_name = st.text_input("Name (optional)")
                fb_email = st.text_input("Email (optional)")
                fb_message = st.text_area(
                    "Your feedback",
                    placeholder="What did you like? What could be better? Feature ideas, bug reports, or just say hi...",
                    height=120,
                )
                fb_submitted = st.form_submit_button(
                    "\U0001f4e8 Send", use_container_width=True
                )
            if fb_submitted:
                if not fb_message.strip():
                    st.warning("Please enter a message.")
                else:
                    emailed = _save_feedback(fb_name.strip(), fb_email.strip(), fb_message.strip())
                    if emailed:
                        st.success("Thanks! Your feedback has been sent.")
                    else:
                        st.success("Thanks! Your feedback has been recorded.")

    # ── MAIN AREA ──

    # API key check
    if not client:
        st.warning(
            "\u26a0\ufe0f **OPENAI_API_KEY not set.** "
            "Add it to `.streamlit/secrets.toml` or set it as an environment variable.\n\n"
            "```toml\n"
            "# .streamlit/secrets.toml\n"
            'OPENAI_API_KEY = "sk-..."\n'
            "```"
        )
        st.stop()

    title_col, info_col = st.columns([6, 1])
    with title_col:
        st.title("Design-for-Manufacturing (DFM) Knowledge Agent")
    with info_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        with st.popover("\u2753 What is this?"):
            st.markdown(
                "### The Research Problem\n\n"
                "When engineers design parts for CNC machining, they often discover "
                "manufacturability issues only after sending drawings to the shop. "
                "A machinist might say *\"you can't hold that tolerance on a wall "
                "this thin\"* or *\"this pocket geometry will chatter with any "
                "reasonable feed rate\"* -- but that feedback comes late, costs "
                "redesign cycles, and depends on having access to an experienced "
                "machinist in the first place.\n\n"
                "The core insight is that **most of this practical manufacturing "
                "knowledge already exists** -- scattered across machinist forums, "
                "Q&A sites, and shop-floor conversations. It's just not structured "
                "in a way that a design tool can use.\n\n"
                "---\n\n"
                "### What Are Knowledge Frames?\n\n"
                "A **knowledge frame** is a structured unit of manufacturing "
                "knowledge extracted from a real practitioner discussion. Each "
                "frame captures one specific insight and classifies it by type:\n\n"
                "| Frame Type | What It Captures | Example |\n"
                "|---|---|---|\n"
                "| **Risk** | A condition that leads to failure | *Thin walls under "
                "2mm deflect under cutting forces, causing chatter* |\n"
                "| **Heuristic** | A practical rule of thumb | *Keep pocket depth "
                "under 4x the tool diameter for rigidity* |\n"
                "| **Principle** | A causal explanation of why something happens | "
                "*Aluminum's high thermal expansion causes dimensional drift "
                "during long cuts* |\n"
                "| **Workaround** | A non-obvious fix for a known limitation | "
                "*Fill thin-walled pockets with wax during machining for support* |\n"
                "| **Observation** | An empirical or factual claim | *6061-T6 "
                "machines significantly better than 7075 at high speeds* |\n"
                "| **Comparison** | Two or more approaches weighed against each other | "
                "*Climb milling gives better finish but conventional is safer "
                "on older machines* |\n"
                "| **Case** | A specific real-world job or incident | *We ran 500 "
                "parts with 0.5mm walls using a vacuum fixture with no scrap* |\n\n"
                "Each frame also records the **original quote**, who said it, "
                "their epistemic confidence, the engineering discipline it "
                "applies to, and any specific numbers or conditions mentioned.\n\n"
                "---\n\n"
                "### How the Agent Uses Frames\n\n"
                "The knowledge graph contains **{:,}** frames extracted from "
                "practitioner discussions across 5 Stack Exchange communities. "
                "Frames are connected by semantic edges -- one frame might "
                "**mitigate** a risk, **contradict** a heuristic, or **specify** "
                "a more general principle.\n\n"
                "When you ask a question, the agent:\n\n"
                "1. **Searches** the frame embeddings to find relevant knowledge\n"
                "2. **Traverses** the graph to pull in connected frames "
                "(e.g., a risk *and* the workaround that addresses it)\n"
                "3. **Synthesizes** a structured report with actionable design "
                "rules, each backed by cited frames\n"
                "4. **Identifies gaps** where the graph has no coverage and "
                "fills them with general engineering guidance\n\n"
                "Every claim in the report links back to the original source "
                "-- click any frame ID to see the verbatim quote, author, "
                "and thread URL.\n\n"
                "---\n\n"
                "**Built by** [Douglas McGowan](https://www.linkedin.com/in/douglas-mcgowan/), "
                "Ph.D. Researcher at UC Berkeley's "
                "[Co-Design Lab](https://codesign.berkeley.edu/)."
            .format(stats['frames']))
    st.caption(
        "Ask a design-for-manufacturability question. The agent retrieves practitioner "
        "knowledge from forum discussions, traverses the knowledge graph for related insights, "
        "and synthesizes a structured report."
    )

    # Inject scroll-dismiss behavior for popovers
    inject_scroll_dismiss(pin_popovers)

    # Query input
    with st.form("query_form", clear_on_submit=False):
        query_text = st.text_input(
            "Ask a DFM question",
            placeholder="e.g. What should I consider when designing thin-walled aluminum parts for CNC machining?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "\U0001f50d Analyze",
            use_container_width=True,
            type="primary",
        )

    # Check for auto-submit from example buttons
    auto_query = st.session_state.pop("_auto_query", None)
    query = None
    if submitted and query_text.strip():
        query = query_text.strip()
    elif auto_query:
        query = auto_query

    # ── Execute pipeline ──
    if query:
        import time as _time

        # Step icons -- unique animation hint per pipeline phase
        _STEP_ICONS = {
            "decompose":  "\U0001f9e9",  # puzzle piece
            "retrieve":   "\U0001f50d",  # magnifying glass
            "traverse":   "\U0001f578\ufe0f",  # spider web
            "rank":       "\U0001f3af",  # bullseye
            "synthesize": "\U0001f9e0",  # brain
            "backfill":   "\U0001f4da",  # books
        }

        # Progress bar + animated step display + elapsed clock
        progress_bar = st.progress(0, text="Starting DFM analysis...")
        step_container = st.empty()

        def _on_progress(step, total, label, step_name=""):
            """Called by query_dfm at each pipeline step."""
            frac = step / total
            icon = _STEP_ICONS.get(step_name, "\u2699\ufe0f")
            elapsed_so_far = _time.time() - _t0
            mins, secs = divmod(int(elapsed_so_far), 60)
            clock = f"{mins}:{secs:02d}" if mins else f"{secs}s"
            progress_bar.progress(frac, text=f"{icon} Step {step}/{total}  |  {clock} elapsed")
            step_container.markdown(
                '<div class="dfm-anim"><div class="dfm-dot"></div>'
                '<div class="dfm-dot"></div><div class="dfm-dot"></div></div>'
                f"### {icon} Step {step} of {total}\n{label}",
                unsafe_allow_html=True,
            )

        _t0 = _time.time()

        # Apply analysis mode: set model and patch reasoning_effort if needed
        if flash_on:
            _agent_module.MODEL = FLASH_MODE["model"]
            _reasoning_effort = None
            top_k = FLASH_MODE["top_k"]
            max_hops = FLASH_MODE["max_hops"]
            max_frames = FLASH_MODE["max_frames"]
            decompose = False
            backfill = False
        else:
            _agent_module.MODEL = mode_cfg["model"]
            _reasoning_effort = mode_cfg["reasoning_effort"]
        _originals = {}

        if _reasoning_effort is not None:
            # Wrap LLM-calling functions to inject reasoning_effort
            from functools import wraps

            _orig_decompose = _agent_module.decompose_query
            _orig_synth = _agent_module.synthesize_report_v2
            _orig_backfill = _agent_module.backfill_knowledge_gaps
            _originals = {"d": _orig_decompose, "s": _orig_synth, "b": _orig_backfill}

            def _patched_decompose(client, question, tracker=None):
                if len(question.split()) <= 5:
                    return [question]
                try:
                    comp = client.beta.chat.completions.parse(
                        model=_agent_module.MODEL,
                        messages=[
                            {"role": "system", "content": DECOMPOSE_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        response_format=QueryDecomposition,
                        reasoning_effort=_reasoning_effort,
                    )
                    if tracker:
                        tracker.record(comp.usage)
                    r = comp.choices[0].message.parsed
                    return r.sub_aspects if r and r.sub_aspects else [question]
                except Exception:
                    return [question]

            def _patched_synth(client, query, ranked_frames, sub_aspects=None, tracker=None):
                frame_blocks = "\n\n".join(
                    _agent_module.format_frame_for_prompt(fid, info)
                    for fid, _, info in ranked_frames
                )
                sub_line = ""
                if sub_aspects and len(sub_aspects) > 1:
                    sub_line = (
                        f"\nSub-aspects investigated: {', '.join(sub_aspects)}\n"
                        "Ensure the report covers findings for each sub-aspect "
                        "where evidence exists.\n"
                    )
                user_msg = (
                    f"Designer's question: {query}\n{sub_line}\n"
                    f"Relevant knowledge frames ({len(ranked_frames)} total):\n\n"
                    f"{frame_blocks}"
                )
                comp = client.beta.chat.completions.parse(
                    model=_agent_module.MODEL,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_PROMPT_V2},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=DFMReportV2,
                    reasoning_effort=_reasoning_effort,
                )
                if tracker:
                    tracker.record(comp.usage)
                return comp.choices[0].message.parsed

            def _patched_backfill(client, report, tracker=None):
                if not report.knowledge_gaps:
                    return None
                gaps_text = "\n".join(f"- {g}" for g in report.knowledge_gaps)
                user_msg = f"Knowledge gaps to backfill:\n{gaps_text}"
                try:
                    comp = client.beta.chat.completions.parse(
                        model=_agent_module.MODEL,
                        messages=[
                            {"role": "system", "content": BACKFILL_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        response_format=BackfillOutput,
                        reasoning_effort=_reasoning_effort,
                    )
                    if tracker:
                        tracker.record(comp.usage)
                    return comp.choices[0].message.parsed
                except Exception:
                    return None

            _agent_module.decompose_query = _patched_decompose
            _agent_module.synthesize_report_v2 = _patched_synth
            _agent_module.backfill_knowledge_gaps = _patched_backfill

        try:
            report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker = query_dfm(
                query,
                top_k=top_k,
                max_frames=max_frames,
                max_hops=max_hops,
                decompose=decompose,
                backfill=backfill,
                verbose=False,
                G=G,
                progress_callback=_on_progress,
            )

            elapsed = _time.time() - _t0
            progress_bar.progress(1.0, text="\u2705 Done!")
            step_container.markdown(f"### \u2705 Pipeline complete in {elapsed:.0f}s\nBuilding frame details...")
            frame_details = get_frame_details(report, enrichment, G)

            token_usage = tracker.as_dict() if tracker else {}

            result = {
                "query": query,
                "report": report,
                "enrichment": enrichment,
                "ranked": ranked,
                "sub_aspects": sub_aspects,
                "diagnostic": diagnostic,
                "backfill_result": backfill_result,
                "frame_details": frame_details,
                "timestamp": _now_pst().strftime("%I:%M:%S %p"),
                "timestamp_dt": _now_pst(),
                "elapsed_s": round(elapsed, 1),
                "token_usage": token_usage,
                "flash_mode": flash_on,
            }

            # Update persistent (global) token tally
            _update_token_tally(token_usage)

            # Update session-level token tally
            st_tok = st.session_state.session_tokens
            st_tok["queries"] += 1
            st_tok["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            st_tok["completion_tokens"] += token_usage.get("completion_tokens", 0)
            st_tok["embedding_tokens"] += token_usage.get("embedding_tokens", 0)
            st_tok["total_tokens"] += token_usage.get("total_tokens", 0)

            st.session_state.query_history.append(result)
            st.session_state.active_idx = len(st.session_state.query_history) - 1
            _log_query(result)

        except Exception as e:
            progress_bar.empty()
            step_container.empty()
            st.error(f"**{type(e).__name__}:** {e}")
            st.stop()
        finally:
            # Restore original functions if patched
            if _originals:
                _agent_module.decompose_query = _originals["d"]
                _agent_module.synthesize_report_v2 = _originals["s"]
                _agent_module.backfill_knowledge_gaps = _originals["b"]

        # Clean up progress elements and rerun to show the report
        progress_bar.empty()
        step_container.empty()
        st.toast("DFM analysis complete!", icon="\u2705")
        st.rerun()

    # Display active report
    if st.session_state.active_idx >= 0 and st.session_state.query_history:
        idx = st.session_state.active_idx
        if idx < len(st.session_state.query_history):
            result = st.session_state.query_history[idx]
            st.markdown(f"---")

            # Query line + relative timestamp + PDF download button
            q_col, dl_col = st.columns([5, 1])
            with q_col:
                ts_label = _relative_time(result.get("timestamp_dt"))
                if ts_label:
                    st.markdown(f"**Query:** {result['query']}")
                    st.caption(ts_label)
                else:
                    st.markdown(f"**Query:** {result['query']}")
            with dl_col:
                pdf_bytes = generate_pdf_report(result)
                st.download_button(
                    "\U0001f4c4 PDF",
                    data=pdf_bytes,
                    file_name="dfm_report.pdf",
                    mime="application/pdf",
                    key=f"pdf_dl_{idx}",
                )

            render_report(result, G, concise=concise)

            # ── Query suggestions from knowledge gaps ──
            report = result.get("report")
            if report and hasattr(report, "knowledge_gaps") and report.knowledge_gaps:
                gaps = report.knowledge_gaps[:3]
                st.markdown("---")
                st.markdown("**Related questions you could ask:**")
                suggestion_cols = st.columns(min(len(gaps), 3))
                for i, gap in enumerate(gaps):
                    # Convert gap text into a question
                    q = gap.strip().rstrip(".")
                    # Strip common prefixes
                    for prefix in ["No coverage of ", "Limited data on ", "The knowledge graph lacks ",
                                   "Missing ", "No data on ", "Insufficient coverage of "]:
                        if q.lower().startswith(prefix.lower()):
                            q = q[len(prefix):]
                            break
                    suggestion = f"What should I know about {q.lower()}?"
                    with suggestion_cols[i]:
                        if st.button(f"\U0001f4a1 {suggestion[:80]}", key=f"suggest_{idx}_{i}",
                                     use_container_width=True):
                            st.session_state["_auto_query"] = suggestion
                            st.rerun()

            # Scroll to report if navigating from history
            if st.session_state.pop("_scroll_to_report", False):
                import streamlit.components.v1 as _comp
                _comp.html("""
                <script>
                (function() {
                    var main = window.parent.document.querySelector('[data-testid="stMain"]');
                    if (main) main.scrollTop = 0;
                })();
                </script>
                """, height=0)
    elif st.session_state.active_idx < 0:
        # Home / welcome state
        st.markdown("---")

        # Usage stats
        tally = _load_token_tally()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries", f"{tally.get('queries', 0):,}")
        col2.metric("Input Tokens", f"{tally.get('prompt_tokens', 0):,}")
        col3.metric("Output Tokens", f"{tally.get('completion_tokens', 0):,}")
        col4.metric("Visitors", f"{len(tally.get('unique_ips', []))}")

        st.markdown("### Getting Started")
        st.markdown(
            "This tool answers design & manufacturing questions using a "
            "knowledge base of **3,879 insights** extracted from machinist "
            "and engineer forums. It's a part of a UC Berkeley design "
            "research project.\n\n"
            "If you're interested in collaborating or learning more, give "
            "some feedback (sidebar) or send me an email!\n\n"
            "**To try it quickly:**\n\n"
            "1. Click one of the **example questions** below.\n"
            "2. Scroll through the report -- you can see the practitioners "
            "whose experience each point is based on.\n"
            "3. Click any **frame ID** (e.g. F-1234) to see the original "
            "quote and source link.\n\n"
            "**Run your own query:**\n\n"
            "1. Type a question in the box above -- anything about "
            "mechanical design, material selection, machining strategy, "
            "tolerances, etc.\n"
            "2. Pick an **analysis mode** in the sidebar "
            "(**Flash** is fastest at ~20 sec, **Deep Reasoning** is most "
            "thorough at 7-10 min).\n"
            "3. Click **Analyze** and the agent will search the knowledge "
            "graph, find relevant practitioner insights, and synthesize "
            "a report with actionable design rules."
        )

        st.markdown("**Example questions:**")
        examples = [
            "What should I consider when designing thin-walled aluminum parts for CNC machining?",
            "What fixturing methods work best for thin flexible parts?",
            "What are the risks of deep pocket milling in steel?",
            "How do I choose between climb and conventional milling?",
        ]
        for ex in examples:
            cached = example_cache.get(ex)
            if cached:
                # Show the analysis mode used for this cached result
                c_model = cached.get("model", "")
                c_effort = cached.get("reasoning_effort")
                if c_model == "gpt-4.1-mini":
                    mode_tag = "\u26a1 Flash Mode"
                elif c_model == "gpt-4.1":
                    mode_tag = "\U0001f50d Quick Search"
                elif c_model == "gpt-5" and c_effort == "low":
                    mode_tag = "\U0001f9e0 Efficient Thinking"
                else:
                    mode_tag = "\U0001f52c Deep Reasoning"
                label = f"{ex}  |  {mode_tag}"
            else:
                label = ex
            if st.button(label, key=f"ex_{ex[:20]}", use_container_width=True):
                if cached:
                    # Serve pre-cached result instantly
                    report = DFMReportV2.model_validate(cached["report"])
                    backfill = (
                        BackfillOutput.model_validate(cached["backfill_result"])
                        if cached.get("backfill_result")
                        else None
                    )
                    result = {
                        "query": ex,
                        "report": report,
                        "enrichment": cached.get("enrichment", {}),
                        "ranked": cached.get("ranked", []),
                        "sub_aspects": cached.get("sub_aspects", []),
                        "diagnostic": cached.get("diagnostic", {}),
                        "backfill_result": backfill,
                        "frame_details": cached.get("frame_details", []),
                        "timestamp": _now_pst().strftime("%I:%M:%S %p"),
                        "timestamp_dt": _now_pst(),
                        "elapsed_s": cached.get("elapsed_s", 0),
                        "token_usage": cached.get("token_usage", {}),
                        "from_cache": True,
                    }
                    st.session_state.query_history.append(result)
                    st.session_state.active_idx = len(st.session_state.query_history) - 1
                    _log_query(result)
                    st.rerun()
                else:
                    # Fall back to live query
                    st.session_state["_auto_query"] = ex
                    st.rerun()

        # ── Admin-only: Visitor & Query Logs at bottom of home page ──
        if _is_admin_visitor():
            st.markdown("---")
            st.markdown("### Admin: Logs")

            import pandas as pd

            tab_v, tab_q = st.tabs(["Visitor Log", "Query Log"])

            with tab_v:
                visitors = _read_log(_VISITOR_LOG_FILE)
                if visitors:
                    df_v = pd.DataFrame(visitors)
                    desired_cols = ["timestamp", "first_name", "last_name", "contact_info", "ip",
                                    "user_agent", "session_id", "ip_hash"]
                    cols = [c for c in desired_cols if c in df_v.columns] + [c for c in df_v.columns if c not in desired_cols]
                    df_v = df_v[cols].iloc[::-1].reset_index(drop=True)
                    st.dataframe(df_v, use_container_width=True, hide_index=True)
                    st.download_button("Download Visitor Log CSV", df_v.to_csv(index=False),
                                       "visitor_log.csv", "text/csv", key="home_dl_visitor_csv")
                else:
                    st.caption("No visitor records yet.")

            with tab_q:
                queries = _read_log(_QUERY_LOG_FILE)
                if queries:
                    df_q = pd.DataFrame(queries)
                    desired_cols = ["timestamp", "visitor_name", "contact_info", "query", "model", "flash_mode",
                                    "from_cache", "elapsed_s", "num_rules", "num_disciplines", "num_gaps",
                                    "ip", "session_id", "summary_preview"]
                    cols = [c for c in desired_cols if c in df_q.columns] + [c for c in df_q.columns if c not in desired_cols]
                    df_q = df_q[cols].iloc[::-1].reset_index(drop=True)
                    st.dataframe(df_q, use_container_width=True, hide_index=True)
                    st.download_button("Download Query Log CSV", df_q.to_csv(index=False),
                                       "query_log.csv", "text/csv", key="home_dl_query_csv")
                else:
                    st.caption("No query records yet.")


if __name__ == "__main__":
    main()
