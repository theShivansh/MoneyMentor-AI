
# ── stdlib ─────────────────────────────────────────────────────────────────
import concurrent.futures
import json
import operator
import re
import urllib.parse
import urllib.request
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

# ── third-party ────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ── LangGraph (optional) ───────────────────────────────────────────────────
try:
    from langgraph.graph import END, StateGraph
    _LG_AVAILABLE = True
except ImportError:
    _LG_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  — MUST be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MoneyMentor AI · 2027",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
#  Four models matched precisely to task requirements.
# ══════════════════════════════════════════════════════════════════════════
MODELS: Dict[str, str] = {
    # ── Budget Analyst ──────────────────────────────────────────────────
    # Llama 3.3 70B Versatile: deep chain-of-thought capable reasoning for
    # exact 50/30/20 breakdowns, Rs-level arithmetic, spending audits.
    "budget": "llama-3.3-70b-versatile",

    # ── Risk Assessor ────────────────────────────────────────────────────
    # Llama 4 Scout 17B: ultra-fast structured scoring with 128k context.
    # Perfect for CRITICAL/HIGH/MEDIUM/LOW classification in milliseconds.
    "risk": "meta-llama/llama-4-scout-17b-16e-instruct",

    # ── Investment Advisor ───────────────────────────────────────────────
    # Llama 3.3 70B Versatile: superior Indian market domain knowledge —
    # SEBI instruments, ELSS, SIP mechanics, NPS Tier-1, PPF math.
    "investment": "llama-3.3-70b-versatile",

    # ── Web Researcher ───────────────────────────────────────────────────
    # Groq Compound Beta: Groq's native agentic model with built-in
    # web search and tool orchestration. No manual DDG scrape needed.
    "research": "compound-beta",

    # ── Supervisor CFO ───────────────────────────────────────────────────
    # GPT-OSS 120B via OpenRouter: largest model in roster for synthesis,
    # executive 30/60/90-day planning, cross-agent report fusion.
    # Falls back to llama-3.3-70b if key unavailable.
    "supervisor": "openai/gpt-oss-120b",

    # ── Chat Advisor ─────────────────────────────────────────────────────
    # Llama 4 Scout: snappy multi-turn Q&A with full context window.
    "chat": "meta-llama/llama-4-scout-17b-16e-instruct",
}

# Fallback for supervisor if GPT-OSS-120B unavailable
_SUPERVISOR_FALLBACK = "llama-3.3-70b-versatile"

# ══════════════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS — one schema per agent task
# ══════════════════════════════════════════════════════════════════════════

# ── Agent 1: Budget Analyst ───────────────────────────────────────────────
BUDGET_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyse_budget_50_30_20",
            "description": (
                "Compute 50/30/20 rule split, savings rate, monthly deficit/surplus, "
                "and flag which expense category exceeds its ideal share."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "income":          {"type": "number", "description": "Gross monthly income in Rs"},
                    "needs_total":     {"type": "number", "description": "Sum of essential expenses (rent, utilities, groceries, transport)"},
                    "wants_total":     {"type": "number", "description": "Sum of discretionary expenses (dining, entertainment, shopping)"},
                    "savings_surplus": {"type": "number", "description": "Income minus all expenses"},
                    "biggest_category":{"type": "string", "description": "Name of the highest-spend category"},
                    "biggest_amount":  {"type": "number", "description": "Amount of highest-spend category in Rs"},
                },
                "required": ["income", "needs_total", "wants_total", "savings_surplus",
                             "biggest_category", "biggest_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_spending_anomalies",
            "description": "Identify categories where spend exceeds 50/30/20 ideal thresholds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "anomalies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category":    {"type": "string"},
                                "actual_pct":  {"type": "number"},
                                "ideal_pct":   {"type": "number"},
                                "excess_rs":   {"type": "number"},
                            },
                        },
                        "description": "List of over-budget categories",
                    }
                },
                "required": ["anomalies"],
            },
        },
    },
]

# ── Agent 2: Risk Assessor ────────────────────────────────────────────────
RISK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_financial_risk",
            "description": (
                "Assign CRITICAL / HIGH / MEDIUM / LOW risk rating based on "
                "expense ratio, emergency fund runway, and goal feasibility gap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level":       {"type": "string", "enum": ["CRITICAL","HIGH","MEDIUM","LOW"]},
                    "expense_ratio_pct":{"type": "number", "description": "Total expenses / income * 100"},
                    "runway_months":    {"type": "number", "description": "Savings / monthly burn"},
                    "emergency_gap_rs": {"type": "number", "description": "Gap between current savings and 6-month fund"},
                    "top_vulnerability":{"type": "string", "description": "Single biggest risk in one sentence"},
                },
                "required": ["risk_level", "expense_ratio_pct", "runway_months",
                             "emergency_gap_rs", "top_vulnerability"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_goal_feasibility",
            "description": "Calculate whether the savings goal is achievable within the timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_required":  {"type": "number"},
                    "monthly_available": {"type": "number"},
                    "feasible":          {"type": "boolean"},
                    "shortfall_rs":      {"type": "number"},
                    "months_to_goal":    {"type": "number"},
                },
                "required": ["monthly_required", "monthly_available", "feasible"],
            },
        },
    },
]

# ── Agent 3: Investment Advisor ───────────────────────────────────────────
INVESTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "build_india_portfolio",
            "description": (
                "Construct a recommended monthly SIP/investment allocation across "
                "Indian instruments: Liquid Fund, ELSS, Nifty 50 Index, PPF, NPS Tier-1, FD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "investable_monthly": {"type": "number"},
                    "allocations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "instrument":  {"type": "string"},
                                "fund_name":   {"type": "string"},
                                "amount_rs":   {"type": "number"},
                                "percent":     {"type": "number"},
                                "rationale":   {"type": "string"},
                            },
                        },
                    },
                    "projected_1yr_corpus": {"type": "number"},
                    "projected_3yr_corpus": {"type": "number"},
                },
                "required": ["investable_monthly", "allocations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_sip_projection",
            "description": "Project SIP corpus using compound growth formula.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_sip":    {"type": "number"},
                    "annual_return":  {"type": "number", "description": "Expected annual return as decimal e.g. 0.12"},
                    "years":          {"type": "number"},
                    "projected_value":{"type": "number"},
                },
                "required": ["monthly_sip", "annual_return", "years", "projected_value"],
            },
        },
    },
]

# ── Agent 4: Web Researcher ───────────────────────────────────────────────
# compound-beta has native web_search — no manual tool schema needed.
# Groq routes it internally. We pass an empty tools list to let it self-direct.
WEB_TOOLS: List[Dict] = []   # compound-beta uses its own built-in search

# ── Agent 5: Supervisor CFO ───────────────────────────────────────────────
SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_90_day_plan",
            "description": (
                "Synthesise all agent reports into a structured 30/60/90-day "
                "financial execution plan with specific Rs amounts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day_30": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3 concrete actions for Month 1 with Rs amounts",
                    },
                    "day_60": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3 concrete actions for Month 2 with Rs amounts",
                    },
                    "day_90": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3 concrete actions for Month 3 with Rs amounts",
                    },
                    "bold_move": {"type": "string", "description": "Single decisive action in CAPS"},
                    "confidence_score": {
                        "type": "integer",
                        "description": "CFO confidence in this plan 0-100",
                    },
                },
                "required": ["day_30", "day_60", "day_90", "bold_move"],
            },
        },
    },
]




def _is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


def _inject_css() -> None:
    d = _is_dark()
    if d:
        bg   = "#050509"; bgc  = "#0b0b14"; bgc2 = "#0f0f1c"; bdr  = "#1c1c36"
        tp   = "#f0eeff"; ts   = "#6b6b99"
        a1   = "#8b5cf6"; a2   = "#10f5a0"; a3   = "#f02d72"; a4   = "#f5c518"
        a1d  = "#6d35d4"; a2d  = "#00c87a"
        g1   = "rgba(139,92,246,.22)"; g2   = "rgba(16,245,160,.15)"
        g3   = "rgba(240,45,114,.22)"
        sb   = "#06060d"; mb   = "rgba(139,92,246,.08)"; tb   = "rgba(139,92,246,.10)"
        ph   = "#3a3a58"; card = "#09091a"; glow = "0 0 40px rgba(139,92,246,.12)"
    else:
        bg   = "#f7f6ff"; bgc  = "#ffffff"; bgc2 = "#f2f0ff"; bdr  = "#e0dbff"
        tp   = "#0d0020"; ts   = "#5b5480"
        a1   = "#6d28d9"; a2   = "#059669"; a3   = "#be123c"; a4   = "#b45309"
        a1d  = "#5b21b6"; a2d  = "#047857"
        g1   = "rgba(109,40,217,.14)"; g2   = "rgba(5,150,105,.12)"
        g3   = "rgba(190,18,60,.14)"
        sb   = "#eeeaff"; mb   = "rgba(109,40,217,.07)"; tb   = "rgba(109,40,217,.08)"
        ph   = "#b0a8d4"; card = "#faf9ff"; glow = "0 4px 24px rgba(109,40,217,.09)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@700;800&display=swap');

:root{{
  --bg:{bg}; --bgc:{bgc}; --bgc2:{bgc2}; --bdr:{bdr};
  --tp:{tp}; --ts:{ts};
  --a1:{a1}; --a2:{a2}; --a3:{a3}; --a4:{a4};
  --a1d:{a1d}; --a2d:{a2d};
  --g1:{g1}; --g2:{g2}; --g3:{g3};
  --mb:{mb}; --tb:{tb}; --sb:{sb}; --card:{card}; --glow:{glow};
}}

/* ── APP SHELL ─────────────────────────────────────── */
html,body{{background:{bg}!important;color:{tp}!important;}}
.stApp,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],.main,.main .block-container,
div.stMainBlockContainer{{background:{bg}!important;}}

/* ── SIDEBAR ───────────────────────────────────────── */
[data-testid="stSidebar"],[data-testid="stSidebar"]>div:first-child,
[data-testid="stSidebarContent"]{{
  background:{sb}!important;
  border-right:1px solid {bdr}!important;
}}
[data-testid="stSidebar"] *{{color:{tp}!important;}}

/* ── TYPOGRAPHY ────────────────────────────────────── */
*{{font-family:'Outfit',sans-serif;color:{tp};}}
h1,h2,h3,h4,h5{{font-family:'Syne',sans-serif!important;color:{tp}!important;}}
p,[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *{{color:{tp}!important;}}
code,pre{{font-family:'JetBrains Mono',monospace!important;}}

/* ── INPUTS ────────────────────────────────────────── */
.stTextInput input,.stNumberInput input,.stTextArea textarea,
[data-baseweb="base-input"],[data-baseweb="input"] input{{
  background:{bgc}!important; color:{tp}!important;
  border:1px solid {bdr}!important; border-radius:10px!important;
  font-family:'JetBrains Mono',monospace!important; font-size:13px!important;
  transition:border-color .18s,box-shadow .18s!important;
}}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{{
  border-color:{a1}!important; box-shadow:0 0 0 3px {g1}!important; outline:none!important;
}}
input::placeholder,textarea::placeholder{{color:{ph}!important;}}
[data-baseweb="select"]>div{{background:{bgc}!important;border:1px solid {bdr}!important;color:{tp}!important;}}
.stNumberInput button{{
  background:{bgc2}!important; border:1px solid {bdr}!important;
  color:{ts}!important; border-radius:8px!important;
  box-shadow:none!important; font-size:14px!important; min-width:26px!important;
}}
.stRadio label{{color:{ts}!important;font-size:13px!important;}}
[data-testid="stWidgetLabel"] p{{color:{ts}!important;font-size:12px!important;font-family:'JetBrains Mono',monospace!important;letter-spacing:.04em!important;}}

/* ── BUTTONS ───────────────────────────────────────── */
.stButton>button{{
  background:linear-gradient(135deg,{a1} 0%,{a3} 100%)!important;
  color:#fff!important; border:none!important; border-radius:10px!important;
  font-family:'Syne',sans-serif!important; font-weight:700!important;
  font-size:11px!important; letter-spacing:.1em!important;
  text-transform:uppercase!important; padding:.6rem 1.4rem!important;
  transition:all .22s cubic-bezier(.34,1.56,.64,1)!important;
  box-shadow:0 4px 20px {g1}!important;
}}
.stButton>button:hover{{
  transform:translateY(-2px) scale(1.02)!important;
  box-shadow:0 8px 32px {g1},0 4px 16px {g3}!important;
}}
.stButton>button:active{{transform:translateY(0) scale(.99)!important;}}

/* ── METRICS ───────────────────────────────────────── */
[data-testid="metric-container"]{{
  background:{mb}!important; border:1px solid {bdr}!important;
  border-radius:14px!important; padding:1rem 1.2rem!important;
  box-shadow:{glow}!important;
}}
[data-testid="stMetricValue"]{{
  font-family:'JetBrains Mono',monospace!important; font-size:1.25rem!important;
  color:{a1}!important; font-weight:700!important;
}}
[data-testid="stMetricLabel"]{{
  font-family:'Syne',sans-serif!important; font-size:9.5px!important;
  text-transform:uppercase!important; letter-spacing:.14em!important; color:{ts}!important;
}}

/* ── TABS ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{{
  background:{bgc}!important; border-radius:12px!important;
  padding:4px!important; border:1px solid {bdr}!important; gap:3px!important;
}}
.stTabs [data-baseweb="tab"]{{
  font-family:'Syne',sans-serif!important; font-weight:700!important;
  font-size:10.5px!important; letter-spacing:.09em!important;
  text-transform:uppercase!important; color:{ts}!important;
  border-radius:9px!important; border:none!important;
  background:transparent!important; padding:8px 22px!important;
  transition:all .17s ease!important;
}}
.stTabs [aria-selected="true"]{{
  background:linear-gradient(135deg,{a1},{a3})!important; color:#fff!important;
  box-shadow:0 2px 12px {g1}!important;
}}
.stTabs [aria-selected="true"] p,.stTabs [aria-selected="true"] *{{color:#fff!important;}}

/* ── EXPANDERS ─────────────────────────────────────── */
[data-testid="stExpander"] summary,.streamlit-expanderHeader{{
  background:{bgc}!important; border:1px solid {bdr}!important;
  border-radius:12px!important; font-family:'Syne',sans-serif!important;
  font-weight:700!important; color:{tp}!important; font-size:13px!important;
  transition:border-color .18s!important;
}}
[data-testid="stExpander"] summary:hover{{border-color:{a1}!important;}}
[data-testid="stExpander"] details[open] summary{{border-radius:12px 12px 0 0!important;}}
.streamlit-expanderContent,[data-testid="stExpander"] .streamlit-expanderContent{{
  background:{bgc2}!important; border:1px solid {bdr}!important;
  border-top:none!important; border-radius:0 0 12px 12px!important;
}}

/* ── FORM ──────────────────────────────────────────── */
[data-testid="stForm"]{{
  background:{bgc}!important; border:1px solid {bdr}!important;
  border-radius:18px!important; padding:1.6rem!important;
  box-shadow:{glow}!important;
}}

/* ── ALERTS ────────────────────────────────────────── */
.stAlert,[data-testid="stAlert"],[data-testid="stAlertContainer"]{{
  background:{bgc}!important; border:1px solid {bdr}!important; border-radius:12px!important;
}}
.stAlert p,[data-testid="stAlertContainer"] p{{color:{tp}!important;}}

/* ── PROGRESS ──────────────────────────────────────── */
.stProgress>div>div>div,[data-testid="stProgressBar"]>div{{
  background:linear-gradient(90deg,{a1},{a2})!important; border-radius:99px!important;
}}
[data-testid="stProgressBar"]{{background:{bgc2}!important;border-radius:99px!important;}}

/* ── SCROLLBAR ─────────────────────────────────────── */
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-track{{background:{bg};}}
::-webkit-scrollbar-thumb{{background:linear-gradient({a1},{a3});border-radius:5px;}}
hr{{border-color:{bdr}!important;opacity:.6!important;margin:1.5rem 0!important;}}
.stSpinner>div{{border-top-color:{a1}!important;}}

/* ══════════════════════════════════════════════════
   BRANDED COMPONENTS
══════════════════════════════════════════════════ */

/* ── Hero Badge ─── */
.mm-badge{{
  display:inline-flex;align-items:center;gap:6px;
  background:{tb};border:1px solid {a1};border-radius:99px;
  padding:4px 16px;font-family:'JetBrains Mono',monospace;
  font-size:9.5px;color:{a1};letter-spacing:.16em;text-transform:uppercase;
  margin-bottom:14px;
}}

/* ── Hero Title ─── */
.mm-title{{
  font-family:'Syne',sans-serif;font-weight:800;
  font-size:clamp(2.2rem,5vw,3.4rem);line-height:1.0;
  background:linear-gradient(135deg,{a1} 0%,{a3} 45%,{a2} 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:12px;letter-spacing:-.01em;
}}

/* ── Subtitle ─── */
.mm-sub{{
  font-family:'Outfit',sans-serif;font-size:15px;color:{ts};
  max-width:620px;line-height:1.7;margin-bottom:24px;font-weight:400;
}}

/* ── Section Labels ─── */
.s-lbl{{
  font-family:'JetBrains Mono',monospace;font-size:9.5px;
  letter-spacing:.22em;text-transform:uppercase;color:{a2};
  margin-bottom:4px;opacity:.85;
}}
.s-ttl{{
  font-family:'Syne',sans-serif;font-weight:700;
  font-size:1.18rem;color:{tp};margin-bottom:16px;
}}

/* ── Agent Cards ─── */
.ag-card{{
  background:{card};border:1px solid {bdr};border-radius:16px;
  padding:1.3rem 1.5rem;margin-bottom:12px;
  position:relative;overflow:hidden;
  transition:border-color .2s,box-shadow .2s;
  box-shadow:{glow};
}}
.ag-card:hover{{border-color:{a1};box-shadow:0 0 32px {g1};}}
.ag-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,{a1},{a3},{a2});
}}
.ag-name{{
  font-family:'Syne',sans-serif;font-weight:700;
  font-size:15px;color:{tp};margin-bottom:3px;
}}
.ag-model{{
  font-family:'JetBrains Mono',monospace;font-size:9px;
  color:{a2};text-transform:uppercase;letter-spacing:.11em;
}}
.ag-out{{
  font-family:'Outfit',sans-serif;font-size:13.5px;color:{ts};
  margin-top:14px;line-height:1.78;
  border-left:3px solid {a1};padding-left:15px;
}}

/* ── Pulse Dot ─── */
.sdot{{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:{a2};box-shadow:0 0 8px {a2};margin-right:7px;
  animation:pulse 2s infinite;
}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.8)}}}}

/* ── Tip Cards ─── */
.tip{{
  background:{card};border:1px solid {bdr};border-radius:13px;
  padding:.95rem 1.2rem;margin-bottom:10px;
  display:flex;gap:14px;align-items:flex-start;
  transition:border-color .18s;
}}
.tip:hover{{border-color:{a1};}}
.tip-ico{{font-size:20px;flex-shrink:0;line-height:1.4;}}
.tip-txt{{font-family:'Outfit',sans-serif;font-size:13.5px;color:{tp};line-height:1.68;}}
.tip-warn{{border-left:3px solid {a4};}}
.tip-ok  {{border-left:3px solid {a2};}}
.tip-info{{border-left:3px solid {a1};}}
.tip-bad {{border-left:3px solid {a3};}}

/* ── Chat Bubbles ─── */
.cb-user{{
  background:linear-gradient(135deg,{a1},{a3});color:#fff!important;
  border-radius:18px 18px 4px 18px;padding:12px 17px;margin-bottom:9px;
  font-family:'Outfit',sans-serif;font-size:14px;max-width:75%;margin-left:auto;
  box-shadow:0 4px 16px {g1};
}}
.cb-ai{{
  background:{card};border:1px solid {bdr};color:{tp}!important;
  border-radius:18px 18px 18px 4px;padding:12px 17px;margin-bottom:9px;
  font-family:'Outfit',sans-serif;font-size:14px;max-width:83%;
}}
.cb-lbl{{
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{a1};
  letter-spacing:.13em;text-transform:uppercase;display:block;margin-bottom:5px;
}}

/* ── Pipeline Steps ─── */
.ps{{
  display:flex;align-items:center;gap:10px;
  padding:9px 15px;background:{bgc2};
  border-radius:10px;border:1px solid {bdr};
  margin-bottom:6px;font-family:'JetBrains Mono',monospace;
  font-size:11px;color:{ts};transition:all .2s;
}}
.ps.run{{border-color:{a1};color:{a1};background:{tb};}}
.ps.ok {{border-color:{a2};color:{a2};background:rgba(16,245,160,.06);}}
.ps.err{{border-color:{a3};color:{a3};background:rgba(240,45,114,.06);}}

/* ── Model Pill ─── */
.mpill{{
  display:inline-block;background:{tb};border:1px solid {a1};
  border-radius:99px;padding:2px 10px;
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{a1};
  letter-spacing:.06em;margin-left:6px;vertical-align:middle;
}}

/* ── Health Grade ─── */
.h-grade{{
  font-family:'Syne',sans-serif;font-size:11px;text-transform:uppercase;
  letter-spacing:.18em;color:{ts};text-align:center;margin-top:6px;
}}

/* ── Stat Chip ─── */
.stat-chip{{
  display:inline-block;background:{tb};border:1px solid {bdr};
  border-radius:99px;padding:4px 14px;
  font-family:'JetBrains Mono',monospace;font-size:11px;color:{a2};
}}

/* ── Web Snippets ─── */
.ws{{
  background:{card};border:1px solid {bdr};border-left:3px solid {a2};
  border-radius:12px;padding:11px 15px;margin-bottom:9px;
  font-family:'Outfit',sans-serif;font-size:13px;color:{tp};line-height:1.65;
}}
.ws-url{{
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{a2};
  letter-spacing:.07em;margin-top:6px;opacity:.6;
}}

/* ── Footer ─── */
.mm-footer{{
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{ts};
  opacity:.3;text-align:center;padding:16px 0 24px;
  letter-spacing:.1em;line-height:2.2;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  GROQ HELPERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _init_groq(key: str) -> Groq:
    return Groq(api_key=key)


def _strip_think(text: str) -> str:
    """Remove chain-of-thought <think> blocks (DeepSeek / R1 style)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _call(
    client: Groq,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 560,
    temp: float = 0.55,
    tools: Optional[List[Dict]] = None,
) -> str:
    """
    Plain single-turn LLM call.
    Pure Python — NO st.* calls — safe for background threads.
    """
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temp,
    )
    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = "auto"
    try:
        r    = client.chat.completions.create(**kwargs)
        text = r.choices[0].message.content or ""
        return _strip_think(text)
    except Exception as exc:
        short = str(exc)[:120]
        # Supervisor fallback: if GPT-OSS-120B unavailable, retry with 70B
        if model == MODELS["supervisor"] and "not found" in short.lower():
            try:
                kwargs["model"] = _SUPERVISOR_FALLBACK
                r2   = client.chat.completions.create(**kwargs)
                text = r2.choices[0].message.content or ""
                return _strip_think(text) + "\n\n*(Supervisor: fallback to llama-3.3-70b)*"
            except Exception as e2:
                return f"[Supervisor fallback error: {str(e2)[:90]}]"
        return f"[{model.split('/')[-1]} — {short}]"


# ══════════════════════════════════════════════════════════════════════════
#  DDG SCRAPE  — pure Python, zero deps, zero key
# ══════════════════════════════════════════════════════════════════════════

def _ddg(query: str, n: int = 5) -> List[Dict]:
    out: List[Dict] = []
    try:
        q   = urllib.parse.quote_plus(query)
        req = urllib.request.Request(
            f"https://html.duckduckgo.com/html/?q={q}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; MoneyMentorBot/2.0)"},
        )
        with urllib.request.urlopen(req, timeout=9) as r:
            html = r.read().decode("utf-8", errors="ignore")
        snips  = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>',        html, re.S)
        hrefs  = re.findall(r'class="result__url"[^>]*>(.*?)</span>',   html, re.S)
        for i, s in enumerate(snips[:n]):
            out.append({
                "title":   re.sub(r"<[^>]+>", "", titles[i]).strip() if i < len(titles) else "",
                "snippet": re.sub(r"<[^>]+>", "", s).strip(),
                "url":     hrefs[i].strip() if i < len(hrefs) else "",
            })
    except Exception as exc:
        out.append({"title": "DDG unavailable", "snippet": str(exc)[:120], "url": ""})
    return out


# ══════════════════════════════════════════════════════════════════════════
#  AGENT STATE (LangGraph TypedDict)
# ══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    income:            float
    expenses:          Dict[str, float]
    savings:           float
    target:            Optional[float]
    timeline:          Optional[int]
    budget_analysis:   str
    risk_assessment:   str
    investment_advice: str
    web_insights:      str
    web_raw:           List[Dict]
    final_report:      str
    messages:          Annotated[List[Dict], operator.add]



def _budget_agent(s: AgentState, c: Groq) -> str:
    """
    Model: llama-3.3-70b-versatile
    Task:  50/30/20 rule audit, Rs-level spending diagnostics
    Tools: analyse_budget_50_30_20, flag_spending_anomalies
    """
    total = sum(s["expenses"].values())
    net   = s["income"] - total
    sr    = net / s["income"] * 100 if s["income"] > 0 else 0
    needs = sum(v for k, v in s["expenses"].items()
                if k in ("Rent", "Utilities", "Groceries", "Transport"))
    wants = sum(v for k, v in s["expenses"].items()
                if k in ("Dining", "Entertainment", "Shopping"))

    system = (
        "You are a precision budget analyst for Indian households. "
        "Use the analyse_budget_50_30_20 and flag_spending_anomalies tools when relevant. "
        "Output EXACTLY 4 numbered insights with Rs amounts. No preamble. No headers."
    )
    user = (
        f"Monthly Income: Rs{s['income']:,.0f}\n"
        f"Expenses breakdown: {json.dumps(s['expenses'])}\n"
        f"Total expenses: Rs{total:,.0f} | Net surplus: Rs{net:,.0f}\n"
        f"Savings rate: {sr:.1f}% | 50/30/20 split →\n"
        f"  Needs (target 50%): Rs{needs:,.0f} ({needs/s['income']*100:.0f}% actual)\n"
        f"  Wants (target 30%): Rs{wants:,.0f} ({wants/s['income']*100:.0f}% actual)\n"
        f"  Savings (target 20%): Rs{net:,.0f} ({sr:.0f}% actual)\n\n"
        "List 4 numbered insights. Identify the single biggest spending inefficiency "
        "with exact Rs waste figure. Be surgical — no generic advice."
    )
    return _call(c, system, user, model=MODELS["budget"],
                 max_tokens=460, tools=BUDGET_TOOLS)


def _risk_agent(s: AgentState, c: Groq) -> str:
    """
    Model: meta-llama/llama-4-scout-17b-16e-instruct
    Task:  CRITICAL/HIGH/MEDIUM/LOW risk scoring, emergency fund gap
    Tools: score_financial_risk, compute_goal_feasibility
    """
    total  = sum(s["expenses"].values())
    runway = s["savings"] / total if total > 0 else 0
    er     = total / s["income"] * 100 if s["income"] > 0 else 0
    gap    = max(0, total * 6 - s["savings"])
    net    = s["income"] - total

    system = (
        "You are a financial risk specialist. Use score_financial_risk and "
        "compute_goal_feasibility tools. Output: RISK LEVEL on line 1, then "
        "3 numbered vulnerabilities with Rs impact, then 1 immediate fix."
    )
    user = (
        f"Expense-to-income ratio: {er:.1f}%\n"
        f"Emergency runway: {runway:.1f} months | 6-month fund gap: Rs{gap:,.0f}\n"
        f"Monthly burn: Rs{total:,.0f} | Net flow: Rs{net:,.0f}\n"
        f"Savings corpus: Rs{s['savings']:,.0f}\n"
        f"Goal: Rs{s.get('target') or 'none'} in {s.get('timeline') or '?'} months\n\n"
        "Rate CRITICAL/HIGH/MEDIUM/LOW. List 3 vulnerabilities with Rs figures. "
        "Give 1 specific immediate fix."
    )
    return _call(c, system, user, model=MODELS["risk"],
                 max_tokens=380, tools=RISK_TOOLS)


def _investment_agent(s: AgentState, c: Groq) -> str:
    """
    Model: llama-3.3-70b-versatile
    Task:  Indian portfolio construction — ELSS, SIP, NPS, PPF
    Tools: build_india_portfolio, compute_sip_projection
    """
    total      = sum(s["expenses"].values())
    investable = max(0, s["income"] - total)

    system = (
        "You are a SEBI-registered investment advisor specialising in Indian retail finance. "
        "Use build_india_portfolio and compute_sip_projection tools. "
        "Name specific real mutual funds. Give exact Rs amounts. No generic statements."
    )
    user = (
        f"Monthly investable surplus: Rs{investable:,.0f}\n"
        f"Existing corpus: Rs{s['savings']:,.0f}\n"
        f"Goal: Rs{s.get('target') or 'wealth creation'} "
        f"in {s.get('timeline') or 'open'} months\n\n"
        "Allocate across: Liquid Fund (emergency top-up), ELSS SIP, "
        "Nifty 50 Index Fund SIP, PPF, NPS Tier-1, FD ladder.\n"
        "Format: [Instrument] — [Specific Fund] — Rs[amount]/mo ([%]) — [1-line reason]\n"
        "Final line: projected 1-year corpus at realistic return rates."
    )
    return _call(c, system, user, model=MODELS["investment"],
                 max_tokens=500, tools=INVESTMENT_TOOLS)


def _web_agent(s: AgentState, c: Groq) -> Tuple[str, List[Dict]]:
    """
    Model: compound-beta (Groq native agentic search)
    Task:  Live RBI rate, SIP categories, CPI, liquid fund yields
    Tools: compound-beta uses its own built-in web search routing
    Fallback: manual DDG scrape if compound-beta unavailable
    """
    total = sum(s["expenses"].values())
    sr    = (s["income"] - total) / s["income"] * 100 if s["income"] > 0 else 0

    system = (
        "You are a macro-finance analyst covering India in real-time. "
        "Search for and report current figures. Cite actual numbers found. Rs signs."
    )
    user = (
        f"User: Rs{s['income']:,.0f}/mo income, {sr:.1f}% savings rate, India.\n"
        "Fetch and report 4 current macro insights:\n"
        "1. RBI repo rate (current) and its impact on FD/savings account rates\n"
        "2. Top-performing SIP category last 3 months (with % returns)\n"
        "3. Latest India CPI headline inflation and household budget impact\n"
        "4. Best liquid fund 7-day yield available right now\n"
        "Use your web search capability. Quote actual data with sources."
    )

    # Try compound-beta first (native agentic search)
    raw: List[Dict] = []
    try:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        r = c.chat.completions.create(
            model=MODELS["research"],
            messages=msgs,
            max_tokens=560,
            temperature=0.4,
        )
        ans = _strip_think(r.choices[0].message.content or "")
        return ans, raw
    except Exception as exc:
        # Fallback: manual DDG + llama-3.3
        short = str(exc)[:100]
        queries = [
            "RBI repo rate India 2025",
            "top SIP mutual fund India 3 months 2025",
            "India CPI inflation 2025",
            "best liquid fund yield India 2025",
        ]
        for q in queries:
            hits = _ddg(q, n=2)
            raw.extend(hits)

        ctx = "\n".join(
            f"[{h['title']}] {h['snippet']}" for h in raw
        ) or "No live data."

        fallback_sys = (
            "You are a macro-finance analyst. Use the web snippets provided to "
            "answer with real figures. Rs signs. Be concise."
        )
        fallback_user = (
            f"Web data:\n{ctx}\n\n{user}\n"
            f"(Note: compound-beta unavailable ({short}); using DDG fallback.)"
        )
        ans = _call(c, fallback_sys, fallback_user,
                    model="llama-3.3-70b-versatile", max_tokens=480)
        return ans, raw


def _supervisor_agent(s: AgentState, c: Groq) -> str:
    """
    Model: openai/gpt-oss-120b (fallback: llama-3.3-70b-versatile)
    Task:  Synthesise all 4 agent reports → 30/60/90-day execution plan
    Tools: generate_90_day_plan
    """
    system = (
        "You are the Chief Financial Officer AI, the most senior intelligence in this system. "
        "You synthesise four expert agent reports into a decisive, numbered action plan. "
        "Use generate_90_day_plan tool structure. Every action must have a specific Rs amount. "
        "Be decisive. No fluff. No repetition."
    )
    user = (
        "Synthesise these four specialist reports:\n\n"
        f"[BUDGET ANALYST — llama-3.3-70b]\n{s['budget_analysis']}\n\n"
        f"[RISK ASSESSOR — llama-4-scout]\n{s['risk_assessment']}\n\n"
        f"[INVESTMENT ADVISOR — llama-3.3-70b]\n{s['investment_advice']}\n\n"
        f"[WEB RESEARCHER — compound-beta (live data)]\n{s['web_insights']}\n\n"
        f"User profile: Rs{s['income']:,.0f}/mo | "
        f"Savings Rs{s['savings']:,.0f} | "
        f"Goal Rs{s.get('target') or 'wealth creation'}\n\n"
        "Output exactly:\n"
        "30-DAY PLAN (3 numbered actions with Rs amounts)\n"
        "60-DAY PLAN (3 numbered actions with Rs amounts)\n"
        "90-DAY PLAN (3 numbered actions with Rs amounts)\n"
        "BOLD MOVE THIS MONTH: [one decisive action in CAPS with Rs amount]\n"
        "CFO CONFIDENCE: [score 0-100]"
    )
    return _call(c, system, user, model=MODELS["supervisor"],
                 max_tokens=700, temp=0.45, tools=SUPERVISOR_TOOLS)


def _run_parallel(
    state:  AgentState,
    client: Groq,
    slots:  Dict[str, Any],  # st.empty() placeholders — main thread only
    prog:   Any,              # st.progress()           — main thread only
) -> AgentState:

    # ── 1. Mark all queued (main thread) ───────────────────────────────
    for name in ("Budget Analyst","Risk Assessor","Investment Advisor","Web Researcher"):
        slots[name].markdown(
            f'<div class="ps">◯ &nbsp; {name} · Queued</div>',
            unsafe_allow_html=True,
        )
    slots["Supervisor CFO"].markdown(
        '<div class="ps">◯ &nbsp; Supervisor CFO · Awaiting parallel agents…</div>',
        unsafe_allow_html=True,
    )
    prog.progress(0.05)

    # ── 2. Submit pure Python functions — NO st.* inside ───────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f_b = pool.submit(_budget_agent,     state, client)
        f_r = pool.submit(_risk_agent,       state, client)
        f_i = pool.submit(_investment_agent, state, client)
        f_w = pool.submit(_web_agent,        state, client)

        
        budget_out     = f_b.result()
        risk_out       = f_r.result()
        invest_out     = f_i.result()
        web_out, w_raw = f_w.result()

    # ── 3. Update UI AFTER threads complete (main thread — safe) ───────
    prog.progress(0.80)
    for name in ("Budget Analyst","Risk Assessor","Investment Advisor","Web Researcher"):
        slots[name].markdown(
            f'<div class="ps ok">✓ &nbsp; {name} · Complete</div>',
            unsafe_allow_html=True,
        )

    # ── 4. Merge results into state ─────────────────────────────────────
    state = {
        **state,
        "budget_analysis":   budget_out,
        "risk_assessment":   risk_out,
        "investment_advice": invest_out,
        "web_insights":      web_out,
        "web_raw":           w_raw,
    }

    # ── 5. Supervisor runs sequentially after all 4 parallel complete ──
    prog.progress(0.88)
    slots["Supervisor CFO"].markdown(
        '<div class="ps run"><span class="sdot"></span>Supervisor CFO · Synthesising…</div>',
        unsafe_allow_html=True,
    )
    final = _supervisor_agent(state, client)  # main thread — fine
    prog.progress(1.0)
    slots["Supervisor CFO"].markdown(
        '<div class="ps ok">✓ &nbsp; Supervisor CFO · Plan Ready</div>',
        unsafe_allow_html=True,
    )

    return {
        **state,
        "final_report": final,
        "messages": [
            {"role": "budget_analyst",    "content": budget_out},
            {"role": "risk_assessor",     "content": risk_out},
            {"role": "investment_advisor","content": invest_out},
            {"role": "web_researcher",    "content": web_out},
            {"role": "supervisor",        "content": final},
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
#  CHAT ADVISOR
# ══════════════════════════════════════════════════════════════════════════

def _chat(
    client:   Groq,
    msg:      str,
    ctx:      Dict,
    history:  List[Dict],
    agent_ctx: str = "",
) -> str:
    exp = sum(ctx.get("expenses", {}).values())
    sys_prompt = (
        "You are MoneyMentor, a sharp and warm 2027-era AI financial advisor for India. "
        "Expert in Indian personal finance — mutual funds, SIPs, PPF, NPS, tax-saving. "
        f"User context: Rs{ctx.get('income',0):,.0f}/mo income, "
        f"Rs{exp:,.0f} expenses, Rs{ctx.get('savings',0):,.0f} savings.\n"
        + (f"Agent analysis snapshot: {agent_ctx[:500]}\n" if agent_ctx else "")
        + "Reply in max 3 sentences. Lead with one emoji. Use Rs signs. Be direct and actionable."
    )
    msgs = [{"role": "system", "content": sys_prompt}]
    for h in history[-8:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": msg})
    try:
        r = client.chat.completions.create(
            model=MODELS["chat"], messages=msgs, max_tokens=300, temperature=0.72,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ {str(e)[:90]}"


# ══════════════════════════════════════════════════════════════════════════
#  RULE-BASED DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════

def _rule_tips(
    income:   float,
    expenses: Dict,
    savings:  float,
    target:   Optional[float],
    timeline: Optional[int],
) -> List[Dict]:
    tips: List[Dict] = []
    total = sum(expenses.values())
    net   = income - total

    if net < 0:
        tips.append({"cls": "tip-bad", "ico": "🚨",
                     "txt": f"Critical overspend — Rs{abs(net):,.0f}/month deficit. "
                            "You are draining savings. Cut discretionary spend immediately."})
    else:
        sr = net / income * 100 if income > 0 else 0
        if sr < 10:
            tips.append({"cls": "tip-warn", "ico": "⚠️",
                         "txt": f"Savings rate {sr:.1f}% is dangerously low. "
                                f"Target 20% = Rs{income * .2:,.0f}/mo. "
                                f"Gap to close: Rs{income * .2 - net:,.0f}."})
        elif sr < 20:
            tips.append({"cls": "tip-warn", "ico": "📊",
                         "txt": f"Savings rate {sr:.1f}% — below 20% benchmark. "
                                f"Rs{income * .2 - net:,.0f}/mo gap. "
                                "Find one category to trim."})
        else:
            tips.append({"cls": "tip-ok", "ico": "✅",
                         "txt": f"Strong {sr:.1f}% savings rate — above the 20% benchmark. "
                                "Step up SIP by 10% annually to maximise compounding."})

    if expenses:
        top = max(expenses, key=expenses.get)
        pct = expenses[top] / income * 100 if income > 0 else 0
        tips.append({"cls": "tip-info", "ico": "🔍",
                     "txt": f"'{top}' is your largest expense: Rs{expenses[top]:,.0f} "
                            f"({pct:.0f}% of income). A 10% cut here saves "
                            f"Rs{expenses[top] * .1:,.0f}/mo — Rs{expenses[top] * 1.2:,.0f}/yr."})

    runway    = savings / total if total > 0 else 0
    em_target = total * 6
    if runway < 3:
        tips.append({"cls": "tip-warn", "ico": "🛡️",
                     "txt": f"Emergency fund covers only {runway:.1f} months. "
                            f"Target: Rs{em_target:,.0f} (6 months). "
                            "Park in a liquid fund — no SIP until this is built."})
    elif runway < 6:
        tips.append({"cls": "tip-info", "ico": "🛡️",
                     "txt": f"{runway:.1f}-month buffer. Push to 6 months "
                            f"(Rs{em_target:,.0f}). You're Rs{em_target - savings:,.0f} away."})
    else:
        tips.append({"cls": "tip-ok", "ico": "🛡️",
                     "txt": f"Solid {runway:.1f}-month emergency fund. "
                            "Surplus beyond 6 months should go into equity SIPs, not savings account."})

    if target and timeline and timeline > 0:
        needed  = max(0, target - savings)
        monthly = needed / timeline
        if monthly <= net:
            tips.append({"cls": "tip-ok", "ico": "🎯",
                         "txt": f"Goal achievable: need Rs{monthly:,.0f}/mo, "
                                f"have Rs{net:,.0f}/mo. Rs{net - monthly:,.0f} surplus "
                                "remaining for additional wealth creation."})
        else:
            tips.append({"cls": "tip-bad", "ico": "🎯",
                         "txt": f"Goal at risk: need Rs{monthly:,.0f}/mo but only "
                                f"Rs{net:,.0f} available. "
                                f"Gap: Rs{monthly - net:,.0f}/mo. "
                                "Extend timeline or increase income."})
    return tips


# ══════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════

_PAL = ["#8b5cf6","#10f5a0","#f02d72","#f5c518","#06b6d4","#f97316","#a78bfa","#34d399"]


def _chart_donut(exp: Dict, dark: bool) -> go.Figure:
    lbls = [k for k, v in exp.items() if v > 0]
    vals = [v for v in exp.values()  if v > 0]
    lc   = "#f0eeff" if dark else "#0d0020"
    bg_c = "rgba(0,0,0,0)"
    fig  = go.Figure(go.Pie(
        labels=lbls, values=vals, hole=0.64,
        marker=dict(colors=_PAL[:len(lbls)],
                    line=dict(color="#050509" if dark else "#f7f6ff", width=3)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>Rs%{value:,.0f} · %{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", font=dict(family="JetBrains Mono", size=9, color=lc),
                    bgcolor=bg_c),
        margin=dict(t=16, b=16, l=10, r=10),
        paper_bgcolor=bg_c, plot_bgcolor=bg_c, height=275,
    )
    return fig


def _chart_waterfall(income: float, exp: Dict, dark: bool) -> go.Figure:
    tc  = "#f0eeff" if dark else "#0d0020"
    gc  = "rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(exp) + ["total"],
        x=["Income"] + list(exp.keys()) + ["Net"],
        y=[income] + [-v for v in exp.values()] + [0],
        connector=dict(line=dict(color="#8b5cf6", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#10f5a0")),
        decreasing=dict(marker=dict(color="#f02d72")),
        totals=dict(marker=dict(color="#8b5cf6")),
        texttemplate="Rs%{y:,.0f}", textposition="outside",
        textfont=dict(family="JetBrains Mono", size=9, color=tc),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=295, margin=dict(t=16, b=16, l=10, r=10),
        xaxis=dict(tickfont=dict(family="JetBrains Mono", size=9, color=tc), gridcolor=gc),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   tickprefix="Rs", gridcolor=gc),
        showlegend=False,
    )
    return fig


def _chart_gauge(score: int, dark: bool) -> go.Figure:
    color = "#10f5a0" if score >= 70 else "#f5c518" if score >= 45 else "#f02d72"
    tc    = "#6b6b99" if dark else "#5b5480"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain=dict(x=[0, 1], y=[0, 1]),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=tc,
                      tickfont=dict(family="JetBrains Mono", size=9)),
            bar=dict(color=color, thickness=0.24),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,  45], color="rgba(240,45,114,.07)"),
                dict(range=[45, 70], color="rgba(245,197,24,.07)"),
                dict(range=[70,100], color="rgba(16,245,160,.07)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
        number=dict(font=dict(family="JetBrains Mono", size=46, color=color)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=185,
        margin=dict(t=8, b=8, l=16, r=16),
    )
    return fig


def _health(
    income: float, exp: Dict, sav: float,
    tgt: Optional[float], tl: Optional[int],
) -> int:
    total = sum(exp.values())
    net   = income - total
    score = 50.0
    if income > 0:
        score += min(26, (net / income) * 88)
        score -= max(0, (total / income - 0.70) * 58)
    runway = sav / total if total > 0 else 0
    score += min(16, runway * 3)
    if tgt and tl and tl > 0:
        needed = (tgt - sav) / tl
        score += 10 if net >= needed else -(min(1.0, (needed - net) / (needed + 1)) * 15)
    return max(0, min(100, round(score)))


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Session defaults ─────────────────────────────────────────────────
    defaults = {
        "dark_mode":      True,
        "analysis_done":  False,
        "agent_state":    None,
        "financial_data": None,
        "chat_history":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


    _inject_css()

    # ══════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            '<div class="mm-badge" style="margin-bottom:20px;">'
            '<span class="sdot"></span>MoneyMentor AI &nbsp;◈&nbsp; 2027</div>',
            unsafe_allow_html=True,
        )

        # Theme toggle
        lbl = "☀️  Light Mode" if _is_dark() else "🌙  Dark Mode"
        if st.button(lbl, key="theme_btn", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("---")

        # API key
        st.markdown('<div class="s-lbl">Groq API Key</div>', unsafe_allow_html=True)
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("API key loaded ✓")
        else:
            api_key = st.text_input(
                "key", type="password", placeholder="gsk_…",
                key="api_in", label_visibility="collapsed",
            )
            if not api_key:
                st.warning("Enter your Groq API key to activate agents.")
                st.stop()

        st.markdown("---")

        # Agent roster
        st.markdown('<div class="s-lbl" style="margin-bottom:12px;">Agent Roster</div>',
                    unsafe_allow_html=True)
        roster_data = [
            ("◈ Budget Analyst",    "llama-3.3-70b-versatile",             "50/30/20 CoT math + anomaly flags"),
            ("◈ Risk Assessor",     "llama-4-scout-17b-16e",               "CRITICAL→LOW scoring + goal gap"),
            ("◈ Investment Advisor","llama-3.3-70b-versatile",             "SEBI instruments + SIP allocation"),
            ("◈ Web Researcher",    "compound-beta (native search)",        "Live RBI · CPI · market data"),
            ("◈ Supervisor CFO",    "openai/gpt-oss-120b → 70B fallback",  "30/60/90-day plan synthesis"),
        ]
        for name, mdl, desc in roster_data:
            st.markdown(
                f'<div style="background:var(--bgc);border:1px solid var(--bdr);'
                f'border-radius:10px;padding:9px 12px;margin-bottom:6px;">'
                f'<div style="font-family:Syne,sans-serif;font-weight:700;'
                f'font-size:12px;color:var(--tp);">{name}</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8.5px;'
                f'color:var(--a2);margin-top:2px;letter-spacing:.05em;">{mdl}</div>'
                f'<div style="font-family:Outfit,sans-serif;font-size:11px;'
                f'color:var(--ts);margin-top:3px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        lg = "✅ LangGraph 1.x" if _LG_AVAILABLE else "⚡ ThreadPool mode"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:var(--ts);letter-spacing:.05em;line-height:2.1;">'
            f'{lg} · true parallel fan-out<br>'
            f'compound-beta native web search<br>'
            f'NoSessionContext-safe ✓<br>'
            f'DDG fallback for web agent</div>',
            unsafe_allow_html=True,
        )

    # ── Init Groq client (cached) ─────────────────────────────────────────
    try:
        groq_client = _init_groq(api_key)
    except Exception as e:
        st.error(f"Groq init error: {e}")
        st.stop()

    # ══════════════════════════════════════════════════════════════════
    #  HERO
    # ══════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="mm-badge">'
        '<span class="sdot"></span>'
        '5-Agent Parallel &nbsp;·&nbsp; 4-Model Roster &nbsp;·&nbsp; '
        'compound-beta Live Search &nbsp;·&nbsp; GPT-OSS-120B Synthesis'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mm-title">MoneyMentor AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-sub">'
        'Four specialised AI models execute in true parallel — Llama 3.3 70B for budget '
        'math and Indian portfolio construction, Llama 4 Scout for instant risk scoring, '
        'Groq Compound Beta for live macroeconomic web search, synthesised by GPT-OSS-120B '
        'into your personalised 30/60/90-day financial execution plan.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════════════════════════════
    t_in, t_an, t_ch = st.tabs(
        ["◈  FINANCIAL INPUT", "◈  AGENT ANALYSIS", "◈  AI ADVISOR CHAT"]
    )

    # ─────────────────────────────────────────────────────────────────
    #  TAB 1 — INPUT
    # ─────────────────────────────────────────────────────────────────
    with t_in:
        st.markdown('<div class="s-lbl">Your Financial Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Enter Monthly Financials</div>', unsafe_allow_html=True)

        with st.form("fin_form"):
            st.markdown('<div class="s-lbl">Monthly Income</div>', unsafe_allow_html=True)
            income = st.number_input(
                "inc", min_value=0.0, step=500.0, value=45000.0,
                label_visibility="collapsed",
            )
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:24px;'
                f'font-weight:700;color:var(--a2);margin:4px 0 22px;">'
                f'Rs{income:,.0f}'
                f'<span style="font-size:11px;font-weight:400;color:var(--ts);"> /month</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="s-lbl">Monthly Expenses</div>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                rent      = st.number_input("🏠 Rent / Mortgage (Rs)", min_value=0.0, step=100.0, value=14000.0)
                utilities = st.number_input("⚡ Utilities (Rs)",        min_value=0.0, step=50.0,  value=2500.0)
                groceries = st.number_input("🛒 Groceries (Rs)",        min_value=0.0, step=50.0,  value=5000.0)
                transport = st.number_input("🚗 Transport (Rs)",        min_value=0.0, step=50.0,  value=3000.0)
            with cb:
                dining        = st.number_input("🍜 Dining Out (Rs)",    min_value=0.0, step=50.0, value=2500.0)
                entertainment = st.number_input("🎮 Entertainment (Rs)", min_value=0.0, step=50.0, value=1500.0)
                shopping      = st.number_input("🛍️ Shopping (Rs)",      min_value=0.0, step=50.0, value=2000.0)
                other         = st.number_input("📦 Other (Rs)",         min_value=0.0, step=50.0, value=1500.0)

            expenses = {
                "Rent": rent, "Utilities": utilities, "Groceries": groceries,
                "Transport": transport, "Dining": dining,
                "Entertainment": entertainment, "Shopping": shopping, "Other": other,
            }
            total_exp = sum(expenses.values())
            net_cf    = income - total_exp
            color_net = "var(--a2)" if net_cf >= 0 else "var(--a3)"
            if income > 0:
                st.markdown(
                    f'<div style="display:flex;gap:28px;margin:18px 0 22px;flex-wrap:wrap;">'
                    f'<div><div class="s-lbl">Total Expenses</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:17px;'
                    f'font-weight:700;color:var(--a3);">Rs{total_exp:,.0f}</div></div>'
                    f'<div><div class="s-lbl">Net Cash Flow</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:17px;'
                    f'font-weight:700;color:{color_net};">Rs{net_cf:,.0f}</div></div>'
                    f'<div><div class="s-lbl">Savings Rate</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:17px;'
                    f'font-weight:700;color:var(--a1);">{net_cf/income*100:.1f}%</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="s-lbl">Savings & Goal</div>', unsafe_allow_html=True)
            cs, cg = st.columns(2)
            with cs:
                savings = st.number_input("💰 Current Savings (Rs)", min_value=0.0, step=1000.0, value=50000.0)
            with cg:
                has_goal = st.radio("🎯 Savings Goal?", ("Yes", "No"), index=1, horizontal=True)

            target = timeline = None
            if has_goal == "Yes":
                ct, cl = st.columns(2)
                with ct:
                    target   = st.number_input("Target Amount (Rs)", min_value=0.0, step=1000.0, value=200000.0)
                with cl:
                    timeline = st.number_input("Timeline (months)", min_value=1, step=1, value=18)

            st.markdown("<br>", unsafe_allow_html=True)
            go_btn = st.form_submit_button(
                "◈  LAUNCH 5-AGENT PARALLEL ANALYSIS", use_container_width=True,
            )

        if go_btn:
            st.session_state.financial_data = dict(
                income=income, expenses=expenses,
                savings=savings, target=target, timeline=timeline,
            )
            init: AgentState = {
                "income": income, "expenses": expenses, "savings": savings,
                "target": target, "timeline": timeline,
                "budget_analysis": "", "risk_assessment": "",
                "investment_advice": "", "web_insights": "",
                "web_raw": [], "final_report": "", "messages": [],
            }

            st.markdown("---")
            st.markdown('<div class="s-lbl">Agent Pipeline · Executing</div>',
                        unsafe_allow_html=True)
            prog  = st.progress(0)
            slots = {n: st.empty() for n in
                     ("Budget Analyst", "Risk Assessor", "Investment Advisor",
                      "Web Researcher", "Supervisor CFO")}

            with st.spinner("Running parallel agents…"):
                result = _run_parallel(init, groq_client, slots, prog)

            st.session_state.agent_state   = result
            st.session_state.analysis_done = True
            st.success("✅ All 5 agents complete — open the **AGENT ANALYSIS** tab")

    # ─────────────────────────────────────────────────────────────────
    #  TAB 2 — ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    with t_an:
        if not st.session_state.analysis_done or not st.session_state.agent_state:
            st.markdown(
                '<div style="text-align:center;padding:80px 20px;">'
                '<div style="font-size:52px;margin-bottom:18px;opacity:.25;">◈</div>'
                '<div class="s-ttl" style="text-align:center;opacity:.5;">No Analysis Yet</div>'
                '<div style="font-family:Outfit,sans-serif;font-size:14px;color:var(--ts);'
                'max-width:320px;margin:0 auto;line-height:1.7;">'
                'Enter your financials in the INPUT tab and launch the agent pipeline.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            s   = st.session_state.agent_state
            d   = st.session_state.financial_data
            inc = d["income"];  exp = d["expenses"]
            sav = d["savings"]; tgt = d["target"]; tl = d["timeline"]
            tot = sum(exp.values()); net = inc - tot
            score = _health(inc, exp, sav, tgt, tl)

            # ── Vitals ─────────────────────────────────────────────────
            st.markdown('<div class="s-lbl">Financial Vitals</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Monthly Income",  f"Rs{inc:,.0f}")
            m2.metric("Total Expenses",  f"Rs{tot:,.0f}",
                      delta=f"{tot/inc*100:.0f}% of income" if inc else None,
                      delta_color="inverse")
            m3.metric("Net Cash Flow",   f"Rs{net:,.0f}",
                      delta=f"{net/inc*100:.1f}% rate" if inc else None,
                      delta_color="normal" if net >= 0 else "inverse")
            m4.metric("Savings Corpus",  f"Rs{sav:,.0f}")
            if tgt:
                m5.metric("Goal Progress", f"{min(100, sav/tgt*100):.0f}%",
                          delta=f"Rs{max(0,tgt-sav):,.0f} to go")
            else:
                m5.metric("Health Score",  f"{score}/100")

            st.markdown("---")

            # ── Charts ──────────────────────────────────────────────────
            cg, cd, cw = st.columns([1, 1.4, 1.9])
            with cg:
                st.markdown('<div class="s-lbl">Health Score</div>', unsafe_allow_html=True)
                st.plotly_chart(_chart_gauge(score, _is_dark()),
                                use_container_width=True, config={"displayModeBar": False})
                grade = ("Excellent" if score >= 80 else "Good" if score >= 65
                         else "Fair" if score >= 45 else "At Risk")
                st.markdown(f'<div class="h-grade">{grade}</div>', unsafe_allow_html=True)
            with cd:
                st.markdown('<div class="s-lbl">Expense Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(_chart_donut(exp, _is_dark()),
                                use_container_width=True, config={"displayModeBar": False})
            with cw:
                st.markdown('<div class="s-lbl">Cash Flow Waterfall</div>', unsafe_allow_html=True)
                st.plotly_chart(_chart_waterfall(inc, exp, _is_dark()),
                                use_container_width=True, config={"displayModeBar": False})

            if tgt and tgt > 0:
                pct = min(1.0, sav / tgt)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                    f'color:var(--ts);margin:6px 0 5px;">'
                    f'<span>Goal Rs{tgt:,.0f}</span><span>{pct*100:.0f}% funded</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(pct)

            st.markdown("---")

            # ── Rule diagnostics ────────────────────────────────────────
            st.markdown('<div class="s-lbl">Instant Diagnostics</div>', unsafe_allow_html=True)
            st.markdown('<div class="s-ttl">Rule-Based Analysis</div>', unsafe_allow_html=True)
            for tip in _rule_tips(inc, exp, sav, tgt, tl):
                st.markdown(
                    f'<div class="tip {tip["cls"]}">'
                    f'<div class="tip-ico">{tip["ico"]}</div>'
                    f'<div class="tip-txt">{tip["txt"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Agent outputs ───────────────────────────────────────────
            st.markdown('<div class="s-lbl">Parallel Agent Intelligence</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="s-ttl">Multi-Model Agent Reports</div>',
                        unsafe_allow_html=True)

            panels = [
                ("Budget Analyst",    "llama-3.3-70b-versatile",             "50/30/20 Budget Diagnostics", "budget_analysis"),
                ("Risk Assessor",     "llama-4-scout-17b-16e-instruct",       "Risk & Vulnerability Scoring","risk_assessment"),
                ("Investment Advisor","llama-3.3-70b-versatile",             "India Portfolio Construction", "investment_advice"),
                ("Web Researcher",    "compound-beta · native web search",    "Live Macro Market Data",       "web_insights"),
            ]
            pl, pr = st.columns(2)
            for i, (name, mdl, role, key) in enumerate(panels):
                out = s.get(key, "") or "No output."
                with (pl if i % 2 == 0 else pr):
                    with st.expander(f"◈  {name}", expanded=(i < 2)):
                        st.markdown(
                            f'<div class="ag-model">{role}</div>'
                            f'<span class="mpill">{mdl}</span>'
                            f'<div class="ag-out">{out.replace(chr(10),"<br>")}</div>',
                            unsafe_allow_html=True,
                        )

            # ── Web raw snippets ────────────────────────────────────────
            raw = s.get("web_raw") or []
            if raw:
                st.markdown("---")
                st.markdown('<div class="s-lbl">DuckDuckGo Fallback Snippets</div>',
                            unsafe_allow_html=True)
                wl, wr = st.columns(2)
                for i, r in enumerate(raw[:6]):
                    with (wl if i % 2 == 0 else wr):
                        st.markdown(
                            f'<div class="ws"><strong>{r.get("title","")}</strong><br>'
                            f'{r.get("snippet","")}'
                            f'<div class="ws-url">↗ {r.get("url","")}</div></div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("---")

            # ── Supervisor plan ─────────────────────────────────────────
            st.markdown(
                '<div class="s-lbl">Supervisor CFO · openai/gpt-oss-120b → llama-3.3-70b fallback</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="s-ttl">◈  30 / 60 / 90-Day Execution Plan</div>',
                        unsafe_allow_html=True)
            final = s.get("final_report", "")
            if final:
                st.markdown(
                    '<div class="ag-card">'
                    '<div class="ag-name">Supervisor CFO — Master Financial Plan</div>'
                    '<span class="mpill">gpt-oss-120B → llama-3.3-70B fallback</span>'
                    f'<div class="ag-out">{final.replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈  Re-run Full Analysis", key="rerun"):
                st.session_state.analysis_done = False
                st.session_state.agent_state   = None
                st.rerun()

    # ─────────────────────────────────────────────────────────────────
    #  TAB 3 — CHAT
    # ─────────────────────────────────────────────────────────────────
    with t_ch:
        st.markdown('<div class="s-lbl">Conversational Finance AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Ask MoneyMentor Anything</div>', unsafe_allow_html=True)

        ctx   = st.session_state.financial_data or {"income": 0, "expenses": {}, "savings": 0}
        exp_t = sum(ctx.get("expenses", {}).values())

        if st.session_state.financial_data:
            st.markdown(
                f'<div style="background:var(--bgc);border:1px solid var(--bdr);'
                f'border-radius:12px;padding:12px 18px;margin-bottom:18px;'
                f'display:flex;gap:30px;flex-wrap:wrap;align-items:center;">'
                f'<div><div class="s-lbl">Income</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:var(--a2);">Rs{ctx["income"]:,.0f}</div></div>'
                f'<div><div class="s-lbl">Expenses</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:var(--a3);">Rs{exp_t:,.0f}</div></div>'
                f'<div><div class="s-lbl">Savings</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:var(--a1);">Rs{ctx["savings"]:,.0f}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Add your financials in the INPUT tab for context-aware answers.")

        ag_ctx = ""
        if st.session_state.agent_state:
            _s = st.session_state.agent_state
            ag_ctx = (
                f"Budget: {_s.get('budget_analysis','')[:180]} | "
                f"Risk: {_s.get('risk_assessment','')[:180]} | "
                f"Investment: {_s.get('investment_advice','')[:180]}"
            )

        st.markdown('<div class="s-lbl" style="margin-bottom:10px;">Quick Questions</div>',
                    unsafe_allow_html=True)
        qs = [
            "Cut my biggest expense?",
            "Best SIPs for Rs5k/month?",
            "Build emergency fund fast?",
            "Invest or clear debt first?",
        ]
        qcols = st.columns(4)
        for col, q, i in zip(qcols, qs, range(4)):
            with col:
                if st.button(q, key=f"sq{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    reply = _chat(groq_client, q, ctx,
                                  st.session_state.chat_history[:-1], ag_ctx)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat history render
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:8px;">'
                    f'<div class="cb-user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin-bottom:8px;">'
                    f'<div class="cb-ai"><span class="cb-lbl">◈ MoneyMentor</span>'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        user_in = st.text_input(
            "ci", placeholder="e.g. How to save Rs1L in 6 months on my salary?",
            key="chat_in", label_visibility="collapsed",
        )
        cs, cc = st.columns([5, 1])
        with cs:
            if st.button("◈  SEND MESSAGE", key="send", use_container_width=True) \
                    and user_in.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_in})
                with st.spinner(""):
                    reply = _chat(groq_client, user_in, ctx,
                                  st.session_state.chat_history[:-1], ag_ctx)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
        with cc:
            if st.button("Clear", key="clr", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="mm-footer">'
        'MONEYMENTOR AI &nbsp;◈&nbsp; 2026 EDITION<br>'
        'LLAMA-3.3-70B-VERSATILE &nbsp;·&nbsp; LLAMA-4-SCOUT-17B &nbsp;·&nbsp; '
        'COMPOUND-BETA &nbsp;·&nbsp; OPENAI/GPT-OSS-120B<br>'
        'TRUE PARALLEL THREADPOOL &nbsp;·&nbsp; COMPOUND-BETA NATIVE SEARCH &nbsp;·&nbsp; '
        'NOSESSIONCONTEXT-SAFE ✓<br>'
        'Not financial advice. For educational purposes only.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
