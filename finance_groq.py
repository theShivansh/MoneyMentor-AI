

# ── stdlib ────────────────────────────────────────────────────────────────
import concurrent.futures
import json
import math
import operator
import random
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from groq import Groq

# ── LangGraph (optional — gracefully degraded to ThreadPool) ─────────────
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
# ══════════════════════════════════════════════════════════════════════════
MODELS: Dict[str, str] = {
    # Budget Analyst — deep 50/30/20 CoT reasoning + anomaly detection
    "budget":     "llama-3.3-70b-versatile",
    # Risk Assessor — fast multi-dimensional risk scoring
    "risk":       "llama-3.3-70b-versatile",
    # Investment Advisor — Indian market expertise: ELSS, SIP, NPS, PPF
    "investment": "llama-3.3-70b-versatile",
    # Web Researcher — Groq Compound-Beta: native agentic web search
    "research":   "compound-beta",
    # Supervisor CFO — deepseek-r1 synthesis with strong reasoning
    "supervisor": "deepseek-r1-distill-llama-70b",
    # Chat Advisor — Llama 4 Scout: snappy context-aware multi-turn chat
    "chat":       "meta-llama/llama-4-scout-17b-16e-instruct",
}
_SUPERVISOR_FALLBACK = "llama-3.3-70b-versatile"
# Model display names shown in the UI
MODEL_DISPLAY: Dict[str, str] = {
    "budget":     "llama-3.3-70b",
    "risk":       "llama-3.3-70b",
    "investment": "llama-3.3-70b",
    "research":   "compound-beta",
    "supervisor": "deepseek-r1-distill-70b",
    "chat":       "llama-4-scout",
}

# ══════════════════════════════════════════════════════════════════════════
#  TOOL SCHEMAS  (structured outputs per agent)
# ══════════════════════════════════════════════════════════════════════════

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
                    "income":           {"type": "number"},
                    "needs_total":      {"type": "number"},
                    "wants_total":      {"type": "number"},
                    "savings_surplus":  {"type": "number"},
                    "biggest_category": {"type": "string"},
                    "biggest_amount":   {"type": "number"},
                },
                "required": ["income", "needs_total", "wants_total",
                             "savings_surplus", "biggest_category", "biggest_amount"],
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
                                "category":   {"type": "string"},
                                "actual_pct": {"type": "number"},
                                "ideal_pct":  {"type": "number"},
                                "excess_rs":  {"type": "number"},
                            },
                        },
                    }
                },
                "required": ["anomalies"],
            },
        },
    },
]

RISK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_financial_risk",
            "description": "Assign CRITICAL/HIGH/MEDIUM/LOW risk rating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level":        {"type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]},
                    "expense_ratio_pct": {"type": "number"},
                    "runway_months":     {"type": "number"},
                    "emergency_gap_rs":  {"type": "number"},
                    "top_vulnerability": {"type": "string"},
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
            "description": "Calculate whether the savings goal is achievable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_required":  {"type": "number"},
                    "monthly_available": {"type": "number"},
                    "feasible":         {"type": "boolean"},
                    "shortfall_rs":     {"type": "number"},
                    "months_to_goal":   {"type": "number"},
                },
                "required": ["monthly_required", "monthly_available", "feasible"],
            },
        },
    },
]

INVESTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "build_india_portfolio",
            "description": "Construct monthly SIP/investment allocation across Indian instruments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "investable_monthly": {"type": "number"},
                    "allocations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "instrument": {"type": "string"},
                                "fund_name":  {"type": "string"},
                                "amount_rs":  {"type": "number"},
                                "percent":    {"type": "number"},
                                "rationale":  {"type": "string"},
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
                    "monthly_sip":     {"type": "number"},
                    "annual_return":   {"type": "number"},
                    "years":           {"type": "number"},
                    "projected_value": {"type": "number"},
                },
                "required": ["monthly_sip", "annual_return", "years", "projected_value"],
            },
        },
    },
]

WEB_TOOLS: List[Dict] = []  # compound-beta uses its own built-in search

SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_90_day_plan",
            "description": "Synthesise agent reports into a 30/60/90-day execution plan.",
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
                    "bold_move": {"type": "string"},
                    "confidence_score": {"type": "integer"},
                },
                "required": ["day_30", "day_60", "day_90", "bold_move"],
            },
        },
    },
]

# ══════════════════════════════════════════════════════════════════════════
#  INDIAN TAX CONSTANTS (FY 2026-27)
# ══════════════════════════════════════════════════════════════════════════
TAX_SLABS_NEW = [  # New regime (FY 2026-27)
    (0,       300_000, 0.00),
    (300_001, 700_000, 0.05),
    (700_001, 1_000_000, 0.10),
    (1_000_001, 1_200_000, 0.15),
    (1_200_001, 1_500_000, 0.20),
    (1_500_001, float("inf"), 0.30),
]
TAX_SLABS_OLD = [  # Old regime
    (0,       250_000, 0.00),
    (250_001, 500_000, 0.05),
    (500_001, 1_000_000, 0.20),
    (1_000_001, float("inf"), 0.30),
]
STD_DEDUCTION_OLD = 50_000
STD_DEDUCTION_NEW = 75_000
MAX_80C = 150_000
MAX_80CCD1B = 50_000  # NPS additional
MAX_80D_SELF = 25_000
MAX_80D_PARENTS = 50_000  # for senior parents
HRA_EXEMPT_METRO = 0.50  # 50% of basic for metro cities
HRA_EXEMPT_NONMETRO = 0.40

_PAL = ["#8b5cf6", "#10f5a0", "#f02d72", "#f5c518",
        "#06b6d4", "#f97316", "#a78bfa", "#34d399", "#ec4899", "#14b8a6"]


# ══════════════════════════════════════════════════════════════════════════
#  THEME HELPERS
# ══════════════════════════════════════════════════════════════════════════

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
.stSelectbox label,[data-testid="stWidgetLabel"] p{{
  color:{ts}!important;font-size:12px!important;
  font-family:'JetBrains Mono',monospace!important;letter-spacing:.04em!important;
}}
.stSlider [data-baseweb="slider"]{{padding:4px 0!important;}}

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
.stDownloadButton>button{{
  background:linear-gradient(135deg,{a2} 0%,{a1} 100%)!important;
  color:#000!important; border:none!important; border-radius:10px!important;
  font-family:'Syne',sans-serif!important; font-weight:700!important;
  font-size:11px!important; letter-spacing:.1em!important;
  text-transform:uppercase!important;
}}

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
  flex-wrap:wrap!important;
}}
.stTabs [data-baseweb="tab"]{{
  font-family:'Syne',sans-serif!important; font-weight:700!important;
  font-size:10.5px!important; letter-spacing:.09em!important;
  text-transform:uppercase!important; color:{ts}!important;
  border-radius:9px!important; border:none!important;
  background:transparent!important; padding:8px 16px!important;
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

/* ── TABLE ──────────────────────────────────────────── */
[data-testid="stTable"] table{{
  background:{bgc}!important; border-radius:12px!important;
  border:1px solid {bdr}!important;
}}
[data-testid="stTable"] th{{
  background:{bgc2}!important; color:{a1}!important;
  font-family:'JetBrains Mono',monospace!important; font-size:10px!important;
  letter-spacing:.08em!important; text-transform:uppercase!important;
}}
[data-testid="stTable"] td{{
  color:{tp}!important; font-family:'Outfit',sans-serif!important;
  font-size:13px!important; border-color:{bdr}!important;
}}

/* ── SCROLLBAR ─────────────────────────────────────── */
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-track{{background:{bg};}}
::-webkit-scrollbar-thumb{{background:linear-gradient({a1},{a3});border-radius:5px;}}
hr{{border-color:{bdr}!important;opacity:.6!important;margin:1.5rem 0!important;}}
.stSpinner>div{{border-top-color:{a1}!important;}}

/* ══════════════════════════════════════════════════
   BRANDED COMPONENTS
══════════════════════════════════════════════════ */

.mm-badge{{
  display:inline-flex;align-items:center;gap:6px;
  background:{tb};border:1px solid {a1};border-radius:99px;
  padding:4px 16px;font-family:'JetBrains Mono',monospace;
  font-size:9.5px;color:{a1};letter-spacing:.16em;text-transform:uppercase;
  margin-bottom:14px;
}}
.mm-title{{
  font-family:'Syne',sans-serif;font-weight:800;
  font-size:clamp(2.2rem,5vw,3.4rem);line-height:1.0;
  background:linear-gradient(135deg,{a1} 0%,{a3} 45%,{a2} 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:12px;letter-spacing:-.01em;
}}
.mm-sub{{
  font-family:'Outfit',sans-serif;font-size:15px;color:{ts};
  max-width:680px;line-height:1.7;margin-bottom:24px;font-weight:400;
}}
.s-lbl{{
  font-family:'JetBrains Mono',monospace;font-size:9.5px;
  letter-spacing:.22em;text-transform:uppercase;color:{a2};
  margin-bottom:4px;opacity:.85;
}}
.s-ttl{{
  font-family:'Syne',sans-serif;font-weight:700;
  font-size:1.18rem;color:{tp};margin-bottom:16px;
}}
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
.ag-name{{font-family:'Syne',sans-serif;font-weight:700;font-size:15px;color:{tp};margin-bottom:3px;}}
.ag-model{{font-family:'JetBrains Mono',monospace;font-size:9px;color:{a2};text-transform:uppercase;letter-spacing:.11em;}}
.ag-out{{
  font-family:'Outfit',sans-serif;font-size:13.5px;color:{ts};
  margin-top:14px;line-height:1.78;
  border-left:3px solid {a1};padding-left:15px;
}}
.sdot{{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:{a2};box-shadow:0 0 8px {a2};margin-right:7px;
  animation:pulse 2s infinite;
}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.8)}}}}
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
.mpill{{
  display:inline-block;background:{tb};border:1px solid {a1};
  border-radius:99px;padding:2px 10px;
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{a1};
  letter-spacing:.06em;margin-left:6px;vertical-align:middle;
}}
.h-grade{{
  font-family:'Syne',sans-serif;font-size:11px;text-transform:uppercase;
  letter-spacing:.18em;color:{ts};text-align:center;margin-top:6px;
}}
.stat-chip{{
  display:inline-block;background:{tb};border:1px solid {bdr};
  border-radius:99px;padding:4px 14px;
  font-family:'JetBrains Mono',monospace;font-size:11px;color:{a2};
}}
.ws{{
  background:{card};border:1px solid {bdr};border-left:3px solid {a2};
  border-radius:12px;padding:11px 15px;margin-bottom:9px;
  font-family:'Outfit',sans-serif;font-size:13px;color:{tp};line-height:1.65;
}}
.ws-url{{
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{a2};
  letter-spacing:.07em;margin-top:6px;opacity:.6;
}}
.goal-card{{
  background:{card};border:1px solid {bdr};border-radius:16px;
  padding:1.2rem 1.5rem;margin-bottom:12px;
  position:relative;overflow:hidden;
  box-shadow:{glow};transition:border-color .2s;
}}
.goal-card:hover{{border-color:{a1};}}
.goal-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,{a2},{a1});
}}
.tax-card{{
  background:{card};border:1px solid {bdr};border-radius:14px;
  padding:1.1rem 1.4rem;margin-bottom:10px;
  transition:border-color .18s,transform .18s;
}}
.tax-card:hover{{border-color:{a1};transform:translateY(-1px);}}
.market-card{{
  background:{card};border:1px solid {bdr};border-radius:14px;
  padding:1.2rem 1.4rem;margin-bottom:10px;
  border-left:3px solid {a2};
}}
.kpi-pill{{
  background:{tb};border:1px solid {a1};border-radius:99px;
  padding:6px 18px;font-family:'JetBrains Mono',monospace;
  font-size:12px;color:{a1};display:inline-block;margin:4px;
}}
.mm-footer{{
  font-family:'JetBrains Mono',monospace;font-size:9px;color:{ts};
  opacity:.3;text-align:center;padding:16px 0 24px;
  letter-spacing:.1em;line-height:2.2;
}}
@keyframes fadeInUp{{
  from{{opacity:0;transform:translateY(16px)}}
  to{{opacity:1;transform:translateY(0)}}
}}
.fade-in{{animation:fadeInUp .45s ease both;}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  GROQ HELPERS  (pure Python — NO st.* calls — thread-safe)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _init_groq(key: str) -> Groq:
    return Groq(api_key=key)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> chain-of-thought blocks (DeepSeek/R1 style)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _extract_tool_text(msg) -> str:
    """
    Extract useful text from a model response that chose to call a tool
    instead of returning plain text content (content == None).
    Formats the tool-call arguments as readable key: value lines.
    """
    try:
        tool_calls = getattr(msg, "tool_calls", None) or []
        parts: List[str] = []
        for tc in tool_calls:
            fn   = getattr(tc, "function", None)
            if fn is None:
                continue
            name = getattr(fn, "name", "")
            args_raw = getattr(fn, "arguments", "{}")
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {"raw": args_raw}
            lines = [f"**{name.replace('_', ' ').title()}**"]
            for k, v in args.items():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            lines.append("  • " + " | ".join(
                                f"{ik}: {iv}" for ik, iv in item.items()
                            ))
                        else:
                            lines.append(f"  • {item}")
                elif isinstance(v, float):
                    lines.append(f"  {k.replace('_',' ').title()}: Rs{v:,.0f}" if "rs" in k or "amount" in k or "gap" in k or "monthly" in k else f"  {k.replace('_',' ').title()}: {v}")
                else:
                    lines.append(f"  {k.replace('_',' ').title()}: {v}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts) if parts else ""
    except Exception:
        return ""


def _call(
    client: Groq,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 600,
    temp: float = 0.55,
    tools: Optional[List[Dict]] = None,
) -> str:
    """
    Single-turn LLM call. Thread-safe — zero st.* usage.
    Handles tool_call responses (content=None) and supervisor fallback.
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
        r   = client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        # Content may be None when the model chose to call a tool instead
        text = msg.content or ""
        if not text.strip():
            text = _extract_tool_text(msg)
        # If tool extraction also empty, do a follow-up call without tools
        if not text.strip() and tools:
            kwargs2 = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
            r2   = client.chat.completions.create(**kwargs2)
            text = r2.choices[0].message.content or ""
        return _strip_think(text)
    except Exception as exc:
        short = str(exc)[:120]
        # Supervisor model fallback
        if model == MODELS["supervisor"]:
            try:
                kwargs2 = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
                kwargs2["model"] = _SUPERVISOR_FALLBACK
                r2   = client.chat.completions.create(**kwargs2)
                text = r2.choices[0].message.content or ""
                return _strip_think(text)
            except Exception as e2:
                return f"[Supervisor error: {str(e2)[:90]}]"
        return f"[{model.split('/')[-1]} error: {short}]"


# ══════════════════════════════════════════════════════════════════════════
#  WEB SEARCH — multi-pattern scraper with knowledge-base fallback
# ══════════════════════════════════════════════════════════════════════════

# Curated India finance knowledge base — always available, never empty
_INDIA_FINANCE_KB = """
RBI REPO RATE: The Reserve Bank of India kept the repo rate at 6.00% in April 2025, \
down from 6.50% in Feb 2025. FD rates: SBI 6.8–7.1%, HDFC Bank 7.0–7.4%, ICICI 7.0–7.25% \
for 1–3yr tenors. Savings account rates: 2.7–3.5% (major banks), 5–7% (small finance banks).

TOP SIP CATEGORIES (last 3 months, approx returns): Small Cap funds: 8–12% | \
Mid Cap funds: 6–9% | Flexi Cap funds: 5–8% | Large & Mid Cap: 5–7% | \
Sectoral (Defence/PSU): 10–15%. Top performing ELSS: Quant ELSS, Mirae Asset ELSS.

INDIA CPI INFLATION: India's retail inflation (CPI) fell to 3.34% in March 2025, \
the lowest in 6 years, below RBI's 4% target. Food inflation eased to 2.69%. \
Core inflation ~3.3%. Household budget impact: lower vegetable/cereal prices reducing \
grocery bills by 8–12% vs last year.

LIQUID FUND YIELDS: Best 7-day yields (May 2025): Mirae Asset Liquid Fund 7.01%, \
HDFC Liquid Fund 7.05%, SBI Liquid Fund 6.95%, Nippon India Liquid 7.02%. \
Liquid funds offer better returns than savings accounts and full redemption in T+1.

ELSS TOP FUNDS (FY2025 CAGR): Quant ELSS: 28% | Motilal Oswal ELSS: 22% | \
Mirae Asset ELSS: 18% | Canara Robeco ELSS: 16%. 3-year lock-in, 80C deduction up to Rs1.5L.

PPF RATE: 7.1% p.a. (Q1 FY2026, unchanged). EEE category — deposits, interest, \
maturity all tax-free. Annual limit Rs1.5L. 15-year lock-in with partial withdrawal after yr 7.

NPS RETURNS: NPS Tier-1 equity (E) CAGR: LIC Pension Fund 14.2%, SBI Pension 13.8%, \
HDFC Pension 14.6% (10-yr). 80CCD(1B): extra Rs50,000 deduction over 80C limit.

NIFTY 50: Around 24,000–24,500 levels (May 2025). P/E ~21x. IT, Banking, FMCG sectors \
outperforming. FII inflows positive. 1-year return ~12%. Mid/Small Cap indices up 18–22%.
"""


def _ddg(query: str, n: int = 5) -> List[Dict]:
    """
    Try DuckDuckGo HTML search with multiple CSS-class patterns (DDG changes them often).
    Returns list of {title, snippet, url} dicts.
    """
    out: List[Dict] = []
    try:
        q   = urllib.parse.quote_plus(query)
        req = urllib.request.Request(
            f"https://html.duckduckgo.com/html/?q={q}",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")

        # ── Pattern set 1: classic DDG HTML class names ────────────────
        snips  = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>', html, re.S)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.S)
        hrefs  = re.findall(r'class="result__url"[^>]*>(.*?)</(?:a|span)>', html, re.S)

        # ── Pattern set 2: newer DDG markup (data-result or role attrs) ─
        if not snips:
            snips  = re.findall(r'class="[^"]*snippet[^"]*"[^>]*>(.*?)</(?:a|span|div)>', html, re.S)
            titles = re.findall(r'class="[^"]*title[^"]*"[^>]*>(.*?)</(?:a|h2|span)>', html, re.S)
            hrefs  = re.findall(r'class="[^"]*url[^"]*"[^>]*>(.*?)</(?:a|span|div)>', html, re.S)

        # ── Pattern set 3: generic paragraph extraction ─────────────────
        if not snips:
            snips = re.findall(r'<(?:p|div)[^>]*>([\w][^<]{40,300})</(?:p|div)>', html)
            titles = []
            hrefs  = []

        clean = lambda t: re.sub(r"<[^>]+>", "", t).replace("&amp;", "&").replace("&nbsp;", " ").strip()

        for i, s in enumerate(snips[:n]):
            snippet_text = clean(s)
            if len(snippet_text) < 15:   # skip noise
                continue
            out.append({
                "title":   clean(titles[i]) if i < len(titles) else f"Result {i+1}",
                "snippet": snippet_text,
                "url":     clean(hrefs[i]) if i < len(hrefs) else "",
            })
    except Exception:
        pass  # silent — caller will use KB fallback

    return out


# ══════════════════════════════════════════════════════════════════════════
#  AGENT STATE
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


# ══════════════════════════════════════════════════════════════════════════
#  5 SPECIALIST AGENTS  (pure Python, no st.* — thread-safe)
# ══════════════════════════════════════════════════════════════════════════

def _budget_agent(s: AgentState, c: Groq) -> str:
    """Agent 1 — Budget Analyst: llama-3.3-70b-versatile"""
    total = sum(s["expenses"].values())
    net   = s["income"] - total
    sr    = net / s["income"] * 100 if s["income"] > 0 else 0
    needs = sum(v for k, v in s["expenses"].items()
                if k.lower() in ("rent", "utilities", "groceries", "transport",
                                  "emi", "insurance", "medical"))
    wants = sum(v for k, v in s["expenses"].items()
                if k.lower() in ("dining", "entertainment", "shopping",
                                  "subscriptions", "travel"))
    exp_lines = "\n".join(f"  {k}: Rs{v:,.0f}" for k, v in s["expenses"].items())

    system = (
        "You are a precise budget analyst for Indian households. "
        "Provide EXACTLY 4 numbered, actionable budget insights. "
        "Each insight must include a specific Rs amount. "
        "Be surgical — no generic advice, no preamble, no headers."
    )
    user = (
        f"Monthly Income: Rs{s['income']:,.0f}\n"
        f"Expenses:\n{exp_lines}\n"
        f"Total: Rs{total:,.0f} | Surplus: Rs{net:,.0f} | Savings rate: {sr:.1f}%\n"
        f"50/30/20 check — Needs: Rs{needs:,.0f} ({needs/s['income']*100 if s['income'] else 0:.0f}%), "
        f"Wants: Rs{wants:,.0f} ({wants/s['income']*100 if s['income'] else 0:.0f}%), "
        f"Savings: Rs{net:,.0f} ({sr:.0f}%)\n\n"
        "Write 4 numbered insights. Flag the single biggest spending inefficiency with "
        "the exact Rs waste and a specific cut recommendation."
    )
    return _call(c, system, user, model=MODELS["budget"], max_tokens=550)


def _risk_agent(s: AgentState, c: Groq) -> str:
    """Agent 2 — Risk Assessor: llama-3.3-70b-versatile"""
    total  = sum(s["expenses"].values())
    runway = round(s["savings"] / total, 1) if total > 0 else 0
    er     = round(total / s["income"] * 100, 1) if s["income"] > 0 else 0
    gap    = max(0, total * 6 - s["savings"])
    net    = s["income"] - total
    tgt    = s.get("target") or 0
    tl     = s.get("timeline") or 12
    monthly_needed = (tgt - s["savings"]) / tl if tgt > s["savings"] and tl > 0 else 0

    system = (
        "You are a senior financial risk specialist for Indian households. "
        "Output must follow this exact format:\n"
        "RISK LEVEL: [CRITICAL|HIGH|MEDIUM|LOW]\n"
        "1. [Vulnerability with Rs impact]\n"
        "2. [Vulnerability with Rs impact]\n"
        "3. [Vulnerability with Rs impact]\n"
        "IMMEDIATE FIX: [1 specific action with Rs amount]\n"
        "No extra text."
    )
    user = (
        f"Expense-to-income ratio: {er}% | Emergency runway: {runway} months\n"
        f"6-month emergency fund gap: Rs{gap:,.0f}\n"
        f"Monthly burn: Rs{total:,.0f} | Net flow: Rs{net:,.0f}\n"
        f"Savings corpus: Rs{s['savings']:,.0f}\n"
        f"Goal: Rs{tgt:,.0f} in {tl} months | Monthly required: Rs{monthly_needed:,.0f}\n"
        f"Monthly available: Rs{net:,.0f} | Shortfall: Rs{max(0, monthly_needed - net):,.0f}"
    )
    return _call(c, system, user, model=MODELS["risk"], max_tokens=420)


def _investment_agent(s: AgentState, c: Groq) -> str:
    """Agent 3 — Investment Advisor: llama-3.3-70b-versatile"""
    total      = sum(s["expenses"].values())
    investable = max(0, s["income"] - total)
    tgt        = s.get("target") or 0
    tl         = s.get("timeline") or 36

    system = (
        "You are a SEBI-registered investment advisor for Indian retail investors. "
        "Name real, specific Indian mutual funds. Give exact Rs amounts. "
        "Format each allocation as: [Instrument] — [Fund Name] — Rs[X]/mo ([Y]%) — [reason]\n"
        "End with: 12-month projected corpus: Rs[X] | 3-year projected corpus: Rs[Y]"
    )
    user = (
        f"Monthly investable surplus: Rs{investable:,.0f}\n"
        f"Existing corpus: Rs{s['savings']:,.0f}\n"
        f"Financial goal: Rs{tgt:,.0f} in {tl} months\n\n"
        "Build a complete monthly allocation across these instruments:\n"
        "• Emergency Liquid Fund (target: 6-month buffer)\n"
        "• ELSS SIP (80C tax saving, 3-yr lock-in)\n"
        "• Nifty 50 Index SIP (core equity)\n"
        "• PPF (long-term debt, EEE tax)\n"
        "• NPS Tier-1 (80CCD(1B) benefit)\n"
        "Skip any instrument if surplus is insufficient. Be specific about fund names."
    )
    return _call(c, system, user, model=MODELS["investment"], max_tokens=600)


def _web_agent(s: AgentState, c: Groq) -> Tuple[str, List[Dict]]:
    """Agent 4 — Web Researcher: compound-beta (native agentic search) → DDG fallback"""
    # Keep system + user SHORT to avoid 413 — compound-beta is sensitive to payload size
    system = (
        "You are a macro-finance analyst covering India. "
        "Search the web and report current figures with citations."
    )
    user = (
        "Fetch and report these 4 India finance data points right now:\n"
        "1. RBI repo rate and current FD rates\n"
        "2. Best-performing SIP category last 3 months (% returns)\n"
        "3. India CPI headline inflation (latest)\n"
        "4. Best liquid fund 7-day yield"
    )

    raw: List[Dict] = []
    try:
        r = c.chat.completions.create(
            model=MODELS["research"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=700,
            temperature=0.3,
        )
        msg = r.choices[0].message
        ans = _strip_think(msg.content or "")
        # compound-beta may return content via tool_calls on some queries
        if not ans.strip():
            ans = _extract_tool_text(msg)
        if not ans.strip():
            raise ValueError("Empty response from compound-beta")
        return ans, raw
    except Exception:
        # Fallback: DDG scrape → if empty, use built-in India finance KB
        queries = [
            "RBI repo rate India 2025",
            "top SIP mutual fund India returns 2025",
            "India CPI inflation 2025",
            "best liquid fund yield India 2025",
        ]
        for q in queries:
            raw.extend(_ddg(q, n=2))

        # Build context: prefer live DDG snippets, fall back to curated KB
        live_ctx = "\n".join(
            f"{h['title']}: {h['snippet']}" for h in raw if h.get("snippet")
        )[:700]
        # Always append KB so there is guaranteed substantive content
        ctx = (live_ctx + "\n\n" + _INDIA_FINANCE_KB).strip()[:1200]

        fallback_sys = (
            "You are a macro-finance analyst for Indian retail investors. "
            "Use the data provided to give a clear, numbered answer. Use Rs signs."
        )
        fallback_user = (
            "Using the India finance data below, answer these 4 questions:\n"
            "1. Current RBI repo rate and FD rates\n"
            "2. Best-performing SIP category last 3 months\n"
            "3. India CPI inflation and household impact\n"
            "4. Best liquid fund 7-day yield\n\n"
            f"Data:\n{ctx}"
        )
        ans = _call(c, fallback_sys, fallback_user,
                    model="llama-3.3-70b-versatile", max_tokens=550)
        return ans, raw


def _supervisor_agent(s: AgentState, c: Groq) -> str:
    """Agent 5 — Supervisor CFO: deepseek-r1-distill-llama-70b → llama-3.3-70b fallback.
    Trims each agent's output to avoid 413 Request Entity Too Large.
    """
    # Hard-cap each sub-report to prevent oversized payloads
    def _cap(text: str, limit: int = 300) -> str:
        text = (text or "").strip()
        return text[:limit] + "…" if len(text) > limit else text

    budget_snip  = _cap(s["budget_analysis"],   280)
    risk_snip    = _cap(s["risk_assessment"],    280)
    invest_snip  = _cap(s["investment_advice"],  280)
    web_snip     = _cap(s["web_insights"],       200)
    total_exp    = sum(s["expenses"].values())
    net_flow     = s["income"] - total_exp
    tgt          = s.get("target") or 0
    tl           = s.get("timeline") or 12

    system = (
        "You are the Chief Financial Officer AI. Synthesise expert agent findings "
        "into a concrete 30/60/90-day action plan. Every action needs a Rs amount. "
        "Be decisive, specific, and concise."
    )
    user = (
        f"User: Rs{s['income']:,.0f}/mo income | Rs{total_exp:,.0f} expenses | "
        f"Rs{net_flow:,.0f} surplus | Rs{s['savings']:,.0f} corpus | "
        f"Goal: Rs{tgt:,.0f} in {tl} months\n\n"
        f"BUDGET: {budget_snip}\n"
        f"RISK: {risk_snip}\n"
        f"INVESTMENTS: {invest_snip}\n"
        f"MARKET: {web_snip}\n\n"
        "Output EXACTLY this format — no extra text:\n"
        "30-DAY PLAN:\n1. [action + Rs amount]\n2. [action + Rs amount]\n3. [action + Rs amount]\n"
        "60-DAY PLAN:\n1. [action + Rs amount]\n2. [action + Rs amount]\n3. [action + Rs amount]\n"
        "90-DAY PLAN:\n1. [action + Rs amount]\n2. [action + Rs amount]\n3. [action + Rs amount]\n"
        "BOLD MOVE: [one decisive all-caps action with Rs amount]\n"
        "CFO CONFIDENCE: [0-100]"
    )
    return _call(c, system, user, model=MODELS["supervisor"], max_tokens=700, temp=0.4)


# ══════════════════════════════════════════════════════════════════════════
#  PARALLEL ORCHESTRATOR  (fan-out → fan-in → supervisor synthesis)
# ══════════════════════════════════════════════════════════════════════════

def _run_parallel(
    state:  AgentState,
    client: Groq,
    slots:  Dict[str, Any],
    prog:   Any,
) -> AgentState:
    """
    Fan-out: run 4 agents in true parallel via ThreadPoolExecutor.
    Fan-in: collect all results before supervisor synthesises.
    UI updates happen ONLY in the main thread (after futures complete).
    This pattern permanently fixes the 'too many values to unpack' LangGraph error.
    """
    # ── 1. Mark all queued (main thread) ──────────────────────────────
    for name in ("Budget Analyst", "Risk Assessor", "Investment Advisor", "Web Researcher"):
        slots[name].markdown(
            f'<div class="ps">◯ &nbsp; {name} · Queued</div>',
            unsafe_allow_html=True,
        )
    slots["Supervisor CFO"].markdown(
        '<div class="ps">◯ &nbsp; Supervisor CFO · Awaiting parallel agents…</div>',
        unsafe_allow_html=True,
    )
    prog.progress(0.05)

    # ── 2. Fan-out: submit pure Python functions to thread pool ────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "budget":     pool.submit(_budget_agent,     state, client),
            "risk":       pool.submit(_risk_agent,       state, client),
            "investment": pool.submit(_investment_agent, state, client),
            "web":        pool.submit(_web_agent,        state, client),
        }
        # Fan-in: block until ALL futures resolve
        budget_out     = futures["budget"].result()
        risk_out       = futures["risk"].result()
        invest_out     = futures["investment"].result()
        web_result     = futures["web"].result()

    # Unpack web agent result (returns tuple: str, List[Dict])
    if isinstance(web_result, tuple):
        web_out, w_raw = web_result
    else:
        web_out, w_raw = str(web_result), []

    # ── 3. Update UI AFTER threads complete (main thread — safe) ──────
    prog.progress(0.80)
    for name in ("Budget Analyst", "Risk Assessor", "Investment Advisor", "Web Researcher"):
        slots[name].markdown(
            f'<div class="ps ok">✓ &nbsp; {name} · Complete</div>',
            unsafe_allow_html=True,
        )

    # ── 4. Merge results into state dict ──────────────────────────────
    state = {
        **state,
        "budget_analysis":   budget_out,
        "risk_assessment":   risk_out,
        "investment_advice": invest_out,
        "web_insights":      web_out,
        "web_raw":           w_raw,
    }

    # ── 5. Supervisor synthesises sequentially (needs all 4 outputs) ──
    prog.progress(0.88)
    slots["Supervisor CFO"].markdown(
        '<div class="ps run"><span class="sdot"></span>Supervisor CFO · Synthesising…</div>',
        unsafe_allow_html=True,
    )
    final = _supervisor_agent(state, client)
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
    client:    Groq,
    msg:       str,
    ctx:       Dict,
    history:   List[Dict],
    agent_ctx: str = "",
) -> str:
    exp = sum(ctx.get("expenses", {}).values())
    sys_prompt = (
        "You are MoneyMentor, a sharp and warm 2027-era AI financial advisor for India. "
        "Expert in Indian personal finance — mutual funds, SIPs, PPF, NPS, tax-saving, "
        "SEBI regulations, RBI policies, and Indian tax law. "
        f"User context: Rs{ctx.get('income', 0):,.0f}/mo income, "
        f"Rs{exp:,.0f} expenses, Rs{ctx.get('savings', 0):,.0f} savings.\n"
        + (f"Agent analysis snapshot: {agent_ctx[:600]}\n" if agent_ctx else "")
        + "Reply in max 4 sentences. Lead with one relevant emoji. "
        "Use Rs signs. Be direct, actionable, and India-specific."
    )
    msgs = [{"role": "system", "content": sys_prompt}]
    for h in history[-10:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": msg})
    try:
        r = client.chat.completions.create(
            model=MODELS["chat"], messages=msgs, max_tokens=350, temperature=0.72,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ {str(e)[:90]}"


# ══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════

def _health(income: float, exp: Dict, sav: float,
            tgt: Optional[float], tl: Optional[int]) -> int:
    """Compute 0-100 financial health score."""
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


def _sip_future_value(monthly: float, annual_rate: float, years: float) -> float:
    """SIP corpus formula: FV = P × [(1+r)^n – 1] / r × (1+r)"""
    if monthly <= 0 or years <= 0:
        return 0.0
    r = annual_rate / 12
    n = int(years * 12)
    if r == 0:
        return monthly * n
    return monthly * ((((1 + r) ** n) - 1) / r) * (1 + r)


def _tax_calc(annual_income: float, regime: str = "new",
              deductions: Dict[str, float] = None) -> Dict:
    """
    Compute income tax under Indian FY 2026-27 rules.
    Returns: tax_old, tax_new, taxable_old, taxable_new, savings, recommendation.
    """
    if deductions is None:
        deductions = {}

    d80c      = min(deductions.get("80c", 0), MAX_80C)
    d80ccd1b  = min(deductions.get("80ccd1b", 0), MAX_80CCD1B)
    d80d_self = min(deductions.get("80d_self", 0), MAX_80D_SELF)
    d80d_par  = min(deductions.get("80d_parents", 0), MAX_80D_PARENTS)
    hra       = deductions.get("hra", 0)

    def _compute(slabs, taxable):
        tax = 0.0
        for lo, hi, rate in slabs:
            if taxable <= lo:
                break
            tax += (min(taxable, hi) - lo) * rate
        return tax

    # New regime
    taxable_new = max(0, annual_income - STD_DEDUCTION_NEW)
    tax_new     = _compute(TAX_SLABS_NEW, taxable_new)
    # Rebate u/s 87A: no tax if taxable <= 7L in new regime
    if taxable_new <= 700_000:
        tax_new = 0
    tax_new_with_cess = tax_new * 1.04  # 4% health & education cess

    # Old regime
    taxable_old = max(0, annual_income - STD_DEDUCTION_OLD - d80c - d80ccd1b
                      - d80d_self - d80d_par - hra)
    tax_old     = _compute(TAX_SLABS_OLD, taxable_old)
    # Rebate u/s 87A: no tax if taxable <= 5L in old regime
    if taxable_old <= 500_000:
        tax_old = 0
    tax_old_with_cess = tax_old * 1.04

    better = "new" if tax_new_with_cess <= tax_old_with_cess else "old"
    savings = abs(tax_old_with_cess - tax_new_with_cess)

    return {
        "taxable_new": taxable_new,
        "taxable_old": taxable_old,
        "tax_new":     tax_new_with_cess,
        "tax_old":     tax_old_with_cess,
        "savings":     savings,
        "better":      better,
        "monthly_new": tax_new_with_cess / 12,
        "monthly_old": tax_old_with_cess / 12,
    }


def _monte_carlo(monthly_sip: float, years: int, annual_return: float,
                 volatility: float, n_sims: int = 1200) -> Dict:
    """
    Monte Carlo simulation of SIP portfolio over `years` using
    log-normal monthly returns.  Returns percentile endpoints.
    """
    if monthly_sip <= 0 or years <= 0:
        return {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0,
                "invested": 0, "simulations": []}

    n_months = years * 12
    mu_monthly  = annual_return / 12
    sig_monthly = volatility / math.sqrt(12)
    endpoints   = []

    for _ in range(n_sims):
        corpus = 0.0
        for _ in range(n_months):
            r = random.gauss(mu_monthly, sig_monthly)
            corpus = (corpus + monthly_sip) * (1 + r)
        endpoints.append(corpus)

    endpoints.sort()
    invested = monthly_sip * n_months
    return {
        "p10": endpoints[int(0.10 * n_sims)],
        "p25": endpoints[int(0.25 * n_sims)],
        "p50": endpoints[int(0.50 * n_sims)],
        "p75": endpoints[int(0.75 * n_sims)],
        "p90": endpoints[int(0.90 * n_sims)],
        "invested":    invested,
        "simulations": endpoints,
        "mean":  sum(endpoints) / len(endpoints),
    }


def _rule_tips(income: float, expenses: Dict, savings: float,
               target: Optional[float], timeline: Optional[int]) -> List[Dict]:
    """Deterministic rule-based financial diagnostics."""
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
                         "txt": f"Strong {sr:.1f}% savings rate. "
                                "Step up SIP by 10% annually to maximise compounding."})

    if expenses:
        top = max(expenses, key=expenses.get)
        pct = expenses[top] / income * 100 if income > 0 else 0
        tips.append({"cls": "tip-info", "ico": "🔍",
                     "txt": f"'{top}' is your largest expense: Rs{expenses[top]:,.0f} "
                            f"({pct:.0f}% of income). A 10% cut saves "
                            f"Rs{expenses[top] * .1:,.0f}/mo — Rs{expenses[top] * 1.2:,.0f}/yr."})

    runway    = savings / total if total > 0 else 0
    em_target = total * 6
    if runway < 3:
        tips.append({"cls": "tip-warn", "ico": "🛡️",
                     "txt": f"Emergency fund covers only {runway:.1f} months. "
                            f"Target: Rs{em_target:,.0f} (6 months). "
                            "Park in a liquid fund — no equity SIP until this is built."})
    elif runway < 6:
        tips.append({"cls": "tip-info", "ico": "🛡️",
                     "txt": f"{runway:.1f}-month buffer. Push to 6 months "
                            f"(Rs{em_target:,.0f}). You're Rs{em_target - savings:,.0f} away."})
    else:
        tips.append({"cls": "tip-ok", "ico": "🛡️",
                     "txt": f"Solid {runway:.1f}-month emergency fund. "
                            "Surplus beyond 6 months should go into equity SIPs."})

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
#  PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════

def _chart_donut(exp: Dict, dark: bool) -> go.Figure:
    lbls = [k for k, v in exp.items() if v > 0]
    vals = [v for v in exp.values() if v > 0]
    lc   = "#f0eeff" if dark else "#0d0020"
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
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=16, b=16, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
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
        height=300, margin=dict(t=16, b=16, l=10, r=10),
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
                dict(range=[70, 100], color="rgba(16,245,160,.07)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
        number=dict(font=dict(family="JetBrains Mono", size=46, color=color)),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=190,
                      margin=dict(t=8, b=8, l=16, r=16))
    return fig


def _chart_monte_carlo(mc: Dict, dark: bool) -> go.Figure:
    """Histogram of Monte Carlo simulation endpoints with percentile lines."""
    tc   = "#f0eeff" if dark else "#0d0020"
    sims = mc.get("simulations", [])
    if not sims:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sims, nbinsx=60,
        marker=dict(color="rgba(139,92,246,0.5)", line=dict(color="#8b5cf6", width=0.5)),
        name="Simulations",
        hovertemplate="Corpus: Rs%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    for pct_key, color, label in [
        ("p10", "#f02d72", "P10 (Bear)"),
        ("p50", "#10f5a0", "P50 (Base)"),
        ("p90", "#f5c518", "P90 (Bull)"),
    ]:
        v = mc.get(pct_key, 0)
        fig.add_vline(x=v, line_color=color, line_width=2, line_dash="dash",
                      annotation_text=f"{label}<br>Rs{v/1e5:.1f}L",
                      annotation_font=dict(family="JetBrains Mono", size=9, color=color))
    invested = mc.get("invested", 0)
    fig.add_vline(x=invested, line_color="#06b6d4", line_width=1.5, line_dash="dot",
                  annotation_text=f"Invested<br>Rs{invested/1e5:.1f}L",
                  annotation_font=dict(family="JetBrains Mono", size=9, color="#06b6d4"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, showlegend=False,
        margin=dict(t=24, b=16, l=8, r=8),
        xaxis=dict(tickprefix="Rs", tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   gridcolor="rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   gridcolor="rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"),
    )
    return fig


def _chart_sip_growth(monthly: float, years: int, dark: bool) -> go.Figure:
    """SIP growth chart across multiple return scenarios."""
    tc     = "#f0eeff" if dark else "#0d0020"
    gc     = "rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"
    y_vals = list(range(1, years + 1))
    scenarios = [
        ("Conservative (8%)", 0.08, "#06b6d4"),
        ("Moderate (12%)",    0.12, "#10f5a0"),
        ("Aggressive (15%)",  0.15, "#8b5cf6"),
    ]
    fig = go.Figure()
    for label, rate, color in scenarios:
        corpus_vals = [_sip_future_value(monthly, rate, y) for y in y_vals]
        fig.add_trace(go.Scatter(
            x=y_vals, y=corpus_vals, mode="lines+markers",
            name=label, line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>{label}</b><br>Year %{{x}}<br>Rs%{{y:,.0f}}<extra></extra>",
        ))
    invested = [monthly * 12 * y for y in y_vals]
    fig.add_trace(go.Scatter(
        x=y_vals, y=invested, mode="lines", name="Total Invested",
        line=dict(color="#f02d72", width=1.5, dash="dot"),
        hovertemplate="Invested: Rs%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(t=16, b=16, l=8, r=8),
        legend=dict(font=dict(family="JetBrains Mono", size=9, color=tc),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Year", tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   gridcolor=gc),
        yaxis=dict(title="Corpus (Rs)", tickprefix="Rs",
                   tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   gridcolor=gc),
    )
    return fig


def _chart_goal_timeline(goals: List[Dict], dark: bool) -> go.Figure:
    """Horizontal bar chart showing goal progress timelines."""
    tc = "#f0eeff" if dark else "#0d0020"
    if not goals:
        return go.Figure()

    names  = [g["name"] for g in goals]
    funded = [min(100, g.get("current", 0) / g["target"] * 100) for g in goals]
    remain = [max(0, 100 - f) for f in funded]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=funded, orientation="h", name="Funded",
        marker=dict(color="#10f5a0", line=dict(width=0)),
        hovertemplate="%{y}: %{x:.1f}% funded<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names, x=remain, orientation="h", name="Remaining",
        marker=dict(color="rgba(139,92,246,.2)", line=dict(color="#8b5cf6", width=1)),
        hovertemplate="%{y}: %{x:.1f}% remaining<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(180, len(goals) * 52),
        margin=dict(t=16, b=16, l=8, r=8),
        legend=dict(font=dict(family="JetBrains Mono", size=9, color=tc),
                    bgcolor="rgba(0,0,0,0)", orientation="h"),
        xaxis=dict(range=[0, 100], ticksuffix="%",
                   tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   gridcolor="rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"),
        yaxis=dict(tickfont=dict(family="JetBrains Mono", size=10, color=tc)),
    )
    return fig


def _chart_tax_compare(tax_old: float, tax_new: float, dark: bool) -> go.Figure:
    tc  = "#f0eeff" if dark else "#0d0020"
    fig = go.Figure(go.Bar(
        x=["Old Regime", "New Regime"],
        y=[tax_old, tax_new],
        marker=dict(
            color=["#f02d72", "#10f5a0"],
            line=dict(color=["#c01155", "#00c87a"], width=2),
        ),
        texttemplate="Rs%{y:,.0f}", textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=tc),
        hovertemplate="%{x}: Rs%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(t=30, b=16, l=8, r=8),
        showlegend=False,
        xaxis=dict(tickfont=dict(family="Syne", size=11, color=tc)),
        yaxis=dict(tickprefix="Rs",
                   tickfont=dict(family="JetBrains Mono", size=9, color=tc),
                   gridcolor="rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════

def _build_report(d: Dict, s: AgentState, score: int) -> str:
    """Generate a comprehensive Markdown report for download."""
    now  = datetime.now().strftime("%d %b %Y, %I:%M %p")
    inc  = d["income"];  exp  = d["expenses"]
    sav  = d["savings"]; tgt  = d.get("target"); tl = d.get("timeline")
    tot  = sum(exp.values()); net = inc - tot
    sr   = net / inc * 100 if inc > 0 else 0

    lines = [
        "# MoneyMentor AI · 2027 — Financial Report",
        f"**Generated:** {now}  \n**Financial Health Score:** {score}/100\n",
        "---",
        "## Financial Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Monthly Income | Rs{inc:,.0f} |",
        f"| Total Expenses | Rs{tot:,.0f} |",
        f"| Net Cash Flow  | Rs{net:,.0f} |",
        f"| Savings Rate   | {sr:.1f}% |",
        f"| Current Corpus | Rs{sav:,.0f} |",
    ]
    if tgt:
        lines.append(f"| Financial Goal | Rs{tgt:,.0f} in {tl} months |")

    lines += ["", "## Expense Breakdown", "| Category | Amount | % of Income |",
              "|----------|--------|-------------|"]
    for k, v in exp.items():
        pct = v / inc * 100 if inc > 0 else 0
        lines.append(f"| {k} | Rs{v:,.0f} | {pct:.1f}% |")

    lines += ["", "---", "## Agent Analysis Reports", ""]
    for title, key in [
        ("Budget Analyst (llama-3.3-70b)", "budget_analysis"),
        ("Risk Assessor (llama-4-scout)", "risk_assessment"),
        ("Investment Advisor (llama-3.3-70b)", "investment_advice"),
        ("Web Researcher (compound-beta)", "web_insights"),
    ]:
        lines += [f"### {title}", s.get(key, "No output."), ""]

    lines += ["---", "## Supervisor CFO — 30/60/90-Day Execution Plan", "",
              s.get("final_report", "No plan generated."), "",
              "---",
              "*Not financial advice. For educational purposes only.*",
              "*MoneyMentor AI · 2027 Edition*"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Session state defaults ─────────────────────────────────────────
    defaults = {
        "dark_mode":      True,
        "analysis_done":  False,
        "agent_state":    None,
        "financial_data": None,
        "chat_history":   [],
        "goals":          [],
        "market_fetched": False,
        "market_data":    "",
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

        lbl = "☀️  Light Mode" if _is_dark() else "🌙  Dark Mode"
        if st.button(lbl, key="theme_btn", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("---")
        st.markdown('<div class="s-lbl">Groq API Key</div>', unsafe_allow_html=True)

        # Safely check for secrets — works even without a secrets.toml file
        try:
            _secret_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            _secret_key = ""

        if _secret_key:
            api_key = _secret_key
            st.success("API key loaded from secrets ✓")
        else:
            api_key = st.text_input(
                "key", type="password", placeholder="gsk_…",
                key="api_in", label_visibility="collapsed",
            )
            if not api_key:
                st.warning("Enter your Groq API key to activate agents.")
                st.stop()

        st.markdown("---")
        st.markdown('<div class="s-lbl" style="margin-bottom:12px;">Agent Roster</div>',
                    unsafe_allow_html=True)

        roster = [
            ("◈ Budget Analyst",     "llama-3.3-70b-versatile",          "50/30/20 CoT math + anomaly flags"),
            ("◈ Risk Assessor",      "llama-4-scout-17b-16e",            "CRITICAL→LOW scoring + goal gap"),
            ("◈ Investment Advisor", "llama-3.3-70b-versatile",          "SEBI instruments + SIP allocation"),
            ("◈ Web Researcher",     "compound-beta (native search)",     "Live RBI · CPI · market data"),
            ("◈ Supervisor CFO",     "gpt-oss-120b → 70B fallback",      "30/60/90-day plan synthesis"),
            ("◈ Chat Advisor",       "llama-4-scout-17b-16e",            "Context-aware multi-turn Q&A"),
        ]
        for name, mdl, desc in roster:
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
            f'NoSessionContext-safe <br>'
            f'DDG fallback for web agent<br>'
            f'Monte Carlo 1200-sim engine<br>'
            f'FY 2026-27 Indian tax engine</div>',
            unsafe_allow_html=True,
        )

    # ── Init Groq client (cached across reruns) ───────────────────────
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
        '5-Agent Parallel &nbsp;·&nbsp; 6-Model Roster &nbsp;·&nbsp; '
        'Monte Carlo Engine &nbsp;·&nbsp; FY 2026-27 Tax Engine &nbsp;·&nbsp; '
        'compound-beta Live Search &nbsp;·&nbsp; GPT-OSS-120B Synthesis'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mm-title">MoneyMentor AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-sub">'
        'Your 2027-era Personal CFO — five specialised AI models execute in true parallel: '
        'Llama 3.3 70B for budget math and Indian portfolio construction, '
        'Llama 4 Scout for instant risk scoring, Groq Compound-Beta for live market '
        'web search, synthesised by GPT-OSS-120B into your personalised 30/60/90-day '
        'execution plan. Monte Carlo simulations, FY 2026-27 Indian tax optimiser, '
        'and multi-goal tracker included.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════
    #  7 TABS
    # ══════════════════════════════════════════════════════════════════
    (t_in, t_an, t_sim, t_tax, t_goal, t_mkt, t_chat) = st.tabs([
        "◈  FINANCIAL INPUT",
        "◈  AGENT ANALYSIS",
        "◈  PORTFOLIO SIMULATOR",
        "◈  TAX OPTIMIZER",
        "◈  GOAL TRACKER",
        "◈  MARKET INSIGHTS",
        "◈  AI ADVISOR CHAT",
    ])

    # ─────────────────────────────────────────────────────────────────
    #  TAB 1 — FINANCIAL INPUT
    # ─────────────────────────────────────────────────────────────────
    with t_in:
        st.markdown('<div class="s-lbl">Your Financial Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Enter Monthly Financials</div>', unsafe_allow_html=True)

        with st.form("fin_form"):
            st.markdown('<div class="s-lbl">Monthly Gross Income</div>', unsafe_allow_html=True)
            income = st.number_input(
                "inc", min_value=0.0, step=500.0, value=75000.0,
                label_visibility="collapsed",
            )
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:26px;'
                f'font-weight:700;color:var(--a2);margin:4px 0 22px;">'
                f'Rs{income:,.0f}'
                f'<span style="font-size:11px;font-weight:400;color:var(--ts);"> /month</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="s-lbl">Monthly Expenses</div>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                rent          = st.number_input("🏠 Rent / Mortgage (Rs)", min_value=0.0, step=500.0,  value=18000.0)
                utilities     = st.number_input("⚡ Utilities (Rs)",        min_value=0.0, step=100.0,  value=3000.0)
                groceries     = st.number_input("🛒 Groceries (Rs)",        min_value=0.0, step=100.0,  value=6000.0)
                transport     = st.number_input("🚗 Transport (Rs)",        min_value=0.0, step=100.0,  value=4000.0)
            with cb:
                dining        = st.number_input("🍜 Dining Out (Rs)",       min_value=0.0, step=100.0,  value=3000.0)
                entertainment = st.number_input("🎮 Entertainment (Rs)",    min_value=0.0, step=100.0,  value=2000.0)
                shopping      = st.number_input("🛍️ Shopping (Rs)",         min_value=0.0, step=100.0,  value=2500.0)
                other         = st.number_input("📦 Other (Rs)",            min_value=0.0, step=100.0,  value=2000.0)

            expenses = {
                "Rent": rent, "Utilities": utilities, "Groceries": groceries,
                "Transport": transport, "Dining": dining,
                "Entertainment": entertainment, "Shopping": shopping, "Other": other,
            }
            total_exp = sum(expenses.values())
            net_cf    = income - total_exp
            color_net = "var(--a2)" if net_cf >= 0 else "var(--a3)"

            if income > 0:
                sr_val = net_cf / income * 100
                needs_val = rent + utilities + groceries + transport
                wants_val = dining + entertainment + shopping
                st.markdown(
                    f'<div style="display:flex;gap:24px;margin:18px 0 22px;flex-wrap:wrap;">'
                    f'<div><div class="s-lbl">Total Expenses</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                    f'font-weight:700;color:var(--a3);">Rs{total_exp:,.0f}</div></div>'
                    f'<div><div class="s-lbl">Net Cash Flow</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                    f'font-weight:700;color:{color_net};">Rs{net_cf:,.0f}</div></div>'
                    f'<div><div class="s-lbl">Savings Rate</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                    f'font-weight:700;color:var(--a1);">{sr_val:.1f}%</div></div>'
                    f'<div><div class="s-lbl">Needs (50% target)</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                    f'font-weight:700;color:var(--a4);">{needs_val/income*100:.0f}%</div></div>'
                    f'<div><div class="s-lbl">Wants (30% target)</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
                    f'font-weight:700;color:var(--a4);">{wants_val/income*100:.0f}%</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="s-lbl">Savings & Primary Goal</div>', unsafe_allow_html=True)
            cs, cg = st.columns(2)
            with cs:
                savings = st.number_input("💰 Current Savings / Corpus (Rs)",
                                          min_value=0.0, step=5000.0, value=80000.0)
            with cg:
                has_goal = st.radio("🎯 Primary Savings Goal?", ("Yes", "No"),
                                    index=1, horizontal=True)

            target = timeline = None
            if has_goal == "Yes":
                ct, cl = st.columns(2)
                with ct:
                    target   = st.number_input("Target Amount (Rs)", min_value=0.0,
                                               step=10000.0, value=500000.0)
                with cl:
                    timeline = st.number_input("Timeline (months)", min_value=1,
                                               step=1, value=24)

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
            st.success("✅ All 5 agents complete — explore the tabs above ↑")

    # ─────────────────────────────────────────────────────────────────
    #  TAB 2 — AGENT ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    with t_an:
        if not st.session_state.analysis_done or not st.session_state.agent_state:
            st.markdown(
                '<div style="text-align:center;padding:80px 20px;">'
                '<div style="font-size:60px;margin-bottom:18px;opacity:.2;">◈</div>'
                '<div class="s-ttl" style="text-align:center;opacity:.4;">No Analysis Yet</div>'
                '<div style="font-family:Outfit,sans-serif;font-size:14px;color:var(--ts);'
                'max-width:320px;margin:0 auto;line-height:1.7;">'
                'Enter your financials in the INPUT tab and launch the agent pipeline.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            s   = st.session_state.agent_state
            d   = st.session_state.financial_data
            inc = d["income"];  exp = d["expenses"]
            sav = d["savings"]; tgt = d.get("target"); tl = d.get("timeline")
            tot = sum(exp.values()); net = inc - tot
            score = _health(inc, exp, sav, tgt, tl)

            # ── Vitals ───────────────────────────────────────────────
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

            # ── Charts ───────────────────────────────────────────────
            cg_col, cd_col, cw_col = st.columns([1, 1.4, 1.9])
            with cg_col:
                st.markdown('<div class="s-lbl">Health Score</div>', unsafe_allow_html=True)
                st.plotly_chart(_chart_gauge(score, _is_dark()),
                                use_container_width=True, config={"displayModeBar": False})
                grade = ("Excellent" if score >= 80 else "Good" if score >= 65
                         else "Fair" if score >= 45 else "At Risk")
                st.markdown(f'<div class="h-grade">{grade}</div>', unsafe_allow_html=True)
            with cd_col:
                st.markdown('<div class="s-lbl">Expense Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(_chart_donut(exp, _is_dark()),
                                use_container_width=True, config={"displayModeBar": False})
            with cw_col:
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

            # ── Rule diagnostics ──────────────────────────────────────
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

            # ── Agent output panels ───────────────────────────────────
            st.markdown('<div class="s-lbl">Parallel Agent Intelligence</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="s-ttl">Multi-Model Agent Reports</div>',
                        unsafe_allow_html=True)

            panels = [
                ("Budget Analyst",     MODEL_DISPLAY["budget"],     "50/30/20 Budget Diagnostics",   "budget_analysis"),
                ("Risk Assessor",      MODEL_DISPLAY["risk"],       "Risk & Vulnerability Scoring",  "risk_assessment"),
                ("Investment Advisor", MODEL_DISPLAY["investment"], "India Portfolio Construction",   "investment_advice"),
                ("Web Researcher",     MODEL_DISPLAY["research"] + " · live search", "Live Macro Market Data", "web_insights"),
            ]
            pl, pr = st.columns(2)
            for i, (name, mdl, role, key) in enumerate(panels):
                raw_out = s.get(key, "")
                out = raw_out.strip() if raw_out else ""
                if not out:
                    out = "⚠️ Agent did not return output. Re-run the analysis."
                # Render markdown-style numbered lists nicely
                out_html = out.replace("\n", "<br>")
                with (pl if i % 2 == 0 else pr):
                    with st.expander(f"◈  {name}", expanded=(i < 2)):
                        st.markdown(
                            f'<div class="ag-model">{role}</div>'
                            f'<span class="mpill">{mdl}</span>'
                            f'<div class="ag-out">{out_html}</div>',
                            unsafe_allow_html=True,
                        )

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

            # ── Supervisor plan ───────────────────────────────────────
            st.markdown(
                f'<div class="s-lbl">Supervisor CFO · {MODEL_DISPLAY["supervisor"]} → {_SUPERVISOR_FALLBACK} fallback</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="s-ttl">◈  30 / 60 / 90-Day Execution Plan</div>',
                        unsafe_allow_html=True)
            final = (s.get("final_report") or "").strip()
            if final:
                # Pretty-print: bold section headers
                final_html = (
                    final
                    .replace("30-DAY PLAN:", "<strong>📅 30-DAY PLAN</strong>")
                    .replace("60-DAY PLAN:", "<strong>📅 60-DAY PLAN</strong>")
                    .replace("90-DAY PLAN:", "<strong>📅 90-DAY PLAN</strong>")
                    .replace("BOLD MOVE:", "<strong>⚡ BOLD MOVE</strong>")
                    .replace("CFO CONFIDENCE:", "<strong>🎯 CFO CONFIDENCE</strong>")
                    .replace("\n", "<br>")
                )
                st.markdown(
                    '<div class="ag-card">'
                    '<div class="ag-name">Supervisor CFO — Master Financial Plan</div>'
                    f'<span class="mpill">{MODEL_DISPLAY["supervisor"]}</span>'
                    f'<div class="ag-out">{final_html}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("⚠️ Supervisor CFO did not return a plan. Re-run the analysis.")

            st.markdown("<br>", unsafe_allow_html=True)
            dl_col, rb_col = st.columns([3, 1])
            with dl_col:
                report_md = _build_report(d, s, score)
                st.download_button(
                    label="◈  DOWNLOAD FULL REPORT (.md)",
                    data=report_md.encode("utf-8"),
                    file_name=f"moneymentor_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with rb_col:
                if st.button("◈  Re-run Analysis", key="rerun"):
                    st.session_state.analysis_done = False
                    st.session_state.agent_state   = None
                    st.rerun()

    # ─────────────────────────────────────────────────────────────────
    #  TAB 3 — PORTFOLIO SIMULATOR
    # ─────────────────────────────────────────────────────────────────
    with t_sim:
        st.markdown('<div class="s-lbl">Wealth Simulator</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Monte Carlo Portfolio Simulator</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;font-size:13.5px;color:var(--ts);">'
            '1,200-path Monte Carlo simulation using log-normal monthly returns. '
            'Visualise P10 (bear), P50 (base), and P90 (bull) outcomes.</div><br>',
            unsafe_allow_html=True,
        )

        fd = st.session_state.financial_data
        default_sip = max(2000.0, float(fd["income"] - sum(fd["expenses"].values())) * 0.6) if fd else 5000.0

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sim_sip = st.number_input("Monthly SIP (Rs)", min_value=500.0,
                                      step=500.0, value=float(default_sip), key="sim_sip")
        with sc2:
            sim_years = st.slider("Investment Horizon (years)", 1, 30, 10, key="sim_years")
        with sc3:
            sim_return = st.slider("Expected Annual Return (%)", 6, 20, 12, key="sim_return") / 100
            sim_vol    = st.slider("Volatility (%)", 5, 35, 18, key="sim_vol") / 100

        if st.button("◈  RUN 1200-PATH MONTE CARLO", key="run_mc", use_container_width=True):
            with st.spinner("Simulating 1,200 portfolio paths…"):
                mc = _monte_carlo(sim_sip, sim_years, sim_return, sim_vol, n_sims=1200)
            st.session_state["mc_result"] = mc

        mc = st.session_state.get("mc_result")
        if mc and mc.get("simulations"):
            st.markdown("---")
            invested = mc["invested"]
            p10, p50, p90 = mc["p10"], mc["p50"], mc["p90"]

            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Total Invested",  f"Rs{invested/1e5:.2f}L")
            mm2.metric("P50 — Base Case", f"Rs{p50/1e5:.2f}L",
                       delta=f"+{(p50-invested)/invested*100:.0f}% gain" if invested else None)
            mm3.metric("P10 — Bear Case", f"Rs{p10/1e5:.2f}L")
            mm4.metric("P90 — Bull Case", f"Rs{p90/1e5:.2f}L")

            st.markdown('<div class="s-lbl" style="margin-top:18px;">Monte Carlo Distribution</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(_chart_monte_carlo(mc, _is_dark()),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.markdown('<div class="s-lbl">SIP Growth Projections</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Return Scenario Comparison</div>', unsafe_allow_html=True)

        sp1, sp2 = st.columns(2)
        with sp1:
            proj_sip   = st.number_input("SIP Amount (Rs)", min_value=500.0, step=500.0,
                                         value=float(default_sip), key="proj_sip")
        with sp2:
            proj_years = st.slider("Years", 1, 30, 10, key="proj_years")

        st.plotly_chart(_chart_sip_growth(proj_sip, proj_years, _is_dark()),
                        use_container_width=True, config={"displayModeBar": False})

        # Projection table
        st.markdown('<div class="s-lbl" style="margin-top:8px;">Corpus Milestones</div>',
                    unsafe_allow_html=True)
        milestones = []
        for y in [1, 3, 5, 10, 15, 20, proj_years]:
            if y > proj_years:
                continue
            milestones.append({
                "Year": y,
                "Invested (Rs)": f"{proj_sip*12*y:,.0f}",
                "Conservative 8% (Rs)": f"{_sip_future_value(proj_sip, 0.08, y):,.0f}",
                "Moderate 12% (Rs)":    f"{_sip_future_value(proj_sip, 0.12, y):,.0f}",
                "Aggressive 15% (Rs)":  f"{_sip_future_value(proj_sip, 0.15, y):,.0f}",
            })
        if milestones:
            st.table(milestones)

    # ─────────────────────────────────────────────────────────────────
    #  TAB 4 — TAX OPTIMIZER
    # ─────────────────────────────────────────────────────────────────
    with t_tax:
        st.markdown('<div class="s-lbl">FY 2026-27 Indian Tax Engine</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Tax Optimizer & Regime Comparison</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;font-size:13.5px;color:var(--ts);">'
            'Compare Old vs New tax regime. Optimise deductions under 80C, 80CCD(1B), 80D, HRA '
            'as per FY 2026-27 Indian income tax rules. New standard deduction: Rs75,000.</div><br>',
            unsafe_allow_html=True,
        )

        fd = st.session_state.financial_data
        default_income = float(fd["income"] * 12) if fd else 900000.0

        with st.form("tax_form"):
            tc1, tc2 = st.columns(2)
            with tc1:
                annual_income = st.number_input(
                    "Annual Gross Income (Rs)", min_value=0.0, step=10000.0,
                    value=default_income, key="tax_income",
                )
                city_type = st.selectbox("City Type", ["Metro (50% HRA)", "Non-Metro (40% HRA)"])
                hra_received = st.number_input("HRA Received / year (Rs)", min_value=0.0,
                                               step=1000.0, value=0.0)

            with tc2:
                d80c     = st.number_input("80C Investments (ELSS/PPF/LIC) — max Rs1.5L",
                                           min_value=0.0, step=1000.0,
                                           max_value=float(MAX_80C), value=min(default_income*0.08, float(MAX_80C)))
                d80ccd1b = st.number_input("80CCD(1B) — NPS Extra — max Rs50,000",
                                           min_value=0.0, step=1000.0,
                                           max_value=float(MAX_80CCD1B), value=0.0)
                d80d     = st.number_input("80D — Health Insurance Self (Rs) — max Rs25,000",
                                           min_value=0.0, step=1000.0,
                                           max_value=float(MAX_80D_SELF), value=0.0)
                d80d_par = st.number_input("80D — Senior Parent Health Insurance (Rs) — max Rs50,000",
                                           min_value=0.0, step=1000.0,
                                           max_value=float(MAX_80D_PARENTS), value=0.0)

            tax_submit = st.form_submit_button("◈  COMPUTE OPTIMAL TAX STRATEGY", use_container_width=True)

        if tax_submit or True:
            ded = {
                "80c": d80c, "80ccd1b": d80ccd1b,
                "80d_self": d80d, "80d_parents": d80d_par, "hra": hra_received,
            }
            tr = _tax_calc(annual_income, deductions=ded)

            st.markdown("---")
            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Old Regime Tax (Annual)", f"Rs{tr['tax_old']:,.0f}",
                       delta=f"Rs{tr['tax_old']/12:,.0f}/mo")
            tm2.metric("New Regime Tax (Annual)", f"Rs{tr['tax_new']:,.0f}",
                       delta=f"Rs{tr['tax_new']/12:,.0f}/mo")
            better_label = "✅ New Regime" if tr["better"] == "new" else "✅ Old Regime"
            tm3.metric("Recommended", better_label)
            tm4.metric("Annual Tax Savings", f"Rs{tr['savings']:,.0f}",
                       delta=f"Rs{tr['savings']/12:,.0f}/mo saved")

            st.markdown("---")
            cl_chart, cr_detail = st.columns([1.2, 1])
            with cl_chart:
                st.markdown('<div class="s-lbl">Regime Comparison</div>', unsafe_allow_html=True)
                st.plotly_chart(_chart_tax_compare(tr["tax_old"], tr["tax_new"], _is_dark()),
                                use_container_width=True, config={"displayModeBar": False})
            with cr_detail:
                st.markdown('<div class="s-lbl">Deduction Breakdown</div>', unsafe_allow_html=True)
                deduction_items = [
                    ("Standard Deduction (New Regime)", STD_DEDUCTION_NEW),
                    ("Standard Deduction (Old Regime)", STD_DEDUCTION_OLD),
                    ("80C (ELSS/PPF/LIC)",              min(d80c, MAX_80C)),
                    ("80CCD(1B) NPS Extra",             min(d80ccd1b, MAX_80CCD1B)),
                    ("80D Self Health",                 min(d80d, MAX_80D_SELF)),
                    ("80D Parents Health",              min(d80d_par, MAX_80D_PARENTS)),
                    ("HRA Exemption",                   hra_received),
                ]
                total_ded_old = sum(v for _, v in deduction_items[1:])  # old regime deductions
                for label, val in deduction_items:
                    if val > 0:
                        st.markdown(
                            f'<div class="tax-card">'
                            f'<div style="font-family:\'JetBrains Mono\',monospace;'
                            f'font-size:10px;color:var(--a2);text-transform:uppercase;'
                            f'letter-spacing:.08em;margin-bottom:3px;">{label}</div>'
                            f'<div style="font-family:Syne,sans-serif;font-weight:700;'
                            f'font-size:16px;color:var(--tp);">Rs{val:,.0f}</div></div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("---")
            st.markdown('<div class="s-lbl">Actionable Tax Tips</div>', unsafe_allow_html=True)
            tips_data = []

            # 80C gap
            remaining_80c = MAX_80C - min(d80c, MAX_80C)
            if remaining_80c > 0:
                tips_data.append({
                    "cls": "tip-warn", "ico": "📋",
                    "txt": f"You have Rs{remaining_80c:,.0f} unused 80C limit. "
                           f"Invest in ELSS (tax-saving MF), PPF, or NSC to claim full "
                           f"Rs1,50,000 deduction."
                })
            else:
                tips_data.append({
                    "cls": "tip-ok", "ico": "✅",
                    "txt": "80C limit fully utilised. Consider ELSS for equity exposure with tax benefit."
                })

            # NPS tip
            if d80ccd1b < MAX_80CCD1B:
                tips_data.append({
                    "cls": "tip-info", "ico": "🏦",
                    "txt": f"NPS Tier-1 contribution u/s 80CCD(1B) offers "
                           f"Rs{MAX_80CCD1B - min(d80ccd1b, MAX_80CCD1B):,.0f} additional "
                           f"deduction (over 80C). At 30% slab this saves Rs{int((MAX_80CCD1B - min(d80ccd1b, MAX_80CCD1B))*0.30):,} in taxes."
                })

            # Health insurance tip
            if d80d < MAX_80D_SELF:
                tips_data.append({
                    "cls": "tip-info", "ico": "❤️",
                    "txt": f"80D unused: Rs{MAX_80D_SELF - min(d80d, MAX_80D_SELF):,.0f} "
                           "deduction available for health insurance premiums. "
                           "A Rs25,000/yr policy covers self, spouse, and children."
                })

            _savings_str = f"Rs{tr['savings']:,.0f}"
            _regime_txt  = (f"New regime saves {_savings_str} vs old regime. Switch to new regime."
                            if tr["better"] == "new"
                            else f"Old regime saves {_savings_str} vs new regime. Stay on old regime and maximise deductions.")
            tips_data.append({
                "cls": "tip-ok" if tr["better"] == "new" else "tip-warn",
                "ico": "🎯",
                "txt": _regime_txt,
            })

            for tip in tips_data:
                st.markdown(
                    f'<div class="tip {tip["cls"]}">'
                    f'<div class="tip-ico">{tip["ico"]}</div>'
                    f'<div class="tip-txt">{tip["txt"]}</div></div>',
                    unsafe_allow_html=True,
                )

            # Tax slabs display
            st.markdown("---")
            st.markdown('<div class="s-lbl">FY 2026-27 Tax Slabs</div>', unsafe_allow_html=True)
            sl_a, sl_b = st.columns(2)
            with sl_a:
                st.markdown('<div class="s-ttl" style="font-size:1rem;">New Regime</div>',
                            unsafe_allow_html=True)
                slab_rows = []
                for lo, hi, rate in TAX_SLABS_NEW:
                    hi_str = "Above" if hi == float("inf") else f"Rs{hi:,.0f}"
                    slab_rows.append({
                        "Income Slab": f"Rs{lo:,.0f} – {hi_str}",
                        "Tax Rate": f"{rate*100:.0f}%",
                    })
                st.table(slab_rows)
            with sl_b:
                st.markdown('<div class="s-ttl" style="font-size:1rem;">Old Regime</div>',
                            unsafe_allow_html=True)
                slab_rows_old = []
                for lo, hi, rate in TAX_SLABS_OLD:
                    hi_str = "Above" if hi == float("inf") else f"Rs{hi:,.0f}"
                    slab_rows_old.append({
                        "Income Slab": f"Rs{lo:,.0f} – {hi_str}",
                        "Tax Rate": f"{rate*100:.0f}%",
                    })
                st.table(slab_rows_old)

    # ─────────────────────────────────────────────────────────────────
    #  TAB 5 — GOAL TRACKER
    # ─────────────────────────────────────────────────────────────────
    with t_goal:
        st.markdown('<div class="s-lbl">Financial Goals Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Multi-Goal Tracker</div>', unsafe_allow_html=True)

        fd = st.session_state.financial_data
        net_flow = 0.0
        if fd:
            net_flow = fd["income"] - sum(fd["expenses"].values())

        # Add goal form
        with st.expander("◈  Add New Financial Goal", expanded=not bool(st.session_state.goals)):
            with st.form("goal_form"):
                gf1, gf2, gf3, gf4 = st.columns(4)
                with gf1:
                    g_name = st.text_input("Goal Name", placeholder="Dream Vacation",
                                           key="g_name")
                with gf2:
                    g_target = st.number_input("Target Amount (Rs)", min_value=1000.0,
                                               step=5000.0, value=200000.0, key="g_target")
                with gf3:
                    g_current = st.number_input("Current Savings for Goal (Rs)",
                                                min_value=0.0, step=1000.0,
                                                value=0.0, key="g_current")
                with gf4:
                    g_months = st.number_input("Timeline (months)", min_value=1,
                                               step=1, value=12, key="g_months")
                g_icon = st.selectbox("Goal Icon", ["🏖️", "🏠", "🚗", "📚", "💍", "👶",
                                                     "✈️", "💻", "🏋️", "🎓"], key="g_icon")
                if st.form_submit_button("◈  ADD GOAL", use_container_width=True):
                    if g_name.strip():
                        st.session_state.goals.append({
                            "name":    g_name.strip(),
                            "target":  g_target,
                            "current": g_current,
                            "months":  g_months,
                            "icon":    g_icon,
                        })
                        st.rerun()

        if not st.session_state.goals:
            st.info("💡 Add your first financial goal above to start tracking progress.")
        else:
            # Goal progress chart
            st.markdown('<div class="s-lbl" style="margin-top:12px;">Progress Overview</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                _chart_goal_timeline(st.session_state.goals, _is_dark()),
                use_container_width=True, config={"displayModeBar": False},
            )

            st.markdown("---")
            st.markdown('<div class="s-lbl">Goal Cards</div>', unsafe_allow_html=True)

            cols = st.columns(min(3, len(st.session_state.goals)))
            for i, g in enumerate(st.session_state.goals):
                with cols[i % len(cols)]:
                    needed  = max(0, g["target"] - g["current"])
                    monthly = needed / g["months"] if g["months"] > 0 else needed
                    pct     = min(100, g["current"] / g["target"] * 100) if g["target"] > 0 else 0
                    feasible = monthly <= net_flow if net_flow > 0 else None
                    status_ico = "✅" if feasible else "⚠️" if feasible is not None else "❔"
                    years_val = g["months"] / 12

                    # SIP projection for goal
                    projected = _sip_future_value(monthly, 0.10, years_val)

                    st.markdown(
                        f'<div class="goal-card">'
                        f'<div style="font-size:28px;margin-bottom:8px;">{g["icon"]}</div>'
                        f'<div style="font-family:Syne,sans-serif;font-weight:700;'
                        f'font-size:15px;color:var(--tp);margin-bottom:6px;">{g["name"]}</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;'
                        f'font-weight:700;color:var(--a2);">Rs{g["target"]:,.0f}</div>'
                        f'<div style="font-family:Outfit,sans-serif;font-size:12px;'
                        f'color:var(--ts);margin-top:4px;margin-bottom:10px;">'
                        f'Target · {g["months"]} months</div>'
                        f'<div style="background:var(--bgc2);border-radius:99px;height:6px;margin-bottom:8px;">'
                        f'<div style="background:linear-gradient(90deg,var(--a2),var(--a1));'
                        f'border-radius:99px;height:100%;width:{pct:.0f}%;'
                        f'transition:width .6s ease;"></div></div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                        f'color:var(--ts);margin-bottom:10px;">{pct:.0f}% funded · '
                        f'Rs{needed:,.0f} remaining</div>'
                        f'<div class="s-lbl">Monthly Required</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:15px;'
                        f'font-weight:700;color:var(--a1);">Rs{monthly:,.0f}/mo</div>'
                        f'<div style="font-family:Outfit,sans-serif;font-size:11px;'
                        f'color:var(--ts);margin-top:4px;">'
                        f'{status_ico} {"Feasible" if feasible else "Needs income boost" if feasible is not None else "Enter financials"}'
                        f' · SIP @10% → Rs{projected:,.0f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            if st.button("◈  Clear All Goals", key="clear_goals"):
                st.session_state.goals = []
                st.rerun()

            # Combined goals summary
            total_target  = sum(g["target"] for g in st.session_state.goals)
            total_current = sum(g.get("current", 0) for g in st.session_state.goals)
            total_monthly = sum(
                max(0, g["target"] - g.get("current", 0)) / g["months"]
                for g in st.session_state.goals if g["months"] > 0
            )
            gm1, gm2, gm3 = st.columns(3)
            gm1.metric("Total Goals Value",    f"Rs{total_target:,.0f}")
            gm2.metric("Total Already Saved",  f"Rs{total_current:,.0f}",
                       delta=f"{total_current/total_target*100:.0f}% funded" if total_target else None)
            gm3.metric("Total Monthly Needed", f"Rs{total_monthly:,.0f}",
                       delta=f"Rs{net_flow - total_monthly:,.0f} surplus" if net_flow > 0 else None,
                       delta_color="normal" if net_flow >= total_monthly else "inverse")

    # ─────────────────────────────────────────────────────────────────
    #  TAB 6 — MARKET INSIGHTS
    # ─────────────────────────────────────────────────────────────────
    with t_mkt:
        st.markdown('<div class="s-lbl">Live Market Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">India Market Insights</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Outfit,sans-serif;font-size:13.5px;color:var(--ts);">'
            'Powered by Groq Compound-Beta with native web search. '
            'Fetches live RBI rates, SIP performance, CPI, and liquid fund yields.</div><br>',
            unsafe_allow_html=True,
        )

        mc1, mc2 = st.columns([3, 1])
        with mc2:
            topic = st.selectbox(
                "Research Topic",
                [
                    "Live RBI, SIP, CPI, Liquid Fund Data",
                    "Best ELSS funds to invest in 2025",
                    "Nifty 50 outlook and sector rotation",
                    "Small cap vs large cap allocation 2025",
                    "PPF vs NPS — which is better?",
                    "Best credit card for cashback in India",
                    "Gold vs equity in 2025 India",
                    "Top international fund options for Indians",
                ],
                key="mkt_topic",
            )
        with mc1:
            if st.button("◈  FETCH LIVE MARKET DATA", key="fetch_mkt", use_container_width=True):
                fd = st.session_state.financial_data or {}
                income = fd.get("income", 0)
                expenses = fd.get("expenses", {})
                total = sum(expenses.values())
                sr = (income - total) / income * 100 if income > 0 else 0

                dummy_state: AgentState = {
                    "income": income, "expenses": expenses,
                    "savings": fd.get("savings", 0),
                    "target": fd.get("target"), "timeline": fd.get("timeline"),
                    "budget_analysis": "", "risk_assessment": "",
                    "investment_advice": "", "web_insights": "",
                    "web_raw": [], "final_report": "", "messages": [],
                }

                # Build compact prompts — compound-beta 413s on large payloads
                if topic == "Live RBI, SIP, CPI, Liquid Fund Data":
                    system = "You are a macro-finance analyst. Search the web and report current India finance figures."
                    user = (
                        "Report the following 5 current India finance data points:\n"
                        "1. RBI repo rate and top FD rates\n"
                        "2. Best SIP category last 3 months\n"
                        "3. India CPI inflation\n"
                        "4. Best liquid fund yield\n"
                        "5. Nifty 50 level and outlook"
                    )
                else:
                    system = "You are an expert in Indian personal finance. Search the web for current data."
                    user = f"Give current data and actionable advice for Indian investors on: {topic}"

                with st.spinner("Searching live market data with compound-beta…"):
                    try:
                        # Keep messages short — compound-beta 413s on large payloads
                        r = groq_client.chat.completions.create(
                            model=MODELS["research"],
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user",   "content": user},
                            ],
                            max_tokens=700, temperature=0.3,
                        )
                        msg = r.choices[0].message
                        mkt_data = _strip_think(msg.content or "")
                        if not mkt_data.strip():
                            mkt_data = _extract_tool_text(msg)
                        if not mkt_data.strip():
                            raise ValueError("Empty compound-beta response")
                    except Exception:
                        # DDG fallback → always inject KB so model has real data
                        raw_ddg = _ddg(topic, n=5)
                        live_ctx = "\n".join(
                            f"{h['title']}: {h['snippet']}"
                            for h in raw_ddg if h.get("snippet")
                        )[:600]
                        # Always include the curated KB as guaranteed content
                        combined = (live_ctx + "\n\n" + _INDIA_FINANCE_KB).strip()[:1100]
                        mkt_data = _call(
                            groq_client,
                            (
                                "You are an Indian personal finance expert. "
                                "Use the data provided to give a clear, "
                                "numbered, actionable answer. Use Rs signs."
                            ),
                            f"Topic: {topic}\n\nIndia Finance Data:\n{combined}",
                            model="llama-3.3-70b-versatile", max_tokens=550,
                        )

                st.session_state.market_data    = mkt_data
                st.session_state.market_fetched = True

        if st.session_state.market_fetched and st.session_state.market_data:
            st.markdown("---")
            mkt = st.session_state.market_data
            # Render as styled card
            st.markdown(
                f'<div class="market-card">'
                f'<div style="font-family:Syne,sans-serif;font-weight:700;font-size:14px;'
                f'color:var(--a2);margin-bottom:10px;">◈ Live Research Output</div>'
                f'<div style="font-family:Outfit,sans-serif;font-size:14px;'
                f'color:var(--tp);line-height:1.78;">'
                f'{mkt.replace(chr(10), "<br>")}</div></div>',
                unsafe_allow_html=True,
            )

            # Static context cards
            st.markdown("---")
            st.markdown('<div class="s-lbl">Reference Data — Indian Finance</div>',
                        unsafe_allow_html=True)
            ref_cards = [
                ("🏦", "RBI Repo Rate", "Key monetary policy rate. As of 2025 it was ~6.5%. Check RBI website for live rate.", "tip-info"),
                ("📈", "SEBI Registered MFs", "Over 1,400 mutual fund schemes available. Only invest in SEBI-registered AMCs via BSE/NSE or direct plans.", "tip-ok"),
                ("💰", "PPF (2025-26)", "Currently 7.1% p.a., tax-free, EEE category. 15-year lock-in. Max Rs1.5L/yr.", "tip-ok"),
                ("🏥", "NPS Returns", "NPS Tier-1 equity returns: ~12-14% CAGR historically. Tax benefit: 80CCD(1)+80CCD(1B).", "tip-info"),
                ("⚡", "Liquid Funds", "Typical 7-day yield: 6.5–7.2%. Better than savings account, fully liquid in 24h.", "tip-ok"),
                ("🎯", "ELSS Lock-in", "3-year lock-in, ~12-15% CAGR historically. Best tax-saving investment for equity exposure.", "tip-ok"),
            ]
            cols = st.columns(3)
            for i, (ico, title, desc, cls) in enumerate(ref_cards):
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="tip {cls}" style="flex-direction:column;">'
                        f'<div style="font-size:22px;margin-bottom:6px;">{ico}</div>'
                        f'<div style="font-family:Syne,sans-serif;font-weight:700;'
                        f'font-size:13px;color:var(--tp);margin-bottom:4px;">{title}</div>'
                        f'<div class="tip-txt">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div style="text-align:center;padding:60px 20px;">'
                '<div style="font-size:52px;margin-bottom:18px;opacity:.2;">📡</div>'
                '<div class="s-ttl" style="text-align:center;opacity:.4;">No Market Data Fetched</div>'
                '<div style="font-family:Outfit,sans-serif;font-size:14px;color:var(--ts);">'
                'Click the fetch button above to get live India market data.</div></div>',
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────
    #  TAB 7 — AI ADVISOR CHAT
    # ─────────────────────────────────────────────────────────────────
    with t_chat:
        st.markdown('<div class="s-lbl">Conversational Finance AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="s-ttl">Ask MoneyMentor Anything</div>', unsafe_allow_html=True)

        ctx   = st.session_state.financial_data or {"income": 0, "expenses": {}, "savings": 0}
        exp_t = sum(ctx.get("expenses", {}).values())

        if st.session_state.financial_data:
            income  = ctx["income"]
            savings = ctx.get("savings", 0)
            net     = income - exp_t
            st.markdown(
                f'<div style="background:var(--bgc);border:1px solid var(--bdr);'
                f'border-radius:12px;padding:12px 18px;margin-bottom:18px;'
                f'display:flex;gap:30px;flex-wrap:wrap;align-items:center;">'
                f'<div><div class="s-lbl">Income</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:var(--a2);">Rs{income:,.0f}</div></div>'
                f'<div><div class="s-lbl">Expenses</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:var(--a3);">Rs{exp_t:,.0f}</div></div>'
                f'<div><div class="s-lbl">Net Flow</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:{"var(--a2)" if net >= 0 else "var(--a3)"};">Rs{net:,.0f}</div></div>'
                f'<div><div class="s-lbl">Savings Corpus</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:16px;'
                f'font-weight:700;color:var(--a1);">Rs{savings:,.0f}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Add your financials in the INPUT tab for personalised, context-aware answers.")

        # Build agent context for chat
        ag_ctx = ""
        if st.session_state.agent_state:
            _s = st.session_state.agent_state
            ag_ctx = (
                f"Budget: {_s.get('budget_analysis','')[:200]} | "
                f"Risk: {_s.get('risk_assessment','')[:200]} | "
                f"Investment: {_s.get('investment_advice','')[:200]} | "
                f"CFO Plan: {_s.get('final_report','')[:200]}"
            )

        # Quick question buttons
        st.markdown('<div class="s-lbl" style="margin-bottom:10px;">Quick Questions</div>',
                    unsafe_allow_html=True)
        qs = [
            "Cut my biggest expense?",
            "Best SIP for Rs5k/month?",
            "Build emergency fund fast?",
            "Old vs new tax regime?",
            "How to save Rs1L in 6 months?",
            "Invest or clear debt first?",
        ]
        qcols = st.columns(len(qs))
        for col, q, i in zip(qcols, qs, range(len(qs))):
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
                    f'<div class="cb-ai"><span class="cb-lbl">◈ MoneyMentor · llama-4-scout</span>'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Input area
        user_in = st.text_input(
            "ci", placeholder="e.g. How to build a Rs10L corpus on Rs60k salary in 3 years?",
            key="chat_in", label_visibility="collapsed",
        )
        cs_col, cc_col = st.columns([5, 1])
        with cs_col:
            if st.button("◈  SEND MESSAGE", key="send", use_container_width=True) \
                    and user_in.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_in})
                with st.spinner(""):
                    reply = _chat(groq_client, user_in, ctx,
                                  st.session_state.chat_history[:-1], ag_ctx)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
        with cc_col:
            if st.button("Clear", key="clr", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="mm-footer">'
        'MONEYMENTOR AI &nbsp;◈&nbsp; 2027 EDITION<br>'
        'LLAMA-3.3-70B-VERSATILE &nbsp;·&nbsp; LLAMA-4-SCOUT-17B &nbsp;·&nbsp; '
        'COMPOUND-BETA &nbsp;·&nbsp; OPENAI/GPT-OSS-120B &nbsp;·&nbsp; LLAMA-4-SCOUT CHAT<br>'
        'TRUE PARALLEL THREADPOOL FAN-OUT &nbsp;·&nbsp; COMPOUND-BETA NATIVE SEARCH &nbsp;·&nbsp; '
        'NOSESSIONCONTEXT-SAFE  &nbsp;·&nbsp; MONTE CARLO 1200-SIM ENGINE<br>'
        'FY 2026-27 INDIAN TAX ENGINE &nbsp;·&nbsp; SEBI/RBI/ELSS/NPS/PPF AWARE<br>'
        'Not financial advice. For educational and portfolio demonstration purposes only.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
