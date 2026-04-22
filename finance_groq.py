"""
MoneyMentor AI 
Multi-Agent LangGraph Financial Intelligence Platform

"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import random
from datetime import datetime
from groq import Groq
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

# ─── LangGraph imports ────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoneyMentor AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "MoneyMentor AI — Next-Gen Agentic Finance Intelligence"
    }
)

# Each agent gets the best model suited to its task
MODELS = {
    "budget_analyst":     "deepseek-r1-distill-llama-70b",   # Reasoning-heavy budget math
    "risk_assessor":      "deepseek-r1-distill-llama-70b",   # Analytical risk scoring
    "investment_advisor": "llama-3.3-70b-versatile",          # Broad India finance knowledge
    "web_researcher":     "llama-3.3-70b-versatile",          # Macro context synthesis
    "financial_planner":  "llama-3.3-70b-versatile",          # Action plan synthesis
    "chat":               "llama-3.3-70b-versatile",          # Conversational fluency
}

# ─── Theme & CSS ──────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg_primary    = "#050508"
        bg_secondary  = "#0d0d18"
        bg_card       = "#10101f"
        bg_card2      = "#13132a"
        border        = "#1e1e40"
        text_primary  = "#e8e8ff"
        text_secondary= "#8888cc"
        accent1       = "#7b2fff"
        accent2       = "#00f5c4"
        accent3       = "#ff3cac"
        accent4       = "#ffbe0b"
        accent5       = "#00b4d8"
        glow1         = "rgba(123,47,255,0.35)"
        glow2         = "rgba(0,245,196,0.25)"
        glow3         = "rgba(255,60,172,0.25)"
        metric_bg     = "#12122a"
        tag_bg        = "rgba(123,47,255,0.15)"
        sidebar_bg    = "#080812"
        # Streamlit native theme override values
        st_base       = "dark"
        st_bg         = "#050508"
        st_secondary  = "#0d0d18"
        st_text       = "#e8e8ff"
        input_bg      = "#10101f"
    else:
        bg_primary    = "#f0effe"
        bg_secondary  = "#fafaff"
        bg_card       = "#ffffff"
        bg_card2      = "#f5f3ff"
        border        = "#ddd8ff"
        text_primary  = "#1a0a3d"
        text_secondary= "#5a4a8a"
        accent1       = "#6a0ef5"
        accent2       = "#00c9a0"
        accent3       = "#e0006e"
        accent4       = "#f59e0b"
        accent5       = "#0077aa"
        glow1         = "rgba(106,14,245,0.12)"
        glow2         = "rgba(0,201,160,0.12)"
        glow3         = "rgba(224,0,110,0.12)"
        metric_bg     = "#f3f0ff"
        tag_bg        = "rgba(106,14,245,0.08)"
        sidebar_bg    = "#ece9ff"
        st_base       = "light"
        st_bg         = "#f0effe"
        st_secondary  = "#fafaff"
        st_text       = "#1a0a3d"
        input_bg      = "#ffffff"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {{
        --bg-primary:     {bg_primary};
        --bg-secondary:   {bg_secondary};
        --bg-card:        {bg_card};
        --bg-card2:       {bg_card2};
        --border:         {border};
        --text-primary:   {text_primary};
        --text-secondary: {text_secondary};
        --accent1:        {accent1};
        --accent2:        {accent2};
        --accent3:        {accent3};
        --accent4:        {accent4};
        --accent5:        {accent5};
        --glow1:          {glow1};
        --glow2:          {glow2};
        --glow3:          {glow3};
        --metric-bg:      {metric_bg};
        --tag-bg:         {tag_bg};
        --sidebar-bg:     {sidebar_bg};
    }}

    /* ── Force override ALL Streamlit theme backgrounds ── */
    html,
    body,
    .stApp,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main,
    .main > div,
    .block-container,
    [data-testid="block-container"],
    section[data-testid="stSidebarContent"],
    .css-1d391kg, .css-18e3th9, .css-fg4pbf,
    [class*="css-"] {{ /* catch-all for generated class names */
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
    }}

    /* Main content area specifically */
    [data-testid="stAppViewContainer"] > .main {{
        background-color: {bg_primary} !important;
    }}
    [data-testid="stAppViewContainer"] > .main > div {{
        background-color: {bg_primary} !important;
    }}
    .block-container {{
        background-color: {bg_primary} !important;
        padding-top: 2rem !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border} !important;
    }}

    /* Header/toolbar */
    [data-testid="stHeader"],
    header[data-testid="stHeader"] {{
        background-color: {bg_primary} !important;
        border-bottom: 1px solid {border} !important;
    }}
    [data-testid="stToolbar"] {{
        background-color: {bg_primary} !important;
    }}
    [data-testid="stDecoration"] {{
        background-image: none !important;
        background-color: {bg_primary} !important;
    }}

    /* Bottom bar */
    [data-testid="stStatusWidget"],
    footer,
    footer > div {{
        background-color: {bg_primary} !important;
        color: {text_secondary} !important;
    }}
    footer {{ visibility: hidden; }}

    /* ── Typography ── */
    html, body, [data-testid="stApp"],
    p, div, span, label {{
        font-family: 'DM Sans', sans-serif;
        color: {text_primary};
    }}
    h1, h2, h3, h4 {{
        font-family: 'Syne', sans-serif;
        color: {text_primary} !important;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {text_primary} !important;
    }}
    label, .stLabel, [data-testid="stWidgetLabel"] p {{
        color: {text_secondary} !important;
        font-size: 13px !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, {accent1}, {accent3}) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.05em !important;
        padding: 0.6rem 1.6rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px {glow1} !important;
        text-transform: uppercase !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 0 36px {glow1}, 0 0 20px {glow3} !important;
    }}
    .stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* ── Inputs ── */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {{
        background: {input_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
    }}
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {{
        border-color: {accent1} !important;
        box-shadow: 0 0 0 2px {glow1} !important;
        outline: none !important;
    }}
    /* Number input stepper buttons */
    .stNumberInput button {{
        background: {bg_card2} !important;
        border-color: {border} !important;
        color: {text_primary} !important;
    }}
    /* Selectbox */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div {{
        background: {input_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
    }}
    /* Select dropdown options */
    [data-baseweb="select"] > div,
    [data-baseweb="popover"] {{
        background: {bg_card} !important;
        border-color: {border} !important;
    }}
    [data-baseweb="menu"] li {{
        background: {bg_card} !important;
        color: {text_primary} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background: {bg_card2} !important;
    }}
    /* Radio */
    .stRadio > div > div > label {{
        color: {text_secondary} !important;
    }}
    .stRadio [data-baseweb="radio"] div:first-child {{
        border-color: {accent1} !important;
    }}

    /* ── Metric cards ── */
    [data-testid="metric-container"] {{
        background: {metric_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Space Mono', monospace !important;
        font-size: 1.5rem !important;
        color: {accent1} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Syne', sans-serif !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {text_secondary} !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'Space Mono', monospace !important;
        font-size: 12px !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {bg_card} !important;
        border-radius: 14px !important;
        padding: 4px !important;
        border: 1px solid {border} !important;
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {text_secondary} !important;
        border-radius: 10px !important;
        border: none !important;
        background: transparent !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {accent1}, {accent3}) !important;
        color: #fff !important;
    }}
    /* Tab content panel */
    [data-testid="stTabsContent"],
    .stTabs [data-baseweb="tab-panel"] {{
        background: {bg_primary} !important;
    }}

    /* ── Expander ── */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {{
        background: {bg_card} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        color: {text_primary} !important;
        font-size: 13px !important;
    }}
    .streamlit-expanderContent,
    [data-testid="stExpander"] > div:last-child {{
        background: {bg_card2} !important;
        border: 1px solid {border} !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }}

    /* ── Form ── */
    [data-testid="stForm"],
    [data-testid="stForm"] > div {{
        background: {bg_card} !important;
        border: 1px solid {border} !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
    }}

    /* ── Alerts / Info boxes ── */
    .stAlert, [data-testid="stAlert"] {{
        background: {bg_card} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        color: {text_primary} !important;
    }}
    .stAlert > div {{
        color: {text_primary} !important;
    }}
    /* Success */
    [data-testid="stAlert"][data-type="success"] {{
        border-left: 4px solid {accent2} !important;
    }}
    /* Warning */
    [data-testid="stAlert"][data-type="warning"] {{
        border-left: 4px solid {accent4} !important;
    }}
    /* Error */
    [data-testid="stAlert"][data-type="error"] {{
        border-left: 4px solid {accent3} !important;
    }}
    /* Info */
    [data-testid="stAlert"][data-type="info"] {{
        border-left: 4px solid {accent5} !important;
    }}

    /* ── Spinner ── */
    .stSpinner > div {{
        border-top-color: {accent1} !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {accent1}, {accent2}) !important;
        border-radius: 99px !important;
    }}
    .stProgress > div > div {{
        background: {bg_card2} !important;
        border-radius: 99px !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {bg_primary}; }}
    ::-webkit-scrollbar-thumb {{ background: {accent1}; border-radius: 6px; }}

    /* ── Divider ── */
    hr {{ border-color: {border} !important; opacity: 0.5; }}

    /* ── Custom component classes ── */
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {tag_bg};
        border: 1px solid {accent1};
        border-radius: 99px;
        padding: 4px 14px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: {accent1};
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .hero-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: clamp(2.2rem, 5vw, 3.2rem);
        line-height: 1.1;
        background: linear-gradient(135deg, {accent1} 0%, {accent3} 55%, {accent2} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
    }}
    .hero-sub {{
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        color: {text_secondary};
        margin-bottom: 24px;
        max-width: 560px;
    }}
    .section-label {{
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: {accent2};
        margin-bottom: 4px;
    }}
    .section-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: {text_primary};
        margin-bottom: 16px;
    }}
    .agent-card {{
        background: {bg_card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }}
    .agent-card:hover {{ border-color: {accent1}; }}
    .agent-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, {accent1}, {accent3});
    }}
    .agent-name {{
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 13px;
        color: {text_primary};
        margin-bottom: 4px;
    }}
    .agent-role {{
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: {accent2};
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .agent-output {{
        font-family: 'DM Sans', sans-serif;
        font-size: 13.5px;
        color: {text_secondary};
        margin-top: 10px;
        line-height: 1.6;
        border-left: 3px solid {accent1};
        padding-left: 12px;
    }}
    .status-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {accent2};
        box-shadow: 0 0 8px {accent2};
        margin-right: 6px;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
    .tip-card {{
        background: {bg_card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 10px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }}
    .tip-icon {{ font-size: 22px; flex-shrink: 0; }}
    .tip-text {{
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: {text_primary};
        line-height: 1.6;
    }}
    .tip-warning  {{ border-left: 4px solid {accent4}; }}
    .tip-success  {{ border-left: 4px solid {accent2}; }}
    .tip-insight  {{ border-left: 4px solid {accent1}; }}
    .tip-danger   {{ border-left: 4px solid {accent3}; }}

    .health-label {{
        font-family: 'Syne', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: {text_secondary};
        text-align: center;
        margin-top: 4px;
    }}
    .chat-bubble-user {{
        background: linear-gradient(135deg, {accent1}, {accent3});
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        max-width: 80%;
        margin-left: auto;
    }}
    .chat-bubble-ai {{
        background: {bg_card};
        border: 1px solid {border};
        color: {text_primary};
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        max-width: 85%;
    }}
    .watermark {{
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: {text_secondary};
        opacity: 0.5;
        text-align: center;
        padding: 12px 0;
        letter-spacing: 0.1em;
    }}
    .flow-step {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: {bg_card2};
        border-radius: 10px;
        border: 1px solid {border};
        margin-bottom: 6px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: {text_secondary};
    }}
    .flow-step.active {{
        border-color: {accent1};
        color: {accent1};
        background: {tag_bg};
    }}
    .model-badge {{
        display: inline-block;
        background: {tag_bg};
        border: 1px solid {accent1};
        border-radius: 6px;
        padding: 2px 8px;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        color: {accent1};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-left: 6px;
        vertical-align: middle;
    }}
    </style>
    """, unsafe_allow_html=True)


# ─── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    income: float
    expenses: Dict[str, float]
    savings: float
    target: Optional[float]
    timeline: Optional[int]
    financial_summary: str
    budget_analysis: str
    risk_assessment: str
    investment_advice: str
    web_insights: str
    final_report: str
    messages: Annotated[List[Dict], operator.add]


# ─── Groq Client ───────────────────────────────────────────────────────────────
@st.cache_resource
def init_groq(api_key: str):
    return Groq(api_key=api_key)


def call_groq(client, system_prompt: str, user_prompt: str,
              agent_key: str = "financial_planner",
              max_tokens: int = 512, temperature: float = 0.65) -> str:
    model = MODELS.get(agent_key, "llama-3.3-70b-versatile")
    # deepseek-r1 models have specific formatting quirks — strip <think> tags
    strip_think = "deepseek" in model
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = resp.choices[0].message.content.strip()
        if strip_think:
            # Remove <think>...</think> blocks from DeepSeek R1 output
            import re
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        return result
    except Exception as e:
        return f"[Agent error: {str(e)[:120]}]"


# ─── Multi-Agent Nodes ─────────────────────────────────────────────────────────

def budget_analyst_node(state: AgentState, client) -> AgentState:
    """Agent 1 — Budget Analyst: breaks down income vs expenses"""
    total_exp = sum(state["expenses"].values())
    net = state["income"] - total_exp
    savings_rate = (net / state["income"] * 100) if state["income"] > 0 else 0
    ratio_50_30_20 = {
        "needs": sum(v for k, v in state["expenses"].items()
                     if k in ["Rent", "Utilities", "Groceries", "Transport"]),
        "wants": sum(v for k, v in state["expenses"].items()
                     if k in ["Dining", "Entertainment", "Shopping"]),
        "savings": max(0, net)
    }
    prompt = (
        f"You are a precision budget analyst AI. Analyze these financials:\n"
        f"Monthly Income: ₹{state['income']:,.0f}\n"
        f"Expenses: {json.dumps(state['expenses'])}\n"
        f"Net Cash Flow: ₹{net:,.0f} ({savings_rate:.1f}% savings rate)\n"
        f"50/30/20 Breakdown: Needs ₹{ratio_50_30_20['needs']:,.0f}, "
        f"Wants ₹{ratio_50_30_20['wants']:,.0f}, Savings ₹{ratio_50_30_20['savings']:,.0f}\n\n"
        "Give a sharp 3-bullet budget diagnosis with exact numbers. "
        "Identify the single biggest inefficiency. Be direct, no fluff. No preamble."
    )
    result = call_groq(client,
                       "You are an elite budget analyst. Use precise numbers. "
                       "Be concise. Use ₹ signs. No preamble, go straight to bullets.",
                       prompt, agent_key="budget_analyst", max_tokens=400)
    return {**state, "budget_analysis": result,
            "messages": [{"role": "budget_analyst", "content": result}]}


def risk_assessor_node(state: AgentState, client) -> AgentState:
    """Agent 2 — Risk Assessor: evaluates financial risk & emergency fund"""
    total_exp = sum(state["expenses"].values())
    emergency_target = total_exp * 6
    runway_months = state["savings"] / total_exp if total_exp > 0 else 0
    debt_ratio = total_exp / state["income"] * 100 if state["income"] > 0 else 0
    prompt = (
        f"Financial Risk Assessment:\n"
        f"- Emergency fund: ₹{state['savings']:,.0f} (6-month target: ₹{emergency_target:,.0f})\n"
        f"- Runway: {runway_months:.1f} months\n"
        f"- Expense-to-income ratio: {debt_ratio:.1f}%\n"
        f"- Monthly burn: ₹{total_exp:,.0f}\n\n"
        "Rate the financial risk as LOW / MEDIUM / HIGH / CRITICAL. "
        "List 2 specific vulnerability points and 1 immediate protection action. "
        "Use bullet points. No preamble."
    )
    result = call_groq(client,
                       "You are a financial risk specialist. Be blunt, data-driven, urgent where needed. "
                       "No preamble — start with the risk rating.",
                       prompt, agent_key="risk_assessor", max_tokens=350)
    return {**state, "risk_assessment": result,
            "messages": [{"role": "risk_assessor", "content": result}]}


def investment_advisor_node(state: AgentState, client) -> AgentState:
    """Agent 3 — Investment Advisor: portfolio allocation for India"""
    total_exp = sum(state["expenses"].values())
    investable = max(0, state["income"] - total_exp)
    prompt = (
        f"Investment Advisory (India-focused):\n"
        f"Investable monthly surplus: ₹{investable:,.0f}\n"
        f"Current savings: ₹{state['savings']:,.0f}\n"
        f"Savings goal: ₹{state.get('target') or 'Not set'}\n"
        f"Timeline: {state.get('timeline') or 'Not set'} months\n\n"
        "Provide a specific India-relevant investment allocation for this surplus. "
        "Name exact instruments (SIP, ELSS, PPF, NPS, FD, Liquid funds) with % allocation and ₹ amounts. "
        "No generic advice. No preamble."
    )
    result = call_groq(client,
                       "You are a SEBI-registered financial advisor for India. "
                       "Be specific with instruments. Use ₹ amounts and percentages. No preamble.",
                       prompt, agent_key="investment_advisor", max_tokens=420)
    return {**state, "investment_advice": result,
            "messages": [{"role": "investment_advisor", "content": result}]}


def web_researcher_node(state: AgentState, client) -> AgentState:
    """Agent 4 — Web Researcher: macro context via LLM knowledge"""
    total_exp = sum(state["expenses"].values())
    savings_rate = ((state["income"] - total_exp) / state["income"] * 100) if state["income"] > 0 else 0
    prompt = (
        f"A person in India earns ₹{state['income']:,.0f}/month with {savings_rate:.1f}% savings rate.\n"
        "Based on current Indian economic context (2026-2027):\n"
        "1. What is the current RBI repo rate impact on savings accounts and FDs?\n"
        "2. Which SIP / mutual fund categories are performing best recently?\n"
        "3. What inflation pressure should they factor into their budget?\n"
        "Give 3 crisp, current macro insights. Use approximate current figures. No preamble."
    )
    result = call_groq(client,
                       "You are a macro-economic researcher specializing in India 2025-2026. "
                       "Cite approximate current figures. Be specific. No preamble.",
                       prompt, agent_key="web_researcher", max_tokens=380)
    return {**state, "web_insights": result,
            "messages": [{"role": "web_researcher", "content": result}]}


def financial_planner_node(state: AgentState, client) -> AgentState:
    """Agent 5 — Financial Planner: synthesizes everything into action plan"""
    prompt = (
        "You are the master financial planner. Synthesize these agent reports into a 30/60/90-day action plan:\n\n"
        f"BUDGET ANALYSIS:\n{state.get('budget_analysis', '')}\n\n"
        f"RISK ASSESSMENT:\n{state.get('risk_assessment', '')}\n\n"
        f"INVESTMENT ADVICE:\n{state.get('investment_advice', '')}\n\n"
        f"MACRO INSIGHTS:\n{state.get('web_insights', '')}\n\n"
        "Create a numbered 30/60/90-day execution plan. "
        "Each phase: 2-3 specific actions with exact rupee amounts. "
        "End with ONE bold financial move for this month. No preamble."
    )
    result = call_groq(client,
                       "You are a CFP synthesizing multiple expert inputs into a clear execution plan. "
                       "Be decisive and specific. Use ₹. No preamble.",
                       prompt, agent_key="financial_planner", max_tokens=550)
    return {**state, "final_report": result,
            "messages": [{"role": "financial_planner", "content": result}]}


# ─── Build LangGraph Pipeline ──────────────────────────────────────────────────
def build_agent_graph(client):
    if not LANGGRAPH_AVAILABLE:
        return None
    graph = StateGraph(AgentState)
    graph.add_node("budget_analyst",     lambda s: budget_analyst_node(s, client))
    graph.add_node("risk_assessor",      lambda s: risk_assessor_node(s, client))
    graph.add_node("investment_advisor", lambda s: investment_advisor_node(s, client))
    graph.add_node("web_researcher",     lambda s: web_researcher_node(s, client))
    graph.add_node("financial_planner",  lambda s: financial_planner_node(s, client))
    graph.set_entry_point("budget_analyst")
    graph.add_edge("budget_analyst",     "risk_assessor")
    graph.add_edge("risk_assessor",      "investment_advisor")
    graph.add_edge("investment_advisor", "web_researcher")
    graph.add_edge("web_researcher",     "financial_planner")
    graph.add_edge("financial_planner",  END)
    return graph.compile()


def run_agents_langgraph(initial_state: AgentState, client, prog_container) -> AgentState:
    """
    Fixed LangGraph stream handler.
    stream_mode='values' yields the full state after each node — a single dict, not a tuple.
    stream_mode='updates' yields (node_name, state_delta) tuples.
    We use 'updates' for progress tracking and collect the final state separately.
    """
    agent_names = [
        "Budget Analyst", "Risk Assessor", "Investment Advisor",
        "Web Researcher", "Financial Planner"
    ]
    graph = build_agent_graph(client)
    result_state = dict(initial_state)

    with prog_container:
        prog   = st.progress(0)
        status = st.empty()
        step   = 0

        try:
            # stream_mode='updates' → yields {node_name: state_delta} dicts
            for chunk in graph.stream(initial_state, stream_mode="updates"):
                # chunk is a dict: { node_name: updated_fields }
                for node_name, node_output in chunk.items():
                    # Merge node output into running state
                    if isinstance(node_output, dict):
                        # Handle messages list specially (it uses add operator)
                        if "messages" in node_output:
                            existing = result_state.get("messages", [])
                            result_state["messages"] = existing + node_output.get("messages", [])
                            node_output_clean = {k: v for k, v in node_output.items()
                                                 if k != "messages"}
                            result_state.update(node_output_clean)
                        else:
                            result_state.update(node_output)

                    step += 1
                    display_name = agent_names[step - 1] if step <= len(agent_names) else node_name
                    prog.progress(step / len(agent_names))
                    status.markdown(
                        f'<div class="flow-step active">'
                        f'<span class="status-dot"></span>'
                        f'⬡ {display_name} · Complete ✓</div>',
                        unsafe_allow_html=True
                    )

            status.markdown(
                '<div class="flow-step active" style="color:var(--accent2);">'
                '✓ All 5 agents complete</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            raise e

   
    return result_state


# ─── Sequential Fallback (no LangGraph) ───────────────────────────────────────
def run_agents_sequential(state: AgentState, client, progress_container) -> AgentState:
    agents = [
        ("⬡ Budget Analyst",     budget_analyst_node,     "budget_analyst"),
        ("⬡ Risk Assessor",      risk_assessor_node,      "risk_assessor"),
        ("⬡ Investment Advisor", investment_advisor_node, "investment_advisor"),
        ("⬡ Web Researcher",     web_researcher_node,     "web_researcher"),
        ("⬡ Financial Planner",  financial_planner_node,  "financial_planner"),
    ]
    with progress_container:
        prog   = st.progress(0)
        status = st.empty()
        for i, (name, fn, key) in enumerate(agents):
            model_label = MODELS.get(key, "llama-3.3-70b").split("-")[0].capitalize()
            status.markdown(
                f'<div class="flow-step active">'
                f'<span class="status-dot"></span>{name} · Running…'
                f'<span class="model-badge">{MODELS.get(key,"").replace("llama-","llama ")[:18]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            state = fn(state, client)
            prog.progress((i + 1) / len(agents))
            time.sleep(0.15)
        status.markdown(
            '<div class="flow-step active" style="color:var(--accent2);">✓ All agents complete</div>',
            unsafe_allow_html=True
        )
    return state


# ─── Health Score ──────────────────────────────────────────────────────────────
def compute_health_score(income, expenses, savings, target, timeline):
    total_exp = sum(expenses.values())
    net = income - total_exp
    score = 50
    if income > 0:
        sr = net / income
        score += min(25, sr * 80)
        er = total_exp / income
        score -= max(0, (er - 0.7) * 60)
    runway = savings / total_exp if total_exp > 0 else 0
    score += min(15, runway * 3)
    if target and timeline and timeline > 0:
        monthly_needed = (target - savings) / timeline
        if net >= monthly_needed:
            score += 10
        else:
            deficit_ratio = (monthly_needed - net) / (monthly_needed + 1)
            score -= deficit_ratio * 15
    return max(0, min(100, round(score)))


# ─── Charts ───────────────────────────────────────────────────────────────────
def make_donut(expenses, dark):
    labels = [k for k, v in expenses.items() if v > 0]
    values = [v for v in expenses.values() if v > 0]
    colors = ["#7b2fff", "#00f5c4", "#ff3cac", "#ffbe0b",
              "#00b4d8", "#ff6b35", "#c77dff", "#4cc9f0"]
    tick_color = "#e8e8ff" if dark else "#1a0a3d"
    paper_bg   = "rgba(0,0,0,0)"
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(
            colors=colors[:len(labels)],
            line=dict(color="#050508" if dark else "#f0effe", width=3)
        ),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>',
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            font=dict(family="Space Mono", size=10, color=tick_color),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        height=300,
    )
    return fig


def make_waterfall(income, expenses, dark):
    tick_color = "#8888cc" if dark else "#5a4a8a"
    text_color = "#e8e8ff" if dark else "#1a0a3d"
    grid_color = "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.04)"
    cats = list(expenses.keys())
    vals = [-v for v in expenses.values()]
    fig = go.Figure(go.Waterfall(
        name="Cash Flow",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(cats) + ["total"],
        x=["Income"] + cats + ["Net"],
        y=[income] + vals + [0],
        connector=dict(line=dict(color="#7b2fff", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#00f5c4")),
        decreasing=dict(marker=dict(color="#ff3cac")),
        totals=dict(marker=dict(color="#7b2fff")),
        texttemplate="₹%{y:,.0f}",
        textposition="outside",
        textfont=dict(family="Space Mono", size=9, color=text_color),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(
            tickfont=dict(family="Space Mono", size=9, color=tick_color),
            gridcolor=grid_color
        ),
        yaxis=dict(
            tickfont=dict(family="Space Mono", size=9, color=tick_color),
            tickprefix="₹",
            gridcolor=grid_color
        ),
        showlegend=False,
    )
    return fig


def make_gauge(score, dark):
    color = "#00f5c4" if score >= 70 else "#ffbe0b" if score >= 45 else "#ff3cac"
    tick_color = "#8888cc" if dark else "#5a4a8a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain=dict(x=[0, 1], y=[0, 1]),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=tick_color,
                tickfont=dict(family="Space Mono", size=9)
            ),
            bar=dict(color=color, thickness=0.2),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 45],   color="rgba(255,60,172,0.08)"),
                dict(range=[45, 70],  color="rgba(255,190,11,0.08)"),
                dict(range=[70, 100], color="rgba(0,245,196,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.7, value=score),
        ),
        number=dict(font=dict(family="Space Mono", size=48, color=color), suffix=""),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(t=10, b=10, l=20, r=20),
    )
    return fig


# ─── Chat Agent ────────────────────────────────────────────────────────────────
def chat_with_advisor(client, user_msg: str, context: dict, history: list) -> str:
    context_str = (
        f"Income: ₹{context['income']:,.0f}, "
        f"Expenses: ₹{sum(context['expenses'].values()):,.0f}, "
        f"Savings: ₹{context['savings']:,.0f}"
    )
    system = (
        "You are MoneyMentor, a 2026-era agentic AI financial advisor. "
        "You are sharp, confident, and data-driven. You know Indian personal finance inside out. "
        f"User context: {context_str}. "
        "Answer concisely in 3-5 sentences. Use ₹ signs. Start with a relevant emoji."
    )
    messages = []
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})
    try:
        resp = client.chat.completions.create(
            model=MODELS["chat"],
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=300,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Chat error: {str(e)[:80]}"


# ─── Rule-based Tips ──────────────────────────────────────────────────────────
def get_rule_tips(income, expenses, savings, target, timeline):
    tips = []
    total_exp = sum(expenses.values())
    net = income - total_exp
    if net < 0:
        tips.append({"type": "danger", "icon": "🚨",
                     "text": f"Critical: Overspending by ₹{abs(net):,.0f}/month. "
                             "Cut discretionary spend immediately."})
    else:
        sr = net / income * 100 if income > 0 else 0
        if sr < 10:
            tips.append({"type": "warning", "icon": "⚠️",
                         "text": f"Savings rate {sr:.1f}% — dangerously low. Target 20%+."})
        elif sr < 20:
            tips.append({"type": "warning", "icon": "📊",
                         "text": f"Savings rate {sr:.1f}% — below 20% benchmark. "
                                 f"Gap to close: ₹{(income * 0.20 - net):,.0f}/month."})
        else:
            tips.append({"type": "success", "icon": "✅",
                         "text": f"Excellent {sr:.1f}% savings rate — above 20% benchmark. Keep compounding."})
    if expenses:
        top = max(expenses, key=expenses.get)
        pct = expenses[top] / income * 100 if income > 0 else 0
        tips.append({"type": "insight", "icon": "🔍",
                     "text": f"'{top}' is your largest expense at ₹{expenses[top]:,.0f} "
                             f"({pct:.0f}% of income). Audit this category."})
    runway = savings / total_exp if total_exp > 0 else 0
    em_target = total_exp * 6
    if runway < 3:
        tips.append({"type": "warning", "icon": "🛡️",
                     "text": f"Emergency fund: {runway:.1f} months. "
                             f"Build to 6 months (₹{em_target:,.0f}) first."})
    elif runway < 6:
        tips.append({"type": "insight", "icon": "🛡️",
                     "text": f"{runway:.1f}-month emergency buffer. "
                             f"6 months (₹{em_target:,.0f}) is the safe target."})
    else:
        tips.append({"type": "success", "icon": "🛡️",
                     "text": f"Strong {runway:.1f}-month emergency fund. "
                             "Deploy surplus into investments."})
    if target and timeline and timeline > 0:
        needed = target - savings
        monthly = needed / timeline
        if monthly <= net:
            surplus = net - monthly
            tips.append({"type": "success", "icon": "🎯",
                         "text": f"Goal achievable: ₹{monthly:,.0f}/month needed. "
                                 f"₹{surplus:,.0f}/month left over."})
        else:
            gap = monthly - net
            tips.append({"type": "danger", "icon": "🎯",
                         "text": f"Goal shortfall: need ₹{monthly:,.0f}/month but only "
                                 f"₹{net:,.0f} available. Gap: ₹{gap:,.0f}."})
    return tips


# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    # ── Session state init ──
    defaults = {
        "analysis_done":  False,
        "agent_state":    None,
        "financial_data": None,
        "chat_history":   [],
        "dark_mode":      True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    dark = st.session_state.dark_mode
    inject_css(dark)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 20px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:11px;
                 color:var(--accent2);letter-spacing:0.15em;
                 text-transform:uppercase;margin-bottom:6px;">⬡ MoneyMentor AI</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                 font-size:1.4rem;background:linear-gradient(135deg,var(--accent1),var(--accent3));
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">Financial OS</div>
        </div>
        """, unsafe_allow_html=True)

        mode_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
        if st.button(mode_label, use_container_width=True):
            st.session_state.dark_mode = not dark
            st.rerun()

        st.markdown("---")

        # API key
        st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)
        if 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
            st.success("✓ API key loaded from secrets")
        else:
            api_key = st.text_input("Groq API Key", type="password",
                                    placeholder="gsk_…", label_visibility="collapsed")
            if not api_key:
                st.warning("Enter your Groq API key to activate agents")
                st.stop()

        st.markdown("---")

        # Agent + model info
        st.markdown('<div class="section-label">Agent Pipeline · Models</div>', unsafe_allow_html=True)
        agents_info = [
            ("⬡", "Budget Analyst",     "budget_analyst",     "Income & expense analysis"),
            ("⬡", "Risk Assessor",      "risk_assessor",      "Emergency & vulnerability"),
            ("⬡", "Investment Advisor", "investment_advisor", "Portfolio allocation"),
            ("⬡", "Web Researcher",     "web_researcher",     "Macro market context"),
            ("⬡", "Financial Planner",  "financial_planner",  "90-day action synthesis"),
        ]
        for icon, name, key, desc in agents_info:
            model_short = MODELS.get(key, "").replace("llama-3.3-70b-versatile", "Llama 3.3 70B") \
                                            .replace("deepseek-r1-distill-llama-70b", "DeepSeek R1 70B") \
                                            .replace("gemma2-9b-it", "Gemma2 9B")
            st.markdown(f"""
            <div style="padding:8px 10px;margin-bottom:5px;
                 background:var(--bg-card);border:1px solid var(--border);border-radius:10px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         font-size:12px;color:var(--text-primary);">{icon} {name}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:8px;
                         color:var(--accent1);letter-spacing:0.04em;">{model_short}</div>
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:11px;
                     color:var(--text-secondary);margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        lg_status = "✅ LangGraph active" if LANGGRAPH_AVAILABLE else "⚡ Sequential mode"
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:10px;
             color:var(--text-secondary);letter-spacing:0.06em;">
             {lg_status}<br>Provider: Groq Cloud
        </div>
        """, unsafe_allow_html=True)

    # ── Init Groq ─────────────────────────────────────────────────────────────
    try:
        groq_client = init_groq(api_key)
    except Exception as e:
        st.error(f"Failed to init Groq: {e}")
        st.stop()

    # ── Hero Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-badge">
        <span class="status-dot"></span>5-Agent Intelligence · LangGraph · Groq Cloud
    </div>
    <div class="hero-title">MoneyMentor AI</div>
    <div class="hero-sub">
        Five specialized AI agents dissect your finances, assess risk, map investments,
        and forge your 90-day execution plan — powered by DeepSeek R1 &amp; Llama 3.3.
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["⬡  FINANCIAL INPUT", "⬡  AGENT ANALYSIS", "⬡  AI ADVISOR CHAT"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — INPUT
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Enter Your Monthly Financials</div>', unsafe_allow_html=True)

        with st.form("finance_form"):
            st.markdown('<div class="section-label" style="margin-top:8px;">Monthly Income</div>',
                        unsafe_allow_html=True)
            income = st.number_input("Total monthly income (₹)",
                                     min_value=0.0, step=500.0, value=45000.0,
                                     label_visibility="collapsed")
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:24px;font-weight:700;
                 color:var(--accent2);margin:4px 0 20px 0;">
                ₹{income:,.0f}<span style="font-size:12px;color:var(--text-secondary);"> /month</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Monthly Expenses</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                rent          = st.number_input("🏠 Rent/Mortgage (₹)",  min_value=0.0, step=100.0, value=14000.0)
                utilities     = st.number_input("⚡ Utilities (₹)",       min_value=0.0, step=50.0,  value=2500.0)
                groceries     = st.number_input("🛒 Groceries (₹)",       min_value=0.0, step=50.0,  value=5000.0)
                transport     = st.number_input("🚗 Transportation (₹)",  min_value=0.0, step=50.0,  value=3000.0)
            with c2:
                dining        = st.number_input("🍜 Dining Out (₹)",      min_value=0.0, step=50.0,  value=2500.0)
                entertainment = st.number_input("🎮 Entertainment (₹)",   min_value=0.0, step=50.0,  value=1500.0)
                shopping      = st.number_input("🛍️ Shopping (₹)",        min_value=0.0, step=50.0,  value=2000.0)
                other         = st.number_input("📦 Other Expenses (₹)",  min_value=0.0, step=50.0,  value=1500.0)

            expenses = {
                "Rent": rent, "Utilities": utilities, "Groceries": groceries,
                "Transport": transport, "Dining": dining,
                "Entertainment": entertainment, "Shopping": shopping, "Other": other
            }

            st.markdown('<div class="section-label" style="margin-top:16px;">Savings & Goals</div>',
                        unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                savings = st.number_input("💰 Current Savings (₹)", min_value=0.0, step=1000.0, value=50000.0)
            with c4:
                has_goal = st.radio("🎯 Savings Goal?", ("Yes", "No"), index=1, horizontal=True)

            target = None
            timeline = None
            if has_goal == "Yes":
                c5, c6 = st.columns(2)
                with c5:
                    target   = st.number_input("Target Amount (₹)", min_value=0.0, step=1000.0, value=200000.0)
                with c6:
                    timeline = st.number_input("Timeline (months)", min_value=1, step=1, value=18)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⬡  LAUNCH 5-AGENT ANALYSIS", use_container_width=True)

        if submitted:
            st.session_state.financial_data = {
                "income": income, "expenses": expenses,
                "savings": savings, "target": target, "timeline": timeline
            }
            initial_state: AgentState = {
                "income": income, "expenses": expenses,
                "savings": savings, "target": target, "timeline": timeline,
                "financial_summary": "", "budget_analysis": "",
                "risk_assessment": "", "investment_advice": "",
                "web_insights": "", "final_report": "",
                "messages": []
            }

            st.markdown("---")
            st.markdown('<div class="section-label">Agent Pipeline · Running</div>', unsafe_allow_html=True)
            prog_container = st.container()

            success = False
            if LANGGRAPH_AVAILABLE:
                try:
                    result_state = run_agents_langgraph(initial_state, groq_client, prog_container)
                    st.session_state.agent_state  = result_state
                    st.session_state.analysis_done = True
                    success = True
                except Exception as e:
                    st.warning(f"LangGraph error ({e}) — switching to sequential mode.")

            if not success:
                result = run_agents_sequential(initial_state, groq_client, prog_container)
                st.session_state.agent_state  = result
                st.session_state.analysis_done = True

            st.success("✅ All 5 agents complete — switch to the **AGENT ANALYSIS** tab")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        if not st.session_state.analysis_done or not st.session_state.agent_state:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;">
                <div style="font-size:48px;margin-bottom:16px;">⬡</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                     font-weight:700;color:var(--text-primary);margin-bottom:8px;">
                     No Analysis Yet</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:14px;
                     color:var(--text-secondary);">
                     Fill out your financials in the INPUT tab and launch the agent pipeline.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            s        = st.session_state.agent_state
            data     = st.session_state.financial_data
            income   = data["income"]
            expenses = data["expenses"]
            savings  = data["savings"]
            target   = data["target"]
            timeline = data["timeline"]
            total_exp = sum(expenses.values())
            net       = income - total_exp

            score = compute_health_score(income, expenses, savings, target, timeline)
            sr    = (net / income * 100) if income > 0 else 0

            # ── Metrics row ───────────────────────────────────────────────
            st.markdown('<div class="section-label">Financial Vitals</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Monthly Income",  f"₹{income:,.0f}")
            m2.metric("Total Expenses",  f"₹{total_exp:,.0f}",
                      delta=f"-{total_exp/income*100:.0f}% of income" if income else None,
                      delta_color="inverse")
            m3.metric("Net Cash Flow",   f"₹{net:,.0f}",
                      delta=f"{sr:.1f}% rate",
                      delta_color="normal" if net >= 0 else "inverse")
            m4.metric("Current Savings", f"₹{savings:,.0f}")
            if target:
                pct_done = min(100, savings / target * 100)
                m5.metric("Goal Progress", f"{pct_done:.0f}%",
                          delta=f"₹{target - savings:,.0f} to go")
            else:
                m5.metric("Health Score", f"{score}/100")

            st.markdown("---")

            # ── Charts row ────────────────────────────────────────────────
            c_gauge, c_donut, c_wfall = st.columns([1, 1.4, 1.8])
            with c_gauge:
                st.markdown('<div class="section-label">Health Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(score, dark), use_container_width=True,
                                config={"displayModeBar": False})
                grade = ("Excellent" if score >= 80 else "Good" if score >= 65
                         else "Fair" if score >= 45 else "At Risk")
                st.markdown(f'<div class="health-label">{grade}</div>', unsafe_allow_html=True)

            with c_donut:
                st.markdown('<div class="section-label">Expense Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(make_donut(expenses, dark), use_container_width=True,
                                config={"displayModeBar": False})

            with c_wfall:
                st.markdown('<div class="section-label">Cash Flow Waterfall</div>', unsafe_allow_html=True)
                st.plotly_chart(make_waterfall(income, expenses, dark), use_container_width=True,
                                config={"displayModeBar": False})

            if target and target > 0:
                pct = min(1.0, savings / target)
                st.markdown(f"""
                <div style="margin:8px 0 4px 0;">
                    <div style="display:flex;justify-content:space-between;
                         font-family:'Space Mono',monospace;font-size:11px;
                         color:var(--text-secondary);margin-bottom:6px;">
                        <span>Goal: ₹{target:,.0f}</span>
                        <span>{pct*100:.0f}% complete</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(pct)

            st.markdown("---")

            # ── Rule-based tips ───────────────────────────────────────────
            st.markdown('<div class="section-label">Instant Diagnostics</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Rule-Based Analysis</div>', unsafe_allow_html=True)
            tips = get_rule_tips(income, expenses, savings, target, timeline)
            for tip in tips:
                st.markdown(f"""
                <div class="tip-card tip-{tip['type']}">
                    <div class="tip-icon">{tip['icon']}</div>
                    <div class="tip-text">{tip['text']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Agent Outputs ─────────────────────────────────────────────
            st.markdown('<div class="section-label">Multi-Agent Intelligence</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Agent Analysis Reports</div>', unsafe_allow_html=True)

            agent_results = [
                ("⬡", "Budget Analyst",     "budget_analyst",     "Income & Expense Diagnostics", s.get("budget_analysis", "")),
                ("⬡", "Risk Assessor",      "risk_assessor",      "Risk & Vulnerability Report",  s.get("risk_assessment", "")),
                ("⬡", "Investment Advisor", "investment_advisor", "Portfolio Allocation Plan",    s.get("investment_advice", "")),
                ("⬡", "Web Researcher",     "web_researcher",     "Macro & Market Context",       s.get("web_insights", "")),
            ]

            a1, a2 = st.columns(2)
            for i, (icon, name, key, role, output) in enumerate(agent_results):
                model_short = MODELS.get(key, "").replace("llama-3.3-70b-versatile", "Llama 3.3 70B") \
                                                  .replace("deepseek-r1-distill-llama-70b", "DeepSeek R1 70B")
                with (a1 if i % 2 == 0 else a2):
                    with st.expander(f"{icon}  {name} — {role}", expanded=(i < 2)):
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <div class="agent-role">{role}</div>
                            <span class="model-badge">{model_short}</span>
                        </div>
                        <div class="agent-output">{output.replace(chr(10), '<br>') if output else 'No output yet.'}</div>
                        """, unsafe_allow_html=True)

            st.markdown("---")

            # ── 90-Day Action Plan ────────────────────────────────────────
            st.markdown('<div class="section-label">Master Synthesis</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⬡ 30/60/90-Day Action Plan</div>', unsafe_allow_html=True)
            final = s.get("final_report", "")
            if final:
                model_short = MODELS.get("financial_planner", "").replace("llama-3.3-70b-versatile", "Llama 3.3 70B")
                st.markdown(f"""
                <div class="agent-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="agent-name">Financial Planner — Master Report</div>
                        <span class="model-badge">{model_short}</span>
                    </div>
                    <div class="agent-output">{final.replace(chr(10), '<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬡  Re-run Agent Analysis", use_container_width=False):
                st.session_state.analysis_done = False
                st.session_state.agent_state   = None
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — CHAT
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-label">Conversational Finance AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ask MoneyMentor Anything</div>', unsafe_allow_html=True)

        if st.session_state.financial_data:
            d = st.session_state.financial_data
            exp_total = sum(d["expenses"].values())
            net_cash  = d["income"] - exp_total
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                 border-radius:12px;padding:10px 16px;margin-bottom:16px;
                 display:flex;gap:24px;flex-wrap:wrap;">
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.1em;">
                         Income</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent2);">₹{d['income']:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.1em;">
                         Expenses</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent3);">₹{exp_total:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.1em;">
                         Net Flow</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent1);">₹{net_cash:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.1em;">
                         Savings</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent4);">₹{d['savings']:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Fill your financials in the INPUT tab first for context-aware responses.")

        st.markdown('<div class="section-label" style="margin-bottom:8px;">Quick Questions</div>',
                    unsafe_allow_html=True)
        suggestions = [
            "How should I reduce my top expense?",
            "Best SIPs for ₹5,000/month?",
            "How to build emergency fund fast?",
            "Should I invest or pay off debt first?",
        ]
        s_cols = st.columns(4)
        for i, (sc, sq) in enumerate(zip(s_cols, suggestions)):
            with sc:
                if st.button(sq, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": sq})
                    ctx = st.session_state.financial_data or {"income": 0, "expenses": {}, "savings": 0}
                    reply = chat_with_advisor(groq_client, sq, ctx,
                                             st.session_state.chat_history[:-1])
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
                    <div class="chat-bubble-user">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-start;margin-bottom:8px;">
                    <div class="chat-bubble-ai">
                        <span style="font-family:'Space Mono',monospace;font-size:9px;
                             color:var(--accent1);letter-spacing:0.1em;
                             text-transform:uppercase;display:block;margin-bottom:4px;">
                             ⬡ MoneyMentor</span>
                        {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input("Ask anything about your finances…",
                                   placeholder="e.g. How can I save ₹1L in 6 months?",
                                   key="chat_input", label_visibility="collapsed")
        col_send, col_clear = st.columns([5, 1])
        with col_send:
            send = st.button("⬡  SEND MESSAGE", use_container_width=True)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            ctx = st.session_state.financial_data or {"income": 0, "expenses": {}, "savings": 0}
            with st.spinner(""):
                reply = chat_with_advisor(groq_client, user_input, ctx,
                                          st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="watermark">
        MONEYMENTOR AI · 2026 EDITION · GROQ × LANGGRAPH × DEEPSEEK R1 × LLAMA 3.3<br>
        Not financial advice. For educational purposes only.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
