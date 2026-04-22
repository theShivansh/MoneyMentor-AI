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
        accent1       = "#7b2fff"   # electric violet
        accent2       = "#00f5c4"   # neon mint
        accent3       = "#ff3cac"   # hot pink
        accent4       = "#ffbe0b"   # amber
        accent5       = "#00b4d8"   # cyan
        glow1         = "rgba(123,47,255,0.35)"
        glow2         = "rgba(0,245,196,0.25)"
        glow3         = "rgba(255,60,172,0.25)"
        metric_bg     = "#12122a"
        tag_bg        = "rgba(123,47,255,0.15)"
        sidebar_bg    = "#080812"
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

    html, body, [data-testid="stApp"] {{
        background: var(--bg-primary) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }}

    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border);
    }}

    /* Headings */
    h1,h2,h3,h4 {{
        font-family: 'Syne', sans-serif;
        color: var(--text-primary);
    }}

    /* Streamlit overrides */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent1), var(--accent3));
        color: #fff;
        border: none;
        border-radius: 12px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.05em;
        padding: 0.6rem 1.6rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px {glow1};
        text-transform: uppercase;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 36px {glow1}, 0 0 20px {glow3};
    }}

    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
    }}
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {{
        border-color: var(--accent1) !important;
        box-shadow: 0 0 0 2px {glow1} !important;
    }}

    .stRadio > div > div > label {{
        color: var(--text-secondary) !important;
    }}

    label, .stMarkdown p {{
        color: var(--text-secondary) !important;
        font-size: 13px;
    }}

    /* Metric cards */
    [data-testid="metric-container"] {{
        background: var(--metric-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Space Mono', monospace !important;
        font-size: 1.5rem !important;
        color: var(--accent1) !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Syne', sans-serif !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary) !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'Space Mono', monospace !important;
        font-size: 12px !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--bg-card) !important;
        border-radius: 14px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-secondary) !important;
        border-radius: 10px !important;
        border: none !important;
        background: transparent !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--accent1), var(--accent3)) !important;
        color: #fff !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 13px !important;
    }}
    .streamlit-expanderContent {{
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }}

    /* Form */
    [data-testid="stForm"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
    }}

    /* Spinner */
    .stSpinner > div {{
        border-top-color: var(--accent1) !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
    ::-webkit-scrollbar-thumb {{ background: var(--accent1); border-radius: 6px; }}

    /* Divider */
    hr {{ border-color: var(--border) !important; }}

    /* Success/Info/Warning/Error */
    .stAlert {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    /* Progress */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
        border-radius: 99px !important;
    }}

    /* ── Custom Components ── */
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--tag-bg);
        border: 1px solid var(--accent1);
        border-radius: 99px;
        padding: 4px 14px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: var(--accent1);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .hero-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: clamp(2.2rem, 5vw, 3.2rem);
        line-height: 1.1;
        background: linear-gradient(135deg, var(--accent1) 0%, var(--accent3) 55%, var(--accent2) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
    }}
    .hero-sub {{
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        color: var(--text-secondary);
        margin-bottom: 24px;
        max-width: 560px;
    }}
    .section-label {{
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--accent2);
        margin-bottom: 4px;
    }}
    .section-title {{
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 16px;
    }}
    .agent-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }}
    .agent-card:hover {{ border-color: var(--accent1); }}
    .agent-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent1), var(--accent3));
    }}
    .agent-name {{
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 13px;
        color: var(--text-primary);
        margin-bottom: 4px;
    }}
    .agent-role {{
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: var(--accent2);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .agent-output {{
        font-family: 'DM Sans', sans-serif;
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 10px;
        line-height: 1.6;
        border-left: 3px solid var(--accent1);
        padding-left: 12px;
    }}
    .status-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent2);
        box-shadow: 0 0 8px var(--accent2);
        margin-right: 6px;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
    .tip-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 10px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }}
    .tip-icon {{
        font-size: 22px;
        flex-shrink: 0;
    }}
    .tip-text {{
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: var(--text-primary);
        line-height: 1.6;
    }}
    .tip-warning  {{ border-left: 4px solid var(--accent4); }}
    .tip-success  {{ border-left: 4px solid var(--accent2); }}
    .tip-insight  {{ border-left: 4px solid var(--accent1); }}
    .tip-danger   {{ border-left: 4px solid var(--accent3); }}

    .health-score {{
        font-family: 'Space Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent1), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        line-height: 1;
    }}
    .health-label {{
        font-family: 'Syne', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--text-secondary);
        text-align: center;
        margin-top: 4px;
    }}
    .chat-bubble-user {{
        background: linear-gradient(135deg, var(--accent1), var(--accent3));
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
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
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
        color: var(--text-secondary);
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
        background: var(--bg-card2);
        border-radius: 10px;
        border: 1px solid var(--border);
        margin-bottom: 6px;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: var(--text-secondary);
    }}
    .flow-step.active {{
        border-color: var(--accent1);
        color: var(--accent1);
        background: var(--tag-bg);
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
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
              model: str = "llama-3.3-70b-versatile",
              max_tokens: int = 512, temperature: float = 0.65) -> str:
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
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Agent error: {str(e)[:120]}]"

# ─── Multi-Agent Nodes ─────────────────────────────────────────────────────────

def budget_analyst_node(state: AgentState, client) -> AgentState:
    """Agent 1 — Budget Analyst: breaks down income vs expenses"""
    total_exp = sum(state["expenses"].values())
    net = state["income"] - total_exp
    savings_rate = (net / state["income"] * 100) if state["income"] > 0 else 0
    ratio_50_30_20 = {
        "needs": sum(v for k,v in state["expenses"].items() if k in ["Rent/Mortgage","Utilities","Groceries","Transportation"]),
        "wants": sum(v for k,v in state["expenses"].items() if k in ["Dining Out","Entertainment","Shopping"]),
        "savings": net
    }
    prompt = (
        f"You are a precision budget analyst AI. Analyze these financials:\n"
        f"Monthly Income: ₹{state['income']:,.0f}\n"
        f"Expenses: {json.dumps(state['expenses'])}\n"
        f"Net Cash Flow: ₹{net:,.0f} ({savings_rate:.1f}% savings rate)\n"
        f"50/30/20 Breakdown: Needs ₹{ratio_50_30_20['needs']:,.0f}, "
        f"Wants ₹{ratio_50_30_20['wants']:,.0f}, Savings ₹{ratio_50_30_20['savings']:,.0f}\n\n"
        "Give a sharp 3-bullet budget diagnosis with exact numbers. "
        "Identify the single biggest inefficiency. Be direct, no fluff."
    )
    result = call_groq(client,
        "You are an elite budget analyst. Use precise numbers. Be concise and direct. Use ₹ signs.",
        prompt, max_tokens=350)
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
        f"- Emergency fund: ₹{state['savings']:,.0f} (target: ₹{emergency_target:,.0f})\n"
        f"- Runway: {runway_months:.1f} months\n"
        f"- Expense-to-income ratio: {debt_ratio:.1f}%\n"
        f"- Monthly burn: ₹{total_exp:,.0f}\n\n"
        "Rate the financial risk (LOW/MEDIUM/HIGH/CRITICAL). "
        "List 2 specific vulnerability points and 1 immediate protection action. Bullet points."
    )
    result = call_groq(client,
        "You are a financial risk specialist. Be blunt, data-driven, urgent where needed.",
        prompt, max_tokens=300)
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
        "Provide a specific India-relevant investment allocation (SIP, PPF, FD, etc) for this surplus. "
        "Name exact instruments with % allocation. No generic advice."
    )
    result = call_groq(client,
        "You are a SEBI-registered financial advisor for India. Be specific with instruments: SIP, ELSS, PPF, NPS, FD, Liquid funds. Use ₹ amounts.",
        prompt, max_tokens=380)
    return {**state, "investment_advice": result,
            "messages": [{"role": "investment_advisor", "content": result}]}

def web_researcher_node(state: AgentState, client) -> AgentState:
    """Agent 4 — Web Researcher: fetches macro context via LLM knowledge"""
    total_exp = sum(state["expenses"].values())
    savings_rate = ((state["income"] - total_exp) / state["income"] * 100) if state["income"] > 0 else 0
    prompt = (
        f"A person in India earns ₹{state['income']:,.0f}/month with {savings_rate:.1f}% savings rate.\n"
        "Based on current Indian economic context (2025-2026):\n"
        "1. What is the current RBI repo rate impact on savings accounts?\n"
        "2. Which SIP categories are performing best recently?\n"
        "3. What inflation pressure should they factor into their budget?\n"
        "Give 3 crisp, current macro insights relevant to their financial situation."
    )
    result = call_groq(client,
        "You are a macro-economic researcher specializing in India. Cite approximate current figures. Be specific.",
        prompt, max_tokens=350)
    return {**state, "web_insights": result,
            "messages": [{"role": "web_researcher", "content": result}]}

def financial_planner_node(state: AgentState, client) -> AgentState:
    """Agent 5 — Financial Planner: synthesizes everything into action plan"""
    prompt = (
        "You are the master financial planner. Synthesize these agent reports into a 30/60/90 day action plan:\n\n"
        f"BUDGET ANALYSIS:\n{state.get('budget_analysis','')}\n\n"
        f"RISK ASSESSMENT:\n{state.get('risk_assessment','')}\n\n"
        f"INVESTMENT ADVICE:\n{state.get('investment_advice','')}\n\n"
        f"MACRO INSIGHTS:\n{state.get('web_insights','')}\n\n"
        "Create a numbered 30/60/90-day execution plan. "
        "Each phase: 2-3 specific actions with exact rupee amounts. "
        "End with ONE bold financial move for this month."
    )
    result = call_groq(client,
        "You are a CFP synthesizing multiple expert inputs into a clear execution plan. Be decisive and specific.",
        prompt, max_tokens=500)
    return {**state, "final_report": result,
            "messages": [{"role": "financial_planner", "content": result}]}

# ─── Build LangGraph Pipeline ──────────────────────────────────────────────────
def build_agent_graph(client):
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(AgentState)

    graph.add_node("budget_analyst",    lambda s: budget_analyst_node(s, client))
    graph.add_node("risk_assessor",     lambda s: risk_assessor_node(s, client))
    graph.add_node("investment_advisor",lambda s: investment_advisor_node(s, client))
    graph.add_node("web_researcher",    lambda s: web_researcher_node(s, client))
    graph.add_node("financial_planner", lambda s: financial_planner_node(s, client))

    graph.set_entry_point("budget_analyst")
    graph.add_edge("budget_analyst",    "risk_assessor")
    graph.add_edge("risk_assessor",     "investment_advisor")
    graph.add_edge("investment_advisor","web_researcher")
    graph.add_edge("web_researcher",    "financial_planner")
    graph.add_edge("financial_planner", END)

    return graph.compile()

# ─── Sequential Fallback (no LangGraph) ───────────────────────────────────────
def run_agents_sequential(state: AgentState, client, progress_container) -> AgentState:
    agents = [
        ("⬡ Budget Analyst",     budget_analyst_node),
        ("⬡ Risk Assessor",      risk_assessor_node),
        ("⬡ Investment Advisor", investment_advisor_node),
        ("⬡ Web Researcher",     web_researcher_node),
        ("⬡ Financial Planner",  financial_planner_node),
    ]
    with progress_container:
        prog = st.progress(0)
        status = st.empty()
        for i, (name, fn) in enumerate(agents):
            status.markdown(
                f'<div class="flow-step active"><span class="status-dot"></span>{name} · Running…</div>',
                unsafe_allow_html=True
            )
            state = fn(state, client)
            prog.progress((i + 1) / len(agents))
            time.sleep(0.2)
        status.markdown(
            '<div class="flow-step active" style="color:var(--accent2)">✓ All agents complete</div>',
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
        score += min(25, sr * 80)                     # savings rate
        er = total_exp / income
        score -= max(0, (er - 0.7) * 60)              # expense ratio penalty

    runway = savings / total_exp if total_exp > 0 else 0
    score += min(15, runway * 3)                      # emergency fund

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
    labels = [k for k,v in expenses.items() if v > 0]
    values = [v for v in expenses.values() if v > 0]
    colors = ["#7b2fff","#00f5c4","#ff3cac","#ffbe0b","#00b4d8","#ff6b35","#c77dff","#4cc9f0"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors[:len(labels)], line=dict(color="#050508" if dark else "#f0effe", width=3)),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>',
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v", font=dict(family="Space Mono", size=10,
            color="#e8e8ff" if dark else "#1a0a3d"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20,b=20,l=20,r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return fig

def make_waterfall(income, expenses, dark):
    total = sum(expenses.values())
    net   = income - total
    cats  = list(expenses.keys())
    vals  = [-v for v in expenses.values()]

    fig = go.Figure(go.Waterfall(
        name="Cash Flow",
        orientation="v",
        measure=["absolute"] + ["relative"]*len(cats) + ["total"],
        x=["Income"] + cats + ["Net"],
        y=[income] + vals + [0],
        connector=dict(line=dict(color="#7b2fff", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#00f5c4")),
        decreasing=dict(marker=dict(color="#ff3cac")),
        totals=dict(marker=dict(color="#7b2fff")),
        texttemplate="₹%{y:,.0f}",
        textposition="outside",
        textfont=dict(family="Space Mono", size=9, color="#e8e8ff" if dark else "#1a0a3d"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=20,b=20,l=20,r=20),
        xaxis=dict(tickfont=dict(family="Space Mono", size=9,
                   color="#8888cc" if dark else "#5a4a8a"),
                   gridcolor="rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.04)"),
        yaxis=dict(tickfont=dict(family="Space Mono", size=9,
                   color="#8888cc" if dark else "#5a4a8a"),
                   tickprefix="₹", gridcolor="rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.04)"),
        showlegend=False,
    )
    return fig

def make_gauge(score, dark):
    color = "#00f5c4" if score >= 70 else "#ffbe0b" if score >= 45 else "#ff3cac"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain=dict(x=[0,1], y=[0,1]),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1,
                      tickcolor="#8888cc" if dark else "#5a4a8a",
                      tickfont=dict(family="Space Mono", size=9)),
            bar=dict(color=color, thickness=0.2),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,45],  color="rgba(255,60,172,0.08)"),
                dict(range=[45,70], color="rgba(255,190,11,0.08)"),
                dict(range=[70,100],color="rgba(0,245,196,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.7, value=score),
        ),
        number=dict(font=dict(family="Space Mono", size=48, color=color), suffix=""),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(t=10,b=10,l=20,r=20),
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
        "You are MoneyMentor, a 2027-era agentic AI financial advisor. "
        "You are sharp, confident, and data-driven. You know Indian personal finance inside out. "
        f"User context: {context_str}. "
        "Answer concisely. Use ₹ signs. Start with a relevant emoji."
    )
    messages = []
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        tips.append({"type":"danger","icon":"🚨","text":
            f"Critical: You're overspending by ₹{abs(net):,.0f}/month. "
            "Immediate action required — cut discretionary spend first."})
    else:
        sr = net / income * 100 if income > 0 else 0
        if sr < 10:
            tips.append({"type":"warning","icon":"⚠️","text":
                f"Savings rate at {sr:.1f}% — dangerously low. Target 20%+ by reducing wants."})
        elif sr < 20:
            tips.append({"type":"warning","icon":"📊","text":
                f"Savings rate {sr:.1f}% — below 20% benchmark. ₹{(income*0.20 - net):,.0f} gap to close."})
        else:
            tips.append({"type":"success","icon":"✅","text":
                f"Excellent {sr:.1f}% savings rate — above 20% benchmark. Keep compounding."})

    if expenses:
        top = max(expenses, key=expenses.get)
        pct = expenses[top] / income * 100 if income > 0 else 0
        tips.append({"type":"insight","icon":"🔍","text":
            f"'{top}' is your largest expense at ₹{expenses[top]:,.0f} ({pct:.0f}% of income). Audit this category."})

    runway = savings / total_exp if total_exp > 0 else 0
    em_target = total_exp * 6
    if runway < 3:
        tips.append({"type":"warning","icon":"🛡️","text":
            f"Emergency fund covers only {runway:.1f} months. Build to 6 months (₹{em_target:,.0f}) first."})
    elif runway < 6:
        tips.append({"type":"insight","icon":"🛡️","text":
            f"{runway:.1f}-month emergency buffer. Good, but 6 months (₹{em_target:,.0f}) is safer."})
    else:
        tips.append({"type":"success","icon":"🛡️","text":
            f"Strong {runway:.1f}-month emergency fund. Consider deploying surplus into investments."})

    if target and timeline and timeline > 0:
        needed = target - savings
        monthly = needed / timeline
        if monthly <= net:
            surplus = net - monthly
            tips.append({"type":"success","icon":"🎯","text":
                f"Goal achievable: save ₹{monthly:,.0f}/month. You'll have ₹{surplus:,.0f}/month left over."})
        else:
            gap = monthly - net
            tips.append({"type":"danger","icon":"🎯","text":
                f"Goal shortfall: need ₹{monthly:,.0f}/month but only ₹{net:,.0f} available. Gap: ₹{gap:,.0f}."})

    return tips

# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    # ── Session state init ──
    if "analysis_done"    not in st.session_state: st.session_state.analysis_done    = False
    if "agent_state"      not in st.session_state: st.session_state.agent_state      = None
    if "financial_data"   not in st.session_state: st.session_state.financial_data   = None
    if "chat_history"     not in st.session_state: st.session_state.chat_history     = []
    if "dark_mode"        not in st.session_state: st.session_state.dark_mode        = True

    dark = st.session_state.dark_mode
    inject_css(dark)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 20px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:11px;
                 color:var(--accent2);letter-spacing:0.15em;text-transform:uppercase;
                 margin-bottom:6px;">⬡ MoneyMentor AI</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;
                 background:linear-gradient(135deg,var(--accent1),var(--accent3));
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">Financial OS </div>
        </div>
        """, unsafe_allow_html=True)

        # Dark/Light toggle
        mode_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
        if st.button(mode_label, use_container_width=True):
            st.session_state.dark_mode = not dark
            st.rerun()

        st.markdown("---")

        # API key
        st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)
        if 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
            st.success("API key loaded from secrets")
        else:
            api_key = st.text_input("Groq API Key", type="password",
                                    placeholder="gsk_…", label_visibility="collapsed")
            if not api_key:
                st.warning("Enter your Groq API key to activate agents")
                st.stop()

        st.markdown("---")

        # Agent graph status
        st.markdown('<div class="section-label">Agent Pipeline</div>', unsafe_allow_html=True)
        agents_info = [
            ("⬡", "Budget Analyst",     "Income & expense analysis"),
            ("⬡", "Risk Assessor",      "Emergency & vulnerability"),
            ("⬡", "Investment Advisor", "Portfolio allocation"),
            ("⬡", "Web Researcher",     "Macro market context"),
            ("⬡", "Financial Planner",  "90-day action synthesis"),
        ]
        for icon, name, desc in agents_info:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;
                 padding:8px 10px;margin-bottom:5px;
                 background:var(--bg-card);border:1px solid var(--border);
                 border-radius:10px;">
                <span style="color:var(--accent1);font-size:16px;margin-top:1px;">{icon}</span>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         font-size:12px;color:var(--text-primary);">{name}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:11px;
                         color:var(--text-secondary);">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        lg_status = "✅ LangGraph active" if LANGGRAPH_AVAILABLE else "⚡ Sequential mode"
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:10px;
             color:var(--text-secondary);letter-spacing:0.06em;">
             {lg_status}<br>Model: llama-3.3-70b-versatile<br>
             Provider: Groq Cloud
        </div>
        """, unsafe_allow_html=True)

    # ── Init Groq ─────────────────────────────────────────────────────────────
    try:
        groq_client = init_groq(api_key)
    except Exception as e:
        st.error(f"Failed to init Groq: {e}")
        st.stop()

    # ── Hero Header ───────────────────────────────────────────────────────────
    col_hero, col_badge = st.columns([3,1])
    with col_hero:
        st.markdown("""
        <div class="hero-badge">
            <span class="status-dot"></span>5-Agent Intelligence · LangGraph · Live
        </div>
        <div class="hero-title">MoneyMentor AI</div>
        <div class="hero-sub">
            The world's first agentic personal finance OS — five specialized AI agents
            dissect your finances, assess risk, map investments, and forge your 90-day plan.
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["⬡  FINANCIAL INPUT", "⬡  AGENT ANALYSIS", "⬡  AI ADVISOR CHAT"])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 — INPUT
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Enter Your Monthly Financials</div>', unsafe_allow_html=True)

        with st.form("finance_form"):
            # Income
            st.markdown('<div class="section-label" style="margin-top:8px;">Monthly Income</div>',
                        unsafe_allow_html=True)
            income = st.number_input("Total monthly income (₹)",
                                     min_value=0.0, step=500.0, value=45000.0,
                                     label_visibility="collapsed")

            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:24px;font-weight:700;
                 color:var(--accent2);margin:4px 0 20px 0;">
                ₹{income:,.0f} <span style="font-size:12px;color:var(--text-secondary);">/month</span>
            </div>
            """, unsafe_allow_html=True)

            # Expenses
            st.markdown('<div class="section-label">Monthly Expenses</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                rent          = st.number_input("🏠 Rent/Mortgage (₹)",      min_value=0.0, step=100.0, value=14000.0)
                utilities     = st.number_input("⚡ Utilities (₹)",           min_value=0.0, step=50.0,  value=2500.0)
                groceries     = st.number_input("🛒 Groceries (₹)",           min_value=0.0, step=50.0,  value=5000.0)
                transport     = st.number_input("🚗 Transportation (₹)",      min_value=0.0, step=50.0,  value=3000.0)
            with c2:
                dining        = st.number_input("🍜 Dining Out (₹)",          min_value=0.0, step=50.0,  value=2500.0)
                entertainment = st.number_input("🎮 Entertainment (₹)",       min_value=0.0, step=50.0,  value=1500.0)
                shopping      = st.number_input("🛍️ Shopping (₹)",            min_value=0.0, step=50.0,  value=2000.0)
                other         = st.number_input("📦 Other Expenses (₹)",      min_value=0.0, step=50.0,  value=1500.0)

            expenses = {
                "Rent":rent,"Utilities":utilities,"Groceries":groceries,
                "Transport":transport,"Dining":dining,"Entertainment":entertainment,
                "Shopping":shopping,"Other":other
            }

            # Savings & Goals
            st.markdown('<div class="section-label" style="margin-top:16px;">Savings & Goals</div>',
                        unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                savings = st.number_input("💰 Current Savings (₹)", min_value=0.0, step=1000.0, value=50000.0)
            with c4:
                has_goal = st.radio("🎯 Savings Goal?", ("Yes","No"), index=1, horizontal=True)

            target = None; timeline = None
            if has_goal == "Yes":
                c5, c6 = st.columns(2)
                with c5:
                    target   = st.number_input("Target Amount (₹)",  min_value=0.0, step=1000.0, value=200000.0)
                with c6:
                    timeline = st.number_input("Timeline (months)",   min_value=1,   step=1,      value=18)

            # Submit
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

            if LANGGRAPH_AVAILABLE:
                try:
                    graph = build_agent_graph(groq_client)
                    with prog_container:
                        prog = st.progress(0)
                        status = st.empty()
                        agent_names = [
                            "Budget Analyst","Risk Assessor","Investment Advisor",
                            "Web Researcher","Financial Planner"
                        ]
                        # LangGraph stream
                        result_state = initial_state
                        for i, (node_name, chunk) in enumerate(
                            graph.stream(initial_state, stream_mode="values")
                        ):
                            result_state = chunk
                            prog.progress((i+1)/5)
                            if i < len(agent_names):
                                status.markdown(
                                    f'<div class="flow-step active">'
                                    f'<span class="status-dot"></span>'
                                    f'⬡ {agent_names[i]} · Complete</div>',
                                    unsafe_allow_html=True
                                )
                        st.session_state.agent_state  = result_state
                        st.session_state.analysis_done = True
                except Exception as e:
                    st.warning(f"LangGraph stream error ({e}), using sequential fallback.")
                    result = run_agents_sequential(initial_state, groq_client, prog_container)
                    st.session_state.agent_state  = result
                    st.session_state.analysis_done = True
            else:
                result = run_agents_sequential(initial_state, groq_client, prog_container)
                st.session_state.agent_state  = result
                st.session_state.analysis_done = True

            st.success("✅ All 5 agents complete — switch to **AGENT ANALYSIS** tab")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANALYSIS
    # ════════════════════════════════════════════════════════════════════════════
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
            s    = st.session_state.agent_state
            data = st.session_state.financial_data
            income   = data["income"]
            expenses = data["expenses"]
            savings  = data["savings"]
            target   = data["target"]
            timeline = data["timeline"]
            total_exp = sum(expenses.values())
            net       = income - total_exp

            # ── Health Score & Metrics ─────────────────────────────────────
            score = compute_health_score(income, expenses, savings, target, timeline)
            sr    = (net / income * 100) if income > 0 else 0

            st.markdown('<div class="section-label">Financial Vitals</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Monthly Income",   f"₹{income:,.0f}")
            m2.metric("Total Expenses",   f"₹{total_exp:,.0f}",
                      delta=f"-{total_exp/income*100:.0f}% of income" if income else None,
                      delta_color="inverse")
            m3.metric("Net Cash Flow",    f"₹{net:,.0f}",
                      delta=f"{sr:.1f}% rate",
                      delta_color="normal" if net >= 0 else "inverse")
            m4.metric("Current Savings",  f"₹{savings:,.0f}")
            if target:
                pct_done = min(100, savings / target * 100)
                m5.metric("Goal Progress", f"{pct_done:.0f}%",
                          delta=f"₹{target-savings:,.0f} to go")
            else:
                m5.metric("Health Score", f"{score}/100")

            st.markdown("---")

            # ── Charts row ─────────────────────────────────────────────────
            c_gauge, c_donut, c_wfall = st.columns([1,1.4,1.8])

            with c_gauge:
                st.markdown('<div class="section-label">Health Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(score, dark), use_container_width=True,
                                config={"displayModeBar":False})
                grade = ("Excellent" if score >= 80 else "Good" if score >= 65
                         else "Fair" if score >= 45 else "At Risk")
                st.markdown(f'<div class="health-label">{grade}</div>', unsafe_allow_html=True)

            with c_donut:
                st.markdown('<div class="section-label">Expense Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(make_donut(expenses, dark), use_container_width=True,
                                config={"displayModeBar":False})

            with c_wfall:
                st.markdown('<div class="section-label">Cash Flow Waterfall</div>', unsafe_allow_html=True)
                st.plotly_chart(make_waterfall(income, expenses, dark), use_container_width=True,
                                config={"displayModeBar":False})

            # Goal progress bar
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

            # ── Rule-based tips ────────────────────────────────────────────
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

            # ── Agent Outputs ──────────────────────────────────────────────
            st.markdown('<div class="section-label">Multi-Agent Intelligence</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Agent Analysis Reports</div>', unsafe_allow_html=True)

            agent_results = [
                ("⬡", "Budget Analyst",     "Income & Expense Diagnostics", s.get("budget_analysis","")),
                ("⬡", "Risk Assessor",      "Risk & Vulnerability Report",  s.get("risk_assessment","")),
                ("⬡", "Investment Advisor", "Portfolio Allocation Plan",    s.get("investment_advice","")),
                ("⬡", "Web Researcher",     "Macro & Market Context",       s.get("web_insights","")),
            ]

            a1, a2 = st.columns(2)
            for i, (icon, name, role, output) in enumerate(agent_results):
                with (a1 if i % 2 == 0 else a2):
                    with st.expander(f"{icon}  {name} — {role}", expanded=(i < 2)):
                        st.markdown(f"""
                        <div class="agent-role">{role}</div>
                        <div class="agent-output">{output.replace(chr(10), '<br>') if output else 'No output yet.'}</div>
                        """, unsafe_allow_html=True)

            st.markdown("---")

            # ── 90-Day Action Plan ─────────────────────────────────────────
            st.markdown('<div class="section-label">Master Synthesis</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⬡ 30/60/90-Day Action Plan</div>', unsafe_allow_html=True)
            final = s.get("final_report","")
            if final:
                st.markdown(f"""
                <div class="agent-card">
                    <div class="agent-name">Financial Planner — Master Report</div>
                    <div class="agent-output">{final.replace(chr(10),'<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Re-run button ──────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬡  Re-run Agent Analysis", use_container_width=False):
                st.session_state.analysis_done = False
                st.session_state.agent_state   = None
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3 — CHAT
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-label">Conversational Finance AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ask MoneyMentor Anything</div>', unsafe_allow_html=True)

        # Context banner
        if st.session_state.financial_data:
            d = st.session_state.financial_data
            exp_total = sum(d["expenses"].values())
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                 border-radius:12px;padding:10px 16px;margin-bottom:16px;
                 display:flex;gap:24px;flex-wrap:wrap;">
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;
                         letter-spacing:0.1em;">Income</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent2);">₹{d['income']:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;
                         letter-spacing:0.1em;">Expenses</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent3);">₹{exp_total:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:10px;
                         color:var(--text-secondary);text-transform:uppercase;
                         letter-spacing:0.1em;">Savings</div>
                    <div style="font-family:'Space Mono',monospace;font-size:14px;
                         font-weight:700;color:var(--accent1);">₹{d['savings']:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Fill your financials in the INPUT tab first for context-aware responses.")

        # Suggested questions
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
                    st.session_state.chat_history.append(
                        {"role":"user","content":sq}
                    )
                    ctx = st.session_state.financial_data or {
                        "income":0,"expenses":{},"savings":0
                    }
                    reply = chat_with_advisor(groq_client, sq, ctx,
                                             st.session_state.chat_history[:-1])
                    st.session_state.chat_history.append(
                        {"role":"assistant","content":reply}
                    )
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat history
        chat_area = st.container()
        with chat_area:
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
                                 text-transform:uppercase;display:block;
                                 margin-bottom:4px;">⬡ MoneyMentor</span>
                            {msg['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Input
        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input("Ask anything about your finances…",
                                   placeholder="e.g. How can I save ₹1L in 6 months?",
                                   key="chat_input", label_visibility="collapsed")

        col_send, col_clear = st.columns([5,1])
        with col_send:
            send = st.button("⬡  SEND MESSAGE", use_container_width=True)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and user_input.strip():
            st.session_state.chat_history.append({"role":"user","content":user_input})
            ctx = st.session_state.financial_data or {"income":0,"expenses":{},"savings":0}
            with st.spinner(""):
                reply = chat_with_advisor(groq_client, user_input, ctx,
                                          st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role":"assistant","content":reply})
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="watermark">
        MONEYMENTOR AI · 2026 EDITION · POWERED BY GROQ × LANGGRAPH × LLAMA-3.3-70B<br>
        Not financial advice. For educational purposes only.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
