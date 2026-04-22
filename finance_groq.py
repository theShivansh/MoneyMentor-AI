"""
MoneyMentor AI — 2027 Edition
Parallel Multi-Agent · Supervisor Architecture · DuckDuckGo Web Search
"""

# ─── stdlib / third-party imports ────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
import json, time, urllib.parse, urllib.request, re, concurrent.futures
from typing import TypedDict, List, Dict, Any, Optional
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# ─── LangGraph ────────────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ─── Groq ─────────────────────────────────────────────────────────────────────
from groq import Groq

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MoneyMentor AI · 2027",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY  — best free Groq models per task
# ═══════════════════════════════════════════════════════════════════════════════
MODELS = {
    # DeepSeek R1 distill — chain-of-thought budget math
    "budget":     "deepseek-r1-distill-llama-70b",
    # Llama 3.3 70B versatile — best general finance + risk
    "risk":       "llama-3.3-70b-versatile",
    # Llama 4 Scout — long-context investment synthesis
    "investment": "meta-llama/llama-4-scout-17b-16e-instruct",
    # Llama 3.3 70B — research + web data synthesis
    "research":   "llama-3.3-70b-versatile",
    # Gemma 2 9B IT — fast CFO supervisor synthesis
    "supervisor": "gemma2-9b-it",
    # Llama 3.3 70B — conversational advisor
    "chat":       "llama-3.3-70b-versatile",
}


# ═══════════════════════════════════════════════════════════════════════════════
# THEME — fully synced with session_state toggle
# ═══════════════════════════════════════════════════════════════════════════════
def _is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


def inject_css():
    dark = _is_dark()
    if dark:
        bg        = "#06060f"
        bg_card   = "#0e0e1e"
        bg_card2  = "#12122a"
        border    = "#1e1e40"
        tp        = "#e8e8ff"
        ts        = "#8080bb"
        a1        = "#7b2fff"
        a2        = "#00f5c4"
        a3        = "#ff3cac"
        a4        = "#ffbe0b"
        glow1     = "rgba(123,47,255,0.4)"
        glow3     = "rgba(255,60,172,0.3)"
        sidebar   = "#080812"
        metric_bg = "#10102a"
        tag_bg    = "rgba(123,47,255,0.15)"
    else:
        bg        = "#f2f0ff"
        bg_card   = "#ffffff"
        bg_card2  = "#f5f3ff"
        border    = "#ddd8ff"
        tp        = "#1a0a3d"
        ts        = "#5a4a8a"
        a1        = "#6200ee"
        a2        = "#00b894"
        a3        = "#c2006e"
        a4        = "#e09000"
        glow1     = "rgba(98,0,238,0.15)"
        glow3     = "rgba(194,0,110,0.15)"
        sidebar   = "#eae6ff"
        metric_bg = "#f3f0ff"
        tag_bg    = "rgba(98,0,238,0.08)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── ROOT VARS ── */
:root {{
  --bg:        {bg};
  --bg-card:   {bg_card};
  --bg-card2:  {bg_card2};
  --border:    {border};
  --tp:        {tp};
  --ts:        {ts};
  --a1:        {a1};
  --a2:        {a2};
  --a3:        {a3};
  --a4:        {a4};
  --glow1:     {glow1};
  --glow3:     {glow3};
  --metric-bg: {metric_bg};
  --tag-bg:    {tag_bg};
}}

/* ── FORCE STREAMLIT THEME SYNC ── */
.stApp, html, body,
[data-testid="stApp"],
[data-testid="stMain"],
[data-testid="stAppViewContainer"],
section[data-testid="stSidebar"] ~ div,
.main .block-container,
.css-1d391kg, .css-18e3th9 {{
  background-color: {bg} !important;
  background: {bg} !important;
  color: {tp} !important;
}}
/* Remove Streamlit's own background injection */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
  background-color: {sidebar} !important;
  background: {sidebar} !important;
  border-right: 1px solid {border} !important;
}}
/* Override all Streamlit theme surfaces */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="base-input"],
[data-baseweb="input"] input {{
  background-color: {bg_card} !important;
  color: {tp} !important;
  border-color: {border} !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 13px !important;
}}
[data-baseweb="select"] > div {{
  background-color: {bg_card} !important;
  color: {tp} !important;
  border-color: {border} !important;
}}
/* All text */
p, span, div, li, td, th, label, h1, h2, h3, h4, h5 {{
  color: {tp};
  font-family: 'DM Sans', sans-serif;
}}
h1, h2, h3, h4 {{ font-family: 'Syne', sans-serif; }}
.stMarkdown p, .stMarkdown li {{ color: {tp} !important; }}
.stRadio label, .stCheckbox label {{ color: {ts} !important; font-size:13px !important; }}
/* Streamlit default text overrides */
[data-testid="stMarkdownContainer"] * {{ color: {tp} !important; }}

/* ── BUTTONS ── */
.stButton > button {{
  background: linear-gradient(135deg, {a1}, {a3}) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 12px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 0 16px {glow1} !important;
  padding: 0.5rem 1.2rem !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 0 28px {glow1}, 0 0 14px {glow3} !important;
}}
/* Number input step arrows — don't make them gradient */
.stNumberInput button {{
  background: {bg_card2} !important;
  border: 1px solid {border} !important;
  color: {ts} !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  text-transform: none !important;
  font-size: 14px !important;
  min-width: 28px !important;
}}

/* ── METRICS ── */
[data-testid="metric-container"] {{
  background: {metric_bg} !important;
  border: 1px solid {border} !important;
  border-radius: 16px !important;
  padding: 1rem !important;
}}
[data-testid="stMetricValue"] {{
  font-family: 'Space Mono', monospace !important;
  font-size: 1.3rem !important;
  color: {a1} !important;
}}
[data-testid="stMetricLabel"] {{
  font-family: 'Syne', sans-serif !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: {ts} !important;
}}
[data-testid="stMetricDelta"] {{
  font-family: 'Space Mono', monospace !important;
  font-size: 11px !important;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
  background: {bg_card} !important;
  border-radius: 14px !important;
  padding: 4px !important;
  border: 1px solid {border} !important;
  gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 11px !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: {ts} !important;
  border-radius: 10px !important;
  border: none !important;
  background: transparent !important;
  padding: 8px 18px !important;
  transition: all 0.2s ease !important;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, {a1}, {a3}) !important;
  color: #fff !important;
}}

/* ── EXPANDER ── */
.streamlit-expanderHeader {{
  background: {bg_card} !important;
  border: 1px solid {border} !important;
  border-radius: 12px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  color: {tp} !important;
  font-size: 13px !important;
}}
.streamlit-expanderContent {{
  background: {bg_card2} !important;
  border: 1px solid {border} !important;
  border-top: none !important;
  border-radius: 0 0 12px 12px !important;
}}

/* ── FORM ── */
[data-testid="stForm"] {{
  background: {bg_card} !important;
  border: 1px solid {border} !important;
  border-radius: 20px !important;
  padding: 1.5rem !important;
}}

/* ── ALERTS ── */
.stAlert, [data-testid="stAlert"] {{
  background: {bg_card} !important;
  border: 1px solid {border} !important;
  border-radius: 12px !important;
  color: {tp} !important;
}}
.stAlert p {{ color: {tp} !important; }}

/* ── PROGRESS ── */
.stProgress > div > div > div {{
  background: linear-gradient(90deg, {a1}, {a2}) !important;
  border-radius: 99px !important;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {bg}; }}
::-webkit-scrollbar-thumb {{ background: {a1}; border-radius: 5px; }}
hr {{ border-color: {border} !important; opacity:1 !important; }}

/* ── SPINNER ── */
.stSpinner > div {{ border-top-color: {a1} !important; }}

/* ── CUSTOM COMPONENTS ── */
.hero-badge {{
  display:inline-flex; align-items:center; gap:6px;
  background:{tag_bg}; border:1px solid {a1};
  border-radius:99px; padding:4px 14px;
  font-family:'Space Mono',monospace; font-size:10px;
  color:{a1}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;
}}
.hero-title {{
  font-family:'Syne',sans-serif; font-weight:800;
  font-size:clamp(2rem,5vw,2.9rem); line-height:1.1;
  background:linear-gradient(135deg,{a1} 0%,{a3} 55%,{a2} 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; margin-bottom:8px;
}}
.hero-sub {{
  font-family:'DM Sans',sans-serif; font-size:14px;
  color:{ts}; margin-bottom:20px; max-width:560px; line-height:1.6;
}}
.sec-lbl {{
  font-family:'Space Mono',monospace; font-size:10px;
  letter-spacing:0.18em; text-transform:uppercase; color:{a2}; margin-bottom:4px;
}}
.sec-ttl {{
  font-family:'Syne',sans-serif; font-weight:700;
  font-size:1.15rem; color:{tp}; margin-bottom:12px;
}}
.agent-card {{
  background:{bg_card}; border:1px solid {border};
  border-radius:16px; padding:1.1rem 1.3rem; margin-bottom:10px;
  position:relative; overflow:hidden;
}}
.agent-card::before {{
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,{a1},{a3});
}}
.agent-name {{
  font-family:'Syne',sans-serif; font-weight:700;
  font-size:13px; color:{tp}; margin-bottom:3px;
}}
.agent-role {{
  font-family:'Space Mono',monospace; font-size:9px;
  color:{a2}; text-transform:uppercase; letter-spacing:0.1em;
}}
.agent-output {{
  font-family:'DM Sans',sans-serif; font-size:13.5px;
  color:{ts}; margin-top:10px; line-height:1.65;
  border-left:3px solid {a1}; padding-left:12px;
}}
.sdot {{
  display:inline-block; width:7px; height:7px; border-radius:50%;
  background:{a2}; box-shadow:0 0 8px {a2};
  margin-right:6px; animation:sdot-pulse 2s infinite;
}}
@keyframes sdot-pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
.tip-card {{
  background:{bg_card}; border:1px solid {border};
  border-radius:14px; padding:0.9rem 1.1rem; margin-bottom:9px;
  display:flex; gap:12px; align-items:flex-start;
}}
.tip-icon {{ font-size:20px; flex-shrink:0; }}
.tip-text {{ font-family:'DM Sans',sans-serif; font-size:14px; color:{tp}; line-height:1.6; }}
.tip-warning  {{ border-left:4px solid {a4}; }}
.tip-success  {{ border-left:4px solid {a2}; }}
.tip-insight  {{ border-left:4px solid {a1}; }}
.tip-danger   {{ border-left:4px solid {a3}; }}
.chat-user {{
  background:linear-gradient(135deg,{a1},{a3}); color:#fff;
  border-radius:18px 18px 4px 18px; padding:10px 16px; margin-bottom:8px;
  font-family:'DM Sans',sans-serif; font-size:14px; max-width:78%; margin-left:auto;
}}
.chat-ai {{
  background:{bg_card}; border:1px solid {border}; color:{tp};
  border-radius:18px 18px 18px 4px; padding:10px 16px; margin-bottom:8px;
  font-family:'DM Sans',sans-serif; font-size:14px; max-width:85%;
}}
.watermark {{
  font-family:'Space Mono',monospace; font-size:9px; color:{ts};
  opacity:0.4; text-align:center; padding:12px 0; letter-spacing:0.1em;
}}
.flow-step {{
  display:flex; align-items:center; gap:8px; padding:7px 12px;
  background:{bg_card2}; border-radius:10px; border:1px solid {border};
  margin-bottom:5px; font-family:'Space Mono',monospace;
  font-size:11px; color:{ts};
}}
.flow-step.active {{
  border-color:{a1}; color:{a1}; background:{tag_bg};
}}
.web-card {{
  background:{bg_card}; border:1px solid {border}; border-left:4px solid {a2};
  border-radius:12px; padding:10px 14px; margin-bottom:8px;
  font-family:'DM Sans',sans-serif; font-size:13px; color:{tp}; line-height:1.6;
}}
.web-src {{
  font-family:'Space Mono',monospace; font-size:9px; color:{a2};
  letter-spacing:0.08em; margin-top:4px; opacity:0.8;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDUCKGO WEB SEARCH  (zero API key)
# ═══════════════════════════════════════════════════════════════════════════════
def ddg_search(query: str, max_results: int = 4) -> List[Dict]:
    results = []
    try:
        q   = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
        titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>',        html, re.S)
        urls_raw = re.findall(r'class="result__url"[^>]*>(.*?)</span>',   html, re.S)
        for i, snip in enumerate(snippets[:max_results]):
            clean = re.sub(r'<[^>]+>', '', snip).strip()
            title = re.sub(r'<[^>]+>', '', titles[i]).strip()   if i < len(titles)   else ""
            url_  = urls_raw[i].strip()                          if i < len(urls_raw) else ""
            if clean:
                results.append({"title": title, "snippet": clean, "url": url_})
    except Exception as e:
        results.append({"title": "DDG unavailable", "snippet": str(e)[:120], "url": ""})
    return results


def fmt_search(results: List[Dict]) -> str:
    return "\n".join(
        f"• [{r['title']}] {r['snippet']} ({r['url']})" for r in results
    ) or "No results."


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ CALL HELPER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_groq(key: str) -> Groq:
    return Groq(api_key=key)


def call_groq(client: Groq, system: str, user: str,
              model: str, max_tokens: int = 512, temp: float = 0.6) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temp,
        )
        text = r.choices[0].message.content or ""
        # Strip DeepSeek R1 <think>…</think> reasoning traces
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        return text
    except Exception as e:
        return f"[{model} error: {str(e)[:120]}]"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════
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
    messages:          List[Dict]


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL AGENT FUNCTIONS  (pure, no Streamlit calls — safe for threads)
# ═══════════════════════════════════════════════════════════════════════════════

def agent_budget(state: AgentState, client: Groq) -> str:
    total = sum(state["expenses"].values())
    net   = state["income"] - total
    sr    = (net / state["income"] * 100) if state["income"] > 0 else 0
    needs = sum(v for k,v in state["expenses"].items()
                if k in ["Rent","Utilities","Groceries","Transport"])
    wants = sum(v for k,v in state["expenses"].items()
                if k in ["Dining","Entertainment","Shopping"])
    prompt = (
        f"Analyse this Indian household budget:\n"
        f"Income: Rs{state['income']:,.0f}/mo\n"
        f"Expenses: {json.dumps(state['expenses'])}\n"
        f"Net: Rs{net:,.0f} | Savings rate: {sr:.1f}%\n"
        f"50/30/20 -> Needs Rs{needs:,.0f} | Wants Rs{wants:,.0f} | Surplus Rs{net:,.0f}\n\n"
        "Give exactly 4 bullet-point insights with Rs figures. "
        "Identify the single biggest spending inefficiency. Be surgical."
    )
    return call_groq(client,
        "You are a precision budget AI. Bullets only. Rs signs. No preamble.",
        prompt, model=MODELS["budget"], max_tokens=400)


def agent_risk(state: AgentState, client: Groq) -> str:
    total  = sum(state["expenses"].values())
    runway = state["savings"] / total if total > 0 else 0
    er     = total / state["income"] * 100 if state["income"] > 0 else 0
    em_gap = max(0, total * 6 - state["savings"])
    prompt = (
        f"Financial Risk Report:\n"
        f"Expense-to-income: {er:.1f}%\n"
        f"Emergency runway: {runway:.1f} months (gap: Rs{em_gap:,.0f} to 6mo fund)\n"
        f"Monthly burn: Rs{total:,.0f} | Savings: Rs{state['savings']:,.0f}\n"
        f"Goal: Rs{state.get('target') or 'none'} in {state.get('timeline') or '?'} months\n\n"
        "Rate risk: CRITICAL/HIGH/MEDIUM/LOW. List 3 specific vulnerabilities with Rs impact. "
        "1 immediate mitigation. Bullets."
    )
    return call_groq(client,
        "You are a financial risk specialist. Blunt and data-driven. Rs signs.",
        prompt, model=MODELS["risk"], max_tokens=340)


def agent_investment(state: AgentState, client: Groq) -> str:
    total      = sum(state["expenses"].values())
    investable = max(0, state["income"] - total)
    prompt = (
        f"Investment plan for India 2025:\n"
        f"Investable surplus: Rs{investable:,.0f}/month\n"
        f"Current corpus: Rs{state['savings']:,.0f}\n"
        f"Goal: Rs{state.get('target') or 'wealth creation'} "
        f"in {state.get('timeline') or 'open'} months\n\n"
        "Give exact % and Rs allocations across: Liquid fund, ELSS SIP, "
        "Nifty 50 index SIP, PPF, NPS, FD. Name 1 specific fund per category. No generic advice."
    )
    return call_groq(client,
        "You are a SEBI-registered advisor for India. Name specific funds. Rs signs.",
        prompt, model=MODELS["investment"], max_tokens=440)


def agent_web_research(state: AgentState, client: Groq) -> tuple:
    """Returns (llm_summary: str, raw_results: list)"""
    total = sum(state["expenses"].values())
    sr    = ((state["income"] - total) / state["income"] * 100) if state["income"] > 0 else 0
    queries = [
        "RBI repo rate India 2025 savings account interest",
        "best SIP mutual funds India 2025 top performers",
        "India CPI inflation 2025 household expenses impact",
    ]
    all_raw    = []
    web_ctx    = ""
    for q in queries:
        res = ddg_search(q, max_results=3)
        all_raw.extend(res)
        web_ctx += f"\n[Query: {q}]\n{fmt_search(res)}\n"

    prompt = (
        f"User: Rs{state['income']:,.0f}/mo income, {sr:.1f}% savings rate, India.\n\n"
        f"Live web data:\n{web_ctx}\n\n"
        "Synthesise 4 crisp macro insights relevant to this user. "
        "Cite actual figures from the web data. Bullets."
    )
    summary = call_groq(client,
        "You are a macro-finance analyst for India. Cite real web numbers. Rs signs.",
        prompt, model=MODELS["research"], max_tokens=380)
    return summary, all_raw


def agent_supervisor(state: AgentState, client: Groq) -> str:
    prompt = (
        "You are the Chief Financial Officer AI. Synthesise four expert reports:\n\n"
        f"BUDGET:\n{state['budget_analysis']}\n\n"
        f"RISK:\n{state['risk_assessment']}\n\n"
        f"INVESTMENT:\n{state['investment_advice']}\n\n"
        f"MACRO RESEARCH:\n{state['web_insights']}\n\n"
        f"User: Rs{state['income']:,.0f}/mo income | "
        f"Savings Rs{state['savings']:,.0f} | "
        f"Goal Rs{state.get('target') or 'none'}\n\n"
        "Create a numbered 30-day / 60-day / 90-day execution plan. "
        "Each phase: 3 specific actions with exact Rs amounts. "
        "End with ONE BOLD MOVE for this month in CAPS. No fluff."
    )
    return call_groq(client,
        "You are a master CFP. Be decisive. Rs signs. Numbered list per phase.",
        prompt, model=MODELS["supervisor"], max_tokens=620, temp=0.55)


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION ENGINE  (ThreadPoolExecutor — fixes the LangGraph stream bug)
# ═══════════════════════════════════════════════════════════════════════════════

def run_parallel_agents(state: AgentState, client: Groq, status_slots: Dict[str, Any], progress_bar: Any) -> AgentState:
    completed = [0]
    ctx = get_script_run_ctx()

    def _run(name: str, fn, *args):
        add_script_run_ctx(threading.current_thread(), ctx)
        status_slots[name].markdown(f'<div class="flow-step active"><span class="sdot"></span>{name.title()} Agent · Running…</div>', unsafe_allow_html=True)
        result = fn(*args)
        completed[0] += 1
        progress_bar.progress(min(0.8, completed[0] / 4 * 0.8))
        status_slots[name].markdown(f'<div class="flow-step active" style="color:var(--a2)">✓ {name.title()} Agent · Done</div>', unsafe_allow_html=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_budget = ex.submit(_run, "budget", agent_budget, state, client)
        f_risk = ex.submit(_run, "risk", agent_risk, state, client)
        f_invest = ex.submit(_run, "investment", agent_investment, state, client)
        f_web = ex.submit(_run, "web", agent_web_research, state, client)

        budget_out = f_budget.result()
        risk_out = f_risk.result()
        investment_out = f_invest.result()
        web_out, web_raw = f_web.result()

    state = {**state, "budget_analysis": budget_out, "risk_assessment": risk_out, "investment_advice": investment_out, "web_insights": web_out, "web_raw": web_raw}

    progress_bar.progress(0.85)
    status_slots["supervisor"].markdown('<div class="flow-step active"><span class="sdot"></span>Supervisor CFO · Synthesising…</div>', unsafe_allow_html=True)
    final = agent_supervisor(state, client)
    progress_bar.progress(1.0)
    status_slots["supervisor"].markdown('<div class="flow-step active" style="color:var(--a2)">✓ Supervisor CFO · Plan Ready</div>', unsafe_allow_html=True)

    return {**state, "final_report": final, "messages": [{"role": "budget_analyst", "content": budget_out}, {"role": "risk_assessor", "content": risk_out}, {"role": "investment_advisor", "content": investment_out}, {"role": "web_researcher", "content": web_out}, {"role": "supervisor", "content": final}]}


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════

def chat_with_advisor(client: Groq, user_msg: str,
                      ctx: Dict, history: List[Dict]) -> str:
    exp_total = sum(ctx.get("expenses", {}).values())
    system = (
        "You are MoneyMentor, a 2027-era AI financial advisor for India. "
        "Sharp, warm, data-driven. Know Indian personal finance inside-out. "
        f"User context: Rs{ctx.get('income',0):,.0f}/mo income, "
        f"Rs{exp_total:,.0f} expenses, Rs{ctx.get('savings',0):,.0f} savings. "
        "Concise replies. Use Rs signs. Lead with emoji. Max 3 sentences."
    )
    msgs = [{"role": "system", "content": system}]
    for h in history[-8:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_msg})
    try:
        r = client.chat.completions.create(
            model=MODELS["chat"], messages=msgs,
            max_tokens=280, temperature=0.72)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def get_rule_tips(income, expenses, savings, target, timeline):
    tips  = []
    total = sum(expenses.values())
    net   = income - total

    if net < 0:
        tips.append({"type":"danger","icon":"🚨","text":
            f"Critical overspend: Rs{abs(net):,.0f}/month deficit. Cut discretionary spend immediately."})
    else:
        sr = (net / income * 100) if income > 0 else 0
        if sr < 10:
            tips.append({"type":"warning","icon":"⚠️","text":
                f"Savings rate {sr:.1f}% — dangerously low. Target 20%+ (gap: Rs{income*0.2-net:,.0f}/mo)."})
        elif sr < 20:
            tips.append({"type":"warning","icon":"📊","text":
                f"Savings rate {sr:.1f}% — below 20% benchmark. Close Rs{income*0.2-net:,.0f}/mo gap."})
        else:
            tips.append({"type":"success","icon":"✅","text":
                f"Excellent {sr:.1f}% savings rate — above benchmark. Start compounding surplus."})

    if expenses:
        top = max(expenses, key=expenses.get)
        pct = (expenses[top] / income * 100) if income > 0 else 0
        tips.append({"type":"insight","icon":"🔍","text":
            f"Largest expense: '{top}' at Rs{expenses[top]:,.0f} ({pct:.0f}% of income). Audit this."})

    runway = savings / total if total > 0 else 0
    em_tgt = total * 6
    if runway < 3:
        tips.append({"type":"warning","icon":"🛡️","text":
            f"Only {runway:.1f}-month runway. Build 6-month fund (Rs{em_tgt:,.0f}) before investing."})
    elif runway < 6:
        tips.append({"type":"insight","icon":"🛡️","text":
            f"{runway:.1f}-month buffer — good. Top up to 6 months (Rs{em_tgt:,.0f})."})
    else:
        tips.append({"type":"success","icon":"🛡️","text":
            f"Strong {runway:.1f}-month emergency fund. Deploy surplus to wealth creation."})

    if target and timeline and timeline > 0:
        needed  = target - savings
        monthly = needed / timeline
        if monthly <= net:
            tips.append({"type":"success","icon":"🎯","text":
                f"Goal on track: save Rs{monthly:,.0f}/mo -> Rs{target:,.0f} in {timeline} months. "
                f"Rs{net-monthly:,.0f}/mo left over."})
        else:
            tips.append({"type":"danger","icon":"🎯","text":
                f"Goal shortfall: need Rs{monthly:,.0f}/mo but only Rs{net:,.0f} free. "
                f"Gap: Rs{monthly-net:,.0f}/mo."})
    return tips


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def health_score(income, expenses, savings, target, timeline) -> int:
    total = sum(expenses.values())
    net   = income - total
    score = 50
    if income > 0:
        score += min(25, (net / income) * 80)
        score -= max(0, (total / income - 0.7) * 60)
    runway = savings / total if total > 0 else 0
    score += min(15, runway * 3)
    if target and timeline and timeline > 0:
        monthly = (target - savings) / timeline
        if net >= monthly:
            score += 10
        else:
            score -= min(15, (monthly - net) / (monthly + 1) * 15)
    return max(0, min(100, round(score)))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS  (theme-aware via _is_dark())
# ═══════════════════════════════════════════════════════════════════════════════
_PALETTE = ["#7b2fff","#00f5c4","#ff3cac","#ffbe0b","#00b4d8","#ff6b35","#c77dff","#4cc9f0"]


def chart_donut(expenses):
    dark  = _is_dark()
    pairs = [(k, v) for k, v in expenses.items() if v > 0]
    if not pairs:
        return None
    labels, values = zip(*pairs)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=_PALETTE[:len(labels)],
                    line=dict(color="#06060f" if dark else "#f2f0ff", width=3)),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Rs%{value:,.0f} · %{percent}<extra></extra>',
    ))
    fig.update_layout(
        showlegend=True, margin=dict(t=8,b=8,l=8,r=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=270,
        legend=dict(font=dict(family="Space Mono", size=9,
                   color="#e8e8ff" if dark else "#1a0a3d"),
                   bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_waterfall(income, expenses):
    dark = _is_dark()
    tc   = "#e8e8ff" if dark else "#1a0a3d"
    gc   = "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.04)"
    cats = list(expenses.keys())
    vals = [-v for v in expenses.values()]
    fig  = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(cats) + ["total"],
        x=["Income"] + cats + ["Net"],
        y=[income] + vals + [0],
        connector=dict(line=dict(color="#7b2fff", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#00f5c4")),
        decreasing=dict(marker=dict(color="#ff3cac")),
        totals=dict(marker=dict(color="#7b2fff")),
        texttemplate="Rs%{y:,.0f}", textposition="outside",
        textfont=dict(family="Space Mono", size=8, color=tc),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(t=8,b=8,l=8,r=8), showlegend=False,
        xaxis=dict(tickfont=dict(family="Space Mono",size=8,color=tc), gridcolor=gc),
        yaxis=dict(tickfont=dict(family="Space Mono",size=8,color=tc),
                   tickprefix="Rs", gridcolor=gc),
    )
    return fig


def chart_gauge(score):
    dark  = _is_dark()
    color = "#00f5c4" if score >= 70 else "#ffbe0b" if score >= 45 else "#ff3cac"
    tc    = "#8888cc" if dark else "#5a4a8a"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=tc,
                      tickfont=dict(family="Space Mono", size=9)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,45],  color="rgba(255,60,172,0.07)"),
                dict(range=[45,70], color="rgba(255,190,11,0.07)"),
                dict(range=[70,100],color="rgba(0,245,196,0.07)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
        number=dict(font=dict(family="Space Mono", size=42, color=color)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=185,
        margin=dict(t=6,b=6,l=12,r=12),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Session init ──────────────────────────────────────────────────────────
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

    inject_css()   # always called after session_state is ready

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:8px 0 16px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:10px;
                 color:var(--a2);letter-spacing:0.15em;text-transform:uppercase;
                 margin-bottom:5px;">⬡ MoneyMentor AI</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;
                 background:linear-gradient(135deg,var(--a1),var(--a3));
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">Financial OS · 2027</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Theme toggle ──────────────────────────────────────────────────────
        # Uses a unique key to avoid widget conflicts on rerun
        dark_label = "☀️  Light Mode" if st.session_state.dark_mode else "🌙  Dark Mode"
        if st.button(dark_label, key="theme_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("---")

        # API Key
        st.markdown('<div class="sec-lbl">API Key</div>', unsafe_allow_html=True)
        if 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
            st.markdown(
                '<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                'color:var(--a2);">✓ Loaded from secrets</div>',
                unsafe_allow_html=True)
        else:
            api_key = st.text_input(
                "Groq API Key", type="password", placeholder="gsk_…",
                key="api_key_input", label_visibility="collapsed")
            if not api_key:
                st.warning("Enter Groq API key to activate agents.")
                st.stop()

        st.markdown("---")

        # Agent roster
        st.markdown('<div class="sec-lbl">Agent Roster</div>', unsafe_allow_html=True)
        roster = [
            ("⬡", "Budget Analyst",    "DeepSeek-R1-Distill-70B", "Chain-of-thought math"),
            ("⬡", "Risk Assessor",     "Llama-3.3-70B",           "Risk mapping"),
            ("⬡", "Investment Advisor","Llama-4-Scout-17B",        "Portfolio allocation"),
            ("⬡", "Web Researcher",    "DDG + Llama-3.3-70B",     "Live market data"),
            ("⬡", "Supervisor CFO",    "Gemma2-9B-IT",            "90-day synthesis"),
        ]
        for icon, name, mdl, desc in roster:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;
                 padding:7px 10px;margin-bottom:5px;
                 background:var(--bg-card);border:1px solid var(--border);
                 border-radius:10px;">
                <span style="color:var(--a1);font-size:14px;margin-top:2px;">{icon}</span>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         font-size:12px;color:var(--tp);">{name}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:9px;
                         color:var(--a2);letter-spacing:0.05em;">{mdl}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:11px;
                         color:var(--ts);">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:9px;
             color:var(--ts);letter-spacing:0.05em;line-height:2;">
            ⬡ 4 Agents parallel (ThreadPool)<br>
            ⬡ Supervisor synthesises all<br>
            ⬡ DuckDuckGo live search<br>
            ⬡ 5 different LLM models<br>
            ⬡ Zero extra API keys needed
        </div>
        """, unsafe_allow_html=True)

    # ── Groq client ───────────────────────────────────────────────────────────
    try:
        groq_client = init_groq(api_key)
    except Exception as e:
        st.error(f"Groq init failed: {e}")
        st.stop()

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-badge">
        <span class="sdot"></span>5 Agents · Parallel · DuckDuckGo · Multi-Model
    </div>
    <div class="hero-title">MoneyMentor AI</div>
    <div class="hero-sub">
        Agentic finance OS — four AI agents run simultaneously, each powered by a
        different best-in-class Groq model, then a Supervisor CFO synthesises
        your personalised 30/60/90-day money plan backed by live web data.
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "⬡  FINANCIAL INPUT",
        "⬡  AGENT ANALYSIS",
        "⬡  AI ADVISOR CHAT",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — INPUT
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="sec-lbl">Financial Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-ttl">Enter Your Monthly Financials</div>', unsafe_allow_html=True)

        with st.form("finance_form"):
            st.markdown('<div class="sec-lbl">Monthly Income</div>', unsafe_allow_html=True)
            income = st.number_input(
                "income_val", min_value=0.0, step=500.0, value=45000.0,
                label_visibility="collapsed")

            st.markdown('<div class="sec-lbl" style="margin-top:12px;">Monthly Expenses</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                rent  = st.number_input("🏠 Rent / Mortgage (Rs)", min_value=0.0, step=100.0, value=14000.0)
                util  = st.number_input("⚡ Utilities (Rs)",        min_value=0.0, step=50.0,  value=2500.0)
                groc  = st.number_input("🛒 Groceries (Rs)",        min_value=0.0, step=50.0,  value=5000.0)
                trans = st.number_input("🚗 Transport (Rs)",        min_value=0.0, step=50.0,  value=3000.0)
            with c2:
                dine  = st.number_input("🍜 Dining Out (Rs)",       min_value=0.0, step=50.0,  value=2500.0)
                entmt = st.number_input("🎮 Entertainment (Rs)",    min_value=0.0, step=50.0,  value=1500.0)
                shop  = st.number_input("🛍️ Shopping (Rs)",         min_value=0.0, step=50.0,  value=2000.0)
                other = st.number_input("📦 Other (Rs)",            min_value=0.0, step=50.0,  value=1000.0)

            expenses = {
                "Rent": rent, "Utilities": util, "Groceries": groc,
                "Transport": trans, "Dining": dine, "Entertainment": entmt,
                "Shopping": shop, "Other": other,
            }

            st.markdown('<div class="sec-lbl" style="margin-top:12px;">Savings & Goals</div>',
                        unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                savings = st.number_input("💰 Current Savings (Rs)", min_value=0.0,
                                          step=1000.0, value=50000.0)
            with c4:
                has_goal = st.radio("🎯 Savings Goal?", ("Yes", "No"),
                                    index=1, horizontal=True)

            target = timeline = None
            if has_goal == "Yes":
                c5, c6 = st.columns(2)
                with c5:
                    target   = st.number_input("Target Amount (Rs)", min_value=0.0,
                                               step=1000.0, value=200000.0)
                with c6:
                    timeline = st.number_input("Timeline (months)",  min_value=1,
                                               step=1, value=18)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "⬡  LAUNCH 5-AGENT PARALLEL ANALYSIS", use_container_width=True)

        if submitted:
            st.session_state.financial_data = {
                "income": income, "expenses": expenses,
                "savings": savings, "target": target, "timeline": timeline,
            }

            init_state: AgentState = {
                "income": income, "expenses": expenses,
                "savings": savings, "target": target, "timeline": timeline,
                "budget_analysis": "", "risk_assessment": "",
                "investment_advice": "", "web_insights": "",
                "web_raw": [], "final_report": "", "messages": [],
            }

            st.markdown("---")
            st.markdown('<div class="sec-lbl">Agent Pipeline · Live Execution</div>',
                        unsafe_allow_html=True)

            # Status placeholders (2-column layout, no rerun conflicts)
            col_a, col_b = st.columns(2)
            with col_a:
                s_budget = st.empty()
                s_risk   = st.empty()
            with col_b:
                s_invest = st.empty()
                s_web    = st.empty()
            s_super = st.empty()
            prog    = st.progress(0.0)

            # Init all to "queued"
            for slot, label in [
                (s_budget,  "Budget"),
                (s_risk,    "Risk"),
                (s_invest,  "Investment"),
                (s_web,     "Web"),
            ]:
                slot.markdown(
                    f'<div class="flow-step">◌  {label} Agent · Queued</div>',
                    unsafe_allow_html=True)
            s_super.markdown(
                '<div class="flow-step">◌  Supervisor CFO · Waiting…</div>',
                unsafe_allow_html=True)

            status_slots = {
                "budget":     s_budget,
                "risk":       s_risk,
                "investment": s_invest,
                "web":        s_web,
                "supervisor": s_super,
            }

            prog.progress(0.05)
            result = run_parallel_agents(init_state, groq_client, status_slots, prog)

            st.session_state.agent_state   = result
            st.session_state.analysis_done = True
            st.success("✅ All 5 agents complete — see results in **AGENT ANALYSIS** tab")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        if not st.session_state.analysis_done or st.session_state.agent_state is None:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;">
                <div style="font-size:42px;margin-bottom:14px;opacity:0.25;">⬡</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.2rem;
                     font-weight:700;color:var(--tp);margin-bottom:8px;">No Analysis Yet</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:14px;color:var(--ts);">
                     Fill financials in INPUT → Launch 5-Agent Analysis.</div>
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
            total_e  = sum(expenses.values())
            net      = income - total_e
            sr       = (net / income * 100) if income > 0 else 0

            # ── Vitals ────────────────────────────────────────────────────
            st.markdown('<div class="sec-lbl">Financial Vitals</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Monthly Income", f"Rs{income:,.0f}")
            m2.metric("Total Expenses", f"Rs{total_e:,.0f}",
                      delta=f"-{total_e/income*100:.0f}% income" if income else None,
                      delta_color="inverse")
            m3.metric("Net Cash Flow",  f"Rs{net:,.0f}",
                      delta=f"{sr:.1f}% saved",
                      delta_color="normal" if net >= 0 else "inverse")
            m4.metric("Current Savings", f"Rs{savings:,.0f}")
            sc = health_score(income, expenses, savings, target, timeline)
            grade = ("Excellent" if sc>=80 else "Good" if sc>=65 else "Fair" if sc>=45 else "At Risk")
            m5.metric("Health Score", f"{sc} / 100", delta=grade)

            st.markdown("---")

            # ── Charts ────────────────────────────────────────────────────
            cg, cd, cw = st.columns([1, 1.4, 1.8])
            with cg:
                st.markdown('<div class="sec-lbl">Health Score</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_gauge(sc), use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                    f'text-transform:uppercase;letter-spacing:0.14em;text-align:center;'
                    f'color:var(--ts);margin-top:-10px;">{grade}</div>',
                    unsafe_allow_html=True)
            with cd:
                st.markdown('<div class="sec-lbl">Expense Breakdown</div>', unsafe_allow_html=True)
                fig_d = chart_donut(expenses)
                if fig_d:
                    st.plotly_chart(fig_d, use_container_width=True,
                                    config={"displayModeBar": False})
            with cw:
                st.markdown('<div class="sec-lbl">Cash Flow Waterfall</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_waterfall(income, expenses),
                                use_container_width=True, config={"displayModeBar": False})

            if target and target > 0:
                pct = min(1.0, savings / target)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:\'Space Mono\',monospace;font-size:10px;'
                    f'color:var(--ts);margin-bottom:4px;">'
                    f'<span>Goal: Rs{target:,.0f}</span>'
                    f'<span>{pct*100:.0f}% done · Rs{max(0,target-savings):,.0f} to go</span>'
                    f'</div>',
                    unsafe_allow_html=True)
                st.progress(pct)

            st.markdown("---")

            # ── Rule Tips ─────────────────────────────────────────────────
            st.markdown('<div class="sec-lbl">Instant Diagnostics</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-ttl">Rule-Based Analysis</div>', unsafe_allow_html=True)
            for tip in get_rule_tips(income, expenses, savings, target, timeline):
                st.markdown(
                    f'<div class="tip-card tip-{tip["type"]}">'
                    f'<div class="tip-icon">{tip["icon"]}</div>'
                    f'<div class="tip-text">{tip["text"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            # ── Parallel Agent Reports ─────────────────────────────────────
            st.markdown('<div class="sec-lbl">Parallel Agent Intelligence</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-ttl">Multi-Model Analysis Reports</div>', unsafe_allow_html=True)

            agents_out = [
                ("Budget Analyst",    "DeepSeek-R1-Distill-70B", "Income & Expense Diagnostics",
                 s.get("budget_analysis","")),
                ("Risk Assessor",     "Llama-3.3-70B",            "Risk & Vulnerability Report",
                 s.get("risk_assessment","")),
                ("Investment Advisor","Llama-4-Scout-17B",         "Portfolio Allocation Plan",
                 s.get("investment_advice","")),
                ("Web Researcher",    "DDG + Llama-3.3-70B",      "Live Macro & Market Data",
                 s.get("web_insights","")),
            ]
            a1c, a2c = st.columns(2)
            for i, (name, mdl, role, output) in enumerate(agents_out):
                with (a1c if i % 2 == 0 else a2c):
                    with st.expander(f"⬡  {name}", expanded=(i < 2)):
                        st.markdown(
                            f'<div class="agent-role">{role} · {mdl}</div>'
                            f'<div class="agent-output">'
                            f'{output.replace(chr(10),"<br>") if output else "No output."}'
                            f'</div>',
                            unsafe_allow_html=True)

            # ── Web Search Raw Results ─────────────────────────────────────
            web_raw = s.get("web_raw", [])
            if web_raw:
                st.markdown("---")
                st.markdown('<div class="sec-lbl">DuckDuckGo Live Data</div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-ttl">Real-Time Market Snippets</div>', unsafe_allow_html=True)
                for r in web_raw[:6]:
                    st.markdown(
                        f'<div class="web-card">'
                        f'<strong>{r.get("title","")}</strong><br>'
                        f'{r.get("snippet","")}'
                        f'<div class="web-src">↗ {r.get("url","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True)

            st.markdown("---")

            # ── Supervisor 90-Day Plan ─────────────────────────────────────
            st.markdown('<div class="sec-lbl">Supervisor CFO · Gemma2-9B-IT</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="sec-ttl">⬡ 30 / 60 / 90-Day Action Plan</div>',
                        unsafe_allow_html=True)
            final = s.get("final_report", "")
            if final:
                st.markdown(
                    f'<div class="agent-card">'
                    f'<div class="agent-name">Supervisor CFO — Master Execution Plan</div>'
                    f'<div class="agent-role">Gemma2-9B-IT · synthesised from 4 parallel agents + live data</div>'
                    f'<div class="agent-output">{final.replace(chr(10),"<br>")}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬡  Re-run Agent Analysis", key="rerun_btn"):
                st.session_state.analysis_done = False
                st.session_state.agent_state   = None
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — CHAT
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="sec-lbl">Conversational Finance AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-ttl">Ask MoneyMentor Anything</div>', unsafe_allow_html=True)

        ctx       = st.session_state.financial_data or {"income": 0, "expenses": {}, "savings": 0}
        exp_total = sum(ctx.get("expenses", {}).values())

        if st.session_state.financial_data:
            st.markdown(
                f'<div style="background:var(--bg-card);border:1px solid var(--border);'
                f'border-radius:12px;padding:10px 16px;margin-bottom:14px;'
                f'display:flex;gap:24px;flex-wrap:wrap;">'
                f'<div><div style="font-family:\'Space Mono\',monospace;font-size:9px;'
                f'color:var(--ts);text-transform:uppercase;letter-spacing:0.1em;">Income</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:14px;'
                f'font-weight:700;color:var(--a2);">Rs{ctx["income"]:,.0f}</div></div>'
                f'<div><div style="font-family:\'Space Mono\',monospace;font-size:9px;'
                f'color:var(--ts);text-transform:uppercase;letter-spacing:0.1em;">Expenses</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:14px;'
                f'font-weight:700;color:var(--a3);">Rs{exp_total:,.0f}</div></div>'
                f'<div><div style="font-family:\'Space Mono\',monospace;font-size:9px;'
                f'color:var(--ts);text-transform:uppercase;letter-spacing:0.1em;">Savings</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:14px;'
                f'font-weight:700;color:var(--a1);">Rs{ctx["savings"]:,.0f}</div></div>'
                f'</div>',
                unsafe_allow_html=True)
        else:
            st.info("💡 Add your financials in INPUT tab for context-aware answers.")

        # Quick suggestions
        st.markdown('<div class="sec-lbl" style="margin-bottom:8px;">Quick Questions</div>',
                    unsafe_allow_html=True)
        suggestions = [
            "Cut my biggest expense?",
            "Best SIPs for Rs5k/month?",
            "Emergency fund fast?",
            "Invest or pay debt first?",
        ]
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, q, idx in zip([sc1, sc2, sc3, sc4], suggestions, range(4)):
            with col:
                if st.button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    reply = chat_with_advisor(groq_client, q, ctx,
                                             st.session_state.chat_history[:-1])
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # History
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:6px;">'
                    f'<div class="chat-user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin-bottom:6px;">'
                    f'<div class="chat-ai">'
                    f'<span style="font-family:\'Space Mono\',monospace;font-size:9px;'
                    f'color:var(--a1);letter-spacing:0.1em;text-transform:uppercase;'
                    f'display:block;margin-bottom:4px;">⬡ MoneyMentor</span>'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input(
            "chat_in", placeholder="e.g. How to save Rs1L in 6 months?",
            key="chat_input", label_visibility="collapsed")

        csend, cclear = st.columns([5, 1])
        with csend:
            send = st.button("⬡  SEND", key="send_btn", use_container_width=True)
        with cclear:
            if st.button("Clear", key="clear_btn", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner(""):
                reply = chat_with_advisor(groq_client, user_input, ctx,
                                          st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="watermark">'
        'MONEYMENTOR AI · 2027 · 5 AGENTS · 4 PARALLEL · DUCKDUCKGO SEARCH · '
        'DEEPSEEK-R1 · LLAMA-4-SCOUT · LLAMA-3.3-70B · GEMMA2 · GROQ CLOUD<br>'
        'Not financial advice. Educational use only.'
        '</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
