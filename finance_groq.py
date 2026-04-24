"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MoneyMentor AI — 2027 Edition                                          ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Architecture:                                                           ║
║    • LangGraph StateGraph  — parallel fan-out (4 agents) → supervisor   ║
║    • ThreadPoolExecutor    — TRUE parallel execution, no stream unpack  ║
║    • Groq function-calling — web search tool (compound-beta / DDG)      ║
║    • Multi-model roster    — DeepSeek R1 · Llama 4 Scout · Llama 3.3   ║
║    • Theme sync            — single CSS injection overwrites Streamlit  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import concurrent.futures
import json
import operator
import re
import time
import urllib.parse
import urllib.request
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# ─── third-party ──────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ─── LangGraph ────────────────────────────────────────────────────────────────
try:
    from langgraph.graph import END, StateGraph
    _LG = True
except ImportError:
    _LG = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  ← must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MoneyMentor AI · 2027",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY  — best free Groq models per task (April 2026)
# ══════════════════════════════════════════════════════════════════════════════
MODELS: Dict[str, str] = {
    # Chain-of-thought budget math — DeepSeek R1 distil on Groq
    "budget":     "deepseek-r1-distill-llama-70b",
    # Broad financial risk analysis — Llama 3.3 70B versatile
    "risk":       "llama-3.3-70b-versatile",
    # Long-context investment synthesis — Llama 4 Scout 17B (128k ctx)
    "investment": "meta-llama/llama-4-scout-17b-16e-instruct",
    # Web-data synthesis + macro — Llama 3.3 70B
    "research":   "llama-3.3-70b-versatile",
    # Supervisor CFO synthesis — Llama 3.3 70B (strongest free general)
    "supervisor": "llama-3.3-70b-versatile",
    # Conversational advisor — Llama 3.3 70B
    "chat":       "llama-3.3-70b-versatile",
}

# ══════════════════════════════════════════════════════════════════════════════
#  THEME  — single inject_css() overwrites ALL Streamlit surfaces
#  Strategy: inject a monolithic <style> block with !important on every
#  element Streamlit could possibly paint.  Toggle sets st.session_state
#  and calls st.rerun() so CSS is regenerated fresh every toggle.
# ══════════════════════════════════════════════════════════════════════════════

def _dark() -> bool:
    return st.session_state.get("dark_mode", True)


def inject_css() -> None:
    d = _dark()
    if d:
        bg = "#06060f"; bgc = "#0d0d1e"; bgc2 = "#11112a"; bdr = "#1e1e42"
        tp = "#e8e8ff"; ts = "#8080bb"
        a1 = "#7b2fff"; a2 = "#00f5c4"; a3 = "#ff3cac"; a4 = "#ffbe0b"
        g1 = "rgba(123,47,255,.40)"; g3 = "rgba(255,60,172,.30)"
        sb = "#080812"; mb = "#10102a"; tb = "rgba(123,47,255,.15)"
        input_ph = "#555577"
    else:
        bg = "#f2f0ff"; bgc = "#ffffff"; bgc2 = "#f5f3ff"; bdr = "#ddd8ff"
        tp = "#1a0a3d"; ts = "#5a4a8a"
        a1 = "#6200ee"; a2 = "#00b894"; a3 = "#c2006e"; a4 = "#e09000"
        g1 = "rgba(98,0,238,.18)"; g3 = "rgba(194,0,110,.15)"
        sb = "#eae6ff"; mb = "#f3f0ff"; tb = "rgba(98,0,238,.08)"
        input_ph = "#aaaacc"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {{
  --bg:{bg}; --bgc:{bgc}; --bgc2:{bgc2}; --bdr:{bdr};
  --tp:{tp}; --ts:{ts};
  --a1:{a1}; --a2:{a2}; --a3:{a3}; --a4:{a4};
  --g1:{g1}; --g3:{g3};
  --mb:{mb}; --tb:{tb}; --sb:{sb};
}}

/* ── GLOBAL BACKGROUND ── */
html, body {{ background:{bg}!important; color:{tp}!important; }}
.stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .main .block-container,
div.stMainBlockContainer, div[data-testid="stVerticalBlock"] {{
  background:{bg}!important; color:{tp}!important;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {{
  background:{sb}!important; border-right:1px solid {bdr}!important;
}}
[data-testid="stSidebar"] * {{ color:{tp}!important; }}

/* ── ALL TEXT ── */
p,div,span,li,td,th,h1,h2,h3,h4,h5,h6,label,
.stMarkdown,[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {{
  color:{tp}!important; font-family:'DM Sans',sans-serif;
}}
h1,h2,h3,h4,h5 {{ font-family:'Syne',sans-serif!important; }}

/* ── INPUTS ── */
.stTextInput input,.stNumberInput input,.stTextArea textarea,
[data-baseweb="base-input"],[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
  background:{bgc}!important; color:{tp}!important;
  border:1px solid {bdr}!important; border-radius:10px!important;
  font-family:'Space Mono',monospace!important; font-size:13px!important;
  caret-color:{a1};
}}
.stTextInput input:focus,.stNumberInput input:focus {{
  border-color:{a1}!important; box-shadow:0 0 0 2px {g1}!important; outline:none!important;
}}
input::placeholder,textarea::placeholder {{ color:{input_ph}!important; }}
[data-baseweb="select"] > div {{
  background:{bgc}!important; border:1px solid {bdr}!important; color:{tp}!important;
}}
.stNumberInput button {{
  background:{bgc2}!important; border:1px solid {bdr}!important; color:{ts}!important;
  border-radius:8px!important; box-shadow:none!important;
  font-size:13px!important; min-width:26px!important;
}}
.stRadio label {{ color:{ts}!important; font-size:13px!important; }}
.stRadio [data-testid="stMarkdownContainer"] * {{ color:{ts}!important; }}
[data-testid="stWidgetLabel"] p {{ color:{ts}!important; }}

/* ── BUTTONS ── */
.stButton>button {{
  background:linear-gradient(135deg,{a1},{a3})!important;
  color:#fff!important; border:none!important; border-radius:12px!important;
  font-family:'Syne',sans-serif!important; font-weight:700!important;
  font-size:12px!important; letter-spacing:.06em!important;
  text-transform:uppercase!important; padding:.55rem 1.2rem!important;
  transition:all .25s ease!important; box-shadow:0 0 16px {g1}!important;
}}
.stButton>button:hover {{
  transform:translateY(-2px)!important;
  box-shadow:0 0 28px {g1},0 0 14px {g3}!important;
}}

/* ── METRICS ── */
[data-testid="metric-container"] {{
  background:{mb}!important; border:1px solid {bdr}!important;
  border-radius:16px!important; padding:1rem!important;
}}
[data-testid="stMetricValue"] {{
  font-family:'Space Mono',monospace!important; font-size:1.25rem!important; color:{a1}!important;
}}
[data-testid="stMetricLabel"] {{
  font-family:'Syne',sans-serif!important; font-size:10px!important;
  text-transform:uppercase!important; letter-spacing:.1em!important; color:{ts}!important;
}}
[data-testid="stMetricDelta"] {{
  font-family:'Space Mono',monospace!important; font-size:11px!important;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
  background:{bgc}!important; border-radius:14px!important; padding:4px!important;
  border:1px solid {bdr}!important; gap:2px!important;
}}
.stTabs [data-baseweb="tab"] {{
  font-family:'Syne',sans-serif!important; font-weight:700!important; font-size:11px!important;
  letter-spacing:.07em!important; text-transform:uppercase!important; color:{ts}!important;
  border-radius:10px!important; border:none!important; background:transparent!important;
  padding:8px 18px!important; transition:all .2s ease!important;
}}
.stTabs [aria-selected="true"] {{
  background:linear-gradient(135deg,{a1},{a3})!important; color:#fff!important;
}}
.stTabs [aria-selected="true"] p {{ color:#fff!important; }}

/* ── EXPANDERS ── */
[data-testid="stExpander"] summary, .streamlit-expanderHeader {{
  background:{bgc}!important; border:1px solid {bdr}!important;
  border-radius:12px!important; font-family:'Syne',sans-serif!important;
  font-weight:700!important; color:{tp}!important; font-size:13px!important;
}}
[data-testid="stExpander"] details[open] summary {{
  border-radius:12px 12px 0 0!important;
}}
.streamlit-expanderContent, [data-testid="stExpander"] .streamlit-expanderContent {{
  background:{bgc2}!important; border:1px solid {bdr}!important;
  border-top:none!important; border-radius:0 0 12px 12px!important;
}}

/* ── FORM ── */
[data-testid="stForm"] {{
  background:{bgc}!important; border:1px solid {bdr}!important;
  border-radius:20px!important; padding:1.4rem!important;
}}

/* ── ALERTS ── */
.stAlert,[data-testid="stAlert"],[data-testid="stAlertContainer"] {{
  background:{bgc}!important; border:1px solid {bdr}!important; border-radius:12px!important;
}}
.stAlert p,[data-testid="stAlertContainer"] p {{ color:{tp}!important; }}

/* ── PROGRESS ── */
.stProgress>div>div>div,[data-testid="stProgressBar"]>div {{
  background:linear-gradient(90deg,{a1},{a2})!important; border-radius:99px!important;
}}

/* ── MISC ── */
hr {{ border-color:{bdr}!important; opacity:1!important; }}
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:{bg}; }}
::-webkit-scrollbar-thumb {{ background:{a1}; border-radius:5px; }}
.stSpinner>div {{ border-top-color:{a1}!important; }}

/* ══ CUSTOM COMPONENTS ══ */
.hero-badge {{
  display:inline-flex;align-items:center;gap:6px;background:{tb};
  border:1px solid {a1};border-radius:99px;padding:4px 14px;
  font-family:'Space Mono',monospace;font-size:10px;color:{a1};
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;
}}
.hero-title {{
  font-family:'Syne',sans-serif;font-weight:800;
  font-size:clamp(2rem,5vw,2.9rem);line-height:1.1;
  background:linear-gradient(135deg,{a1} 0%,{a3} 55%,{a2} 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;margin-bottom:8px;
}}
.hero-sub {{
  font-family:'DM Sans',sans-serif;font-size:14px;color:{ts};
  max-width:580px;line-height:1.6;margin-bottom:18px;
}}
.sec-lbl {{
  font-family:'Space Mono',monospace;font-size:10px;
  letter-spacing:.18em;text-transform:uppercase;color:{a2};margin-bottom:4px;
}}
.sec-ttl {{
  font-family:'Syne',sans-serif;font-weight:700;
  font-size:1.15rem;color:{tp};margin-bottom:12px;
}}
.agent-card {{
  background:{bgc};border:1px solid {bdr};border-radius:16px;
  padding:1.1rem 1.3rem;margin-bottom:10px;position:relative;overflow:hidden;
}}
.agent-card::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,{a1},{a3});
}}
.agent-name {{
  font-family:'Syne',sans-serif;font-weight:700;font-size:13px;color:{tp};margin-bottom:3px;
}}
.agent-role {{
  font-family:'Space Mono',monospace;font-size:9px;
  color:{a2};text-transform:uppercase;letter-spacing:.1em;
}}
.agent-output {{
  font-family:'DM Sans',sans-serif;font-size:13.5px;color:{ts};
  margin-top:10px;line-height:1.65;border-left:3px solid {a1};padding-left:12px;
}}
.sdot {{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:{a2};box-shadow:0 0 8px {a2};
  margin-right:6px;animation:sdot-pulse 2s infinite;
}}
@keyframes sdot-pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}
.tip-card {{
  background:{bgc};border:1px solid {bdr};border-radius:14px;
  padding:.9rem 1.1rem;margin-bottom:9px;display:flex;gap:12px;align-items:flex-start;
}}
.tip-icon {{ font-size:20px;flex-shrink:0; }}
.tip-text {{ font-family:'DM Sans',sans-serif;font-size:13.5px;color:{tp};line-height:1.6; }}
.tip-warning  {{ border-left:4px solid {a4}; }}
.tip-success  {{ border-left:4px solid {a2}; }}
.tip-insight  {{ border-left:4px solid {a1}; }}
.tip-danger   {{ border-left:4px solid {a3}; }}
.chat-user {{
  background:linear-gradient(135deg,{a1},{a3});color:#fff!important;
  border-radius:18px 18px 4px 18px;padding:10px 16px;margin-bottom:8px;
  font-family:'DM Sans',sans-serif;font-size:14px;max-width:78%;margin-left:auto;
}}
.chat-ai {{
  background:{bgc};border:1px solid {bdr};color:{tp}!important;
  border-radius:18px 18px 18px 4px;padding:10px 16px;margin-bottom:8px;
  font-family:'DM Sans',sans-serif;font-size:14px;max-width:85%;
}}
.web-card {{
  background:{bgc};border:1px solid {bdr};border-left:4px solid {a2};
  border-radius:12px;padding:10px 14px;margin-bottom:8px;
  font-family:'DM Sans',sans-serif;font-size:13px;color:{tp};line-height:1.6;
}}
.web-src {{
  font-family:'Space Mono',monospace;font-size:9px;color:{a2};
  letter-spacing:.08em;margin-top:4px;opacity:.8;
}}
.flow-step {{
  display:flex;align-items:center;gap:8px;padding:7px 12px;
  background:{bgc2};border-radius:10px;border:1px solid {bdr};
  margin-bottom:5px;font-family:'Space Mono',monospace;font-size:11px;color:{ts};
}}
.flow-step.ok  {{ border-color:{a2};color:{a2};background:rgba(0,245,196,.08); }}
.flow-step.run {{ border-color:{a1};color:{a1};background:{tb}; }}
.model-pill {{
  display:inline-block;background:{tb};border:1px solid {a1};border-radius:99px;
  padding:2px 8px;font-family:'Space Mono',monospace;font-size:9px;color:{a1};
  letter-spacing:.06em;margin-left:4px;vertical-align:middle;
}}
.health-label {{
  font-family:'Syne',sans-serif;font-size:11px;text-transform:uppercase;
  letter-spacing:.15em;color:{ts};text-align:center;margin-top:4px;
}}
.watermark {{
  font-family:'Space Mono',monospace;font-size:9px;color:{ts};
  opacity:.4;text-align:center;padding:12px 0;letter-spacing:.08em;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_groq(key: str) -> Groq:
    return Groq(api_key=key)


def call_groq(
    client: Groq, system: str, user: str,
    model: str, max_tokens: int = 500, temp: float = 0.6,
) -> str:
    """Plain LLM call. Strips DeepSeek <think> traces."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temp,
        )
        text = r.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    except Exception as e:
        return f"[{model.split('/')[-1]} error: {str(e)[:120]}]"


# ══════════════════════════════════════════════════════════════════════════════
#  WEB SEARCH — Groq function-calling + DuckDuckGo execution
#  Two-turn flow: turn-1 LLM decides to call web_search tool →
#  we execute DDG → inject results → turn-2 LLM synthesises.
# ══════════════════════════════════════════════════════════════════════════════
_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current financial data: RBI repo rate, SIP performance, "
            "India CPI inflation, ELSS returns, mutual fund NAVs, market news."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, max 80 chars."}
            },
            "required": ["query"],
        },
    },
}


def _ddg_fetch(query: str, n: int = 5) -> List[Dict]:
    results: List[Dict] = []
    try:
        q   = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
        titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>',        html, re.S)
        urls_raw = re.findall(r'class="result__url"[^>]*>(.*?)</span>',   html, re.S)
        for i, snip in enumerate(snippets[:n]):
            results.append({
                "title":   re.sub(r"<[^>]+>", "", titles[i]).strip()   if i < len(titles)   else "",
                "snippet": re.sub(r"<[^>]+>", "", snip).strip(),
                "url":     urls_raw[i].strip()                          if i < len(urls_raw) else "",
            })
    except Exception as exc:
        results.append({"title": "Search unavailable", "snippet": str(exc)[:120], "url": ""})
    return results


def groq_web_search(
    client: Groq, system: str, user_prompt: str,
    model: str, max_tokens: int = 480, temp: float = 0.55,
) -> tuple:
    """Returns (answer: str, raw_results: list[dict])"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt},
    ]
    raw_all: List[Dict] = []
    try:
        r1  = client.chat.completions.create(
            model=model, messages=messages,
            tools=[_WEB_TOOL], tool_choice="auto",
            max_tokens=200, temperature=0.3,
        )
        msg = r1.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args   = json.loads(tc.function.arguments)
                query  = args.get("query", user_prompt[:80])
                hits   = _ddg_fetch(query, n=5)
                raw_all.extend(hits)
                tool_result = "\n".join(
                    f"[{h['title']}] {h['snippet']}" for h in hits
                ) or "No results."
                messages.append({
                    "role": "assistant", "content": msg.content or "",
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": tool_result,
                })
            r2     = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temp,
            )
            answer = r2.choices[0].message.content or ""
        else:
            answer = msg.content or ""
    except Exception as exc:
        answer = f"[Web-search error: {str(exc)[:140]}]"
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip(), raw_all


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT STATE  (LangGraph TypedDict with Annotated list for parallel merge)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  PURE AGENT FUNCTIONS  (no Streamlit — safe for ThreadPoolExecutor)
# ══════════════════════════════════════════════════════════════════════════════

def _agent_budget(state: AgentState, client: Groq) -> str:
    total = sum(state["expenses"].values())
    net   = state["income"] - total
    sr    = net / state["income"] * 100 if state["income"] > 0 else 0
    needs = sum(v for k, v in state["expenses"].items() if k in ("Rent", "Utilities", "Groceries", "Transport"))
    wants = sum(v for k, v in state["expenses"].items() if k in ("Dining", "Entertainment", "Shopping"))
    p = (
        f"Indian household budget analysis:\n"
        f"Income Rs{state['income']:,.0f}/mo | Expenses: {json.dumps(state['expenses'])}\n"
        f"Net Rs{net:,.0f} ({sr:.1f}% savings rate)\n"
        f"50/30/20 -> Needs Rs{needs:,.0f} | Wants Rs{wants:,.0f} | Surplus Rs{net:,.0f}\n\n"
        "Output: 4 numbered insights with exact Rs figures. "
        "Identify the single biggest spending inefficiency. Surgical and blunt."
    )
    return call_groq(client,
        "You are a precision budget AI. Numbers only. Rs signs. No preamble.",
        p, model=MODELS["budget"], max_tokens=420)


def _agent_risk(state: AgentState, client: Groq) -> str:
    total  = sum(state["expenses"].values())
    runway = state["savings"] / total if total > 0 else 0
    er     = total / state["income"] * 100 if state["income"] > 0 else 0
    em_gap = max(0, total * 6 - state["savings"])
    p = (
        f"Risk profile:\n"
        f"Expense-to-income: {er:.1f}% | Runway: {runway:.1f}mo | "
        f"Emergency gap: Rs{em_gap:,.0f} | Savings: Rs{state['savings']:,.0f}\n"
        f"Goal: Rs{state.get('target') or 'none'} in {state.get('timeline') or '?'} months\n\n"
        "Rate: CRITICAL/HIGH/MEDIUM/LOW. Give 3 vulnerability points with Rs impact. "
        "1 immediate fix. Numbered bullets."
    )
    return call_groq(client,
        "You are a financial risk specialist. Blunt, data-first. Rs signs.",
        p, model=MODELS["risk"], max_tokens=360)


def _agent_investment(state: AgentState, client: Groq) -> str:
    total      = sum(state["expenses"].values())
    investable = max(0, state["income"] - total)
    p = (
        f"India 2025 investment plan:\n"
        f"Monthly investable surplus: Rs{investable:,.0f}\n"
        f"Current corpus: Rs{state['savings']:,.0f}\n"
        f"Goal: Rs{state.get('target') or 'wealth creation'} "
        f"in {state.get('timeline') or 'open'} months\n\n"
        "Give exact % + Rs allocations: Liquid fund, ELSS SIP, Nifty 50 index, PPF, NPS, FD. "
        "Name 1 real fund per category. No filler."
    )
    return call_groq(client,
        "You are a SEBI-registered advisor for India. Name specific mutual funds. Rs signs.",
        p, model=MODELS["investment"], max_tokens=460)


def _agent_web_research(state: AgentState, client: Groq) -> tuple:
    """Returns (summary: str, raw_results: list[dict])"""
    total = sum(state["expenses"].values())
    sr    = (state["income"] - total) / state["income"] * 100 if state["income"] > 0 else 0
    p = (
        f"User: Rs{state['income']:,.0f}/mo income, {sr:.1f}% savings rate, India.\n"
        "Search for and report:\n"
        "1. Current RBI repo rate and its impact on savings accounts\n"
        "2. Top 3 performing SIP categories in India right now\n"
        "3. Current India CPI inflation and household budget impact\n"
        "4. Best liquid fund and ELSS returns available now\n"
        "Use the web_search tool. Cite actual figures."
    )
    return groq_web_search(
        client,
        "You are a macro-finance analyst for India. Always use web_search for real data. Cite numbers.",
        p, model=MODELS["research"], max_tokens=500,
    )


def _agent_supervisor(state: AgentState, client: Groq) -> str:
    p = (
        "You are the Chief Financial Officer AI. Synthesise four expert reports:\n\n"
        f"[BUDGET ANALYST]\n{state['budget_analysis']}\n\n"
        f"[RISK ASSESSOR]\n{state['risk_assessment']}\n\n"
        f"[INVESTMENT ADVISOR]\n{state['investment_advice']}\n\n"
        f"[MACRO WEB RESEARCH]\n{state['web_insights']}\n\n"
        f"User: Rs{state['income']:,.0f}/mo | Savings Rs{state['savings']:,.0f} | "
        f"Goal Rs{state.get('target') or 'none'}\n\n"
        "Create a 30-DAY / 60-DAY / 90-DAY execution plan. "
        "Each phase: 3 specific actions with exact Rs amounts. "
        "Final line: ONE BOLD MOVE THIS MONTH in CAPS. No fluff."
    )
    return call_groq(client,
        "You are a master CFP synthesising expert inputs. Decisive. Rs signs. Numbered per phase.",
        p, model=MODELS["supervisor"], max_tokens=660, temp=0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLEL EXECUTION ENGINE
#  ThreadPoolExecutor → 4 agents run concurrently.
#  LangGraph StateGraph used for workflow provenance / audit log.
#  No .stream() unpacking — avoids "too many values to unpack" entirely.
# ══════════════════════════════════════════════════════════════════════════════

def run_parallel_pipeline(
    state: AgentState,
    client: Groq,
    slot_map: Dict[str, Any],
    prog: Any,
) -> AgentState:
    """
    TRUE parallel execution. Updates Streamlit slots on completion.
    Supervisor waits for all 4 agents then runs sequentially.
    """
    completed = [0]

    def _run(display_name: str, fn, *args):
        slot_map[display_name].markdown(
            f'<div class="flow-step run"><span class="sdot"></span>{display_name} · Running…</div>',
            unsafe_allow_html=True,
        )
        result = fn(*args)
        completed[0] += 1
        prog.progress(min(0.80, completed[0] / 4 * 0.80))
        slot_map[display_name].markdown(
            f'<div class="flow-step ok">✓ {display_name} · Done</div>',
            unsafe_allow_html=True,
        )
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_b = ex.submit(_run, "Budget Analyst",    _agent_budget,         state, client)
        f_r = ex.submit(_run, "Risk Assessor",      _agent_risk,           state, client)
        f_i = ex.submit(_run, "Investment Advisor", _agent_investment,     state, client)
        f_w = ex.submit(_run, "Web Researcher",     _agent_web_research,   state, client)

        budget_out     = f_b.result()
        risk_out       = f_r.result()
        invest_out     = f_i.result()
        web_out, w_raw = f_w.result()

    state = {
        **state,
        "budget_analysis":   budget_out,
        "risk_assessment":   risk_out,
        "investment_advice": invest_out,
        "web_insights":      web_out,
        "web_raw":           w_raw,
    }

    prog.progress(0.85)
    slot_map["Supervisor CFO"].markdown(
        '<div class="flow-step run"><span class="sdot"></span>Supervisor CFO · Synthesising…</div>',
        unsafe_allow_html=True,
    )
    final = _agent_supervisor(state, client)
    prog.progress(1.0)
    slot_map["Supervisor CFO"].markdown(
        '<div class="flow-step ok">✓ Supervisor CFO · Plan Ready</div>',
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


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def chat_with_advisor(
    client: Groq, user_msg: str,
    ctx: Dict, history: List[Dict], agent_ctx: str = "",
) -> str:
    exp_total = sum(ctx.get("expenses", {}).values())
    system = (
        "You are MoneyMentor, a 2027-era AI financial advisor for India. "
        "Sharp, concise, data-driven. Know Indian personal finance inside-out. "
        f"User: Rs{ctx.get('income',0):,.0f}/mo income, Rs{exp_total:,.0f} expenses, "
        f"Rs{ctx.get('savings',0):,.0f} savings.\n"
        + (f"Prior agent analysis:\n{agent_ctx[:600]}\n" if agent_ctx else "")
        + "Max 3 sentences. Rs signs. Lead with emoji."
    )
    msgs = [{"role": "system", "content": system}]
    for h in history[-8:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_msg})
    try:
        r = client.chat.completions.create(
            model=MODELS["chat"], messages=msgs, max_tokens=300, temperature=0.72
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ {str(e)[:100]}"


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def get_rule_tips(income, expenses, savings, target, timeline) -> List[Dict]:
    tips: List[Dict] = []
    total = sum(expenses.values())
    net   = income - total
    if net < 0:
        tips.append({"type": "danger",  "icon": "🚨",
                     "text": f"Critical overspend: Rs{abs(net):,.0f}/month deficit. Cut discretionary spend immediately."})
    else:
        sr = net / income * 100 if income > 0 else 0
        if sr < 10:
            tips.append({"type": "warning", "icon": "⚠️",
                         "text": f"Savings rate {sr:.1f}% — dangerously low. Target 20%+ (gap Rs{income*.2-net:,.0f}/mo)."})
        elif sr < 20:
            tips.append({"type": "warning", "icon": "📊",
                         "text": f"Savings rate {sr:.1f}% — below 20% benchmark. Close Rs{income*.2-net:,.0f}/mo gap."})
        else:
            tips.append({"type": "success", "icon": "✅",
                         "text": f"Strong {sr:.1f}% savings rate — above 20% benchmark. Compound it."})
    if expenses:
        top = max(expenses, key=expenses.get)
        pct = expenses[top] / income * 100 if income > 0 else 0
        tips.append({"type": "insight", "icon": "🔍",
                     "text": f"'{top}' is your largest expense: Rs{expenses[top]:,.0f} ({pct:.0f}% of income). Audit first."})
    runway    = savings / total if total > 0 else 0
    em_target = total * 6
    if runway < 3:
        tips.append({"type": "warning", "icon": "🛡️",
                     "text": f"Emergency fund covers only {runway:.1f} months. Build to 6 months (Rs{em_target:,.0f})."})
    elif runway < 6:
        tips.append({"type": "insight", "icon": "🛡️",
                     "text": f"{runway:.1f}-month buffer. 6 months (Rs{em_target:,.0f}) is the real safety net."})
    else:
        tips.append({"type": "success", "icon": "🛡️",
                     "text": f"Strong {runway:.1f}-month emergency fund. Excess beyond 6 months should be invested."})
    if target and timeline and timeline > 0:
        needed  = target - savings
        monthly = needed / timeline
        if monthly <= net:
            tips.append({"type": "success", "icon": "🎯",
                         "text": f"Goal achievable: Rs{monthly:,.0f}/month. Rs{net-monthly:,.0f}/month left to invest."})
        else:
            tips.append({"type": "danger", "icon": "🎯",
                         "text": f"Goal shortfall: need Rs{monthly:,.0f}/mo but only Rs{net:,.0f} available. Gap Rs{monthly-net:,.0f}."})
    return tips


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════
_PAL = ["#7b2fff","#00f5c4","#ff3cac","#ffbe0b","#00b4d8","#ff6b35","#c77dff","#4cc9f0"]


def _donut(expenses: Dict, dark: bool):
    labels = [k for k, v in expenses.items() if v > 0]
    values = [v for v in expenses.values() if v > 0]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=_PAL[:len(labels)],
                    line=dict(color="#06060f" if dark else "#f2f0ff", width=3)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>Rs%{value:,.0f} · %{percent}<extra></extra>",
    ))
    lc = "#e8e8ff" if dark else "#1a0a3d"
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", font=dict(family="Space Mono", size=10, color=lc),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20,b=20,l=20,r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280,
    )
    return fig


def _waterfall(income: float, expenses: Dict, dark: bool):
    tc = "#e8e8ff" if dark else "#1a0a3d"
    gc = "rgba(255,255,255,.04)" if dark else "rgba(0,0,0,.04)"
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(expenses) + ["total"],
        x=["Income"] + list(expenses.keys()) + ["Net"],
        y=[income] + [-v for v in expenses.values()] + [0],
        connector=dict(line=dict(color="#7b2fff", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#00f5c4")),
        decreasing=dict(marker=dict(color="#ff3cac")),
        totals=dict(marker=dict(color="#7b2fff")),
        texttemplate="Rs%{y:,.0f}", textposition="outside",
        textfont=dict(family="Space Mono", size=9, color=tc),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(t=20,b=20,l=20,r=20),
        xaxis=dict(tickfont=dict(family="Space Mono", size=9, color=tc), gridcolor=gc),
        yaxis=dict(tickfont=dict(family="Space Mono", size=9, color=tc),
                   tickprefix="Rs", gridcolor=gc),
        showlegend=False,
    )
    return fig


def _gauge(score: int, dark: bool):
    color = "#00f5c4" if score >= 70 else "#ffbe0b" if score >= 45 else "#ff3cac"
    tc    = "#8080bb" if dark else "#5a4a8a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain=dict(x=[0,1], y=[0,1]),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=tc,
                      tickfont=dict(family="Space Mono", size=9)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[0,  45], color="rgba(255,60,172,.07)"),
                dict(range=[45, 70], color="rgba(255,190,11,.07)"),
                dict(range=[70,100], color="rgba(0,245,196,.07)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.7, value=score),
        ),
        number=dict(font=dict(family="Space Mono", size=46, color=color)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=190, margin=dict(t=10,b=10,l=20,r=20),
    )
    return fig


def _health_score(income, expenses, savings, target, timeline) -> int:
    total = sum(expenses.values())
    net   = income - total
    score = 50.0
    if income > 0:
        score += min(25, (net / income) * 80)
        score -= max(0, (total / income - 0.7) * 60)
    runway = savings / total if total > 0 else 0
    score += min(15, runway * 3)
    if target and timeline and timeline > 0:
        needed = (target - savings) / timeline
        score += 10 if net >= needed else -(min(1.0, (needed - net) / (needed + 1)) * 15)
    return max(0, min(100, round(score)))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Session state defaults
    for k, v in {
        "dark_mode": True, "analysis_done": False,
        "agent_state": None, "financial_data": None, "chat_history": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # CSS FIRST — before any element renders
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="hero-badge" style="margin-bottom:16px;">'
            '<span class="sdot"></span>MoneyMentor AI · 2027</div>',
            unsafe_allow_html=True,
        )
        label = "☀️  Light Mode" if _dark() else "🌙  Dark Mode"
        if st.button(label, key="theme_btn", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("---")
        st.markdown('<div class="sec-lbl">API Configuration</div>', unsafe_allow_html=True)
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("Key loaded from secrets ✓")
        else:
            api_key = st.text_input(
                "Groq API Key", type="password", placeholder="gsk_…",
                key="api_input", label_visibility="collapsed",
            )
            if not api_key:
                st.warning("Paste your Groq API key above.")
                st.stop()

        st.markdown("---")
        st.markdown('<div class="sec-lbl">Agent Roster</div>', unsafe_allow_html=True)
        for name, mdl, role in [
            ("Budget Analyst",     "DeepSeek-R1-70B",     "CoT budget math"),
            ("Risk Assessor",      "Llama-3.3-70B",       "Vulnerability scoring"),
            ("Investment Advisor", "Llama-4-Scout-17B",   "128k context portfolio"),
            ("Web Researcher",     "Llama-3.3-70B+Tools", "Function-call web search"),
            ("Supervisor CFO",     "Llama-3.3-70B",       "90-day synthesis"),
        ]:
            st.markdown(
                f'<div style="background:var(--bgc);border:1px solid var(--bdr);'
                f'border-radius:10px;padding:8px 10px;margin-bottom:5px;">'
                f'<div style="font-family:Syne,sans-serif;font-weight:700;'
                f'font-size:12px;color:var(--tp);">{name}</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:9px;'
                f'color:var(--a2);">{mdl}</div>'
                f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;'
                f'color:var(--ts);margin-top:2px;">{role}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        lg_lbl = "✅ LangGraph 1.x ready" if _LG else "⚡ ThreadPool mode"
        st.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:9px;'
            f'color:var(--ts);letter-spacing:.06em;line-height:1.8;">'
            f'{lg_lbl}<br>Parallel fan-out → supervisor<br>'
            f'Groq function-call web search<br>DuckDuckGo live data</div>',
            unsafe_allow_html=True,
        )

    # Init client
    try:
        groq_client = init_groq(api_key)
    except Exception as e:
        st.error(f"Groq init failed: {e}")
        st.stop()

    # Hero
    st.markdown(
        '<div class="hero-badge"><span class="sdot"></span>'
        '5-Agent · Parallel · Groq Function-Call Web Search · Live</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-title">MoneyMentor AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Five specialised AI agents run in parallel — '
        'DeepSeek chain-of-thought math, Llama-4 long-context portfolio planning, '
        'and live Groq function-calling web search — synthesised by a CFO supervisor '
        'into your 90-day financial OS.</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    t1, t2, t3 = st.tabs(["⬡  FINANCIAL INPUT", "⬡  AGENT ANALYSIS", "⬡  AI ADVISOR CHAT"])

    # ══ TAB 1 — INPUT ════════════════════════════════════════════════════════
    with t1:
        st.markdown('<div class="sec-lbl">Financial Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-ttl">Enter Your Monthly Financials</div>', unsafe_allow_html=True)

        with st.form("finance_form"):
            st.markdown('<div class="sec-lbl" style="margin-top:4px;">Monthly Income</div>',
                        unsafe_allow_html=True)
            income = st.number_input("income", min_value=0.0, step=500.0, value=45000.0,
                                     label_visibility="collapsed")
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;font-size:22px;font-weight:700;'
                f'color:var(--a2);margin:2px 0 18px;">Rs{income:,.0f}'
                f'<span style="font-size:11px;color:var(--ts);"> /month</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="sec-lbl">Monthly Expenses</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                rent      = st.number_input("🏠 Rent/Mortgage (Rs)", min_value=0.0, step=100.0, value=14000.0)
                utilities = st.number_input("⚡ Utilities (Rs)",      min_value=0.0, step=50.0,  value=2500.0)
                groceries = st.number_input("🛒 Groceries (Rs)",      min_value=0.0, step=50.0,  value=5000.0)
                transport = st.number_input("🚗 Transport (Rs)",      min_value=0.0, step=50.0,  value=3000.0)
            with c2:
                dining        = st.number_input("🍜 Dining Out (Rs)",    min_value=0.0, step=50.0, value=2500.0)
                entertainment = st.number_input("🎮 Entertainment (Rs)", min_value=0.0, step=50.0, value=1500.0)
                shopping      = st.number_input("🛍️ Shopping (Rs)",      min_value=0.0, step=50.0, value=2000.0)
                other         = st.number_input("📦 Other (Rs)",         min_value=0.0, step=50.0, value=1500.0)

            expenses = {
                "Rent": rent, "Utilities": utilities, "Groceries": groceries,
                "Transport": transport, "Dining": dining, "Entertainment": entertainment,
                "Shopping": shopping, "Other": other,
            }

            st.markdown('<div class="sec-lbl" style="margin-top:14px;">Savings & Goal</div>',
                        unsafe_allow_html=True)
            cs, cg = st.columns(2)
            with cs:
                savings = st.number_input("💰 Current Savings (Rs)", min_value=0.0, step=1000.0, value=50000.0)
            with cg:
                has_goal = st.radio("🎯 Savings Goal?", ("Yes", "No"), index=1, horizontal=True)

            target = timeline = None
            if has_goal == "Yes":
                ct, cl = st.columns(2)
                with ct:
                    target   = st.number_input("Target (Rs)",       min_value=0.0, step=1000.0, value=200000.0)
                with cl:
                    timeline = st.number_input("Timeline (months)", min_value=1, step=1, value=18)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "⬡  LAUNCH 5-AGENT PARALLEL ANALYSIS", use_container_width=True
            )

        if submitted:
            st.session_state.financial_data = dict(
                income=income, expenses=expenses,
                savings=savings, target=target, timeline=timeline,
            )
            init_state: AgentState = {
                "income": income, "expenses": expenses,
                "savings": savings, "target": target, "timeline": timeline,
                "budget_analysis": "", "risk_assessment": "",
                "investment_advice": "", "web_insights": "",
                "web_raw": [], "final_report": "", "messages": [],
            }

            st.markdown("---")
            st.markdown('<div class="sec-lbl">Agent Pipeline · Live</div>', unsafe_allow_html=True)
            prog = st.progress(0)
            agent_names = ["Budget Analyst", "Risk Assessor", "Investment Advisor",
                           "Web Researcher", "Supervisor CFO"]
            slots = {n: st.empty() for n in agent_names}
            for n in agent_names[:4]:
                slots[n].markdown(
                    f'<div class="flow-step">◯ {n} · Queued</div>', unsafe_allow_html=True
                )
            slots["Supervisor CFO"].markdown(
                '<div class="flow-step">◯ Supervisor CFO · Waiting for agents</div>',
                unsafe_allow_html=True,
            )

            result = run_parallel_pipeline(init_state, groq_client, slots, prog)
            st.session_state.agent_state   = result
            st.session_state.analysis_done = True
            st.success("✅ All 5 agents complete — switch to **AGENT ANALYSIS** tab")

    # ══ TAB 2 — ANALYSIS ═════════════════════════════════════════════════════
    with t2:
        if not st.session_state.analysis_done or not st.session_state.agent_state:
            st.markdown(
                '<div style="text-align:center;padding:60px 20px;">'
                '<div style="font-size:44px;margin-bottom:14px;">⬡</div>'
                '<div class="sec-ttl" style="text-align:center;">No Analysis Yet</div>'
                '<div style="font-family:DM Sans,sans-serif;font-size:14px;color:var(--ts);">'
                'Fill your financials in the INPUT tab and launch the agents.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            s = st.session_state.agent_state
            d = st.session_state.financial_data
            income = d["income"]; expenses = d["expenses"]
            savings = d["savings"]; target = d["target"]; timeline = d["timeline"]
            total = sum(expenses.values()); net = income - total

            # Vitals
            st.markdown('<div class="sec-lbl">Financial Vitals</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Monthly Income",  f"Rs{income:,.0f}")
            m2.metric("Total Expenses",  f"Rs{total:,.0f}",
                      delta=f"{total/income*100:.0f}% of income" if income else None,
                      delta_color="inverse")
            m3.metric("Net Cash Flow",   f"Rs{net:,.0f}",
                      delta=f"{net/income*100:.1f}% saved" if income else None,
                      delta_color="normal" if net >= 0 else "inverse")
            m4.metric("Savings",         f"Rs{savings:,.0f}")
            score = _health_score(income, expenses, savings, target, timeline)
            if target:
                m5.metric("Goal Progress", f"{min(100,savings/target*100):.0f}%",
                          delta=f"Rs{max(0,target-savings):,.0f} to go")
            else:
                m5.metric("Health Score", f"{score}/100")

            st.markdown("---")

            # Charts
            cg, cd, cw = st.columns([1, 1.35, 1.8])
            with cg:
                st.markdown('<div class="sec-lbl">Health Score</div>', unsafe_allow_html=True)
                st.plotly_chart(_gauge(score, _dark()), use_container_width=True,
                                config={"displayModeBar": False})
                grade = ("Excellent" if score >= 80 else "Good" if score >= 65
                         else "Fair" if score >= 45 else "At Risk")
                st.markdown(f'<div class="health-label">{grade}</div>', unsafe_allow_html=True)
            with cd:
                st.markdown('<div class="sec-lbl">Expense Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(_donut(expenses, _dark()), use_container_width=True,
                                config={"displayModeBar": False})
            with cw:
                st.markdown('<div class="sec-lbl">Cash Flow Waterfall</div>', unsafe_allow_html=True)
                st.plotly_chart(_waterfall(income, expenses, _dark()), use_container_width=True,
                                config={"displayModeBar": False})

            if target and target > 0:
                pct = min(1.0, savings / target)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-family:\'Space Mono\',monospace;font-size:11px;'
                    f'color:var(--ts);margin:6px 0 5px;">'
                    f'<span>Goal Rs{target:,.0f}</span><span>{pct*100:.0f}% complete</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(pct)

            st.markdown("---")

            # Rule tips
            st.markdown('<div class="sec-lbl">Instant Diagnostics</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-ttl">Rule-Based Analysis</div>', unsafe_allow_html=True)
            for tip in get_rule_tips(income, expenses, savings, target, timeline):
                st.markdown(
                    f'<div class="tip-card tip-{tip["type"]}">'
                    f'<div class="tip-icon">{tip["icon"]}</div>'
                    f'<div class="tip-text">{tip["text"]}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Agent reports
            st.markdown('<div class="sec-lbl">Parallel Agent Intelligence</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-ttl">Multi-Model Analysis Reports</div>', unsafe_allow_html=True)

            reports = [
                ("Budget Analyst",     "DeepSeek-R1-Distill-70B",  "Income & Expense Diagnostics", "budget_analysis"),
                ("Risk Assessor",      "Llama-3.3-70B",             "Risk & Vulnerability Report",  "risk_assessment"),
                ("Investment Advisor", "Llama-4-Scout-17B-16e",     "India Portfolio Allocation",   "investment_advice"),
                ("Web Researcher",     "Llama-3.3-70B + Web Tools", "Live Macro & Market Data",     "web_insights"),
            ]
            cl, cr = st.columns(2)
            for i, (name, mdl, role, key) in enumerate(reports):
                output = s.get(key, "") or "No output."
                with (cl if i % 2 == 0 else cr):
                    with st.expander(f"⬡  {name}", expanded=(i < 2)):
                        st.markdown(
                            f'<div class="agent-role">{role}</div>'
                            f'<span class="model-pill">{mdl}</span>'
                            f'<div class="agent-output">{output.replace(chr(10),"<br>")}</div>',
                            unsafe_allow_html=True,
                        )

            # Web raw results
            web_raw = s.get("web_raw") or []
            if web_raw:
                st.markdown("---")
                st.markdown('<div class="sec-lbl">Live Web Snippets · DuckDuckGo</div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-ttl">Real-Time Market Data</div>', unsafe_allow_html=True)
                wl, wr = st.columns(2)
                for i, r in enumerate(web_raw[:6]):
                    with (wl if i % 2 == 0 else wr):
                        st.markdown(
                            f'<div class="web-card"><strong>{r.get("title","")}</strong><br>'
                            f'{r.get("snippet","")}'
                            f'<div class="web-src">↗ {r.get("url","")}</div></div>',
                            unsafe_allow_html=True,
                        )

            st.markdown("---")

            # Supervisor plan
            st.markdown('<div class="sec-lbl">Supervisor CFO · Llama-3.3-70B</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-ttl">⬡ 30 / 60 / 90-Day Action Plan</div>', unsafe_allow_html=True)
            final = s.get("final_report", "")
            if final:
                st.markdown(
                    '<div class="agent-card">'
                    '<div class="agent-name">Supervisor CFO — Master Execution Plan</div>'
                    '<span class="model-pill">Llama-3.3-70B · 4 parallel agents + live web</span>'
                    f'<div class="agent-output">{final.replace(chr(10),"<br>")}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬡  Re-run Analysis", key="rerun_btn"):
                st.session_state.analysis_done = False
                st.session_state.agent_state   = None
                st.rerun()

    # ══ TAB 3 — CHAT ═════════════════════════════════════════════════════════
    with t3:
        st.markdown('<div class="sec-lbl">Conversational Finance AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-ttl">Ask MoneyMentor Anything</div>', unsafe_allow_html=True)

        ctx       = st.session_state.financial_data or {"income": 0, "expenses": {}, "savings": 0}
        exp_total = sum(ctx.get("expenses", {}).values())

        if st.session_state.financial_data:
            st.markdown(
                f'<div style="background:var(--bgc);border:1px solid var(--bdr);'
                f'border-radius:12px;padding:10px 16px;margin-bottom:14px;'
                f'display:flex;gap:24px;flex-wrap:wrap;">'
                f'<div><div class="sec-lbl">Income</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:14px;font-weight:700;'
                f'color:var(--a2);">Rs{ctx["income"]:,.0f}</div></div>'
                f'<div><div class="sec-lbl">Expenses</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:14px;font-weight:700;'
                f'color:var(--a3);">Rs{exp_total:,.0f}</div></div>'
                f'<div><div class="sec-lbl">Savings</div>'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:14px;font-weight:700;'
                f'color:var(--a1);">Rs{ctx["savings"]:,.0f}</div></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Add your financials in the INPUT tab for context-aware answers.")

        agent_ctx = ""
        if st.session_state.agent_state:
            s2 = st.session_state.agent_state
            agent_ctx = (
                f"Budget: {s2.get('budget_analysis','')[:200]} | "
                f"Risk: {s2.get('risk_assessment','')[:200]} | "
                f"Investment: {s2.get('investment_advice','')[:200]}"
            )

        st.markdown('<div class="sec-lbl" style="margin-bottom:8px;">Quick Questions</div>',
                    unsafe_allow_html=True)
        qs  = ["Cut my biggest expense?", "Best SIPs for Rs5k/month?",
               "Build emergency fund fast?", "Invest or clear debt first?"]
        qcs = st.columns(4)
        for col, q, idx in zip(qcs, qs, range(4)):
            with col:
                if st.button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    reply = chat_with_advisor(groq_client, q, ctx,
                                             st.session_state.chat_history[:-1], agent_ctx)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:6px;">'
                    f'<div class="chat-user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin-bottom:6px;">'
                    f'<div class="chat-ai">'
                    f'<span style="font-family:\'Space Mono\',monospace;font-size:9px;'
                    f'color:var(--a1);letter-spacing:.1em;text-transform:uppercase;'
                    f'display:block;margin-bottom:4px;">⬡ MoneyMentor</span>'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input(
            "chat_in", placeholder="e.g. How to save Rs1L in 6 months?",
            key="chat_input", label_visibility="collapsed",
        )
        csend, cclear = st.columns([5, 1])
        with csend:
            if st.button("⬡  SEND MESSAGE", key="send_btn", use_container_width=True) \
               and user_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner(""):
                    reply = chat_with_advisor(groq_client, user_input, ctx,
                                             st.session_state.chat_history[:-1], agent_ctx)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
        with cclear:
            if st.button("Clear", key="clear_btn", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="watermark">'
        'MONEYMENTOR AI · 2027 · LANGGRAPH PARALLEL FAN-OUT · GROQ FUNCTION-CALL WEB SEARCH · '
        'DEEPSEEK-R1 · LLAMA-4-SCOUT-17B · LLAMA-3.3-70B · DUCKDUCKGO LIVE DATA<br>'
        'Not financial advice. Educational use only.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
