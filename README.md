
<div align="center">

# ◈ MoneyMentor AI 

### Personal CFO AI Platform — Multi-Agent · Production-Grade · India-First

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge)](https://groq.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-10f5a0?style=for-the-badge)](LICENSE)

> **Not another AI chatbot wrapper.**
> A production-grade multi-agent financial intelligence system with parallel LLM orchestration, domain-specific financial computation engines.
</div>

---

## Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Why This Stands Out](#-why-this-stands-out)
3. [System Architecture](#-system-architecture)
4. [Agent Intelligence Layer](#-agent-intelligence-layer)
5. [Feature Showcase](#-feature-showcase-7-tabs)
6. [Financial Computation Engines](#-financial-computation-engines)
7. [Technical Deep Dives](#-technical-deep-dives)
8. [Model Registry & Routing](#-model-registry--routing)
9. [Quick Start](#-quick-start)

---

## 🎯 Executive Summary

MoneyMentor AI is a **full-stack AI engineering project** that simulates a personal Chief Financial Officer for Indian households. It goes far beyond calling an LLM API — it implements:

- A **fan-out / fan-in multi-agent orchestration** pattern with true parallel execution
- **Domain-specific financial computation** (Monte Carlo simulation, Indian tax engine, SIP compounding)
- **Graceful model routing** with automatic fallback chains across 4 different LLMs
- **Live web intelligence** via Groq Compound-Beta's native agentic search, with a multi-layer fallback strategy (DDG scrape → curated knowledge base → LLM synthesis)
- **Production-quality UI** with dark/light theming, custom CSS design system, and Plotly interactive charts


---

## 🚀 Why This Stands Out

| Typical "AI Wrapper" Project | MoneyMentor AI 2027 |
|---|---|
| Single LLM call, renders text | 5 specialized agents, each with a role, prompt, and model |
| Sequential execution | True parallel fan-out via `ThreadPoolExecutor` |
| Generic Q&A | Indian tax law (FY 2026-27), SEBI regulations, PPF/NPS/ELSS |
| No fallback | 3-layer fallback chain per agent; graceful degradation at every level |
| Static UI | Dynamic dark/light CSS design system, micro-animations, Plotly charts |
| `st.write(response)` | Custom-rendered agent panels, formatted 30/60/90-day plans, goal cards |
| No computation | Monte Carlo engine, SIP FV formula, tax slab calculator, health scorer |
| Single model | 4-model registry with automatic routing and model-specific prompt engineering |
| Hardcoded | Tool-call response extraction, `_extract_tool_text()`, multi-pattern HTML scraper |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INPUT (Tab 1)                           │
│          Income · Expenses · Savings · Goal · Timeline              │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  AgentState TypedDict
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PARALLEL FAN-OUT  (ThreadPoolExecutor, 4 workers)      │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │  Budget Analyst  │  │  Risk Assessor  │  │ Invest Advisor  │  │ Web Researcher │ │
│  │  llama-3.3-70b  │  │  llama-3.3-70b  │  │  llama-3.3-70b  │  │ compound-beta  │ │
│  │                 │  │                 │  │                 │  │  (agentic web) │ │
│  │ 50/30/20 split  │  │ CRITICAL/HIGH/  │  │ ELSS · SIP ·   │  │ RBI repo rate  │ │
│  │ anomaly detect  │  │ MEDIUM/LOW risk │  │ PPF · NPS ·    │  │ CPI · SIP perf │ │
│  │ Rs-level cuts   │  │ runway calc     │  │ Nifty Index    │  │ liquid yields  │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘ │
│           └───────────────────┴───────────────────┴──────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  FAN-IN — all 4 futures resolved
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SUPERVISOR CFO                                  │
│              deepseek-r1-distill-llama-70b                          │
│     → fallback: llama-3.3-70b-versatile                            │
│                                                                     │
│  Synthesises all 4 reports → 30 / 60 / 90-Day Execution Plan       │
│  Payload-trimmed (280 chars/agent) to prevent 413 overflows         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────┼───────────────┐
                    ▼              ▼               ▼
             Tab 2: Analysis  Tab 3: Charts   Tab 7: Chat
             Agent Panels     MC Simulator    llama-4-scout
```

### Key Architectural Decisions

**1. ThreadPoolExecutor Fan-Out (not LangGraph)**
LangGraph's graph-based approach produced a `"too many values to unpack (expected 2)"` error with stateful parallel edges. The solution: use `concurrent.futures.ThreadPoolExecutor` directly. All agent functions are pure Python (zero `st.*` calls), making them fully thread-safe. The UI updates happen *only* after all futures resolve — correct fan-in semantics.

**2. Strict UI / Logic Separation**
Every agent function (`_budget_agent`, `_risk_agent`, etc.) is a pure function: `(AgentState, Groq) → str`. No Streamlit globals, no session state, no side effects. This makes them independently testable and safely parallelisable.

**3. Tool-Call Response Extraction**
When LLMs respond with a `function_call` instead of text (`message.content == None`), a naive implementation shows "No output." `_extract_tool_text()` parses `message.tool_calls[].function.arguments` and formats the JSON as readable prose. A no-tools retry is used as a final safety net.

**4. Three-Layer Fallback for Web Data**
```
compound-beta (native search)
    ↓ fails (timeout / 413 / empty)
DuckDuckGo HTML scraper (multi-pattern regex)
    ↓ DDG HTML changed / blocked
_INDIA_FINANCE_KB (curated hardcoded knowledge base)
    ↓ always available
llama-3.3-70b synthesises answer
```

---

## 🤖 Agent Intelligence Layer

### Agent 1 — Budget Analyst
**Model:** `llama-3.3-70b-versatile`

Performs 50/30/20 rule decomposition with category-level anomaly detection. The prompt is engineered to produce **exactly 4 numbered, Rs-denominated insights** with no preamble — output is deterministic in structure, enabling reliable UI rendering. Needs/wants classification uses a case-insensitive keyword matcher against Indian household expense categories.

### Agent 2 — Risk Assessor
**Model:** `llama-3.3-70b-versatile`

Computes emergency runway (`savings ÷ monthly_burn`), 6-month fund gap, expense-to-income ratio, and goal monthly requirement vs available surplus. Prompt enforces strict output format (`RISK LEVEL:` header + 3 numbered vulnerabilities + `IMMEDIATE FIX:`) — no tools used, ensuring content is never `None`.

### Agent 3 — Investment Advisor
**Model:** `llama-3.3-70b-versatile`

India-specific portfolio construction against real investable surplus. System prompt specifies allocation format verbatim: `[Instrument] — [Fund Name] — Rs[X]/mo ([Y]%) — [reason]`. Forces naming real SEBI-registered AMCs (Mirae Asset, Quant MF, Motilal Oswal, SBI, HDFC) rather than generic placeholders. Ends with 12-month and 3-year corpus projections.

### Agent 4 — Web Researcher
**Model:** `compound-beta` (Groq's native agentic search model)

Payload is intentionally minimal (system: 1 sentence, user: 4 bullet points) to avoid the **413 Request Entity Too Large** error that compound-beta triggers with verbose prompts. Returns live RBI rates, SIP performance, CPI, and liquid fund yields with citations. Falls back to DuckDuckGo + curated KB on failure.

### Agent 5 — Supervisor CFO
**Model:** `deepseek-r1-distill-llama-70b` → `llama-3.3-70b-versatile`

Synthesises all 4 agent outputs. Each sub-report is hard-capped to 280 characters (`_cap()` helper) before being included in the supervisor prompt — this keeps the total token count well within limits while preserving the key insight from each agent. Outputs a structured 30/60/90-day plan with bold formatting applied at render time.

### Agent 6 — Chat Advisor
**Model:** `meta-llama/llama-4-scout-17b-16e-instruct`

Multi-turn conversational advisor with rolling 10-message history. System prompt is dynamically constructed with live financial context (income, expenses, agent analysis snapshot). Responds in max 4 sentences with an emoji lead and Rs-denominated advice.

---

## 📊 Feature Showcase — 7 Tabs

### Tab 1 · Financial Input
- Sidebar with income/expense sliders, savings corpus, goal amount and timeline
- Real-time 50/30/20 health preview before running agents
- One-click 5-agent pipeline launcher with live progress slots per agent
- Agent status: Queued → Running (animated) → Complete ✓

### Tab 2 · Agent Analysis
- **Financial Vitals** — 5 KPI metrics (income, expenses, net flow, corpus, goal %)
- **Health Score Gauge** — 0–100 score with `Excellent / Good / Fair / At Risk` grade
- **Expense Donut** — Plotly donut chart with Indian expense categories
- **Cash Flow Waterfall** — Income minus each expense category, net surplus bar
- **Instant Diagnostics** — 8 deterministic rule-based tips (50/30/20, emergency fund, goal feasibility)
- **4 Agent Panels** — Expandable cards with model badge and formatted output
- **30/60/90-Day Plan** — Supervisor CFO output with bold section headers
- **Markdown Report Export** — Full analysis as downloadable `.md` file

### Tab 3 · Portfolio Simulator
- **1,200-path Monte Carlo** — Log-normal monthly returns with configurable µ and σ
- P10 (bear) / P50 (base) / P90 (bull) outcome percentiles
- Plotly fan chart of all 1,200 simulation paths
- **SIP Growth Projections** — Multi-line chart: Conservative 8% / Moderate 12% / Aggressive 15%
- **Corpus Milestone Table** — Year 1, 3, 5, 10, 15, 20 projections

### Tab 4 · Tax Optimizer
- **FY 2026-27 Indian Tax Engine** — Full slab computation (Old & New regime)
- Deductions: 80C (ELSS/PPF/LIC), 80CCD(1B) NPS, 80D health insurance, HRA exemption
- Section 87A rebate logic (≤7L new regime, ≤5L old regime)
- 4% health & education cess applied
- Bar chart comparison, deduction breakdown cards, actionable tips
- Tax slab tables for both regimes

### Tab 5 · Goal Tracker
- Add multiple financial goals with icon, target, current savings, timeline
- Horizontal progress bar chart (Plotly) across all goals
- Per-goal card: monthly SIP required, feasibility vs net flow, @10% SIP projection
- Aggregate metrics: total goals value, funded %, combined monthly need vs surplus

### Tab 6 · Market Insights
- 8 research topics (RBI data, ELSS, Nifty, Small/Large cap, PPF vs NPS, Gold vs equity)
- compound-beta live web search → DDG fallback → curated KB fallback
- Styled output card with line-by-line rendering
- 6 reference cards (RBI, SEBI MFs, PPF, NPS, Liquid Funds, ELSS)

### Tab 7 · AI Advisor Chat
- 6 pre-set quick-question buttons for common queries
- Context-aware: injects live financial data + agent analysis snapshot
- Chat history with user/AI styled bubbles
- Rolling 10-message memory window

---

## ⚙️ Financial Computation Engines

### Monte Carlo Simulation Engine
```python
# Log-normal monthly returns — 1,200 paths
mu_monthly  = annual_return / 12
sig_monthly = volatility / math.sqrt(12)

for _ in range(n_sims):          # 1,200 simulations
    corpus = 0.0
    for _ in range(n_months):    # up to 360 months
        r = random.gauss(mu_monthly, sig_monthly)
        corpus = (corpus + monthly_sip) * (1 + r)
    endpoints.append(corpus)

# Output: P10 / P25 / P50 / P75 / P90 percentiles
```
Produces statistically valid P10 (bear), P50 (base), P90 (bull) projections. Configurable: 1–30 year horizon, 6–20% annual return, 5–35% volatility.

### SIP Future Value Formula
```python
# Standard SIP compounding — Rs return including monthly contributions
FV = P × [(1 + r)^n – 1] / r × (1 + r)
# r = annual_rate / 12  |  n = years × 12
```
Used across Tab 3 projections, Tab 5 goal cards, and Tab 2 investment advice validation.

### FY 2026-27 Indian Tax Engine
```python
# New regime — taxable after Rs75,000 standard deduction
taxable_new = max(0, annual_income - STD_DEDUCTION_NEW)
# 87A rebate: zero tax if taxable ≤ Rs7,00,000
# Cess: 4% health & education on computed tax

# Old regime — taxable after all Section 80 deductions
taxable_old = annual_income - STD_DEDUCTION_OLD - 80C - 80CCD1B - 80D - HRA
```
Computes both regimes, recommends the better one, and shows exact annual savings — using the actual FY 2026-27 slab structure as notified by the Finance Act.

### Financial Health Score (0–100)
```python
score = 50.0                                    # base
score += min(26, (net / income) * 88)           # savings rate bonus
score -= max(0, (total / income - 0.70) * 58)   # overspend penalty
score += min(16, runway * 3)                     # emergency buffer bonus
score ± goal_feasibility_adjustment(10–15)       # goal delta
```

---

## 🔬 Technical Deep Dives

### Tool-Call Response Extraction
When an LLM responds with a `function_call` (tool use) instead of free text, `message.content` is `None`. Most tutorials fail here with "No output." MoneyMentor handles this with `_extract_tool_text()`:

```python
def _extract_tool_text(msg) -> str:
    for tc in (msg.tool_calls or []):
        args = json.loads(tc.function.arguments)
        # Recursively format: lists → bullet points, floats → Rs-formatted
        # Returns human-readable prose from structured tool arguments
```

If tool extraction also fails, a clean retry *without* the tools parameter is issued automatically.

### Multi-Pattern DuckDuckGo Scraper
DDG frequently changes its HTML class names. The scraper tries 3 independent patterns:

```python
# Pattern 1 — classic class names
snips = re.findall(r'class="result__snippet"...', html)

# Pattern 2 — any class containing "snippet"
snips = re.findall(r'class="[^"]*snippet[^"]*"...', html)

# Pattern 3 — raw paragraph extraction (40–300 char blocks)
snips = re.findall(r'<(?:p|div)[^>]*>([\w][^<]{40,300})</(?:p|div)>', html)
```

### Thread-Safe Agent Execution
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = {
        "budget":     pool.submit(_budget_agent,     state, client),
        "risk":       pool.submit(_risk_agent,       state, client),
        "investment": pool.submit(_investment_agent, state, client),
        "web":        pool.submit(_web_agent,        state, client),
    }
    # Block until ALL complete — true fan-in
    budget_out = futures["budget"].result()
    ...

# UI updates ONLY happen here (main thread) — not inside threads
slots["Budget Analyst"].markdown(...)
```

No Streamlit globals are accessed from worker threads — a common mistake that causes crashes.

---

## 🧠 Model Registry & Routing

```python
MODELS = {
    "budget":     "llama-3.3-70b-versatile",          # Rich CoT, reliable text
    "risk":       "llama-3.3-70b-versatile",          # Structured format output
    "investment": "llama-3.3-70b-versatile",          # Domain knowledge depth
    "research":   "compound-beta",                    # Native agentic web search
    "supervisor": "deepseek-r1-distill-llama-70b",   # Chain-of-thought reasoning
    "chat":       "meta-llama/llama-4-scout-17b-16e-instruct",  # Fast multi-turn
}
```

**Why these models?**
- `llama-3.3-70b-versatile` — Best balance of instruction-following and domain knowledge on Groq's free tier; produces structured, Rs-denominated output reliably
- `compound-beta` — Groq's only model with **native tool-use web search**; no API keys needed for live data
- `deepseek-r1-distill-llama-70b` — Chain-of-thought reasoning for synthesis tasks; produces more coherent multi-step plans than instruction-tuned models
- `llama-4-scout-17b` — Optimised for short, snappy responses; ideal for chat where latency matters

**Fallback chain:**
```
deepseek-r1 (supervisor) → llama-3.3-70b (always available)
compound-beta (web)      → DDG scrape → India Finance KB → llama-3.3-70b
```

---

## 📄 License

MIT License — free to use, modify, and showcase in your portfolio.

---

<div align="center">

*Every architectural decision in this codebase is intentional.*
*The goal was not to build the fastest AI app — it was to build the most defensible one.*

---

`Python` · `Groq LPU Inference` · `Multi-Agent Systems` · `Indian FinTech` · `Streamlit`

</div>
