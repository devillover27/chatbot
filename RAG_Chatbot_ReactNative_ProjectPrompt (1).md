# 📱 Portfolio RAG Chatbot — React Native App
### Complete Build Prompt & Project Specification
> Powered by Anthropic Claude API · RAG Architecture · Python FastAPI Backend · Financial Portfolio Intelligence

---

## 🎯 Project Overview

Build a **React Native mobile application** that acts as an intelligent financial portfolio assistant. Users can ask natural language questions about their **holdings** and **trades** data, and the app answers using a RAG (Retrieval-Augmented Generation) pipeline backed by **Anthropic's Claude API**.

Backend is written in **Python (FastAPI)** — perfect for data science workflows, pandas, and ML libraries.

---

## 📦 Data Schema (Your Knowledge Base)

### `holdings.csv` — 1,022 rows
| Column | Description |
|---|---|
| `AsOfDate` | Snapshot date of holding |
| `OpenDate` / `CloseDate` | Position open/close dates |
| `ShortName` | Portfolio short name |
| `PortfolioName` | Full portfolio name |
| `StrategyRefShortName` / `Strategy1RefShortName` / `Strategy2RefShortName` | Strategy labels |
| `CustodianName` | Custodian bank/institution |
| `DirectionName` | Long / Short |
| `SecurityId` / `SecurityTypeName` / `SecName` | Security details |
| `StartQty` / `Qty` | Position quantity |
| `StartPrice` / `Price` | Entry & current price |
| `StartFXRate` / `FXRate` | FX conversion rates |
| `MV_Local` / `MV_Base` | Market value (local currency & base currency) |
| `PL_DTD` / `PL_QTD` / `PL_MTD` / `PL_YTD` | P&L across time periods |

### `trades.csv` — 649 rows
| Column | Description |
|---|---|
| `id` / `RevisionId` / `AllocationId` | Trade identifiers |
| `TradeTypeName` | Buy / Sell / etc. |
| `SecurityId` / `SecurityType` / `Name` / `Ticker` | Security info |
| `CUSIP` / `ISIN` | Security identifiers |
| `TradeDate` / `SettleDate` | Trade and settlement dates |
| `Quantity` / `Price` / `TradeFXRate` | Trade terms |
| `Principal` / `Interest` / `TotalCash` | Cash flows |
| `AllocationQTY` / `AllocationPrincipal` / `AllocationInterest` / `AllocationFees` / `AllocationCash` | Allocation breakdown |
| `PortfolioName` / `CustodianName` | Institutional details |
| `StrategyName` / `Strategy1Name` / `Strategy2Name` | Strategy assignment |
| `Counterparty` | Trading counterparty |
| `AllocationRule` / `IsCustomAllocation` | Allocation rule info |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│            React Native App             │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │Chat UI  │  │Dashboard │  │History │ │
│  └────┬────┘  └──────────┘  └────────┘ │
└───────┼─────────────────────────────────┘
        │ User Query (HTTP/SSE)
        ▼
┌───────────────────────────────────────────┐
│     Python FastAPI Backend (port 8000)    │
│                                           │
│  POST /api/chat   → RAG + Claude stream   │
│  GET  /api/summary → Portfolio stats      │
│  GET  /api/search  → Debug chunk search   │
└──────────────┬────────────────────────────┘
               │
        ┌──────┴──────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌───────────────────┐
│ Vector Store │  │  Anthropic Claude │
│  (numpy      │  │  claude-sonnet-4  │
│  in-memory)  │  │  (streaming SSE)  │
└──────────────┘  └───────────────────┘
        ▲
┌───────┴──────────┐
│  holdings.csv    │
│  trades.csv      │
│  (pandas loaded) │
└──────────────────┘
```

---

## 🤖 Full Build Prompt for AI Code Generation

> **Copy this entire prompt into Claude.ai, Cursor, or any AI coding assistant.**

---

```
You are an expert Python developer and ML engineer specializing in RAG systems,
FastAPI, and financial data applications. Build a complete, production-ready
backend + React Native frontend with the following specifications:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT: Portfolio Intelligence RAG Chatbot
BACKEND: Python 3.11 + FastAPI + Anthropic Claude API
FRONTEND: React Native (Expo) with TypeScript
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

════════════════════════════════════════════════════
PART 1: PYTHON FASTAPI BACKEND
════════════════════════════════════════════════════

## 1.1 — Project Structure

  backend/
  ├── main.py
  ├── requirements.txt
  ├── .env
  ├── data/
  │   ├── holdings.csv
  │   └── trades.csv
  ├── rag/
  │   ├── __init__.py
  │   ├── loader.py
  │   ├── embedder.py
  │   └── retriever.py
  ├── routes/
  │   ├── __init__.py
  │   ├── chat.py
  │   ├── summary.py
  │   └── search.py
  └── models/
      ├── __init__.py
      └── schemas.py

## 1.2 — requirements.txt

  fastapi==0.111.0
  uvicorn[standard]==0.30.0
  python-dotenv==1.0.1
  pandas==2.2.2
  numpy==1.26.4
  anthropic==0.28.0
  sentence-transformers==3.0.1
  scikit-learn==1.5.0
  pydantic==2.7.4
  sse-starlette==2.1.0

## 1.3 — .env

  ANTHROPIC_API_KEY=sk-ant-your-key-here
  TOP_K=8
  EMBEDDING_MODEL=all-MiniLM-L6-v2
  PORT=8000

## 1.4 — models/schemas.py

  from pydantic import BaseModel

  class Message(BaseModel):
      role: str  # "user" or "assistant"
      content: str

  class ChatRequest(BaseModel):
      query: str
      conversation_history: list[Message] = []

  class PortfolioBreakdown(BaseModel):
      portfolio_name: str
      holdings_count: int
      total_mv_base: float
      total_pl_ytd: float

  class SecurityStat(BaseModel):
      sec_name: str
      value: float

  class TradeTypeStat(BaseModel):
      trade_type: str
      count: int
      total_cash: float

  class PortfolioSummary(BaseModel):
      total_holdings: int
      total_trades: int
      total_mv_base: float
      total_pl_ytd: float
      total_pl_mtd: float
      total_pl_qtd: float
      portfolios: list[PortfolioBreakdown]
      top_securities_by_mv: list[SecurityStat]
      top_securities_by_pl: list[SecurityStat]
      trades_by_type: list[TradeTypeStat]

## 1.5 — rag/loader.py

Load both CSVs with pandas. Convert every row into a natural language string.
Also create aggregate summary chunks.

  HOLDINGS CSV columns:
  AsOfDate, OpenDate, CloseDate, ShortName, PortfolioName,
  StrategyRefShortName, Strategy1RefShortName, Strategy2RefShortName,
  CustodianName, DirectionName, SecurityId, SecurityTypeName, SecName,
  StartQty, Qty, StartPrice, Price, StartFXRate, FXRate,
  MV_Local, MV_Base, PL_DTD, PL_QTD, PL_MTD, PL_YTD

  Row-to-text format:
  "Portfolio '{PortfolioName}' holds {Qty} units of {SecName}
   (type: {SecurityTypeName}), direction: {DirectionName},
   custodian: {CustodianName}, strategy: {StrategyRefShortName},
   as of {AsOfDate}, price: {Price}, FX rate: {FXRate},
   market value base: {MV_Base}, PL_DTD: {PL_DTD},
   PL_MTD: {PL_MTD}, PL_QTD: {PL_QTD}, PL_YTD: {PL_YTD}."

  TRADES CSV columns:
  id, RevisionId, AllocationId, TradeTypeName, SecurityId, SecurityType,
  Name, Ticker, CUSIP, ISIN, TradeDate, SettleDate, Quantity, Price,
  TradeFXRate, Principal, Interest, TotalCash, AllocationQTY,
  AllocationPrincipal, AllocationInterest, AllocationFees, AllocationCash,
  PortfolioName, CustodianName, StrategyName, Strategy1Name, Strategy2Name,
  Counterparty, AllocationRule, IsCustomAllocation

  Row-to-text format:
  "Trade {id}: {TradeTypeName} of {Quantity} units of {Name}
   (ticker: {Ticker}, ISIN: {ISIN}), trade date: {TradeDate},
   settle date: {SettleDate}, price: {Price}, total cash: {TotalCash},
   portfolio: {PortfolioName}, custodian: {CustodianName},
   strategy: {StrategyName}, counterparty: {Counterparty},
   allocation rule: {AllocationRule}."

  Aggregate chunks to generate:
  - Per-portfolio summary: "Portfolio X: N holdings, total MV: Y, total YTD PL: Z"
  - Top 10 holdings by MV_Base
  - Trade counts by TradeTypeName
  - Long vs short position totals

  Export: load_all_chunks() -> tuple[list[str], pd.DataFrame, pd.DataFrame]

## 1.6 — rag/embedder.py

  Use sentence-transformers (all-MiniLM-L6-v2, local model, no API key needed).
  Store embeddings as L2-normalized numpy matrix.
  Cosine similarity = dot product on normalized vectors.

  class VectorStore:
    - __init__(model_name): load SentenceTransformer
    - build(chunks: list[str]): encode all chunks, store matrix
    - search(query: str, top_k: int) -> list[str]: return top-K chunks

  Expose singleton: vector_store = VectorStore(...)

## 1.7 — rag/retriever.py

  Use anthropic Python SDK for Claude API (streaming).

  SYSTEM_PROMPT = """
  You are PortfolioIQ, an expert financial portfolio analyst assistant.
  You answer questions about the user's investment portfolio using the
  provided context from their holdings and trades data.
  - Be precise with numbers and always specify P&L time periods (DTD/MTD/QTD/YTD).
  - If a question cannot be answered from context, say so clearly.
  - Use markdown formatting: tables for comparisons, bullets for lists.
  """

  async def stream_rag_response(query, conversation_history, top_k=8):
    1. vector_store.search(query, top_k) → context chunks
    2. Build messages array with history + current query+context
    3. Use client.messages.stream(...) with claude-sonnet-4-20250514
    4. yield each text token as async generator

## 1.8 — routes/chat.py

  POST /api/chat
  Body: ChatRequest { query: str, conversation_history: list[Message] }
  Returns: SSE stream (EventSourceResponse from sse-starlette)

  SSE events:
    event: message  →  data: {"text": "<token>"}
    event: done     →  data: {"status": "complete"}
    event: error    →  data: {"error": "<message>"}

  Use EventSourceResponse wrapping an async generator that calls
  stream_rag_response() and yields formatted SSE dicts.

## 1.9 — routes/summary.py

  GET /api/summary
  Returns: PortfolioSummary (Pydantic model)

  Compute using pandas groupby on freshly loaded holdings + trades:
  - total_holdings, total_trades
  - total_mv_base (sum of MV_Base)
  - total_pl_ytd, total_pl_mtd, total_pl_qtd
  - portfolios: group by PortfolioName → count + sum MV_Base + sum PL_YTD
  - top_securities_by_mv: nlargest(5, "MV_Base")
  - top_securities_by_pl: nlargest(5, "PL_YTD")
  - trades_by_type: groupby TradeTypeName → count + sum TotalCash

## 1.10 — routes/search.py

  GET /api/search?q=<query>&top_k=5
  Returns: {"query": str, "results": list[str]}
  Calls vector_store.search(q, top_k)
  (Debug/preview endpoint)

## 1.11 — main.py

  FastAPI app with:
  - asynccontextmanager lifespan: on startup call load_all_chunks()
    then vector_store.build(chunks). Log progress.
  - CORSMiddleware with allow_origins=["*"]
  - Include all 3 routers with prefix="/api"
  - Run with uvicorn on port from .env

  Auto-generated Swagger docs at: http://localhost:8000/docs

## 1.12 — Run Commands

  python -m venv venv
  source venv/bin/activate       # Windows: venv\Scripts\activate
  pip install -r requirements.txt

  # Place CSV files in backend/data/
  # Set ANTHROPIC_API_KEY in .env

  python main.py
  # OR: uvicorn main:app --reload --port 8000

  # Test:
  curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"query":"What is my total YTD P&L?","conversation_history":[]}'

  # Visit docs:
  open http://localhost:8000/docs

════════════════════════════════════════════════════
PART 2: REACT NATIVE APP (Expo + TypeScript)
════════════════════════════════════════════════════

## 2.1 — Setup

  npx create-expo-app portfolio-chatbot --template expo-template-blank-typescript
  cd portfolio-chatbot
  npx expo install expo-router expo-haptics @react-native-async-storage/async-storage
  npm install react-native-markdown-display @expo/vector-icons

## 2.2 — Screens

### Chat Screen — app/(tabs)/index.tsx

  Layout:
  - Header: "PortfolioIQ 📊" + settings icon
  - Suggested chips (when empty): "Total YTD P&L?", "Top holdings",
    "List buy trades", "Highest P&L portfolio", "Summarize activity"
  - FlatList (inverted) for messages:
      User: right-aligned blue bubble (#1D4ED8)
      AI: left-aligned dark card with "IQ" avatar, markdown rendered
      Animated 3-dot typing indicator while streaming
  - Fixed bottom input bar with send button + haptic feedback

  SSE streaming:
    fetch POST /api/chat with ReadableStream reader
    Parse "data: {json}" lines → append json.text to last AI message
    On event:done → setIsStreaming(false)

  Persist messages to AsyncStorage per conversation.

### Dashboard — app/(tabs)/dashboard.tsx

  Fetch GET /api/summary on mount. Show:
  A) 4 stat cards (horizontal scroll):
     Total Market Value | YTD P&L (green/red) | Holdings | Trades
  B) Portfolio table: Name | Holdings | Market Value | YTD P&L
  C) Top 5 securities bar chart (react-native-svg)
  D) Trades by type grid cards
  E) "Ask about this →" button → chat with pre-filled summary question

### History — app/(tabs)/history.tsx

  List saved conversations from AsyncStorage.
  Show first message preview + timestamp.
  Tap → load in chat. Swipe → delete. "Clear All" button.

### Settings — app/settings.tsx

  Backend URL input (default: http://localhost:8000)
  Theme toggle. Clear history button. App version.

## 2.3 — Design System

  Colors:
    BACKGROUND:   '#0A0E1A'   // deep navy
    SURFACE:      '#131929'   // card
    BORDER:       '#1E2D4A'
    ACCENT:       '#3B82F6'   // blue
    POSITIVE:     '#10B981'   // green P&L
    NEGATIVE:     '#EF4444'   // red P&L
    TEXT_PRIMARY: '#F1F5F9'
    TEXT_MUTED:   '#94A3B8'
    USER_BUBBLE:  '#1D4ED8'

## 2.4 — TypeScript Types — src/types/index.ts

  interface Message { id, role, content, timestamp }
  interface Conversation { id, messages, createdAt, updatedAt }
  interface PortfolioBreakdown { portfolio_name, holdings_count, total_mv_base, total_pl_ytd }
  interface SecurityStat { sec_name, value }
  interface TradeTypeStat { trade_type, count, total_cash }
  interface PortfolioSummary {
    total_holdings, total_trades, total_mv_base,
    total_pl_ytd, total_pl_mtd, total_pl_qtd,
    portfolios, top_securities_by_mv, top_securities_by_pl, trades_by_type
  }

## 2.5 — src/api/client.ts

  API_BASE = configurable (default 'http://localhost:8000')

  sendChatMessage(query, history, onToken, onDone, onError): void
    → fetch + ReadableStream SSE consumer

  getPortfolioSummary(): Promise<PortfolioSummary>
    → fetch GET /api/summary

## 2.6 — File Structure

  portfolio-chatbot/
  ├── app/
  │   ├── (tabs)/
  │   │   ├── _layout.tsx
  │   │   ├── index.tsx          (Chat)
  │   │   ├── dashboard.tsx      (Dashboard)
  │   │   └── history.tsx        (History)
  │   ├── settings.tsx
  │   └── _layout.tsx
  ├── src/
  │   ├── api/client.ts
  │   ├── components/
  │   │   ├── chat/
  │   │   │   ├── MessageBubble.tsx
  │   │   │   ├── TypingIndicator.tsx
  │   │   │   ├── SuggestedQuestions.tsx
  │   │   │   └── ChatInput.tsx
  │   │   ├── dashboard/
  │   │   │   ├── StatCard.tsx
  │   │   │   ├── PortfolioTable.tsx
  │   │   │   └── MiniBarChart.tsx
  │   │   └── shared/Screen.tsx
  │   ├── hooks/
  │   │   ├── useChat.ts
  │   │   └── useSummary.ts
  │   ├── store/conversations.ts
  │   ├── theme/colors.ts
  │   └── types/index.ts
  ├── app.json
  └── package.json

════════════════════════════════════════════════════
PART 3: SAMPLE QUESTIONS THE RAG MUST HANDLE
════════════════════════════════════════════════════

  1.  "What is my total YTD P&L across all portfolios?"
  2.  "Which security has the highest market value?"
  3.  "Show me all trades for portfolio HoldCo-1"
  4.  "Which securities had positive day-over-day P&L?"
  5.  "How many buy trades vs sell trades do I have?"
  6.  "Which custodian holds the most assets?"
  7.  "What strategies are assigned to my holdings?"
  8.  "List the top 5 securities by YTD P&L"
  9.  "What is total market value of long vs short positions?"
  10. "Which counterparty had the most trade volume?"

════════════════════════════════════════════════════
PART 4: BUILD ORDER
════════════════════════════════════════════════════

  Step 1:  Create backend/ structure + requirements.txt + .env
  Step 2:  Write models/schemas.py (Pydantic models)
  Step 3:  Write rag/loader.py (CSV loading + text conversion)
  Step 4:  Write rag/embedder.py (SentenceTransformer + numpy store)
  Step 5:  Write rag/retriever.py (search + Claude streaming)
  Step 6:  Write routes/chat.py (SSE endpoint)
  Step 7:  Write routes/summary.py (stats endpoint)
  Step 8:  Write routes/search.py (debug endpoint)
  Step 9:  Write main.py (FastAPI app + lifespan startup)
  Step 10: Test: uvicorn main:app --reload, hit /docs, run curl tests
  Step 11: Create Expo app, install dependencies
  Step 12: Build Chat screen + SSE streaming
  Step 13: Build Dashboard screen
  Step 14: Build History + Settings screens
  Step 15: Test on iOS/Android

Generate complete working code for every file.
Start with backend files (steps 1-9), then frontend (steps 11-14).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🛠️ Tech Stack Summary

| Layer | Tool | Why |
|---|---|---|
| **Backend language** | Python 3.11 | Best ecosystem for ML/data |
| **API framework** | FastAPI | Async, fast, auto Swagger docs at /docs |
| **Data loading** | pandas | Native CSV handling + groupby aggregations |
| **Embeddings** | sentence-transformers | Local, free, no extra API key |
| **Vector search** | numpy cosine similarity | Zero setup, works in-memory |
| **LLM** | Anthropic Claude `claude-sonnet-4-20250514` | Powerful, streaming |
| **Streaming** | sse-starlette | SSE responses from FastAPI |
| **Mobile** | Expo + React Native | Cross-platform iOS/Android |
| **Routing** | expo-router | File-based navigation |
| **Storage** | AsyncStorage | Persist chat history locally |

---

## 🔑 Getting Your Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. **API Keys** → **Create Key**
3. Add to `backend/.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
> ⚠️ Never put this key in your React Native app — always call through your Python backend.

---

## 🚀 Quick Start

```bash
# ── Backend ──────────────────────────────
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Drop your CSVs in backend/data/
# Add ANTHROPIC_API_KEY to .env

python main.py
# Swagger docs → http://localhost:8000/docs

# ── Frontend (new terminal) ───────────────
cd portfolio-chatbot
npx expo start
# Scan QR with Expo Go app on your phone
```

---

## 💡 RAG Flow Example

```
User: "Which portfolio has the highest YTD P&L?"

Python Backend:
  1. Embed query → float[384] vector (all-MiniLM-L6-v2)
  2. Dot product vs 1,700+ stored vectors → top 8 matches:
       "Portfolio HoldCo-1 summary: 120 holdings, total YTD P&L: 41,054.59"
       "Portfolio HoldCo-3 summary: 85 holdings, total YTD P&L: 28,300.12"
       ... (6 more chunks)
  3. Build Claude prompt with context
  4. Stream SSE tokens → React Native

Claude responds (streamed token by token):
  "Based on your portfolio data, HoldCo-1 leads with a YTD P&L
   of $41,054.59, followed by HoldCo-3 at $28,300..."
```

---

*Generated for: Portfolio RAG Chatbot · Python FastAPI + React Native + Anthropic Claude · March 2026*
