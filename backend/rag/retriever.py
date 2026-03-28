import os
import re
import json
from typing import List, AsyncGenerator
from dotenv import load_dotenv

from .embedder import vector_store
from models.schemas import Message

load_dotenv()

# -----------------------------------------------------------------------------
# SURGICAL ANALYST DIRECTIVE (Highly Trained Mode)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are PortfolioIQ, a world-class financial analysts' terminal engine.
Your objective is to provide surgical, data-only answers with **zero conversational filler**.

STRICT MANDATES:
1. **Zero Filler**: NEVER say "Hello," "I can help," "Based on the data," or "Certainly." Start with the answer immediately.
2. **Data-First**: If the answer is a metric, provide the value (e.g., "$1.2M USD"). If it's plural, use a Markdown table.
3. **No Hallucinations**: If a data point is missing from the [CONTEXT], output exactly: "Data point missing." No apologies.
4. **Visual Triggers**: If a breakdown or distribution is implied, include a Chart block at the END of your response in this format:
   :::chart {
     "type": "bar" | "pie",
     "labels": ["Label1", "Label2"], 
     "values": [numeric_value1, numeric_value2],
     "title": "Title"
   } :::
   Only provide ONE chart. Use it to replace text when possible to save space.
5. **Precision**: Specific time periods (DTD/YTD) must match the context exactly.
"""

def format_smart_answer(query: str, chunks: List[str]) -> str:
    """Surgical Local fallback engine for immediate response when AI is offline."""
    q = query.lower()
    
    # 1. Total Market Value lookup
    if any(k in q for k in ["total mv", "total value", "total market value", "total account"]):
        for chunk in chunks:
            match = re.search(r"total market value base: ([\d\.]+)", chunk)
            if match:
                return f"${float(match.group(1)):,.2f} USD"
    
    # 2. Portfolio Summaries logic (Surgical Table)
    if "portfolio" in q:
        summaries = [c for c in chunks if "Summary:" in c]
        if summaries:
            table = "| Portfolio | MV Base | YTD P&L |\n|---|---|---|"
            for s in summaries:
                # Extract: Portfolio 'Name' Summary: X holdings, total market value base: Y, total YTD P&L: Z.
                try:
                    name = re.search(r"Portfolio '(.*?)'", s).group(1)
                    mv = re.search(r"market value base: ([\d\.]+)", s).group(1)
                    pl = re.search(r"YTD P&L: ([\-]?[\d\.]+)", s).group(1)
                    table += f"\n| {name} | ${float(mv):,.2f} | ${float(pl):,.2f} |"
                except: continue
            return table

    # 3. Clean fallback (Bullet points, but no filler text)
    top_chunks = [c.split(": ")[1] if ": " in c else c for c in chunks[:3]]
    return "Data points:\n- " + "\n- ".join(top_chunks)

def get_api_keys() -> List[str]:
    """Parses keys from environment."""
    raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
    return [k.strip() for k in raw_keys.split(",") if k.strip() and not k.startswith("YOUR_")]

async def stream_rag_response(query: str, conversation_history: List[Message], top_k: int = 8) -> AsyncGenerator[str, None]:
    """Streams a surgical analyst response with failover."""
    
    # Retrieve & Section Context
    context_chunks = vector_store.search(query, top_k)
    holdings = [c for c in context_chunks if "hold" in c.lower() or "summary" in c.lower()]
    trades = [c for c in context_chunks if "trade" in c.lower()]
    
    structured_context = ""
    if holdings: structured_context += "[SECTION: HOLDINGS]\n" + "\n".join(holdings) + "\n\n"
    if trades: structured_context += "[SECTION: TRADES]\n" + "\n".join(trades) + "\n\n"
    if not structured_context: structured_context = "[CONTEXT]\n" + "\n".join(context_chunks)

    # Try AI Engine
    api_keys = get_api_keys()
    if api_keys:
        from google import genai
        from google.genai import types

        for key_index, current_key in enumerate(api_keys):
            try:
                client = genai.Client(api_key=current_key)
                
                history = []
                for msg in conversation_history:
                    role = "user" if msg.role == "user" else "model"
                    history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

                user_content = f"CONTEXT:\n{structured_context}\n\nUSER QUERY: {query}"
                history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                
                found_text = False
                for chunk in response:
                    text = getattr(chunk, "text", "") or ""
                    if text:
                        found_text = True
                        yield text
                
                if found_text: return

            except Exception as e:
                # Silently rotate keys on quota error
                if ("quota" in str(e).lower() or "429" in str(e)) and key_index < len(api_keys) - 1:
                    continue
                break

    # Final Fallback (Surgical)
    yield "⚠️ *AI Engine Offline. Local Precision Active.*\n\n"
    answer = format_smart_answer(query, context_chunks)
    for word in answer.split(" "):
        yield word + " "
