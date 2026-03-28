import os
import re
import json
from typing import List, AsyncGenerator
from dotenv import load_dotenv

from .embedder import vector_store
from models.schemas import Message

load_dotenv()

# -----------------------------------------------------------------------------
# Optimized System Instruction for PortfolioIQ
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are PortfolioIQ, a world-class financial portfolio analyst.
Your goal is to provide **highly optimized, concise, and professional** answers from the provided context.

RULES:
1. **Be Concise**: Answer the specific question immediately. Avoid conversational filler like "Based on the data provided" or "Certainly! I can help with that."
2. **Strict Data Access**: Only use the provided context. If the data is missing, say "I don't have that specific data in your portfolio records yet."
3. **Optimized Numbers**: Always specify currency (USD) and P&L periods (YTD/MTD/etc.). Use tables for multi-row data.
4. **Visual Insight**: If the user asks for a breakdown, comparative status, or historical distribution, include a Chart block at the END of your response in this EXACT format:
   :::chart {
     "type": "bar" | "pie",
     "labels": ["string", "string"], 
     "values": [number, number],
     "title": "Short Title"
   } :::
   Only provide ONE chart block if it adds significant value. Do not repeat the same data in text if the chart is clear.
"""

def get_api_keys() -> List[str]:
    """Retrieves and parses the list of Gemini API Keys from environment."""
    raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
    if not raw_keys:
        return []
    return [k.strip() for k in raw_keys.split(",") if k.strip()]

async def stream_rag_response(query: str, conversation_history: List[Message], top_k: int = 8) -> AsyncGenerator[str, None]:
    """Streams an optimized response with API key rotation for maximum resilience."""
    
    # 1. Retrieve context
    context_chunks = vector_store.search(query, top_k)
    context_text = "\n\n".join(context_chunks)

    # 2. Get keys for rotation
    api_keys = get_api_keys()
    
    if api_keys:
        from google import genai
        from google.genai import types

        # Try each available key if Rate Limit occurs
        for key_index, current_key in enumerate(api_keys):
            try:
                client = genai.Client(api_key=current_key)
                
                history = []
                for msg in conversation_history:
                    role = "user" if msg.role == "user" else "model"
                    history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

                user_content = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"
                history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

                # Preference for gemini-2.0-flash (fastest & smartest for RAG)
                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1, # Low temperature for precision
                    ),
                )
                
                for chunk in response:
                    text = getattr(chunk, "text", "") or ""
                    if text:
                        yield text
                
                return # Successful completion, exit

            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = any(kw in error_str for kw in ["quota", "429", "resource_exhausted", "limit"])
                
                if is_quota_error and key_index < len(api_keys) - 1:
                    # Silently fail over to the next key
                    continue
                else:
                    # Final key failed or non-quota error
                    # yield f"⚠️ Analysis Engine Error: {str(e)}\n\n"
                    break

    # 3. Fallback: Intelligent Local Formatter (if LLM fails)
    yield "--- *Local PortfolioIQ Engine Fallback* ---\n"
    from .retriever import format_smart_answer
    answer = format_smart_answer(query, context_chunks)
    for word in answer.split(" "):
        yield word + " "
