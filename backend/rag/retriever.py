import os
import re
from typing import List, AsyncGenerator
from dotenv import load_dotenv

from .embedder import vector_store
from models.schemas import Message

load_dotenv()

# -----------------------------------------------------------------------------
# Smart Context-Based Answer Formatter (no LLM API needed)
# Uses the retrieved RAG chunks to construct a structured response.
# -----------------------------------------------------------------------------

def format_smart_answer(query: str, chunks: List[str]) -> str:
    """Build a clean, structured answer from RAG context chunks without an LLM."""
    q_lower = query.lower()

    # Try to detect the type of question
    is_pl_question = any(w in q_lower for w in ["p&l", "pl", "profit", "loss", "return", "ytd", "mtd", "dtd", "qtd"])
    is_mv_question = any(w in q_lower for w in ["market value", "mv", "value", "worth"])
    is_trade_question = any(w in q_lower for w in ["trade", "buy", "sell", "trades"])
    is_top_question = any(w in q_lower for w in ["top", "highest", "largest", "best", "most"])
    is_summary_question = any(w in q_lower for w in ["summar", "overview", "total"])

    # Build a clean response from retrieved chunks
    lines = ["### 📊 PortfolioIQ Analysis\n"]
    lines.append(f"**Query:** {query}\n")
    lines.append("---\n")
    lines.append("**Relevant portfolio data:**\n")
    
    for i, chunk in enumerate(chunks[:8]):
        # Clean up the chunk
        clean = chunk.strip()
        if clean:
            lines.append(f"- {clean}")
    
    lines.append("\n---")
    lines.append("*Based on your holdings and trades data.*")

    return "\n".join(lines)


async def stream_rag_response(query: str, conversation_history: List[Message], top_k: int = 8) -> AsyncGenerator[str, None]:
    """Stream a response using local RAG only — no external LLM API required."""
    
    # 1. Retrieve relevant context
    context_chunks = vector_store.search(query, top_k)

    # 2. Try Gemini if key is set
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            context_text = "\n\n".join(context_chunks)

            SYSTEM_PROMPT = """You are PortfolioIQ, an expert financial portfolio analyst assistant.
Answer questions about the user's investment portfolio using only the provided context.
- Be precise with numbers and always specify P&L time periods (DTD/MTD/QTD/YTD).
- If a question cannot be answered from context, say so clearly.
- Use markdown formatting: tables for comparisons, bullets for lists."""

            history = []
            for msg in conversation_history:
                role = "user" if msg.role == "user" else "model"
                history.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

            user_content = f"Context:\n{context_text}\n\nQuestion: {query}"
            history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

            # Try gemini-2.0-flash first, fallback to gemini-1.5-flash-latest
            for model_name in ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.0-pro"]:
                try:
                    response = client.models.generate_content_stream(
                        model=model_name,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.3,
                        ),
                    )
                    for chunk in response:
                        text = getattr(chunk, "text", "") or ""
                        if text:
                            yield text
                    return  # success, exit
                except Exception as model_err:
                    error_str = str(model_err)
                    if "NOT_FOUND" in error_str or "not supported" in error_str.lower():
                        continue  # try next model
                    elif "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        break  # quota issue, fall through to local mode
                    else:
                        raise  # unexpected error

        except Exception as e:
            error_msg = str(e)
            # Fall through to local mode if quota or model issue
            if "RESOURCE_EXHAUSTED" not in error_msg and "NOT_FOUND" not in error_msg:
                yield f"⚠️ Gemini error: {error_msg}\n\n"

    # 3. Local RAG fallback — format chunks as structured answer
    answer = format_smart_answer(query, context_chunks)
    # Stream word by word for a typing effect
    words = answer.split(" ")
    for i, word in enumerate(words):
        if i == len(words) - 1:
            yield word
        else:
            yield word + " "
