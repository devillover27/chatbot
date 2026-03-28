from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv

from rag.loader import load_all_chunks
from rag.embedder import vector_store
from routes import chat, summary, search

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load data and build vector store
    print("🚀 Starting PortfolioIQ Backend...")
    chunks, _, _ = load_all_chunks()
    print(f"📦 Loaded {len(chunks)} data chunks. Building vector store...")
    vector_store.build(chunks)
    print("✅ Vector store ready.")
    yield
    # Shutdown: Clean up if needed
    print("🛑 Shutting down PortfolioIQ Backend...")

app = FastAPI(
    title="PortfolioIQ RAG API",
    description="Backend for the Portfolio Intelligence RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(summary.router, prefix="/api", tags=["Summary"])
app.include_router(search.router, prefix="/api", tags=["Search"])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
