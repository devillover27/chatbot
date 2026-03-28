from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import asyncio
from dotenv import load_dotenv

from rag.loader import load_all_chunks
from rag.embedder import vector_store
from routes import chat, summary, search

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load data and build vector store in background to avoid blocking server start
    print("🚀 Starting PortfolioIQ Backend...")
    
    # We build the vector store immediately, but ensure the server can start listening
    def init_rag():
        chunks, _, _ = load_all_chunks()
        print(f"📦 Loaded {len(chunks)} data chunks. Building vector store...")
        vector_store.build(chunks)
        print("✅ Vector store ready and listening on Port 8000.")

    # Run the expensive build in a thread to keep the main event loop spinning
    # Note: For simple local dev, we do it here synchronously, but uvicorn will start 
    # the process once this lifespan yields or completes.
    init_rag()
    yield
    print("🛑 Shutting down PortfolioIQ Backend...")

app = FastAPI(
    title="PortfolioIQ RAG API",
    description="Backend for the Portfolio Intelligence RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow everything for local dev stability
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "PortfolioIQ API is live", "port": 8000}

# Routes
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(summary.router, prefix="/api", tags=["Summary"])
app.include_router(search.router, prefix="/api", tags=["Search"])

if __name__ == "__main__":
    # Use 127.0.0.1 for maximum reliability on Windows local development
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
