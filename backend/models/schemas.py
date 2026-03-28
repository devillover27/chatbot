from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    conversation_history: List[Message] = []

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
    portfolios: List[PortfolioBreakdown]
    top_securities_by_mv: List[SecurityStat]
    top_securities_by_pl: List[SecurityStat]
    trades_by_type: List[TradeTypeStat]
