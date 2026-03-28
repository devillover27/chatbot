from fastapi import APIRouter
from models.schemas import PortfolioSummary, PortfolioBreakdown, SecurityStat, TradeTypeStat
from rag.loader import load_all_chunks
import pandas as pd

router = APIRouter()

@router.get("/summary", response_model=PortfolioSummary)
async def get_summary():
    # In a real app, we'd cache these. For now, re-load.
    _, holdings_df, trades_df = load_all_chunks()

    if holdings_df.empty:
        return PortfolioSummary(
            total_holdings=0, total_trades=0, total_mv_base=0, 
            total_pl_ytd=0, total_pl_mtd=0, total_pl_qtd=0,
            portfolios=[], top_securities_by_mv=[], 
            top_securities_by_pl=[], trades_by_type=[]
        )

    # basic metrics
    total_holdings = len(holdings_df)
    total_trades = len(trades_df)
    total_mv_base = holdings_df["MV_Base"].sum()
    total_pl_ytd = holdings_df["PL_YTD"].sum()
    total_pl_mtd = holdings_df["PL_MTD"].sum()
    total_pl_qtd = holdings_df["PL_QTD"].sum()

    # breakdown by portfolio
    portfolio_groups = holdings_df.groupby("PortfolioName").agg({
        "MV_Base": "sum",
        "PL_YTD": "sum"
    })
    portfolios = [
        PortfolioBreakdown(
            portfolio_name=name,
            holdings_count=len(holdings_df[holdings_df["PortfolioName"] == name]),
            total_mv_base=row["MV_Base"],
            total_pl_ytd=row["PL_YTD"]
        ) for name, row in portfolio_groups.iterrows()
    ]

    # top securities
    top_mv = holdings_df.nlargest(5, "MV_Base")
    top_securities_by_mv = [
        SecurityStat(sec_name=row["SecName"], value=row["MV_Base"]) 
        for _, row in top_mv.iterrows()
    ]

    top_pl = holdings_df.nlargest(5, "PL_YTD")
    top_securities_by_pl = [
        SecurityStat(sec_name=row["SecName"], value=row["PL_YTD"]) 
        for _, row in top_pl.iterrows()
    ]

    # trades by type
    if not trades_df.empty:
        trades_by_type_group = trades_df.groupby("TradeTypeName").agg({
            "TotalCash": "sum"
        })
        trades_by_type = [
            TradeTypeStat(
                trade_type=name,
                count=len(trades_df[trades_df["TradeTypeName"] == name]),
                total_cash=row["TotalCash"]
            ) for name, row in trades_by_type_group.iterrows()
        ]
    else:
        trades_by_type = []

    return PortfolioSummary(
        total_holdings=total_holdings,
        total_trades=total_trades,
        total_mv_base=total_mv_base,
        total_pl_ytd=total_pl_ytd,
        total_pl_mtd=total_pl_mtd,
        total_pl_qtd=total_pl_qtd,
        portfolios=portfolios,
        top_securities_by_mv=top_securities_by_mv,
        top_securities_by_pl=top_securities_by_pl,
        trades_by_type=trades_by_type
    )
