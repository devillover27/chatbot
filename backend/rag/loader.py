import pandas as pd
import os
from typing import List, Tuple

def load_all_chunks() -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    holdings_path = os.path.join(data_dir, "holdings.csv")
    trades_path = os.path.join(data_dir, "trades.csv")

    chunks = []

    # 1. Load Holdings
    if os.path.exists(holdings_path):
        holdings_df = pd.read_csv(holdings_path)
        for _, row in holdings_df.iterrows():
            chunk = (
                f"Portfolio '{row['PortfolioName']}' holds {row['Qty']} units of {row['SecName']} "
                f"(type: {row['SecurityTypeName']}), direction: {row['DirectionName']}, "
                f"custodian: {row['CustodianName']}, strategy: {row['StrategyRefShortName']}, "
                f"as of {row['AsOfDate']}, price: {row['Price']}, FX rate: {row['FXRate']}, "
                f"market value base: {row['MV_Base']}, PL_DTD: {row['PL_DTD']}, "
                f"PL_MTD: {row['PL_MTD']}, PL_QTD: {row['PL_QTD']}, PL_YTD: {row['PL_YTD']}."
            )
            chunks.append(chunk)
            
        # Per-portfolio summary chunks
        portfolio_groups = holdings_df.groupby("PortfolioName")
        for name, group in portfolio_groups:
            summary = (
                f"Portfolio '{name}' Summary: {len(group)} holdings, "
                f"total market value base: {group['MV_Base'].sum():.2f}, "
                f"total YTD P&L: {group['PL_YTD'].sum():.2f}."
            )
            chunks.append(summary)
            
        # Top 10 holdings by MV_Base
        top_10_mv = holdings_df.nlargest(10, "MV_Base")
        top_mv_chunk = "Top 10 holdings by Market Value (Base): " + ", ".join(
            [f"{row['SecName']} ({row['MV_Base']:.2f})" for _, row in top_10_mv.iterrows()]
        )
        chunks.append(top_mv_chunk)
        
        # Long vs Short
        direction_summary = holdings_df.groupby("DirectionName")["MV_Base"].sum()
        for direction, val in direction_summary.items():
            chunks.append(f"Total Market Value for {direction} positions: {val:.2f}")

    else:
        holdings_df = pd.DataFrame()

    # 2. Load Trades
    if os.path.exists(trades_path):
        trades_df = pd.read_csv(trades_path)
        for _, row in trades_df.iterrows():
            chunk = (
                f"Trade {row['id']}: {row['TradeTypeName']} of {row['Quantity']} units of {row.get('Name', row.get('Ticker', 'Security'))} "
                f"(ticker: {row['Ticker']}, ISIN: {row['ISIN']}), trade date: {row['TradeDate']}, "
                f"settle date: {row['SettleDate']}, price: {row['Price']}, total cash: {row['TotalCash']}, "
                f"portfolio: {row['PortfolioName']}, custodian: {row['CustodianName']}, "
                f"strategy: {row.get('StrategyName', 'N/A')}, counterparty: {row.get('Counterparty', 'N/A')}, "
                f"allocation rule: {row.get('AllocationRule', 'N/A')}."
            )
            chunks.append(chunk)
            
        # Trade counts by type
        trade_type_counts = trades_df.groupby("TradeTypeName").size()
        for ttype, count in trade_type_counts.items():
            chunks.append(f"Total {ttype} trades: {count}")

    else:
        trades_df = pd.DataFrame()

    return chunks, holdings_df, trades_df
