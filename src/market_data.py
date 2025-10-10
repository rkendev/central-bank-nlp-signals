"""
Market Data Module using yfinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np


class MarketDataFetcher:
    """
    Fetch and process market data using yfinance.
    """
    
    def __init__(self, tickers: Dict[str, List[str]] = None):
        """
        Initialize market data fetcher.
        
        Args:
            tickers: Dictionary of ticker categories and their symbols
        """
        if tickers is None:
            # Default tickers
            self.tickers = {
                "bonds": ["^TNX", "^TYX"],  # 10-year and 30-year Treasury yields
                "equities": ["^GSPC", "^DJI", "^IXIC"],  # S&P 500, Dow Jones, NASDAQ
                "volatility": ["^VIX"]  # VIX volatility index
            }
        else:
            self.tickers = tickers
    
    def fetch_data(self, 
                   start_date: str, 
                   end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """
        Fetch market data for specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with market data
        """
        all_data = {}
        
        for category, ticker_list in self.tickers.items():
            for ticker in ticker_list:
                try:
                    data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date,
                        interval=interval,
                        progress=False
                    )
                    
                    if not data.empty:
                        # Store closing prices
                        all_data[f"{category}_{ticker}"] = data['Close']
                except Exception as e:
                    print(f"Error fetching {ticker}: {str(e)}")
                    continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data into single DataFrame
        df = pd.DataFrame(all_data)
        return df
    
    def calculate_returns(self, prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame with price data
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame with returns
        """
        returns = prices.pct_change(periods=periods)
        return returns
    
    def get_market_reaction(self, 
                           event_date: datetime,
                           lookback_days: int = 5,
                           lookforward_days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get market reaction around a specific event (e.g., speech date).
        
        Args:
            event_date: Date of the event
            lookback_days: Days before event to include
            lookforward_days: Days after event to analyze
            
        Returns:
            Dictionary with market data and metrics
        """
        start_date = event_date - timedelta(days=lookback_days)
        end_date = event_date + timedelta(days=lookforward_days)
        
        # Fetch data
        prices = self.fetch_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if prices.empty:
            return {}
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Calculate cumulative returns from event date
        event_idx = prices.index.get_indexer([event_date], method='nearest')[0]
        
        cumulative_returns = {}
        for col in prices.columns:
            if event_idx < len(prices):
                post_event_prices = prices[col].iloc[event_idx:]
                post_event_returns = (post_event_prices / post_event_prices.iloc[0] - 1) * 100
                cumulative_returns[col] = post_event_returns
        
        cumulative_returns_df = pd.DataFrame(cumulative_returns)
        
        return {
            'prices': prices,
            'returns': returns,
            'cumulative_returns': cumulative_returns_df
        }
    
    def calculate_volatility(self, returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            returns: DataFrame with returns
            window: Rolling window size
            
        Returns:
            DataFrame with volatility metrics
        """
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def get_summary_statistics(self, 
                               prices: pd.DataFrame,
                               event_date: datetime,
                               periods: List[int] = [1, 5, 10, 30]) -> pd.DataFrame:
        """
        Get summary statistics for different periods after an event.
        
        Args:
            prices: Price data
            event_date: Event date
            periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with summary statistics
        """
        try:
            event_idx = prices.index.get_indexer([event_date], method='nearest')[0]
        except:
            return pd.DataFrame()
        
        summary = []
        
        for period in periods:
            if event_idx + period < len(prices):
                period_returns = {}
                for col in prices.columns:
                    start_price = prices[col].iloc[event_idx]
                    end_price = prices[col].iloc[event_idx + period]
                    if pd.notna(start_price) and pd.notna(end_price) and start_price != 0:
                        ret = ((end_price - start_price) / start_price) * 100
                        period_returns[col] = ret
                
                if period_returns:
                    avg_return = np.mean(list(period_returns.values()))
                    summary.append({
                        'period_days': period,
                        'avg_return': avg_return,
                        **period_returns
                    })
        
        return pd.DataFrame(summary)
