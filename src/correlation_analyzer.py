"""
Correlation Analysis Module
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class CorrelationAnalyzer:
    """
    Analyze correlations between NLP signals and market movements.
    """
    
    def __init__(self):
        """Initialize correlation analyzer."""
        self.results = []
    
    def calculate_correlation(self, 
                             x: pd.Series, 
                             y: pd.Series,
                             method: str = 'pearson') -> Tuple[float, float]:
        """
        Calculate correlation between two series.
        
        Args:
            x: First series
            y: Second series
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Tuple of (correlation, p-value)
        """
        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:  # Need at least 3 points
            return 0.0, 1.0
        
        if method == 'pearson':
            corr, pval = stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, pval = stats.spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, pval = stats.kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return corr, pval
    
    def sentiment_market_correlation(self,
                                     sentiment_scores: pd.DataFrame,
                                     market_returns: pd.DataFrame,
                                     lags: List[int] = [0, 1, 5, 10, 30]) -> pd.DataFrame:
        """
        Analyze correlation between sentiment and market returns at different lags.
        
        Args:
            sentiment_scores: DataFrame with sentiment scores indexed by date
            market_returns: DataFrame with market returns indexed by date
            lags: List of lag periods to test (in days)
            
        Returns:
            DataFrame with correlation results
        """
        results = []
        
        for sentiment_col in sentiment_scores.columns:
            for market_col in market_returns.columns:
                for lag in lags:
                    try:
                        # Shift market returns to align with sentiment
                        if lag > 0:
                            shifted_returns = market_returns[market_col].shift(-lag)
                        else:
                            shifted_returns = market_returns[market_col]
                        
                        # Calculate correlation
                        corr, pval = self.calculate_correlation(
                            sentiment_scores[sentiment_col],
                            shifted_returns
                        )
                        
                        results.append({
                            'sentiment_metric': sentiment_col,
                            'market_metric': market_col,
                            'lag_days': lag,
                            'correlation': corr,
                            'p_value': pval,
                            'significant': pval < 0.05
                        })
                    except Exception as e:
                        print(f"Error calculating correlation for {sentiment_col} vs {market_col} at lag {lag}: {e}")
                        continue
        
        return pd.DataFrame(results)
    
    def topic_market_correlation(self,
                                 topic_distributions: pd.DataFrame,
                                 market_returns: pd.DataFrame,
                                 lags: List[int] = [0, 1, 5, 10, 30]) -> pd.DataFrame:
        """
        Analyze correlation between topics and market returns.
        
        Args:
            topic_distributions: DataFrame with topic distributions indexed by date
            market_returns: DataFrame with market returns indexed by date
            lags: List of lag periods to test
            
        Returns:
            DataFrame with correlation results
        """
        results = []
        
        for topic_col in topic_distributions.columns:
            for market_col in market_returns.columns:
                for lag in lags:
                    try:
                        if lag > 0:
                            shifted_returns = market_returns[market_col].shift(-lag)
                        else:
                            shifted_returns = market_returns[market_col]
                        
                        corr, pval = self.calculate_correlation(
                            topic_distributions[topic_col],
                            shifted_returns,
                            method='spearman'  # Use Spearman for topic distributions
                        )
                        
                        results.append({
                            'topic': topic_col,
                            'market_metric': market_col,
                            'lag_days': lag,
                            'correlation': corr,
                            'p_value': pval,
                            'significant': pval < 0.05
                        })
                    except Exception as e:
                        print(f"Error calculating correlation for {topic_col} vs {market_col}: {e}")
                        continue
        
        return pd.DataFrame(results)
    
    def find_predictive_signals(self,
                               correlations: pd.DataFrame,
                               min_correlation: float = 0.3,
                               max_pvalue: float = 0.05) -> pd.DataFrame:
        """
        Identify statistically significant predictive signals.
        
        Args:
            correlations: DataFrame with correlation results
            min_correlation: Minimum absolute correlation threshold
            max_pvalue: Maximum p-value threshold
            
        Returns:
            DataFrame with significant predictive signals
        """
        significant = correlations[
            (abs(correlations['correlation']) >= min_correlation) &
            (correlations['p_value'] <= max_pvalue)
        ].copy()
        
        # Sort by absolute correlation
        significant['abs_correlation'] = abs(significant['correlation'])
        significant = significant.sort_values('abs_correlation', ascending=False)
        
        return significant
    
    def analyze_event_impact(self,
                           event_dates: List[datetime],
                           sentiment_scores: List[float],
                           market_data: pd.DataFrame,
                           windows: List[int] = [1, 5, 10, 30]) -> pd.DataFrame:
        """
        Analyze market impact around specific events (speeches).
        
        Args:
            event_dates: List of event dates
            sentiment_scores: List of sentiment scores for each event
            market_data: DataFrame with market returns
            windows: List of window sizes to analyze
            
        Returns:
            DataFrame with event impact analysis
        """
        results = []
        
        for i, (event_date, sentiment) in enumerate(zip(event_dates, sentiment_scores)):
            for window in windows:
                try:
                    # Find closest market date
                    closest_idx = market_data.index.get_indexer([event_date], method='nearest')[0]
                    
                    # Calculate returns in the window after event
                    if closest_idx + window < len(market_data):
                        window_returns = {}
                        for col in market_data.columns:
                            returns = market_data[col].iloc[closest_idx:closest_idx + window + 1]
                            if len(returns) > 1:
                                cumulative_return = ((returns.iloc[-1] / returns.iloc[0]) - 1) * 100
                                window_returns[col] = cumulative_return
                        
                        if window_returns:
                            avg_return = np.mean(list(window_returns.values()))
                            
                            results.append({
                                'event_date': event_date,
                                'sentiment': sentiment,
                                'window_days': window,
                                'avg_market_return': avg_return,
                                **window_returns
                            })
                except Exception as e:
                    print(f"Error analyzing event {i} at {event_date}: {e}")
                    continue
        
        df = pd.DataFrame(results)
        
        # Calculate correlation between sentiment and returns for each window
        if not df.empty:
            summary = []
            for window in windows:
                window_df = df[df['window_days'] == window]
                if len(window_df) >= 3:
                    corr, pval = self.calculate_correlation(
                        window_df['sentiment'],
                        window_df['avg_market_return']
                    )
                    summary.append({
                        'window_days': window,
                        'correlation': corr,
                        'p_value': pval,
                        'n_events': len(window_df)
                    })
            
            df.summary = pd.DataFrame(summary)
        
        return df
    
    def generate_report(self, 
                       sentiment_correlations: pd.DataFrame,
                       topic_correlations: pd.DataFrame) -> str:
        """
        Generate a text report of correlation findings.
        
        Args:
            sentiment_correlations: Sentiment correlation results
            topic_correlations: Topic correlation results
            
        Returns:
            Text report
        """
        report = []
        report.append("=" * 80)
        report.append("CENTRAL BANK NLP SIGNALS - CORRELATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Sentiment findings
        report.append("SENTIMENT ANALYSIS FINDINGS")
        report.append("-" * 80)
        significant_sentiment = self.find_predictive_signals(sentiment_correlations)
        
        if not significant_sentiment.empty:
            report.append(f"\nFound {len(significant_sentiment)} significant sentiment-market correlations:")
            report.append("")
            for _, row in significant_sentiment.head(10).iterrows():
                report.append(
                    f"  • {row['sentiment_metric']} → {row['market_metric']} "
                    f"(lag: {row['lag_days']} days): "
                    f"r={row['correlation']:.3f}, p={row['p_value']:.4f}"
                )
        else:
            report.append("\nNo significant sentiment-market correlations found.")
        
        report.append("")
        report.append("")
        
        # Topic findings
        report.append("TOPIC ANALYSIS FINDINGS")
        report.append("-" * 80)
        significant_topics = self.find_predictive_signals(topic_correlations)
        
        if not significant_topics.empty:
            report.append(f"\nFound {len(significant_topics)} significant topic-market correlations:")
            report.append("")
            for _, row in significant_topics.head(10).iterrows():
                report.append(
                    f"  • {row['topic']} → {row['market_metric']} "
                    f"(lag: {row['lag_days']} days): "
                    f"r={row['correlation']:.3f}, p={row['p_value']:.4f}"
                )
        else:
            report.append("\nNo significant topic-market correlations found.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
