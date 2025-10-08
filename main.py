"""
Main Analysis Pipeline for Central Bank NLP Signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.speech_collector import SpeechCollector, SpeechData
from src.sentiment_analyzer import LoughranMcDonaldSentiment
from src.topic_modeler import TopicModeler
from src.market_data import MarketDataFetcher
from src.correlation_analyzer import CorrelationAnalyzer
import config


class CentralBankNLPPipeline:
    """
    Complete pipeline for analyzing central bank speeches and correlating with market data.
    """
    
    def __init__(self, 
                 n_topics: int = None,
                 min_topic_size: int = None,
                 market_tickers: Dict[str, List[str]] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            n_topics: Number of topics for BERTopic
            min_topic_size: Minimum topic size for BERTopic
            market_tickers: Dictionary of market tickers to track
        """
        self.speech_collector = SpeechCollector()
        self.sentiment_analyzer = LoughranMcDonaldSentiment()
        
        n_topics = n_topics or config.N_TOPICS
        min_topic_size = min_topic_size or config.MIN_TOPIC_SIZE
        self.topic_modeler = TopicModeler(n_topics=n_topics, min_topic_size=min_topic_size)
        
        market_tickers = market_tickers or config.MARKET_TICKERS
        self.market_fetcher = MarketDataFetcher(tickers=market_tickers)
        
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Storage for results
        self.sentiment_results = None
        self.topic_results = None
        self.market_data = None
        self.correlations = None
    
    def load_speeches(self, source: str = 'sample', filepath: Optional[str] = None):
        """
        Load speech data.
        
        Args:
            source: 'sample', 'csv', or 'dataframe'
            filepath: Path to CSV file (if source='csv')
        """
        if source == 'sample':
            df = self.speech_collector.create_sample_data()
            self.speech_collector.load_from_dataframe(df)
            print(f"Loaded {len(self.speech_collector.speeches)} sample speeches")
        elif source == 'csv' and filepath:
            self.speech_collector.load_from_csv(filepath)
            print(f"Loaded {len(self.speech_collector.speeches)} speeches from {filepath}")
        else:
            raise ValueError("Invalid source or missing filepath")
    
    def analyze_sentiment(self) -> pd.DataFrame:
        """
        Analyze sentiment for all speeches.
        
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for speech in self.speech_collector.speeches:
            sentiment = self.sentiment_analyzer.analyze_sentiment(speech.text)
            stance, confidence = self.sentiment_analyzer.classify_monetary_stance(speech.text)
            
            results.append({
                'date': speech.date,
                'speaker': speech.speaker,
                'title': speech.title,
                'central_bank': speech.central_bank,
                'positive': sentiment['positive'],
                'negative': sentiment['negative'],
                'uncertainty': sentiment['uncertainty'],
                'litigious': sentiment['litigious'],
                'net_sentiment': sentiment['net_sentiment'],
                'sentiment_ratio': sentiment['sentiment_ratio'],
                'monetary_stance': stance,
                'stance_confidence': confidence
            })
        
        self.sentiment_results = pd.DataFrame(results)
        self.sentiment_results['date'] = pd.to_datetime(self.sentiment_results['date'])
        self.sentiment_results = self.sentiment_results.sort_values('date')
        
        print(f"\nSentiment Analysis Complete:")
        print(f"  - Analyzed {len(results)} speeches")
        print(f"  - Average net sentiment: {self.sentiment_results['net_sentiment'].mean():.4f}")
        print(f"  - Monetary stance distribution:")
        print(self.sentiment_results['monetary_stance'].value_counts().to_string())
        
        return self.sentiment_results
    
    def analyze_topics(self) -> pd.DataFrame:
        """
        Perform topic modeling on speeches.
        
        Returns:
            DataFrame with topic analysis results
        """
        texts = [speech.text for speech in self.speech_collector.speeches]
        dates = [speech.date for speech in self.speech_collector.speeches]
        
        print(f"\nPerforming topic modeling on {len(texts)} speeches...")
        
        topics, probabilities = self.topic_modeler.fit_transform(texts)
        
        # Get topic info
        topic_info = self.topic_modeler.get_topic_info()
        print(f"\nExtracted {len(topic_info)} topics")
        
        # Extract themes
        themes = self.topic_modeler.extract_topic_themes()
        
        # Create results DataFrame
        results = []
        for i, (text, date) in enumerate(zip(texts, dates)):
            topic_dist = self.topic_modeler.get_topic_distribution(text)
            
            result = {
                'date': date,
                'dominant_topic': topics[i],
                'topic_confidence': probabilities[i][topics[i]] if topics[i] != -1 else 0.0
            }
            result.update(topic_dist)
            results.append(result)
        
        self.topic_results = pd.DataFrame(results)
        self.topic_results['date'] = pd.to_datetime(self.topic_results['date'])
        self.topic_results = self.topic_results.sort_values('date')
        
        print(f"\nTopic Analysis Complete:")
        print(f"  - Topic distribution across speeches:")
        for topic_id, theme in themes.items():
            count = sum(1 for t in topics if t == topic_id)
            print(f"    • Topic {topic_id} ({theme}): {count} speeches")
        
        return self.topic_results
    
    def fetch_market_data(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch market data for analysis period.
        
        Args:
            start_date: Start date (YYYY-MM-DD), defaults to earliest speech date - 30 days
            end_date: End date (YYYY-MM-DD), defaults to latest speech date + 60 days
            
        Returns:
            DataFrame with market data
        """
        if self.speech_collector.speeches:
            dates = [speech.date for speech in self.speech_collector.speeches]
            min_date = min(dates) - timedelta(days=30)
            max_date = max(dates) + timedelta(days=60)
            
            start_date = start_date or min_date.strftime('%Y-%m-%d')
            end_date = end_date or max_date.strftime('%Y-%m-%d')
        else:
            raise ValueError("No speeches loaded. Load speeches first.")
        
        print(f"\nFetching market data from {start_date} to {end_date}...")
        
        self.market_data = self.market_fetcher.fetch_data(start_date, end_date)
        
        print(f"  - Fetched data for {len(self.market_data.columns)} market indicators")
        print(f"  - Date range: {self.market_data.index[0]} to {self.market_data.index[-1]}")
        
        return self.market_data
    
    def calculate_correlations(self, lags: List[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlations between NLP signals and market data.
        
        Args:
            lags: List of lag periods to test (in days)
            
        Returns:
            Dictionary with correlation results
        """
        lags = lags or [0, 1, 5, 10, 30]
        
        if self.sentiment_results is None:
            raise ValueError("Run analyze_sentiment() first")
        if self.topic_results is None:
            raise ValueError("Run analyze_topics() first")
        if self.market_data is None:
            raise ValueError("Run fetch_market_data() first")
        
        print(f"\nCalculating correlations with lags: {lags}")
        
        # Prepare sentiment data aligned with market data
        sentiment_aligned = self.sentiment_results.set_index('date')
        sentiment_cols = ['net_sentiment', 'sentiment_ratio', 'positive', 'negative', 'uncertainty']
        sentiment_series = sentiment_aligned[sentiment_cols].reindex(self.market_data.index)
        
        # Forward fill sentiment (assume sentiment persists until next speech)
        sentiment_series = sentiment_series.fillna(method='ffill').fillna(0)
        
        # Calculate market returns
        market_returns = self.market_fetcher.calculate_returns(self.market_data)
        
        # Sentiment-market correlations
        print("  - Analyzing sentiment-market correlations...")
        sentiment_corr = self.correlation_analyzer.sentiment_market_correlation(
            sentiment_series,
            market_returns,
            lags=lags
        )
        
        # Topic-market correlations
        print("  - Analyzing topic-market correlations...")
        topic_aligned = self.topic_results.set_index('date')
        topic_cols = [col for col in topic_aligned.columns 
                     if col not in ['dominant_topic', 'topic_confidence']]
        topic_series = topic_aligned[topic_cols].reindex(self.market_data.index)
        topic_series = topic_series.fillna(method='ffill').fillna(0)
        
        topic_corr = self.correlation_analyzer.topic_market_correlation(
            topic_series,
            market_returns,
            lags=lags
        )
        
        self.correlations = {
            'sentiment': sentiment_corr,
            'topics': topic_corr
        }
        
        # Find significant signals
        print("\n  - Identifying significant predictive signals...")
        significant_sentiment = self.correlation_analyzer.find_predictive_signals(sentiment_corr)
        significant_topics = self.correlation_analyzer.find_predictive_signals(topic_corr)
        
        print(f"    • Found {len(significant_sentiment)} significant sentiment signals")
        print(f"    • Found {len(significant_topics)} significant topic signals")
        
        return self.correlations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Report text
        """
        if self.correlations is None:
            raise ValueError("Run calculate_correlations() first")
        
        report = self.correlation_analyzer.generate_report(
            self.correlations['sentiment'],
            self.correlations['topics']
        )
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")
        
        return report
    
    def run_full_analysis(self, 
                         speech_source: str = 'sample',
                         speech_filepath: Optional[str] = None,
                         output_report: Optional[str] = None) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            speech_source: Source of speech data ('sample' or 'csv')
            speech_filepath: Path to speech CSV file
            output_report: Path to save the report
            
        Returns:
            Dictionary with all results
        """
        print("=" * 80)
        print("CENTRAL BANK NLP SIGNALS - FULL ANALYSIS PIPELINE")
        print("=" * 80)
        
        # Load speeches
        print("\n[1/5] Loading speech data...")
        self.load_speeches(source=speech_source, filepath=speech_filepath)
        
        # Analyze sentiment
        print("\n[2/5] Analyzing sentiment...")
        sentiment_df = self.analyze_sentiment()
        
        # Analyze topics
        print("\n[3/5] Analyzing topics...")
        topic_df = self.analyze_topics()
        
        # Fetch market data
        print("\n[4/5] Fetching market data...")
        market_df = self.fetch_market_data()
        
        # Calculate correlations
        print("\n[5/5] Calculating correlations...")
        correlations = self.calculate_correlations()
        
        # Generate report
        print("\n" + "=" * 80)
        report = self.generate_report(output_file=output_report)
        print(report)
        
        return {
            'sentiment': sentiment_df,
            'topics': topic_df,
            'market_data': market_df,
            'correlations': correlations,
            'report': report
        }


if __name__ == "__main__":
    # Run the analysis
    pipeline = CentralBankNLPPipeline()
    results = pipeline.run_full_analysis(
        speech_source='sample',
        output_report='analysis_report.txt'
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete! Results saved.")
    print("=" * 80)
