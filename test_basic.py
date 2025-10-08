"""
Test script to verify basic functionality without heavy dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sentiment_analyzer import LoughranMcDonaldSentiment
from src.speech_collector import SpeechCollector
import pandas as pd


def test_sentiment_analyzer():
    """Test sentiment analysis"""
    print("Testing Sentiment Analyzer...")
    print("-" * 60)
    
    analyzer = LoughranMcDonaldSentiment()
    
    # Test hawkish text
    hawkish_text = """The Federal Reserve is committed to bringing inflation back down 
    to our 2 percent goal. We are prepared to raise interest rates further if needed. 
    The economy continues to grow at a solid pace and remains strong."""
    
    sentiment = analyzer.analyze_sentiment(hawkish_text)
    stance, confidence = analyzer.classify_monetary_stance(hawkish_text)
    
    print(f"Hawkish Speech Analysis:")
    print(f"  Positive Score: {sentiment['positive']:.4f}")
    print(f"  Negative Score: {sentiment['negative']:.4f}")
    print(f"  Net Sentiment: {sentiment['net_sentiment']:.4f}")
    print(f"  Monetary Stance: {stance} (confidence: {confidence:.2f})")
    print()
    
    # Test dovish text
    dovish_text = """The economy faces significant challenges with declining growth 
    and uncertainty. There are concerns about deteriorating labor market conditions. 
    We need to remain cautious about risks and downside pressures."""
    
    sentiment2 = analyzer.analyze_sentiment(dovish_text)
    stance2, confidence2 = analyzer.classify_monetary_stance(dovish_text)
    
    print(f"Dovish Speech Analysis:")
    print(f"  Positive Score: {sentiment2['positive']:.4f}")
    print(f"  Negative Score: {sentiment2['negative']:.4f}")
    print(f"  Net Sentiment: {sentiment2['net_sentiment']:.4f}")
    print(f"  Monetary Stance: {stance2} (confidence: {confidence2:.2f})")
    print()
    
    return True


def test_speech_collector():
    """Test speech data collection"""
    print("\nTesting Speech Collector...")
    print("-" * 60)
    
    collector = SpeechCollector()
    
    # Create sample data
    sample_df = collector.create_sample_data()
    print(f"Created {len(sample_df)} sample speeches")
    print(f"Columns: {list(sample_df.columns)}")
    print()
    
    # Load into collector
    collector.load_from_dataframe(sample_df)
    print(f"Loaded {len(collector.speeches)} speeches into collector")
    
    # Test retrieving speeches
    if collector.speeches:
        first_speech = collector.speeches[0]
        print(f"\nFirst speech:")
        print(f"  Date: {first_speech.date}")
        print(f"  Speaker: {first_speech.speaker}")
        print(f"  Title: {first_speech.title}")
        print(f"  Text length: {len(first_speech.text)} characters")
    
    return True


def test_market_data():
    """Test market data fetching (requires yfinance)"""
    print("\nTesting Market Data Fetcher...")
    print("-" * 60)
    
    try:
        from src.market_data import MarketDataFetcher
        from datetime import datetime, timedelta
        
        fetcher = MarketDataFetcher()
        
        # Try to fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Fetching market data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        market_data = fetcher.fetch_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not market_data.empty:
            print(f"Successfully fetched data for {len(market_data.columns)} market indicators")
            print(f"Date range: {market_data.index[0]} to {market_data.index[-1]}")
            print(f"Market indicators: {list(market_data.columns)}")
        else:
            print("Warning: No market data fetched (might be network issue)")
        
        return True
    except Exception as e:
        print(f"Market data test skipped or failed: {str(e)}")
        return False


def test_integration():
    """Test basic integration"""
    print("\nTesting Integration...")
    print("-" * 60)
    
    # Collect speeches
    collector = SpeechCollector()
    sample_df = collector.create_sample_data()
    collector.load_from_dataframe(sample_df)
    
    # Analyze sentiment for all speeches
    analyzer = LoughranMcDonaldSentiment()
    
    results = []
    for speech in collector.speeches:
        sentiment = analyzer.analyze_sentiment(speech.text)
        stance, confidence = analyzer.classify_monetary_stance(speech.text)
        
        results.append({
            'date': speech.date,
            'speaker': speech.speaker,
            'net_sentiment': sentiment['net_sentiment'],
            'stance': stance,
            'confidence': confidence
        })
    
    results_df = pd.DataFrame(results)
    print(f"Analyzed {len(results_df)} speeches")
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CENTRAL BANK NLP SIGNALS - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    print()
    
    try:
        # Run tests
        test_sentiment_analyzer()
        test_speech_collector()
        test_market_data()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ All basic tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
