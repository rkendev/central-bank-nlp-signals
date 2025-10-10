"""
Example usage of the Central Bank NLP Signals pipeline
"""

import pandas as pd
from datetime import datetime
from main import CentralBankNLPPipeline


def example_basic_analysis():
    """
    Example 1: Run basic analysis with sample data
    """
    print("Example 1: Basic Analysis with Sample Data")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = CentralBankNLPPipeline()
    
    # Run full analysis
    results = pipeline.run_full_analysis(
        speech_source='sample',
        output_report='example_report.txt'
    )
    
    # Access specific results
    sentiment_df = results['sentiment']
    print("\nSentiment Results:")
    print(sentiment_df[['date', 'speaker', 'net_sentiment', 'monetary_stance']])
    
    return results


def example_custom_speeches():
    """
    Example 2: Analyze custom speech data
    """
    print("\n\nExample 2: Analysis with Custom Speeches")
    print("=" * 80)
    
    # Create custom speech data
    custom_speeches = pd.DataFrame([
        {
            'date': '2024-01-15',
            'speaker': 'Janet Yellen',
            'title': 'Economic Outlook',
            'text': '''The economy continues to show resilience with strong employment 
            and declining inflation. We remain committed to our dual mandate of maximum 
            employment and price stability. The recent data suggests we are making progress 
            toward our inflation target.''',
            'central_bank': 'FED',
            'url': ''
        },
        {
            'date': '2024-02-20',
            'speaker': 'Christine Lagarde',
            'title': 'Monetary Policy Update',
            'text': '''Inflation remains a concern for the euro area. We must maintain 
            our restrictive monetary policy stance to ensure price stability. Economic 
            growth has slowed, but labor markets remain tight. Uncertainty about the 
            economic outlook persists.''',
            'central_bank': 'ECB',
            'url': ''
        }
    ])
    
    # Save to CSV
    custom_speeches.to_csv('custom_speeches.csv', index=False)
    
    # Run analysis
    pipeline = CentralBankNLPPipeline()
    results = pipeline.run_full_analysis(
        speech_source='csv',
        speech_filepath='custom_speeches.csv',
        output_report='custom_report.txt'
    )
    
    return results


def example_component_analysis():
    """
    Example 3: Use individual components separately
    """
    print("\n\nExample 3: Using Individual Components")
    print("=" * 80)
    
    from src.sentiment_analyzer import LoughranMcDonaldSentiment
    from src.topic_modeler import TopicModeler
    
    # Sentiment analysis on single text
    sentiment_analyzer = LoughranMcDonaldSentiment()
    
    text = """The Federal Reserve is committed to achieving maximum employment and 
    price stability. Recent economic data shows strong growth but elevated inflation 
    remains a concern. We will continue to monitor economic conditions carefully."""
    
    sentiment = sentiment_analyzer.analyze_sentiment(text)
    stance, confidence = sentiment_analyzer.classify_monetary_stance(text)
    
    print(f"\nSentiment Analysis:")
    print(f"  Net Sentiment: {sentiment['net_sentiment']:.4f}")
    print(f"  Monetary Stance: {stance} (confidence: {confidence:.2f})")
    
    # Topic modeling on multiple texts
    texts = [
        "Inflation concerns continue to dominate monetary policy discussions.",
        "Employment data shows strong labor market conditions.",
        "Financial stability risks are being monitored closely by the committee."
    ]
    
    topic_modeler = TopicModeler(n_topics=3, min_topic_size=1)
    topics, probs = topic_modeler.fit_transform(texts)
    
    print(f"\nTopic Modeling:")
    print(f"  Identified topics: {set(topics)}")
    
    return sentiment, topics


def example_market_correlation():
    """
    Example 4: Focus on market correlation analysis
    """
    print("\n\nExample 4: Market Correlation Analysis")
    print("=" * 80)
    
    pipeline = CentralBankNLPPipeline()
    
    # Load and analyze speeches
    pipeline.load_speeches(source='sample')
    sentiment_df = pipeline.analyze_sentiment()
    topic_df = pipeline.analyze_topics()
    
    # Fetch market data
    market_df = pipeline.fetch_market_data()
    
    # Calculate correlations with custom lags
    correlations = pipeline.calculate_correlations(lags=[0, 1, 3, 5, 7, 14, 30])
    
    # Find top predictive signals
    from src.correlation_analyzer import CorrelationAnalyzer
    analyzer = CorrelationAnalyzer()
    
    top_sentiment_signals = analyzer.find_predictive_signals(
        correlations['sentiment'],
        min_correlation=0.2,  # Lower threshold for demo
        max_pvalue=0.1
    )
    
    print("\nTop Sentiment-Market Correlations:")
    print(top_sentiment_signals.head(5).to_string())
    
    return correlations


def example_visualization():
    """
    Example 5: Generate visualizations
    """
    print("\n\nExample 5: Visualization")
    print("=" * 80)
    
    # Note: This would require matplotlib/plotly
    # Just showing the data preparation here
    
    pipeline = CentralBankNLPPipeline()
    results = pipeline.run_full_analysis(speech_source='sample')
    
    sentiment_df = results['sentiment']
    
    # Prepare data for plotting
    print("\nSentiment over time:")
    print(sentiment_df[['date', 'net_sentiment', 'monetary_stance']])
    
    print("\nThis data can be visualized using matplotlib or plotly")
    print("Example: plt.plot(sentiment_df['date'], sentiment_df['net_sentiment'])")
    
    return sentiment_df


if __name__ == "__main__":
    # Run examples
    print("CENTRAL BANK NLP SIGNALS - USAGE EXAMPLES")
    print("=" * 80)
    
    # Example 1: Basic analysis
    example_basic_analysis()
    
    # Example 2: Component analysis
    # example_component_analysis()
    
    # Uncomment to run other examples:
    # example_custom_speeches()
    # example_market_correlation()
    # example_visualization()
    
    print("\n\nAll examples completed successfully!")
