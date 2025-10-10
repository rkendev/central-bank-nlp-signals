"""
Simple demo script that works without network access or heavy dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sentiment_analyzer import LoughranMcDonaldSentiment
from src.speech_collector import SpeechCollector
import pandas as pd


def main():
    print("=" * 80)
    print("CENTRAL BANK NLP SIGNALS - SIMPLE DEMO")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing sentiment analyzer...")
    sentiment_analyzer = LoughranMcDonaldSentiment()
    
    print("Loading sample speeches...")
    speech_collector = SpeechCollector()
    sample_df = speech_collector.create_sample_data()
    speech_collector.load_from_dataframe(sample_df)
    print(f"Loaded {len(speech_collector.speeches)} sample speeches\n")
    
    # Analyze each speech
    print("-" * 80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("-" * 80)
    
    results = []
    for i, speech in enumerate(speech_collector.speeches, 1):
        print(f"\nSpeech #{i}")
        print(f"Date: {speech.date.strftime('%Y-%m-%d')}")
        print(f"Speaker: {speech.speaker}")
        print(f"Title: {speech.title}")
        print()
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze_sentiment(speech.text)
        stance, confidence = sentiment_analyzer.classify_monetary_stance(speech.text)
        
        print(f"Sentiment Metrics:")
        print(f"  Positive Score:    {sentiment['positive']:.4f}")
        print(f"  Negative Score:    {sentiment['negative']:.4f}")
        print(f"  Net Sentiment:     {sentiment['net_sentiment']:.4f}")
        print(f"  Uncertainty:       {sentiment['uncertainty']:.4f}")
        print(f"  Sentiment Ratio:   {sentiment['sentiment_ratio']:.4f}")
        print()
        print(f"Monetary Policy Stance: {stance.upper()}")
        print(f"Confidence: {confidence:.2%}")
        
        results.append({
            'date': speech.date,
            'speaker': speech.speaker,
            'positive': sentiment['positive'],
            'negative': sentiment['negative'],
            'net_sentiment': sentiment['net_sentiment'],
            'uncertainty': sentiment['uncertainty'],
            'stance': stance,
            'confidence': confidence
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    print(f"\nTotal speeches analyzed: {len(results_df)}")
    print(f"\nMonetary stance distribution:")
    print(results_df['stance'].value_counts().to_string())
    
    print(f"\nAverage sentiment metrics:")
    print(f"  Net Sentiment: {results_df['net_sentiment'].mean():.4f}")
    print(f"  Uncertainty:   {results_df['uncertainty'].mean():.4f}")
    
    # Identify most hawkish and dovish speeches
    most_hawkish = results_df.loc[results_df['net_sentiment'].idxmax()]
    most_dovish = results_df.loc[results_df['net_sentiment'].idxmin()]
    
    print(f"\nMost Hawkish Speech:")
    print(f"  Date: {most_hawkish['date'].strftime('%Y-%m-%d')}")
    print(f"  Net Sentiment: {most_hawkish['net_sentiment']:.4f}")
    
    print(f"\nMost Dovish Speech:")
    print(f"  Date: {most_dovish['date'].strftime('%Y-%m-%d')}")
    print(f"  Net Sentiment: {most_dovish['net_sentiment']:.4f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
Hawkish Stance: Indicates restrictive monetary policy (tightening, rate hikes)
  - Typically associated with fighting inflation
  - May lead to higher bond yields
  - Can put pressure on equity markets
  
Dovish Stance: Indicates accommodative monetary policy (easing, rate cuts)
  - Focused on supporting economic growth and employment
  - May lead to lower bond yields
  - Generally supportive of equity markets

Neutral Stance: Balanced or wait-and-see approach
  - No clear policy direction signal
  - Data-dependent decision making

Uncertainty: Higher values indicate more uncertain or cautious language
  - May signal policy hesitation or economic unpredictability
    """)
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
