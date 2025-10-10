"""
Example: Analyze your own central bank speeches
This script shows how to prepare and analyze custom speech data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime
from src.sentiment_analyzer import LoughranMcDonaldSentiment
from src.speech_collector import SpeechCollector


def create_custom_speech_csv():
    """
    Create a sample CSV file with custom speeches.
    This shows the format for your own data.
    """
    
    # Example speeches (you would replace these with your own)
    speeches_data = [
        {
            'date': '2024-03-20',
            'speaker': 'Christine Lagarde',
            'title': 'ECB Monetary Policy Decision',
            'text': '''Today the Governing Council decided to keep the three key ECB 
            interest rates unchanged. Inflation has been declining but remains too high. 
            The disinflation process is well on track. Economic activity remains weak but 
            should pick up over time. We are determined to ensure that inflation returns 
            to our 2% medium-term target in a timely manner.''',
            'central_bank': 'ECB',
            'url': ''
        },
        {
            'date': '2024-03-15',
            'speaker': 'Andrew Bailey',
            'title': 'BOE Inflation Report',
            'text': '''Inflation has fallen significantly from its peak, but core 
            inflation remains persistent. The labor market shows signs of cooling with 
            wage growth moderating. We need to see more evidence that inflation pressures 
            are sustainably weakening before considering any policy changes. Risks to the 
            outlook remain elevated given global uncertainty.''',
            'central_bank': 'BOE',
            'url': ''
        },
        {
            'date': '2024-02-28',
            'speaker': 'Jerome Powell',
            'title': 'Semiannual Monetary Policy Report',
            'text': '''The U.S. economy has been performing better than expected with 
            strong job gains and robust GDP growth. However, progress on inflation has 
            been slower recently. We remain committed to returning inflation to 2 percent 
            and will keep policy restrictive until we are confident inflation is moving 
            sustainably toward our goal. The economy appears to be landing softly.''',
            'central_bank': 'FED',
            'url': ''
        },
        {
            'date': '2024-01-25',
            'speaker': 'Kazuo Ueda',
            'title': 'BOJ Policy Statement',
            'text': '''Japan\'s economy has recovered moderately. However, inflation 
            remains below our target and underlying price pressures are still weak. We 
            will continue with monetary easing to achieve the price stability target in 
            a sustainable manner. Financial conditions remain accommodative and we are 
            carefully monitoring economic and price developments.''',
            'central_bank': 'BOJ',
            'url': ''
        }
    ]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(speeches_data)
    csv_filename = 'custom_speeches_example.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"Created example CSV file: {csv_filename}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of speeches: {len(df)}\n")
    
    return csv_filename


def analyze_custom_speeches(csv_file):
    """
    Analyze speeches from a CSV file.
    
    Args:
        csv_file: Path to CSV file with speech data
    """
    
    print("=" * 80)
    print("CUSTOM SPEECH ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    sentiment_analyzer = LoughranMcDonaldSentiment()
    speech_collector = SpeechCollector()
    
    # Load speeches from CSV
    print(f"Loading speeches from {csv_file}...")
    speech_collector.load_from_csv(csv_file)
    print(f"Loaded {len(speech_collector.speeches)} speeches\n")
    
    # Analyze each speech
    results = []
    
    print("-" * 80)
    print("DETAILED ANALYSIS")
    print("-" * 80)
    
    for i, speech in enumerate(speech_collector.speeches, 1):
        print(f"\n{i}. {speech.central_bank} - {speech.speaker}")
        print(f"   Date: {speech.date.strftime('%Y-%m-%d')}")
        print(f"   Title: {speech.title}")
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze_sentiment(speech.text)
        stance, confidence = sentiment_analyzer.classify_monetary_stance(speech.text)
        
        print(f"   Net Sentiment: {sentiment['net_sentiment']:.4f}")
        print(f"   Stance: {stance.upper()} (confidence: {confidence:.2%})")
        print(f"   Uncertainty: {sentiment['uncertainty']:.4f}")
        
        results.append({
            'date': speech.date,
            'central_bank': speech.central_bank,
            'speaker': speech.speaker,
            'title': speech.title,
            'positive': sentiment['positive'],
            'negative': sentiment['negative'],
            'net_sentiment': sentiment['net_sentiment'],
            'uncertainty': sentiment['uncertainty'],
            'stance': stance,
            'confidence': confidence
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    print("\n1. Stance by Central Bank:")
    stance_by_bank = results_df.groupby(['central_bank', 'stance']).size().unstack(fill_value=0)
    print(stance_by_bank)
    
    print("\n2. Average Sentiment by Central Bank:")
    sentiment_by_bank = results_df.groupby('central_bank').agg({
        'net_sentiment': 'mean',
        'uncertainty': 'mean'
    }).round(4)
    print(sentiment_by_bank)
    
    print("\n3. Most Hawkish to Most Dovish:")
    ranked = results_df.sort_values('net_sentiment', ascending=False)
    for _, row in ranked.iterrows():
        print(f"   {row['central_bank']:6s} ({row['date'].strftime('%Y-%m-%d')}): "
              f"{row['net_sentiment']:+.4f} - {row['stance'].upper()}")
    
    print("\n4. Highest Uncertainty:")
    uncertain = results_df.nlargest(3, 'uncertainty')
    for _, row in uncertain.iterrows():
        print(f"   {row['central_bank']:6s} ({row['speaker']:20s}): "
              f"{row['uncertainty']:.4f}")
    
    # Save results
    output_file = 'analysis_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Trading implications (example)
    print("\n" + "=" * 80)
    print("POTENTIAL MARKET IMPLICATIONS")
    print("=" * 80)
    
    print("\nHawkish Speeches (positive net sentiment):")
    hawkish = results_df[results_df['stance'] == 'hawkish']
    if len(hawkish) > 0:
        print("  → May signal rate hikes or tighter policy")
        print("  → Could lead to higher bond yields")
        print("  → Potential headwinds for equities")
        for _, row in hawkish.iterrows():
            print(f"     • {row['central_bank']} ({row['date'].strftime('%Y-%m-%d')})")
    else:
        print("  None detected")
    
    print("\nDovish Speeches (negative net sentiment):")
    dovish = results_df[results_df['stance'] == 'dovish']
    if len(dovish) > 0:
        print("  → May signal rate cuts or easier policy")
        print("  → Could lead to lower bond yields")
        print("  → Generally supportive for equities")
        for _, row in dovish.iterrows():
            print(f"     • {row['central_bank']} ({row['date'].strftime('%Y-%m-%d')})")
    else:
        print("  None detected")
    
    print("\nHigh Uncertainty Speeches:")
    high_uncertainty = results_df[results_df['uncertainty'] > results_df['uncertainty'].median()]
    if len(high_uncertainty) > 0:
        print("  → Indicates cautious or data-dependent approach")
        print("  → May lead to increased market volatility")
        print("  → Policy direction unclear")
        for _, row in high_uncertainty.iterrows():
            print(f"     • {row['central_bank']} ({row['date'].strftime('%Y-%m-%d')})")
    
    print("\n" + "=" * 80)
    
    return results_df


def main():
    """
    Main function to demonstrate custom speech analysis.
    """
    
    print("CENTRAL BANK SPEECH ANALYZER - CUSTOM DATA EXAMPLE")
    print()
    
    # Step 1: Create example CSV (or use your own)
    print("Step 1: Creating example CSV file...")
    csv_file = create_custom_speech_csv()
    
    print("Step 2: Analyzing speeches...")
    print()
    
    # Step 2: Analyze the speeches
    results = analyze_custom_speeches(csv_file)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Replace the example speeches with your own data
2. Ensure your CSV has these columns: date, speaker, title, text
3. Optional columns: central_bank, url
4. Run this script to analyze your speeches
5. Check analysis_results.csv for detailed output

To analyze market correlations, you'll need:
- Market data (via yfinance or CSV)
- Full dependencies: pip install -r requirements.txt
- Use main.py for complete pipeline
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
