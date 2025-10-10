# Usage Guide

This guide provides detailed instructions for using the Central Bank NLP Signals toolkit.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Analyzing Speeches](#analyzing-speeches)
3. [Market Data Analysis](#market-data-analysis)
4. [Correlation Analysis](#correlation-analysis)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)

## Quick Start

### 1. Basic Sentiment Analysis

The simplest way to analyze speeches:

```python
from src.sentiment_analyzer import LoughranMcDonaldSentiment

analyzer = LoughranMcDonaldSentiment()

speech = """The Federal Reserve is committed to bringing inflation back 
down to our 2 percent goal. Economic growth remains strong."""

# Get sentiment scores
sentiment = analyzer.analyze_sentiment(speech)
print(f"Net Sentiment: {sentiment['net_sentiment']:.4f}")

# Get monetary policy stance
stance, confidence = analyzer.classify_monetary_stance(speech)
print(f"Stance: {stance} (confidence: {confidence:.2%})")
```

### 2. Analyzing Multiple Speeches

Use the demo script for quick analysis:

```bash
python demo.py
```

### 3. Analyzing Custom Data

Use the custom analysis script:

```bash
python analyze_custom.py
```

## Analyzing Speeches

### Preparing Your Data

Create a CSV file with these columns:

| Column | Description | Required |
|--------|-------------|----------|
| date | Speech date (YYYY-MM-DD) | Yes |
| speaker | Speaker name | Yes |
| title | Speech title | Yes |
| text | Full speech text | Yes |
| central_bank | Bank (FED/ECB/BOE/BOJ) | No |
| url | Source URL | No |

Example CSV:
```csv
date,speaker,title,text,central_bank,url
2024-01-15,Jerome Powell,Policy Update,"Text here...",FED,https://...
```

### Loading and Analyzing

```python
from src.speech_collector import SpeechCollector
from src.sentiment_analyzer import LoughranMcDonaldSentiment

# Load speeches
collector = SpeechCollector()
collector.load_from_csv('my_speeches.csv')

# Analyze each speech
analyzer = LoughranMcDonaldSentiment()

for speech in collector.speeches:
    sentiment = analyzer.analyze_sentiment(speech.text)
    stance, confidence = analyzer.classify_monetary_stance(speech.text)
    
    print(f"{speech.date}: {stance.upper()}")
    print(f"  Net Sentiment: {sentiment['net_sentiment']:.4f}")
    print(f"  Uncertainty: {sentiment['uncertainty']:.4f}")
```

## Market Data Analysis

### Fetching Market Data

```python
from src.market_data import MarketDataFetcher
from datetime import datetime, timedelta

# Initialize fetcher
fetcher = MarketDataFetcher()

# Define date range
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

# Fetch data
market_data = fetcher.fetch_data(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

print(f"Fetched data for: {list(market_data.columns)}")
```

### Analyzing Market Reaction

```python
# Analyze market reaction around a speech
event_date = datetime(2024, 1, 15)

reaction = fetcher.get_market_reaction(
    event_date,
    lookback_days=5,
    lookforward_days=30
)

# Get returns
returns = reaction['cumulative_returns']
print(returns)
```

## Correlation Analysis

### Full Pipeline

For complete correlation analysis:

```python
from main import CentralBankNLPPipeline

# Initialize
pipeline = CentralBankNLPPipeline()

# Run full analysis
results = pipeline.run_full_analysis(
    speech_source='csv',
    speech_filepath='my_speeches.csv',
    output_report='my_report.txt'
)

# Access results
sentiment_results = results['sentiment']
correlations = results['correlations']
```

### Custom Correlation Analysis

```python
from src.correlation_analyzer import CorrelationAnalyzer
import pandas as pd

analyzer = CorrelationAnalyzer()

# Your sentiment and market data
sentiment_df = pd.DataFrame({
    'net_sentiment': [0.05, 0.02, -0.03],
    'uncertainty': [0.01, 0.02, 0.03]
})

market_returns = pd.DataFrame({
    'stocks': [0.01, -0.02, 0.03],
    'bonds': [-0.01, 0.02, -0.01]
})

# Calculate correlations
correlations = analyzer.sentiment_market_correlation(
    sentiment_df,
    market_returns,
    lags=[0, 1, 5, 10]
)

# Find significant signals
signals = analyzer.find_predictive_signals(correlations)
print(signals)
```

## Understanding Results

### Sentiment Scores

- **Positive Score**: Proportion of positive words (0-1)
- **Negative Score**: Proportion of negative words (0-1)
- **Net Sentiment**: Positive - Negative (can be negative)
- **Uncertainty**: Proportion of uncertainty words (0-1)
- **Sentiment Ratio**: (Pos - Neg) / (Pos + Neg) (-1 to 1)

### Monetary Stance

- **Hawkish**: Restrictive policy (rate hikes, fighting inflation)
  - Positive net sentiment
  - Words like: strong, growth, inflation, tightening
  
- **Dovish**: Accommodative policy (rate cuts, supporting growth)
  - Negative net sentiment
  - Words like: weak, decline, concerns, risks
  
- **Neutral**: Balanced or unclear
  - Near-zero net sentiment
  - Mixed signals

### Correlation Interpretation

- **Correlation > 0.3**: Moderate to strong relationship
- **P-value < 0.05**: Statistically significant
- **Lag**: Days between speech and market reaction
  - Lag 0: Same day
  - Lag 5: 5 days after speech
  - Lag 30: 30 days after speech

## Advanced Usage

### Custom Sentiment Dictionary

Extend the Loughran-McDonald dictionary:

```python
from src.sentiment_analyzer import LoughranMcDonaldSentiment

analyzer = LoughranMcDonaldSentiment()

# Add custom words
analyzer.positive_words.update(['bullish', 'optimistic'])
analyzer.negative_words.update(['bearish', 'pessimistic'])

# Analyze with custom dictionary
sentiment = analyzer.analyze_sentiment(your_text)
```

### Custom Market Tickers

Track different markets:

```python
from src.market_data import MarketDataFetcher

# Define custom tickers
custom_tickers = {
    "currencies": ["EURUSD=X", "GBPUSD=X"],
    "commodities": ["GC=F", "CL=F"],  # Gold, Oil
    "crypto": ["BTC-USD", "ETH-USD"]
}

fetcher = MarketDataFetcher(tickers=custom_tickers)
data = fetcher.fetch_data('2024-01-01', '2024-03-31')
```

### Topic Modeling

For advanced topic analysis (requires full dependencies):

```python
from src.topic_modeler import TopicModeler

# Initialize
modeler = TopicModeler(n_topics=10, min_topic_size=3)

# Fit on speeches
texts = [speech.text for speech in speeches]
topics, probs = modeler.fit_transform(texts)

# Get topic information
topic_info = modeler.get_topic_info()
print(topic_info)

# Get top words for a topic
words = modeler.get_topic_words(topic_id=0, n_words=10)
print(words)
```

### Batch Processing

Process multiple files:

```python
import glob
from main import CentralBankNLPPipeline

# Process all CSV files in a directory
csv_files = glob.glob('speeches/*.csv')

for csv_file in csv_files:
    print(f"Processing {csv_file}...")
    
    pipeline = CentralBankNLPPipeline()
    results = pipeline.run_full_analysis(
        speech_source='csv',
        speech_filepath=csv_file,
        output_report=f'report_{csv_file.split("/")[-1].replace(".csv", ".txt")}'
    )
```

### Exporting Results

Save results in different formats:

```python
# Save to CSV
sentiment_df.to_csv('sentiment_results.csv', index=False)

# Save to Excel
sentiment_df.to_excel('sentiment_results.xlsx', index=False)

# Save to JSON
sentiment_df.to_json('sentiment_results.json', orient='records')

# Save correlations
correlations['sentiment'].to_csv('sentiment_correlations.csv', index=False)
correlations['topics'].to_csv('topic_correlations.csv', index=False)
```

## Tips and Best Practices

1. **Data Quality**: Ensure speeches are complete and properly formatted
2. **Sample Size**: Need at least 5-10 speeches for meaningful topic modeling
3. **Time Alignment**: Match speech dates with market data dates
4. **Lag Selection**: Test multiple lags to find optimal prediction window
5. **Statistical Significance**: Focus on p-value < 0.05 for reliable signals
6. **Combine Signals**: Use both sentiment and topic analysis together
7. **Market Context**: Consider broader market conditions, not just speeches
8. **Backtesting**: Validate findings on historical data before using

## Troubleshooting

### "No module named 'bertopic'"
Solution: Install full dependencies or use minimal features
```bash
pip install -r requirements.txt
```

### "Market data not available"
Solution: Check internet connection or use cached data

### "Not enough speeches for topic modeling"
Solution: Need at least 5 speeches, or reduce `min_topic_size`

### "Correlation analysis fails"
Solution: Ensure date ranges overlap between speeches and market data

## Getting Help

- Check the [README.md](README.md) for overview
- See [INSTALL.md](INSTALL.md) for installation issues
- Review example scripts: `demo.py`, `analyze_custom.py`
- Open an issue on GitHub for bugs or questions
