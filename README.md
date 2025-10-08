# Central Bank NLP Signals

Analyze central bank speeches using NLP to extract sentiment and topics, then correlate with market data to uncover predictive signals in monetary policy language.

## Overview

This project uses advanced natural language processing (NLP) techniques to analyze central bank speeches and communications, extracting sentiment and thematic content. It then correlates these NLP-derived signals with market data (bond yields, stock indices, volatility) to identify potential predictive relationships between monetary policy language and market movements.

## Key Features

- **Sentiment Analysis**: Uses Loughran-McDonald financial dictionary to analyze sentiment in central bank speeches
- **Topic Modeling**: Employs BERTopic to extract key themes and topics from monetary policy communications
- **Market Data Integration**: Fetches real-time market data using yfinance
- **Correlation Analysis**: Identifies statistically significant relationships between NLP signals and market movements
- **Monetary Stance Classification**: Classifies speeches as hawkish, dovish, or neutral
- **Predictive Signal Detection**: Discovers leading indicators in policy language

## Technologies Used

- **spaCy**: Advanced NLP processing
- **BERTopic**: State-of-the-art topic modeling using transformer embeddings
- **Loughran-McDonald Dictionary**: Financial sentiment analysis
- **yfinance**: Market data retrieval
- **Sentence-Transformers**: High-quality text embeddings
- **pandas & numpy**: Data manipulation and analysis
- **scipy**: Statistical analysis and correlation testing

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rkendev/central-bank-nlp-signals.git
cd central-bank-nlp-signals
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy language model (optional, for advanced text processing):
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Usage

Run the complete analysis pipeline with sample data:

```python
from main import CentralBankNLPPipeline

# Initialize pipeline
pipeline = CentralBankNLPPipeline()

# Run full analysis
results = pipeline.run_full_analysis(
    speech_source='sample',
    output_report='analysis_report.txt'
)

# Access results
sentiment_df = results['sentiment']
topic_df = results['topics']
correlations = results['correlations']
```

### Command Line

```bash
python main.py
```

This will run the analysis with sample speeches and generate a comprehensive report.

## Usage Examples

### 1. Analyze Custom Speeches

```python
import pandas as pd
from main import CentralBankNLPPipeline

# Prepare your speech data
speeches = pd.DataFrame([
    {
        'date': '2024-01-15',
        'speaker': 'Jerome Powell',
        'title': 'Monetary Policy Update',
        'text': 'Your speech text here...',
        'central_bank': 'FED',
        'url': 'https://...'
    }
])

# Save to CSV
speeches.to_csv('my_speeches.csv', index=False)

# Run analysis
pipeline = CentralBankNLPPipeline()
results = pipeline.run_full_analysis(
    speech_source='csv',
    speech_filepath='my_speeches.csv'
)
```

### 2. Sentiment Analysis Only

```python
from src.sentiment_analyzer import LoughranMcDonaldSentiment

analyzer = LoughranMcDonaldSentiment()

text = "The economy shows strong growth but inflation remains elevated."
sentiment = analyzer.analyze_sentiment(text)
stance, confidence = analyzer.classify_monetary_stance(text)

print(f"Net Sentiment: {sentiment['net_sentiment']:.4f}")
print(f"Monetary Stance: {stance} (confidence: {confidence:.2f})")
```

### 3. Topic Modeling

```python
from src.topic_modeler import TopicModeler

texts = [
    "Inflation concerns dominate policy discussions.",
    "Employment data shows strong labor market.",
    "Financial stability risks are monitored closely."
]

modeler = TopicModeler(n_topics=5, min_topic_size=2)
topics, probabilities = modeler.fit_transform(texts)

# Get topic information
topic_info = modeler.get_topic_info()
print(topic_info)
```

### 4. Market Data Analysis

```python
from src.market_data import MarketDataFetcher
from datetime import datetime, timedelta

fetcher = MarketDataFetcher()

# Fetch data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

market_data = fetcher.fetch_data(
    start_date.strftime('%Y-%m-%d'),
    end_date.strftime('%Y-%m-%d')
)

# Analyze market reaction around an event
event_date = datetime(2024, 1, 15)
reaction = fetcher.get_market_reaction(
    event_date,
    lookback_days=5,
    lookforward_days=30
)
```

### 5. Correlation Analysis

```python
from src.correlation_analyzer import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()

# Find correlations with different time lags
correlations = analyzer.sentiment_market_correlation(
    sentiment_scores,
    market_returns,
    lags=[0, 1, 5, 10, 30]
)

# Identify significant signals
signals = analyzer.find_predictive_signals(
    correlations,
    min_correlation=0.3,
    max_pvalue=0.05
)

print(signals)
```

## Project Structure

```
central-bank-nlp-signals/
├── config.py                  # Configuration settings
├── main.py                    # Main analysis pipeline
├── example_usage.py           # Usage examples
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                # Git ignore rules
└── src/
    ├── sentiment_analyzer.py  # Loughran-McDonald sentiment analysis
    ├── topic_modeler.py       # BERTopic topic modeling
    ├── market_data.py         # yfinance market data fetching
    ├── speech_collector.py    # Speech data management
    └── correlation_analyzer.py # Statistical correlation analysis
```

## Configuration

Edit `config.py` to customize:

- Market tickers to track
- Number of topics for BERTopic
- Sentiment analysis thresholds
- Time windows for correlation analysis
- Central bank sources

Example:
```python
MARKET_TICKERS = {
    "bonds": ["^TNX", "^TYX"],
    "equities": ["^GSPC", "^DJI"],
    "volatility": ["^VIX"]
}

N_TOPICS = 10
MIN_TOPIC_SIZE = 5
```

## Data Format

### Speech Data CSV Format

Your CSV file should contain these columns:

| Column | Description | Required |
|--------|-------------|----------|
| date | Date of speech (YYYY-MM-DD) | Yes |
| speaker | Name of speaker | Yes |
| title | Speech title | Yes |
| text | Full speech text | Yes |
| central_bank | Bank identifier (FED, ECB, etc.) | No |
| url | URL to original speech | No |

Example:
```csv
date,speaker,title,text,central_bank,url
2024-01-15,Jerome Powell,Policy Update,"The economy continues...",FED,https://...
```

## Output

The pipeline generates:

1. **Sentiment Analysis Results**: DataFrame with sentiment scores, monetary stance classification
2. **Topic Analysis Results**: Topic distributions, dominant themes
3. **Market Data**: Historical price and return data
4. **Correlation Analysis**: Statistical relationships between NLP signals and markets
5. **Comprehensive Report**: Text report with key findings and predictive signals

## Methodology

### Sentiment Analysis

Uses the Loughran-McDonald financial dictionary, specifically designed for financial text analysis. Calculates:
- Positive/negative sentiment ratios
- Uncertainty and constraint indicators
- Net sentiment scores
- Monetary policy stance (hawkish/dovish/neutral)

### Topic Modeling

Employs BERTopic with transformer-based embeddings:
- Extracts semantic themes from speeches
- Identifies topics like inflation, interest rates, employment, growth
- Provides topic distributions for each speech

### Correlation Analysis

Performs statistical correlation tests:
- Pearson and Spearman correlations
- Multiple time lag analysis (0-30 days)
- P-value significance testing
- Identifies leading vs. lagging indicators

## Research Applications

This tool can be used for:

- **Academic Research**: Study relationships between central bank communication and markets
- **Trading Strategy Development**: Identify potential trading signals from policy language
- **Risk Management**: Anticipate market volatility based on policy tone
- **Economic Analysis**: Track evolution of monetary policy themes over time
- **Policy Analysis**: Compare communication patterns across different central banks

## Limitations

- Sample speeches provided are for demonstration only
- Market data depends on yfinance availability and data quality
- Correlations do not imply causation
- Past patterns may not predict future relationships
- Requires sufficient speech data for meaningful topic modeling

## Contributing

Contributions are welcome! Areas for improvement:

- Additional sentiment dictionaries
- More sophisticated NLP models (e.g., fine-tuned BERT)
- Real-time speech scraping from central bank websites
- Enhanced visualization tools
- Additional market data sources
- Machine learning prediction models

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Loughran-McDonald financial sentiment dictionary
- BERTopic by Maarten Grootendorst
- yfinance for market data access
- spaCy and Hugging Face for NLP tools

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Disclaimer**: This tool is for research and educational purposes only. It does not constitute financial advice. Market predictions based on NLP signals should be used at your own risk.