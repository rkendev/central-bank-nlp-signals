# Quick Reference Card

## Common Commands

### Get Started
```bash
# Minimal install (sentiment analysis only)
pip install pandas numpy scipy
python demo.py

# Full install (all features)
pip install -r requirements.txt
python main.py
```

### Run Examples
```bash
# Demo with sample speeches
python demo.py

# Analyze custom speeches
python analyze_custom.py

# Run basic tests
python test_basic.py
```

## Quick Code Snippets

### Analyze Single Speech
```python
from src.sentiment_analyzer import LoughranMcDonaldSentiment

analyzer = LoughranMcDonaldSentiment()
sentiment = analyzer.analyze_sentiment("Your speech text here")
stance, confidence = analyzer.classify_monetary_stance("Your speech text here")

print(f"Stance: {stance}, Net Sentiment: {sentiment['net_sentiment']:.4f}")
```

### Load and Analyze CSV
```python
from src.speech_collector import SpeechCollector
from src.sentiment_analyzer import LoughranMcDonaldSentiment

collector = SpeechCollector()
collector.load_from_csv('speeches.csv')

analyzer = LoughranMcDonaldSentiment()
for speech in collector.speeches:
    sentiment = analyzer.analyze_sentiment(speech.text)
    print(f"{speech.date}: {sentiment['net_sentiment']:.4f}")
```

### Fetch Market Data
```python
from src.market_data import MarketDataFetcher

fetcher = MarketDataFetcher()
data = fetcher.fetch_data('2024-01-01', '2024-03-31')
print(data.head())
```

### Full Pipeline
```python
from main import CentralBankNLPPipeline

pipeline = CentralBankNLPPipeline()
results = pipeline.run_full_analysis(
    speech_source='csv',
    speech_filepath='my_speeches.csv'
)
```

## Understanding Output

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Net Sentiment | -1 to +1 | Negative = dovish, Positive = hawkish |
| Uncertainty | 0 to 1 | Higher = more uncertain language |
| Stance | categorical | hawkish / dovish / neutral |

### Stance Meanings
- **Hawkish**: Restrictive policy, rate hikes, fighting inflation
- **Dovish**: Accommodative policy, rate cuts, supporting growth  
- **Neutral**: Balanced, wait-and-see, data-dependent

## File Structure
```
├── src/                    # Core modules
├── config.py              # Configuration
├── main.py                # Full pipeline
├── demo.py                # Quick demo
├── analyze_custom.py      # Custom analysis
├── README.md              # Full documentation
├── USAGE.md               # Detailed usage guide
├── INSTALL.md             # Installation help
└── requirements*.txt      # Dependencies
```

## Common Issues

**ModuleNotFoundError**: Install dependencies
```bash
pip install -r requirements-minimal.txt  # or requirements.txt
```

**Market data fails**: Check internet or use offline mode

**Not enough data**: Need 5+ speeches for topic modeling

## Key URLs

- GitHub: https://github.com/rkendev/central-bank-nlp-signals
- Loughran-McDonald: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
- yfinance: https://github.com/ranaroussi/yfinance
- BERTopic: https://maartengr.github.io/BERTopic/

## Support

- Check README.md for overview
- See USAGE.md for detailed examples
- See INSTALL.md for installation help
- Open GitHub issue for bugs

---

**Disclaimer**: For research/educational purposes only. Not financial advice.
