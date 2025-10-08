# Dependencies Installation Guide

This project has different dependency levels depending on which features you need.

## Quick Start (Minimal Dependencies)

For basic sentiment analysis and speech processing:

```bash
pip install -r requirements-minimal.txt
```

This installs:
- pandas, numpy (data processing)
- scipy (statistical analysis)
- yfinance (market data - optional but recommended)

## Full Installation

For complete functionality including advanced topic modeling:

```bash
pip install -r requirements.txt
```

This includes all dependencies for:
- BERTopic topic modeling
- Sentence transformers for embeddings
- Deep learning models (PyTorch)
- Visualization tools

**Note**: Full installation may take several minutes and requires ~2-3GB of disk space.

## Feature-Specific Dependencies

### Sentiment Analysis Only
```bash
pip install pandas numpy
```

### Market Data Analysis
```bash
pip install pandas numpy yfinance scipy
```

### Topic Modeling
```bash
pip install pandas numpy bertopic sentence-transformers
```

### Visualization
```bash
pip install matplotlib seaborn plotly
```

## Testing Installation

After installation, verify your setup:

```bash
# Test basic functionality (sentiment analysis)
python demo.py

# Test all available features
python test_basic.py

# Run full pipeline (requires all dependencies)
python main.py
```

## Common Issues

### ImportError: No module named 'bertopic'

**Solution**: The topic modeling features require BERTopic. Either:
1. Install it: `pip install bertopic sentence-transformers`
2. Or use only sentiment analysis features (they work independently)

### Market data fetching fails

**Solution**: This requires internet access. If offline:
1. The sentiment analysis still works
2. You can provide pre-downloaded market data as CSV files

### Slow installation

**Solution**: 
- Use minimal requirements first: `pip install -r requirements-minimal.txt`
- Add features as needed later
- Consider using a virtual environment

## Docker Installation (Alternative)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "demo.py"]
```

## Development Setup

For development with all tools:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Optional
```
