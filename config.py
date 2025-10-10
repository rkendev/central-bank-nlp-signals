"""
Central Bank NLP Signals Configuration
"""

# Loughran-McDonald sentiment word lists URLs
LOUGHRAN_MCDONALD_URL = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"

# Market data configuration
MARKET_TICKERS = {
    "bonds": ["^TNX", "^TYX"],  # 10-year and 30-year Treasury yields
    "equities": ["^GSPC", "^DJI", "^IXIC"],  # S&P 500, Dow Jones, NASDAQ
    "volatility": ["^VIX"]  # VIX volatility index
}

# Date ranges for analysis
DEFAULT_LOOKBACK_DAYS = 30  # Days to look back for market impact

# NLP Configuration
MIN_TOPIC_SIZE = 5  # Minimum size for BERTopic clustering
N_TOPICS = 10  # Number of topics to extract

# Sentiment analysis thresholds
SENTIMENT_THRESHOLDS = {
    "hawkish": 0.1,  # Positive threshold for hawkish sentiment
    "dovish": -0.1   # Negative threshold for dovish sentiment
}

# Central bank sources
CENTRAL_BANKS = {
    "FED": "Federal Reserve",
    "ECB": "European Central Bank",
    "BOE": "Bank of England",
    "BOJ": "Bank of Japan"
}
