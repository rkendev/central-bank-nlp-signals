"""
Central Bank NLP Signals Package
"""

__version__ = '1.0.0'

# Import modules conditionally to handle missing dependencies gracefully
__all__ = []

try:
    from .sentiment_analyzer import LoughranMcDonaldSentiment
    __all__.append('LoughranMcDonaldSentiment')
except ImportError:
    pass

try:
    from .topic_modeler import TopicModeler
    __all__.append('TopicModeler')
except ImportError:
    pass

try:
    from .market_data import MarketDataFetcher
    __all__.append('MarketDataFetcher')
except ImportError:
    pass

try:
    from .speech_collector import SpeechCollector, SpeechData
    __all__.extend(['SpeechCollector', 'SpeechData'])
except ImportError:
    pass

try:
    from .correlation_analyzer import CorrelationAnalyzer
    __all__.append('CorrelationAnalyzer')
except ImportError:
    pass
