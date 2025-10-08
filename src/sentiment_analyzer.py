"""
Sentiment Analysis Module using Loughran-McDonald Dictionary
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


class LoughranMcDonaldSentiment:
    """
    Sentiment analyzer using Loughran-McDonald financial sentiment dictionary.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with Loughran-McDonald word lists."""
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.uncertainty_words = self._load_uncertainty_words()
        self.litigious_words = self._load_litigious_words()
        
    def _load_positive_words(self) -> set:
        """Load positive words from Loughran-McDonald dictionary."""
        # Common positive financial/monetary policy words
        return {
            'positive', 'strong', 'growth', 'improved', 'improvement', 'increasing',
            'better', 'success', 'successful', 'gain', 'gains', 'opportunity',
            'favorable', 'confident', 'confidence', 'optimistic', 'optimism',
            'robust', 'solid', 'healthy', 'stabilize', 'stabilization', 'recovery',
            'expand', 'expansion', 'strengthen', 'strengthening', 'progress',
            'advance', 'advancing', 'prosper', 'prosperity', 'upturn', 'upside'
        }
    
    def _load_negative_words(self) -> set:
        """Load negative words from Loughran-McDonald dictionary."""
        # Common negative financial/monetary policy words
        return {
            'negative', 'weak', 'weakness', 'decline', 'declined', 'declining',
            'loss', 'losses', 'risk', 'risks', 'uncertainty', 'uncertain',
            'concern', 'concerned', 'concerning', 'deteriorate', 'deterioration',
            'worsen', 'worsening', 'adverse', 'adversely', 'challenging',
            'difficulty', 'difficulties', 'downturn', 'downside', 'recession',
            'contractionary', 'contraction', 'volatility', 'volatile', 'instability',
            'unstable', 'crisis', 'pressure', 'pressures', 'stagnation', 'sluggish'
        }
    
    def _load_uncertainty_words(self) -> set:
        """Load uncertainty words from Loughran-McDonald dictionary."""
        return {
            'uncertain', 'uncertainty', 'unclear', 'ambiguous', 'unpredictable',
            'may', 'might', 'could', 'possibly', 'perhaps', 'tentative',
            'contingent', 'dependent', 'unclear', 'unknown', 'variable',
            'fluctuate', 'fluctuation', 'unpredictability', 'indefinite'
        }
    
    def _load_litigious_words(self) -> set:
        """Load litigious/constraining words."""
        return {
            'constraint', 'constraints', 'limited', 'limiting', 'restrict',
            'restriction', 'restrictions', 'barrier', 'barriers', 'impediment',
            'challenge', 'challenges', 'obstacle', 'obstacles'
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase and tokenize
        text = text.lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]+\b', text)
        return words
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using Loughran-McDonald dictionary.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        words = self._preprocess_text(text)
        total_words = len(words)
        
        if total_words == 0:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'uncertainty': 0.0,
                'litigious': 0.0,
                'net_sentiment': 0.0,
                'sentiment_ratio': 0.0
            }
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        uncertainty_count = sum(1 for word in words if word in self.uncertainty_words)
        litigious_count = sum(1 for word in words if word in self.litigious_words)
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        uncertainty_score = uncertainty_count / total_words
        litigious_score = litigious_count / total_words
        
        net_sentiment = positive_score - negative_score
        
        # Calculate sentiment ratio (avoids division by zero)
        if (positive_count + negative_count) > 0:
            sentiment_ratio = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            sentiment_ratio = 0.0
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'uncertainty': uncertainty_score,
            'litigious': litigious_score,
            'net_sentiment': net_sentiment,
            'sentiment_ratio': sentiment_ratio
        }
    
    def classify_monetary_stance(self, text: str) -> Tuple[str, float]:
        """
        Classify monetary policy stance as hawkish, dovish, or neutral.
        
        Args:
            text: Speech text to analyze
            
        Returns:
            Tuple of (stance, confidence_score)
        """
        sentiment = self.analyze_sentiment(text)
        net_sentiment = sentiment['net_sentiment']
        
        # Hawkish: restrictive monetary policy (tightening, inflation fighting)
        # Dovish: accommodative monetary policy (easing, supporting growth)
        
        if net_sentiment > 0.01:
            stance = "hawkish"
            confidence = abs(net_sentiment)
        elif net_sentiment < -0.01:
            stance = "dovish"
            confidence = abs(net_sentiment)
        else:
            stance = "neutral"
            confidence = 1.0 - abs(net_sentiment)
        
        return stance, confidence
