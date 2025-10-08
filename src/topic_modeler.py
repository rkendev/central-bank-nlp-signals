"""
Topic Modeling Module using BERTopic
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class TopicModeler:
    """
    Topic modeling using BERTopic for central bank speeches.
    """
    
    def __init__(self, n_topics: int = 10, min_topic_size: int = 5):
        """
        Initialize BERTopic model.
        
        Args:
            n_topics: Number of topics to extract
            min_topic_size: Minimum size for topic clustering
        """
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        
        # Use a financial domain-tuned embedding model if available,
        # otherwise use general-purpose model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize BERTopic
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=self.min_topic_size,
            nr_topics=self.n_topics,
            verbose=False
        )
        
        self.topics = None
        self.probabilities = None
        
    def fit_transform(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fit the topic model and transform documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            Tuple of (topics, probabilities)
        """
        if len(documents) < self.min_topic_size:
            print(f"Warning: Only {len(documents)} documents provided, "
                  f"which is less than min_topic_size={self.min_topic_size}")
        
        self.topics, self.probabilities = self.model.fit_transform(documents)
        return self.topics, self.probabilities
    
    def get_topic_info(self) -> pd.DataFrame:
        """
        Get information about all topics.
        
        Returns:
            DataFrame with topic information
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        
        return self.model.get_topic_info()
    
    def get_topic_words(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """
        Get top words for a specific topic.
        
        Args:
            topic_id: Topic ID
            n_words: Number of top words to return
            
        Returns:
            List of (word, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        
        topic = self.model.get_topic(topic_id)
        if topic:
            return topic[:n_words]
        return []
    
    def get_document_topic(self, document: str) -> Tuple[int, float]:
        """
        Get the topic for a single document.
        
        Args:
            document: Text document
            
        Returns:
            Tuple of (topic_id, probability)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        
        topics, probs = self.model.transform([document])
        return topics[0], probs[0][topics[0]] if topics[0] != -1 else 0.0
    
    def extract_topic_themes(self) -> Dict[int, str]:
        """
        Extract interpretable themes from topics based on top words.
        
        Returns:
            Dictionary mapping topic IDs to theme descriptions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        
        topic_info = self.get_topic_info()
        themes = {}
        
        # Define monetary policy related themes based on keywords
        theme_keywords = {
            'inflation': ['inflation', 'price', 'prices', 'cpi', 'pce'],
            'interest_rates': ['rate', 'rates', 'interest', 'policy', 'fed'],
            'employment': ['employment', 'jobs', 'labor', 'unemployment', 'wage'],
            'growth': ['growth', 'gdp', 'economy', 'economic', 'output'],
            'financial_stability': ['financial', 'stability', 'market', 'credit', 'bank'],
            'monetary_policy': ['monetary', 'policy', 'committee', 'decision', 'stance']
        }
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Get top words for this topic
            top_words = self.get_topic_words(topic_id, n_words=20)
            words = [word.lower() for word, _ in top_words]
            
            # Find best matching theme
            best_theme = 'general_economic_conditions'
            max_matches = 0
            
            for theme, keywords in theme_keywords.items():
                matches = sum(1 for word in words if any(kw in word for kw in keywords))
                if matches > max_matches:
                    max_matches = matches
                    best_theme = theme
            
            themes[topic_id] = best_theme
        
        return themes
    
    def get_topic_distribution(self, document: str) -> Dict[str, float]:
        """
        Get topic distribution for a document with theme labels.
        
        Args:
            document: Text document
            
        Returns:
            Dictionary mapping theme names to probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        
        # Get all topic probabilities for the document
        topics, probs = self.model.transform([document])
        
        # Get theme mappings
        themes = self.extract_topic_themes()
        
        # Create distribution with theme names
        distribution = {}
        for topic_id, theme in themes.items():
            if topic_id < len(probs[0]):
                prob = float(probs[0][topic_id])
                if theme in distribution:
                    distribution[theme] += prob
                else:
                    distribution[theme] = prob
        
        return distribution
