"""
Speech Data Collection Module
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import re


class SpeechData:
    """
    Class to manage central bank speech data.
    """
    
    def __init__(self, 
                 date: datetime,
                 speaker: str,
                 title: str,
                 text: str,
                 central_bank: str = "FED",
                 url: Optional[str] = None):
        """
        Initialize speech data.
        
        Args:
            date: Date of the speech
            speaker: Name of the speaker
            title: Title of the speech
            text: Full text of the speech
            central_bank: Central bank identifier (FED, ECB, BOE, BOJ)
            url: URL to the original speech
        """
        self.date = date
        self.speaker = speaker
        self.title = title
        self.text = text
        self.central_bank = central_bank
        self.url = url
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'date': self.date,
            'speaker': self.speaker,
            'title': self.title,
            'text': self.text,
            'central_bank': self.central_bank,
            'url': self.url
        }


class SpeechCollector:
    """
    Collect and manage central bank speeches.
    """
    
    def __init__(self):
        """Initialize speech collector."""
        self.speeches = []
    
    def add_speech(self, speech: SpeechData):
        """Add a speech to the collection."""
        self.speeches.append(speech)
    
    def load_from_dataframe(self, df: pd.DataFrame):
        """
        Load speeches from a pandas DataFrame.
        
        Args:
            df: DataFrame with columns: date, speaker, title, text, central_bank, url
        """
        required_cols = ['date', 'speaker', 'title', 'text']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        
        for _, row in df.iterrows():
            speech = SpeechData(
                date=pd.to_datetime(row['date']) if not isinstance(row['date'], datetime) else row['date'],
                speaker=row['speaker'],
                title=row['title'],
                text=row['text'],
                central_bank=row.get('central_bank', 'FED'),
                url=row.get('url', None)
            )
            self.add_speech(speech)
    
    def load_from_csv(self, filepath: str):
        """
        Load speeches from a CSV file.
        
        Args:
            filepath: Path to CSV file
        """
        df = pd.read_csv(filepath)
        self.load_from_dataframe(df)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert speeches to DataFrame.
        
        Returns:
            DataFrame with all speeches
        """
        return pd.DataFrame([speech.to_dict() for speech in self.speeches])
    
    def get_speeches_by_date_range(self, 
                                    start_date: datetime,
                                    end_date: datetime) -> List[SpeechData]:
        """
        Get speeches within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of speeches within the date range
        """
        return [
            speech for speech in self.speeches
            if start_date <= speech.date <= end_date
        ]
    
    def get_speeches_by_speaker(self, speaker: str) -> List[SpeechData]:
        """
        Get speeches by a specific speaker.
        
        Args:
            speaker: Speaker name
            
        Returns:
            List of speeches by the speaker
        """
        return [
            speech for speech in self.speeches
            if speech.speaker.lower() == speaker.lower()
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean speech text for analysis.
        
        Args:
            text: Raw speech text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        return text.strip()
    
    def create_sample_data(self) -> pd.DataFrame:
        """
        Create sample speech data for demonstration.
        
        Returns:
            DataFrame with sample speeches
        """
        sample_speeches = [
            {
                'date': '2023-11-01',
                'speaker': 'Jerome Powell',
                'title': 'Monetary Policy and Economic Outlook',
                'text': '''The Federal Reserve is committed to bringing inflation back down to our 
                2 percent goal. Recent economic data shows continued strength in the labor market, 
                with unemployment remaining low. However, inflation remains elevated and we are 
                prepared to raise interest rates further if needed. The economy continues to grow 
                at a solid pace, but we remain vigilant about financial stability risks.''',
                'central_bank': 'FED',
                'url': 'https://www.federalreserve.gov/sample'
            },
            {
                'date': '2023-09-15',
                'speaker': 'Jerome Powell',
                'title': 'Price Stability and the Economy',
                'text': '''We have made progress in reducing inflation, but it remains too high. 
                The FOMC is strongly committed to returning inflation to our 2 percent objective. 
                Economic growth has been resilient despite our policy tightening. The labor market 
                shows signs of better balance between supply and demand. We will continue to make 
                our decisions meeting by meeting based on the totality of incoming data.''',
                'central_bank': 'FED',
                'url': 'https://www.federalreserve.gov/sample2'
            },
            {
                'date': '2023-07-26',
                'speaker': 'Jerome Powell',
                'title': 'FOMC Press Conference',
                'text': '''The Committee decided to raise the target range for the federal funds 
                rate to 5.25 to 5.5 percent. Inflation has moderated somewhat since the middle of 
                last year, but remains well above our longer-run goal of 2 percent. Economic activity 
                has been expanding at a moderate pace, and job gains have been robust. The economic 
                outlook remains uncertain and we remain attentive to inflation risks.''',
                'central_bank': 'FED',
                'url': 'https://www.federalreserve.gov/sample3'
            }
        ]
        
        return pd.DataFrame(sample_speeches)
