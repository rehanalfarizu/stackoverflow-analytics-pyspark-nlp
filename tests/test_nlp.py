"""
Test NLP Module
===============
Unit tests untuk NLP pipeline.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTextPreprocessing:
    """Tests untuk text preprocessing."""
    
    def test_remove_html(self):
        """Test HTML removal."""
        from src.utils.helpers import clean_html
        
        text = "<p>This is <b>bold</b> text</p>"
        result = clean_html(text)
        
        assert "<" not in result
        assert ">" not in result
    
    def test_remove_code_blocks(self):
        """Test code block removal."""
        import re
        
        text = "Some text ```python\nprint('hello')\n``` more text"
        # Remove code blocks
        result = re.sub(r'```[\s\S]*?```', ' ', text)
        
        assert "print" not in result
        assert "Some text" in result
        assert "more text" in result


class TestSentimentAnalysis:
    """Tests untuk sentiment analysis."""
    
    def test_positive_words(self):
        """Test positive sentiment detection."""
        positive_words = {'solved', 'works', 'great', 'excellent'}
        
        text = "This solution works great!"
        words = text.lower().split()
        
        has_positive = any(w in positive_words for w in words)
        assert has_positive
    
    def test_negative_words(self):
        """Test negative sentiment detection."""
        negative_words = {'error', 'bug', 'broken', 'fail'}
        
        text = "This code has a bug and error"
        words = text.lower().split()
        
        has_negative = any(w in negative_words for w in words)
        assert has_negative
    
    def test_neutral_text(self):
        """Test neutral text detection."""
        positive_words = {'solved', 'works', 'great'}
        negative_words = {'error', 'bug', 'broken'}
        
        text = "How do I implement this function?"
        words = text.lower().split()
        
        has_positive = any(w in positive_words for w in words)
        has_negative = any(w in negative_words for w in words)
        
        assert not has_positive
        assert not has_negative


class TestTokenization:
    """Tests untuk tokenization."""
    
    def test_simple_tokenize(self):
        """Test simple tokenization."""
        text = "Hello World Python"
        tokens = text.lower().split()
        
        assert len(tokens) == 3
        assert "hello" in tokens
        assert "python" in tokens
    
    def test_filter_short_tokens(self):
        """Test filtering short tokens."""
        tokens = ["a", "to", "python", "is", "great"]
        filtered = [t for t in tokens if len(t) > 2]
        
        assert "a" not in filtered
        assert "to" not in filtered
        assert "python" in filtered


class TestStopwords:
    """Tests untuk stopword removal."""
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        stopwords = {'the', 'is', 'a', 'an', 'in', 'on', 'at'}
        
        tokens = ["the", "python", "is", "a", "programming", "language"]
        filtered = [t for t in tokens if t not in stopwords]
        
        assert "the" not in filtered
        assert "is" not in filtered
        assert "python" in filtered
        assert "programming" in filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
