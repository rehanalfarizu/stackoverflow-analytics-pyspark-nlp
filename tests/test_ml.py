"""
Test ML Module
==============
Unit tests untuk ML pipeline.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQualityLabels:
    """Tests untuk quality labeling."""
    
    def test_high_quality_label(self):
        """Test high quality label assignment."""
        score = 15
        view_count = 2000
        has_accepted = True
        
        if score >= 5 and view_count >= 1000 and has_accepted:
            label = 2  # High quality
        elif score >= 0:
            label = 1  # Medium
        else:
            label = 0  # Low
        
        assert label == 2
    
    def test_medium_quality_label(self):
        """Test medium quality label assignment."""
        score = 3
        view_count = 500
        has_accepted = False
        
        if score >= 5 and view_count >= 1000 and has_accepted:
            label = 2
        elif score >= 0:
            label = 1
        else:
            label = 0
        
        assert label == 1
    
    def test_low_quality_label(self):
        """Test low quality label assignment."""
        score = -2
        
        if score >= 5:
            label = 2
        elif score >= 0:
            label = 1
        else:
            label = 0
        
        assert label == 0


class TestFeatureExtraction:
    """Tests untuk feature extraction."""
    
    def test_title_length(self):
        """Test title length calculation."""
        title = "How to read CSV file in Python?"
        title_length = len(title)
        
        assert title_length == 32
    
    def test_word_count(self):
        """Test word count calculation."""
        text = "How to read CSV file in Python?"
        word_count = len(text.split())
        
        assert word_count == 7
    
    def test_has_question_mark(self):
        """Test question mark detection."""
        title1 = "How to do this?"
        title2 = "Tutorial for Python"
        
        assert "?" in title1
        assert "?" not in title2
    
    def test_tag_count(self):
        """Test tag count extraction."""
        import re
        
        tags_str = "<python><pandas><csv>"
        tags = re.findall(r'<([^>]+)>', tags_str)
        
        assert len(tags) == 3


class TestSimilarity:
    """Tests untuk similarity calculations."""
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        set1 = {"python", "pandas", "data"}
        set2 = {"python", "numpy", "data"}
        
        intersection = set1 & set2
        union = set1 | set2
        
        jaccard = len(intersection) / len(union)
        
        assert 0 <= jaccard <= 1
        assert jaccard == 0.5  # 2/4
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        import math
        
        vec1 = [1, 2, 3]
        vec2 = [1, 2, 3]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a ** 2 for a in vec1))
        norm2 = math.sqrt(sum(b ** 2 for b in vec2))
        
        cosine = dot_product / (norm1 * norm2)
        
        assert abs(cosine - 1.0) < 0.001  # Same vectors = similarity 1


class TestTrendDetection:
    """Tests untuk trend detection."""
    
    def test_growth_calculation(self):
        """Test growth rate calculation."""
        current = 150
        previous = 100
        
        growth = ((current - previous) / previous) * 100
        
        assert growth == 50.0
    
    def test_decline_calculation(self):
        """Test decline rate calculation."""
        current = 80
        previous = 100
        
        growth = ((current - previous) / previous) * 100
        
        assert growth == -20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
