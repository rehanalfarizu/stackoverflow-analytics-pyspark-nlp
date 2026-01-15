"""
Test ETL Module
===============
Unit tests untuk ETL pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestXMLParser:
    """Tests untuk XMLParser."""
    
    def test_parse_tags(self):
        """Test parsing tags dari format Stack Overflow."""
        from src.utils.helpers import parse_tags
        
        tags_str = "<python><pandas><dataframe>"
        result = parse_tags(tags_str)
        
        assert result == ["python", "pandas", "dataframe"]
    
    def test_parse_tags_empty(self):
        """Test parsing empty tags."""
        from src.utils.helpers import parse_tags
        
        assert parse_tags("") == []
        assert parse_tags(None) == []


class TestDataTransformer:
    """Tests untuk DataTransformer."""
    
    def test_clean_html(self):
        """Test HTML cleaning."""
        from src.utils.helpers import clean_html
        
        html = "<p>Hello <b>World</b></p>"
        result = clean_html(html)
        
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_clean_html_empty(self):
        """Test HTML cleaning dengan empty input."""
        from src.utils.helpers import clean_html
        
        assert clean_html("") == ""
        assert clean_html(None) == ""


class TestHelperFunctions:
    """Tests untuk helper functions."""
    
    def test_format_number(self):
        """Test number formatting."""
        from src.utils.helpers import format_number
        
        assert format_number(500) == "500"
        assert format_number(1500) == "1.5K"
        assert format_number(1500000) == "1.5M"
        assert format_number(1500000000) == "1.5B"
    
    def test_format_duration(self):
        """Test duration formatting."""
        from src.utils.helpers import format_duration
        
        assert "s" in format_duration(30)
        assert "m" in format_duration(90)
        assert "h" in format_duration(3700)
    
    def test_truncate_text(self):
        """Test text truncation."""
        from src.utils.helpers import truncate_text
        
        text = "This is a very long text that needs to be truncated"
        result = truncate_text(text, max_length=20)
        
        assert len(result) == 20
        assert result.endswith("...")
    
    def test_calculate_percentage(self):
        """Test percentage calculation."""
        from src.utils.helpers import calculate_percentage
        
        assert calculate_percentage(25, 100) == 25.0
        assert calculate_percentage(1, 3, decimals=1) == 33.3
        assert calculate_percentage(0, 0) == 0.0
    
    def test_safe_divide(self):
        """Test safe division."""
        from src.utils.helpers import safe_divide
        
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1


class TestChunkList:
    """Tests untuk chunk_list function."""
    
    def test_chunk_list(self):
        """Test list chunking."""
        from src.utils.helpers import chunk_list
        
        lst = [1, 2, 3, 4, 5, 6, 7]
        chunks = chunk_list(lst, 3)
        
        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7]
    
    def test_chunk_list_empty(self):
        """Test chunking empty list."""
        from src.utils.helpers import chunk_list
        
        assert chunk_list([], 3) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
