"""Tests for minima_llm_dspy module, particularly TolerantChatAdapter."""

import pytest
from typing import List, Optional, Union


class TestIsListStr:
    """Tests for TolerantChatAdapter.is_list_str() type detection."""

    @pytest.fixture
    def adapter_class(self):
        """Import TolerantChatAdapter (requires dspy)."""
        from trec_auto_judge.llm.minima_llm_dspy import TolerantChatAdapter
        return TolerantChatAdapter

    def test_list_str_lowercase(self, adapter_class):
        """list[str] should be detected."""
        assert adapter_class.is_list_str(list[str]) is True

    def test_list_str_typing(self, adapter_class):
        """List[str] from typing should be detected."""
        assert adapter_class.is_list_str(List[str]) is True

    def test_optional_list_str_lowercase(self, adapter_class):
        """Optional[list[str]] should be detected."""
        assert adapter_class.is_list_str(Optional[list[str]]) is True

    def test_optional_list_str_typing(self, adapter_class):
        """Optional[List[str]] should be detected."""
        assert adapter_class.is_list_str(Optional[List[str]]) is True

    def test_union_list_str_none(self, adapter_class):
        """Union[List[str], None] should be detected."""
        assert adapter_class.is_list_str(Union[List[str], None]) is True

    def test_plain_str_not_detected(self, adapter_class):
        """str should not be detected as list[str]."""
        assert adapter_class.is_list_str(str) is False

    def test_list_int_not_detected(self, adapter_class):
        """list[int] should not be detected as list[str]."""
        assert adapter_class.is_list_str(list[int]) is False

    def test_optional_str_not_detected(self, adapter_class):
        """Optional[str] should not be detected as list[str]."""
        assert adapter_class.is_list_str(Optional[str]) is False

    def test_optional_list_int_not_detected(self, adapter_class):
        """Optional[List[int]] should not be detected as list[str]."""
        assert adapter_class.is_list_str(Optional[List[int]]) is False

    def test_union_multiple_types_not_detected(self, adapter_class):
        """Union[List[str], int] should not be detected (multiple non-None types)."""
        assert adapter_class.is_list_str(Union[List[str], int]) is False


class TestTryParseListStr:
    """Tests for TolerantChatAdapter.try_parse_list_str() parsing."""

    @pytest.fixture
    def adapter_class(self):
        from trec_auto_judge.llm.minima_llm_dspy import TolerantChatAdapter
        return TolerantChatAdapter

    # === JSON (double-quote) tests ===

    def test_valid_json_array(self, adapter_class):
        """Valid JSON array should parse correctly."""
        result = adapter_class.try_parse_list_str('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_array_with_whitespace(self, adapter_class):
        """JSON array items should be stripped."""
        result = adapter_class.try_parse_list_str('["  a  ", "b", "c  "]')
        assert result == ["a", "b", "c"]

    def test_json_array_filters_empty(self, adapter_class):
        """Empty strings should be filtered out."""
        result = adapter_class.try_parse_list_str('["a", "", "b", null, "c"]')
        assert result == ["a", "b", "c"]

    def test_json_with_apostrophe(self, adapter_class):
        """JSON handles apostrophes in strings correctly."""
        result = adapter_class.try_parse_list_str('["What is the article\'s main point?"]')
        assert result == ["What is the article's main point?"]

    def test_invalid_json_raises(self, adapter_class):
        """Invalid input should raise ValueError."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            adapter_class.try_parse_list_str("not json")

    def test_json_object_raises(self, adapter_class):
        """JSON object (not array) should raise ValueError."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            adapter_class.try_parse_list_str('{"key": "value"}')

    def test_json_string_raises(self, adapter_class):
        """JSON string (not array) should raise ValueError."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            adapter_class.try_parse_list_str('"just a string"')

    # === Python (single-quote) tests ===

    def test_python_single_quote_array(self, adapter_class):
        """Python-style single-quote list should parse correctly."""
        result = adapter_class.try_parse_list_str("['a', 'b', 'c']")
        assert result == ["a", "b", "c"]

    def test_python_single_quote_with_whitespace(self, adapter_class):
        """Python list items should be stripped."""
        result = adapter_class.try_parse_list_str("['  a  ', 'b', 'c  ']")
        assert result == ["a", "b", "c"]

    def test_python_with_escaped_apostrophe(self, adapter_class):
        """Python list with properly escaped apostrophe should parse."""
        result = adapter_class.try_parse_list_str("['What is the article\\'s main point?']")
        assert result == ["What is the article's main point?"]

    def test_unbalanced_quotes_raises(self, adapter_class):
        """Odd number of single quotes should raise ValueError (apostrophe issue)."""
        # This simulates: ['What about article's thing'] with unescaped apostrophe
        with pytest.raises(ValueError, match="Unbalanced single quotes"):
            adapter_class.try_parse_list_str("['What about article's thing']")

    # === Empty/edge cases ===

    def test_empty_list(self, adapter_class):
        """Empty list should parse correctly."""
        result = adapter_class.try_parse_list_str("[]")
        assert result == []

    def test_empty_list_with_spaces(self, adapter_class):
        """Empty list with whitespace should parse correctly."""
        result = adapter_class.try_parse_list_str("  [  ]  ")
        assert result == []