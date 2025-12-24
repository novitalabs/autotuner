"""
Tests for parameter specification normalization.
"""

import pytest
from src.utils.param_spec import (
    normalize_choice_values,
    extract_max_value,
    extract_min_value,
)


def test_normalize_dict_with_values():
    """Test normalization of dict with 'values' key."""
    spec = {"values": [1, 2, 3]}
    result = normalize_choice_values(spec)
    assert result == [1, 2, 3]


def test_normalize_list():
    """Test normalization of list input."""
    spec = [1, 2, 3]
    result = normalize_choice_values(spec)
    assert result == [1, 2, 3]


def test_normalize_scalar():
    """Test normalization of scalar input."""
    result = normalize_choice_values(5)
    assert result == [5]
    
    result = normalize_choice_values("auto")
    assert result == ["auto"]


def test_normalize_empty_list():
    """Test normalization of empty list."""
    result = normalize_choice_values([])
    assert result == []


def test_normalize_dict_without_values():
    """Test normalization of dict without 'values' key."""
    spec = {"key": "value"}
    result = normalize_choice_values(spec)
    assert result == [{"key": "value"}]


def test_extract_max_value_from_dict():
    """Test extracting max value from dict."""
    spec = {"values": [1, 2, 4, 3]}
    result = extract_max_value(spec)
    assert result == 4


def test_extract_max_value_from_list():
    """Test extracting max value from list."""
    spec = [1, 2, 4, 3]
    result = extract_max_value(spec)
    assert result == 4


def test_extract_max_value_from_scalar():
    """Test extracting max value from scalar."""
    result = extract_max_value(8)
    assert result == 8


def test_extract_max_value_empty():
    """Test extracting max value from empty spec."""
    result = extract_max_value([])
    assert result is None


def test_extract_max_value_strings():
    """Test extracting max from non-numeric values."""
    spec = ["a", "b", "c"]
    result = extract_max_value(spec)
    # Should return first value for non-comparable types
    assert result == "c"


def test_extract_min_value_from_dict():
    """Test extracting min value from dict."""
    spec = {"values": [4, 2, 1, 3]}
    result = extract_min_value(spec)
    assert result == 1


def test_extract_min_value_from_list():
    """Test extracting min value from list."""
    spec = [4, 2, 1, 3]
    result = extract_min_value(spec)
    assert result == 1


def test_extract_min_value_from_scalar():
    """Test extracting min value from scalar."""
    result = extract_min_value(8)
    assert result == 8


def test_mixed_types():
    """Test with float and int values."""
    spec = {"values": [1, 2.5, 3, 4.2]}
    result = extract_max_value(spec)
    assert result == 4.2
    
    result = extract_min_value(spec)
    assert result == 1
