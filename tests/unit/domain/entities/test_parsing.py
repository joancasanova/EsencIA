# tests/unit/domain/entities/test_parsing.py

import pytest
from app.domain.model.entities.parsing import (
    ParseMode,
    ParseRule,
    ParseRequest,
    ParseResult,
    ParseMatch,
    ParseResponse
)


class TestParseRule:
    """Tests for ParseRule entity validation"""

    def test_valid_parse_rule_creation(self):
        """Test creating a valid ParseRule"""
        # Arrange & Act
        rule = ParseRule(
            name="test_rule",
            pattern=r"\d+",
            mode=ParseMode.REGEX,
            fallback_value="N/A"
        )

        # Assert
        assert rule.name == "test_rule"
        assert rule.pattern == r"\d+"
        assert rule.mode == ParseMode.REGEX
        assert rule.fallback_value == "N/A"

    def test_parse_rule_empty_name_raises_error(self):
        """Test that empty name raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="name cannot be empty"):
            ParseRule(
                name="",
                pattern=r"\d+",
                mode=ParseMode.REGEX
            )

    def test_parse_rule_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="name cannot be empty"):
            ParseRule(
                name="   ",
                pattern=r"\d+",
                mode=ParseMode.REGEX
            )

    def test_parse_rule_empty_pattern_raises_error(self):
        """Test that empty pattern raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            ParseRule(
                name="test",
                pattern="",
                mode=ParseMode.REGEX
            )

    def test_parse_rule_whitespace_pattern_raises_error(self):
        """Test that whitespace-only pattern raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="pattern cannot be empty"):
            ParseRule(
                name="test",
                pattern="   ",
                mode=ParseMode.REGEX
            )

    def test_parse_rule_keyword_mode(self):
        """Test ParseRule with KEYWORD mode"""
        # Arrange & Act
        rule = ParseRule(
            name="keyword_rule",
            pattern="Name:",
            mode=ParseMode.KEYWORD,
            secondary_pattern="\n"
        )

        # Assert
        assert rule.mode == ParseMode.KEYWORD
        assert rule.secondary_pattern == "\n"


class TestParseRequest:
    """Tests for ParseRequest entity validation"""

    def test_valid_parse_request_creation(self):
        """Test creating a valid ParseRequest"""
        # Arrange
        rules = [
            ParseRule(name="rule1", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        # Act
        request = ParseRequest(
            rules=rules,
            text="Test text 123",
            output_filter="all"
        )

        # Assert
        assert request.text == "Test text 123"
        assert len(request.rules) == 1
        assert request.output_filter == "all"

    def test_parse_request_empty_rules_raises_error(self):
        """Test that empty rules list raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="rules list cannot be empty"):
            ParseRequest(
                rules=[],
                text="Test text",
                output_filter="all"
            )

    def test_parse_request_first_n_without_limit_raises_error(self):
        """Test that first_n filter without limit raises ValueError"""
        # Arrange
        rules = [
            ParseRule(name="rule1", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="output_limit is required when output_filter is 'first_n'"):
            ParseRequest(
                rules=rules,
                text="Test text",
                output_filter="first_n",
                output_limit=None
            )

    def test_parse_request_first_n_with_zero_limit_raises_error(self):
        """Test that first_n filter with zero limit raises ValueError"""
        # Arrange
        rules = [
            ParseRule(name="rule1", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="output_limit must be >= 1"):
            ParseRequest(
                rules=rules,
                text="Test text",
                output_filter="first_n",
                output_limit=0
            )

    def test_parse_request_first_n_with_valid_limit(self):
        """Test that first_n filter with valid limit works"""
        # Arrange
        rules = [
            ParseRule(name="rule1", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        # Act
        request = ParseRequest(
            rules=rules,
            text="Test text",
            output_filter="first_n",
            output_limit=5
        )

        # Assert
        assert request.output_limit == 5


class TestParseResult:
    """Tests for ParseResult functionality"""

    def test_parse_result_to_list_of_dicts(self):
        """Test conversion of ParseResult to list of dicts"""
        # Arrange
        entries = [
            {"name": "John", "age": "30"},
            {"name": "Jane", "age": "25"}
        ]
        result = ParseResult(entries=entries)

        # Act
        output = result.to_list_of_dicts()

        # Assert
        assert output == entries
        assert len(output) == 2

    def test_parse_result_get_all_matches_for_rule(self):
        """Test getting all matches for a specific rule"""
        # Arrange
        entries = [
            {"name": "John", "age": "30"},
            {"name": "Jane", "age": "25"},
            {"city": "NYC"}  # Missing 'name' key
        ]
        result = ParseResult(entries=entries)

        # Act
        names = result.get_all_matches_for_rule("name")
        ages = result.get_all_matches_for_rule("age")
        missing = result.get_all_matches_for_rule("nonexistent")

        # Assert
        assert names == ["John", "Jane"]
        assert ages == ["30", "25"]
        assert missing == []


class TestParseMatch:
    """Tests for ParseMatch entity"""

    def test_parse_match_creation(self):
        """Test creating a ParseMatch"""
        # Arrange & Act
        match = ParseMatch(rule_name="test_rule", value="matched_value")

        # Assert
        assert match.rule_name == "test_rule"
        assert match.value == "matched_value"

    def test_parse_match_immutability(self):
        """Test that ParseMatch is immutable"""
        # Arrange
        match = ParseMatch(rule_name="test_rule", value="matched_value")

        # Act & Assert
        with pytest.raises(AttributeError):
            match.value = "new_value"


class TestParseResponse:
    """Tests for ParseResponse entity"""

    def test_parse_response_creation(self):
        """Test creating a ParseResponse"""
        # Arrange
        parse_result = ParseResult(entries=[{"name": "John"}])

        # Act
        response = ParseResponse(parse_result=parse_result)

        # Assert
        assert response.parse_result == parse_result
        assert len(response.parse_result.entries) == 1
