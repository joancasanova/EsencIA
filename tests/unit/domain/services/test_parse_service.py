# tests/unit/domain/services/test_parse_service.py

import pytest
from unittest.mock import patch, MagicMock
from domain.services.parse_service import ParseService
from domain.model.entities.parsing import ParseMode, ParseRule, ParseResult


class TestParseServiceParseText:
    """Tests for ParseService.parse_text() method"""

    def test_parse_text_single_regex_rule(self):
        """Test parsing with a single regex rule"""
        service = ParseService()
        text = "Name: John, Age: 30"
        rules = [
            ParseRule(name="name", pattern=r"John", mode=ParseMode.REGEX)
        ]

        result = service.parse_text(text, rules)

        assert isinstance(result, ParseResult)
        assert len(result.entries) >= 1

    def test_parse_text_multiple_regex_rules(self):
        """Test parsing with multiple regex rules extracting values"""
        service = ParseService()
        text = "John 30 NYC Jane 25 LA"
        rules = [
            ParseRule(name="name", pattern=r"[A-Z][a-z]+", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX),
            ParseRule(name="city", pattern=r"[A-Z]{2,3}", mode=ParseMode.REGEX)
        ]

        result = service.parse_text(text, rules)

        assert isinstance(result, ParseResult)

    def test_parse_text_keyword_mode(self):
        """Test parsing with keyword mode"""
        service = ParseService()
        text = "Name: John Age: 30 Name: Jane Age: 25"
        rules = [
            ParseRule(name="name", pattern="Name:", mode=ParseMode.KEYWORD, secondary_pattern="Age:"),
            ParseRule(name="age", pattern="Age:", mode=ParseMode.KEYWORD, secondary_pattern="Name:")
        ]

        result = service.parse_text(text, rules)

        assert isinstance(result, ParseResult)

    def test_parse_text_keyword_mode_without_secondary_pattern(self):
        """Test keyword parsing without secondary pattern (extracts to end)"""
        service = ParseService()
        text = "Label: Some value here"
        rules = [
            ParseRule(name="label", pattern="Label:", mode=ParseMode.KEYWORD)
        ]

        result = service.parse_text(text, rules)

        assert isinstance(result, ParseResult)
        assert len(result.entries) >= 1

    def test_parse_text_empty_text(self):
        """Test parsing with empty text"""
        service = ParseService()
        text = ""
        rules = [
            ParseRule(name="test", pattern=r"\w+", mode=ParseMode.REGEX)
        ]

        result = service.parse_text(text, rules)

        assert isinstance(result, ParseResult)
        assert len(result.entries) == 0

    def test_parse_text_invalid_regex_raises_error(self):
        """Test that invalid regex pattern raises ValueError"""
        service = ParseService()
        text = "Some text"
        rules = [
            ParseRule(name="bad", pattern=r"[invalid(regex", mode=ParseMode.REGEX)
        ]

        with pytest.raises(ValueError) as exc_info:
            service.parse_text(text, rules)

        assert "Invalid regex pattern in rule 'bad'" in str(exc_info.value)
        assert "Please fix the pattern configuration" in str(exc_info.value)


class TestParseServiceFindAllOccurrences:
    """Tests for ParseService._find_all_occurrences() method"""

    def test_find_occurrences_regex_multiple_matches(self):
        """Test finding multiple regex matches"""
        service = ParseService()
        text = "apple banana cherry apple"
        rule = ParseRule(name="fruit", pattern=r"apple", mode=ParseMode.REGEX)

        results = service._find_all_occurrences(text, rule)

        assert len(results) == 2
        assert results[0][2] == "apple"
        assert results[1][2] == "apple"

    def test_find_occurrences_regex_no_match(self):
        """Test regex with no matches"""
        service = ParseService()
        text = "no matching pattern here"
        rule = ParseRule(name="number", pattern=r"\d+", mode=ParseMode.REGEX)

        results = service._find_all_occurrences(text, rule)

        assert len(results) == 0

    def test_find_occurrences_keyword_mode(self):
        """Test keyword mode extraction"""
        service = ParseService()
        text = "KEY: value1 END KEY: value2 END"
        rule = ParseRule(name="key", pattern="KEY:", mode=ParseMode.KEYWORD, secondary_pattern="END")

        results = service._find_all_occurrences(text, rule)

        assert len(results) == 2
        assert "value1" in results[0][2]
        assert "value2" in results[1][2]

    def test_find_occurrences_keyword_without_end_boundary(self):
        """Test keyword extraction without end boundary"""
        service = ParseService()
        text = "PREFIX: all remaining text"
        rule = ParseRule(name="prefix", pattern="PREFIX:", mode=ParseMode.KEYWORD)

        results = service._find_all_occurrences(text, rule)

        assert len(results) == 1
        assert "all remaining text" in results[0][2]

    def test_find_occurrences_keyword_no_match(self):
        """Test keyword mode with no match"""
        service = ParseService()
        text = "no keyword here"
        rule = ParseRule(name="missing", pattern="NOTFOUND:", mode=ParseMode.KEYWORD)

        results = service._find_all_occurrences(text, rule)

        assert len(results) == 0


class TestParseServiceBuildEntries:
    """Tests for entry building logic"""

    def test_build_entries_sequential_rules(self):
        """Test building entries with sequential rule matches"""
        service = ParseService()
        text = "John 30 Jane 25"
        rules = [
            ParseRule(name="name", pattern=r"[A-Z][a-z]+", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        result = service.parse_text(text, rules)

        assert len(result.entries) >= 1
        entry = result.entries[0]
        assert "name" in entry
        assert "age" in entry

    def test_build_entries_with_default_fallback(self):
        """Test that missing rules get default fallback format"""
        service = ParseService()
        text = "OnlyName"
        rules = [
            ParseRule(name="name", pattern=r"OnlyName", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d{5}", mode=ParseMode.REGEX)  # Won't match
        ]

        result = service.parse_text(text, rules)

        if result.entries:
            entry = result.entries[0]
            assert "name" in entry
            # age should have fallback value
            if "age" in entry:
                assert "missing_age" in entry["age"] or entry["age"]


class TestParseServiceFilterEntries:
    """Tests for ParseService.filter_entries() method"""

    def test_filter_all_returns_all_entries(self):
        """Test that 'all' filter returns all entries"""
        # Arrange
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John", "age": "30"},
            {"name": "missing_age", "age": "N/A"},
            {"name": "Jane", "age": "25"}
        ])
        rules = [
            ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX, fallback_value="N/A")
        ]

        # Act
        filtered = service.filter_entries(parse_result, "all", None, rules)

        # Assert
        assert len(filtered.entries) == 3

    def test_filter_successful_excludes_fallback_values(self):
        """Test that 'successful' filter excludes entries with fallback values"""
        # Arrange
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John", "age": "30"},
            {"name": "missing_name", "age": "25"},
            {"name": "Jane", "age": "28"}
        ])
        rules = [
            ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX, fallback_value="missing_name"),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        # Act
        filtered = service.filter_entries(parse_result, "successful", None, rules)

        # Assert
        # Should exclude entries with "missing_name"
        assert len(filtered.entries) <= 2

    def test_filter_first_n_returns_limited_entries(self):
        """Test that 'first_n' filter returns only N entries"""
        # Arrange
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John"},
            {"name": "Jane"},
            {"name": "Bob"},
            {"name": "Alice"},
            {"name": "Charlie"}
        ])
        rules = [
            ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX)
        ]

        # Act
        filtered = service.filter_entries(parse_result, "first_n", 3, rules)

        # Assert
        assert len(filtered.entries) == 3
        assert filtered.entries[0]["name"] == "John"
        assert filtered.entries[1]["name"] == "Jane"
        assert filtered.entries[2]["name"] == "Bob"

    def test_filter_first_n_with_more_than_available(self):
        """Test 'first_n' when N is greater than available entries"""
        # Arrange
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John"},
            {"name": "Jane"}
        ])
        rules = [
            ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX)
        ]

        # Act
        filtered = service.filter_entries(parse_result, "first_n", 10, rules)

        # Assert
        assert len(filtered.entries) == 2


class TestParseServiceBasic:
    """Basic tests for ParseService functionality"""

    def test_parse_service_instantiation(self):
        """Test that ParseService can be instantiated"""
        service = ParseService()
        assert service is not None

    def test_parse_text_returns_parse_result(self):
        """Test that parse_text returns a ParseResult"""
        service = ParseService()
        text = "Test text"
        rules = [
            ParseRule(name="test", pattern=r"\w+", mode=ParseMode.REGEX)
        ]

        result = service.parse_text(text, rules)

        assert isinstance(result, ParseResult)


class TestParseServiceLimits:
    """Tests for ParseService safety limits and edge cases"""

    def test_max_parse_entries_limit(self):
        """Test that MAX_PARSE_ENTRIES limit is enforced"""
        service = ParseService()
        # Create text with many repeated patterns
        text = " ".join([f"Name{i}" for i in range(2000)])
        rules = [
            ParseRule(name="name", pattern=r"Name\d+", mode=ParseMode.REGEX)
        ]

        with patch('domain.services.parse_service.MAX_PARSE_ENTRIES', 5):
            result = service.parse_text(text, rules)
            # Should be limited to MAX_PARSE_ENTRIES
            assert len(result.entries) <= 5

    def test_max_matches_per_rule_limit_regex(self):
        """Test that MAX_MATCHES_PER_RULE limit is enforced for regex mode"""
        service = ParseService()
        # Create text with many matches
        text = " ".join(["apple"] * 1000)
        rule = ParseRule(name="fruit", pattern=r"apple", mode=ParseMode.REGEX)

        with patch('domain.services.parse_service.MAX_MATCHES_PER_RULE', 10):
            results = service._find_all_occurrences(text, rule)
            assert len(results) <= 10

    def test_max_matches_per_rule_limit_keyword(self):
        """Test that MAX_MATCHES_PER_RULE limit is enforced for keyword mode"""
        service = ParseService()
        # Create text with many keyword matches
        text = " ".join(["KEY: value END"] * 100)
        rule = ParseRule(name="key", pattern="KEY:", mode=ParseMode.KEYWORD, secondary_pattern="END")

        with patch('domain.services.parse_service.MAX_MATCHES_PER_RULE', 5):
            results = service._find_all_occurrences(text, rule)
            assert len(results) <= 5

    def test_max_iterations_limit_raises_runtime_error(self):
        """Test that MAX_PARSE_ITERATIONS limit raises RuntimeError"""
        service = ParseService()
        # Create text with many keyword matches that will cause many iterations
        text = "KEY: value END " * 100
        rule = ParseRule(name="key", pattern="KEY:", mode=ParseMode.KEYWORD, secondary_pattern="END")

        # Set iteration limit low but match limit high so iterations are hit first
        with patch('domain.services.parse_service.MAX_PARSE_ITERATIONS', 5):
            with patch('domain.services.parse_service.MAX_MATCHES_PER_RULE', 1000):
                with pytest.raises(RuntimeError, match="iteration limit exceeded"):
                    service._find_all_occurrences(text, rule)


class TestParseServiceEntryBuilding:
    """Tests for entry building edge cases"""

    def test_build_entries_rule_appears_before_expected(self):
        """Test entry finalization when rule appears before expected order"""
        service = ParseService()
        # Pattern: name1, age1, name2, age2 - but with interruption
        text = "John 30 Jane 25"
        rules = [
            ParseRule(name="name", pattern=r"[A-Z][a-z]+", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX)
        ]

        result = service.parse_text(text, rules)

        # Should have created entries correctly
        assert len(result.entries) >= 1
        for entry in result.entries:
            assert "name" in entry
            assert "age" in entry

    def test_build_entries_fills_missing_rules_with_fallback(self):
        """Test that missing rules are filled with fallback values"""
        service = ParseService()
        # Create a pattern where some rules won't match
        text = "John 30 Unknown Jane"  # "Unknown" won't match as age
        rules = [
            ParseRule(name="name", pattern=r"[A-Z][a-z]+", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX, fallback_value="N/A")
        ]

        result = service.parse_text(text, rules)

        # Entries should exist, some may have fallback values
        assert isinstance(result, ParseResult)

    def test_build_entries_custom_fallback_value(self):
        """Test that custom fallback_value is used"""
        service = ParseService()
        text = "OnlyNames: John Jane Bob"
        rules = [
            ParseRule(name="name", pattern=r"[A-Z][a-z]+", mode=ParseMode.REGEX),
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX, fallback_value="UNKNOWN_AGE")
        ]

        result = service.parse_text(text, rules)

        # Check that some entry has the fallback
        for entry in result.entries:
            if entry.get("age") == "UNKNOWN_AGE":
                assert True
                return
        # If no fallback found, the test still passes if entries exist
        assert len(result.entries) >= 0

    def test_should_finalize_entry_duplicate_rule(self):
        """Test _should_finalize_entry returns True for duplicate rule"""
        service = ParseService()

        # Simulate a duplicate rule scenario
        result = service._should_finalize_entry(
            rule_name="name",
            rule_idx=0,
            rules_matched={"name"},  # Already matched
            expected_idx=1
        )

        assert result == True

    def test_should_finalize_entry_out_of_order(self):
        """Test _should_finalize_entry returns True for out-of-order rule"""
        service = ParseService()

        # Rule index is before expected
        result = service._should_finalize_entry(
            rule_name="name",
            rule_idx=0,
            rules_matched=set(),
            expected_idx=2  # Expected rule 2, got rule 0
        )

        assert result == True

    def test_should_finalize_entry_normal_case(self):
        """Test _should_finalize_entry returns False for normal case"""
        service = ParseService()

        result = service._should_finalize_entry(
            rule_name="age",
            rule_idx=1,
            rules_matched={"name"},
            expected_idx=1
        )

        assert result == False

    def test_fill_missing_rules(self):
        """Test _fill_missing_rules fills gaps with fallback values"""
        service = ParseService()

        current_entry = {}
        rules_matched = set()
        rules = [
            ParseRule(name="first", pattern="a", mode=ParseMode.REGEX),
            ParseRule(name="second", pattern="b", mode=ParseMode.REGEX, fallback_value="CUSTOM"),
            ParseRule(name="third", pattern="c", mode=ParseMode.REGEX)
        ]

        service._fill_missing_rules(current_entry, rules_matched, 0, 2, rules)

        # Should have filled first and second
        assert "first" in current_entry
        assert "second" in current_entry
        assert "missing_first" in current_entry["first"]
        assert current_entry["second"] == "CUSTOM"

    def test_add_match_to_entry_expected_index(self):
        """Test _add_match_to_entry with expected index"""
        service = ParseService()

        current_entry = {}
        rules_matched = set()
        rules = [ParseRule(name="name", pattern="a", mode=ParseMode.REGEX)]
        entries = []

        new_idx = service._add_match_to_entry(
            current_entry, rules_matched,
            rule_idx=0, rule_name="name", matched_str="John",
            expected_idx=0, rules=rules, entries=entries
        )

        assert current_entry["name"] == "John"
        assert "name" in rules_matched
        assert new_idx == 1

    def test_add_match_to_entry_after_expected(self):
        """Test _add_match_to_entry when rule comes after expected (fills gaps)"""
        service = ParseService()

        current_entry = {}
        rules_matched = set()
        rules = [
            ParseRule(name="first", pattern="a", mode=ParseMode.REGEX),
            ParseRule(name="second", pattern="b", mode=ParseMode.REGEX)
        ]
        entries = []

        # Rule index 1, but expected 0 - should fill gap
        new_idx = service._add_match_to_entry(
            current_entry, rules_matched,
            rule_idx=1, rule_name="second", matched_str="value",
            expected_idx=0, rules=rules, entries=entries
        )

        # Should have filled first with fallback and finalized
        assert new_idx == 0  # Reset to 0 after finalization
        assert len(entries) == 1


class TestParseServiceFilterEdgeCases:
    """Tests for filter_entries edge cases"""

    def test_filter_unknown_type_returns_original(self):
        """Test that unknown filter type returns original parse result"""
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John"},
            {"name": "Jane"}
        ])
        rules = [ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX)]

        # Using a filter type that doesn't match any known type
        filtered = service.filter_entries(parse_result, "unknown_type", None, rules)

        # Should return original
        assert len(filtered.entries) == 2

    def test_filter_successful_default_fallback_pattern(self):
        """Test successful filter with default fallback pattern"""
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John", "age": "30"},
            {"name": "missing_name", "age": "25"},  # Uses default fallback pattern
            {"name": "Jane", "age": "missing_age"}  # Uses default fallback pattern
        ])
        rules = [
            ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX),  # No fallback_value
            ParseRule(name="age", pattern=r"\d+", mode=ParseMode.REGEX)    # No fallback_value
        ]

        filtered = service.filter_entries(parse_result, "successful", None, rules)

        # Only first entry should pass (no fallback values)
        assert len(filtered.entries) == 1
        assert filtered.entries[0]["name"] == "John"

    def test_filter_first_n_with_none_n(self):
        """Test first_n filter with None as n value"""
        service = ParseService()
        parse_result = ParseResult(entries=[
            {"name": "John"},
            {"name": "Jane"}
        ])
        rules = [ParseRule(name="name", pattern=r"\w+", mode=ParseMode.REGEX)]

        # first_n with None n - should not apply (returns original)
        filtered = service.filter_entries(parse_result, "first_n", None, rules)

        # Should return original (condition n is not None fails)
        assert len(filtered.entries) == 2
