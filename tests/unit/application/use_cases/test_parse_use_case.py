# tests/unit/application/use_cases/test_parse_use_case.py

import pytest
from unittest.mock import Mock, patch
from app.application.use_cases.parse_use_case import ParseUseCase, ParseRequestValidationError
from app.domain.model.entities.parsing import ParseRequest, ParseRule, ParseMode


@pytest.fixture
def sample_parse_rules():
    """Fixture for sample parse rules"""
    return [
        ParseRule(name="name", pattern=r"Name:\s*(\w+)", mode=ParseMode.REGEX),
        ParseRule(name="age", pattern=r"Age:\s*(\d+)", mode=ParseMode.REGEX)
    ]


class TestParseUseCase:
    """Tests for ParseUseCase"""

    def test_execute_with_valid_request(self, sample_parse_rules):
        """Test executing parse use case with valid request"""
        # Arrange
        use_case = ParseUseCase()
        text = "Name: John Age: 30 City: NYC"
        request = ParseRequest(
            text=text,
            rules=sample_parse_rules,
            output_filter="all"
        )

        # Act
        response = use_case.execute(request)

        # Assert
        assert response is not None
        assert hasattr(response, 'parse_result')
        assert response.parse_result is not None

    def test_execute_with_empty_text_raises_error(self, sample_parse_rules):
        """Test that empty text raises validation error"""
        # Arrange
        use_case = ParseUseCase()
        request = ParseRequest(
            text="",
            rules=sample_parse_rules,
            output_filter="all"
        )

        # Act & Assert
        with pytest.raises(ParseRequestValidationError, match="Text input cannot be empty"):
            use_case.execute(request)

    def test_execute_with_whitespace_text_raises_error(self, sample_parse_rules):
        """Test that whitespace-only text raises validation error"""
        # Arrange
        use_case = ParseUseCase()
        request = ParseRequest(
            text="   \n\t  ",
            rules=sample_parse_rules,
            output_filter="all"
        )

        # Act & Assert
        with pytest.raises(ParseRequestValidationError, match="Text input cannot be empty"):
            use_case.execute(request)

    def test_execute_with_successful_filter(self):
        """Test executing with 'successful' output filter"""
        # Arrange
        use_case = ParseUseCase()
        rules = [
            ParseRule(
                name="number",
                pattern=r"\d+",
                mode=ParseMode.REGEX,
                fallback_value="N/A"
            )
        ]
        text = "Numbers: 123, 456, 789"
        request = ParseRequest(
            text=text,
            rules=rules,
            output_filter="successful"
        )

        # Act
        response = use_case.execute(request)

        # Assert
        assert response is not None
        # Successful filter should only return entries without fallback values
        for entry in response.parse_result.entries:
            assert entry.get("number") != "N/A"

    def test_execute_with_first_n_filter(self):
        """Test executing with 'first_n' output filter"""
        # Arrange
        use_case = ParseUseCase()
        rules = [
            ParseRule(
                name="word",
                pattern=r"\w+",
                mode=ParseMode.REGEX
            )
        ]
        text = "one two three four five six"
        request = ParseRequest(
            text=text,
            rules=rules,
            output_filter="first_n",
            output_limit=3
        )

        # Act
        response = use_case.execute(request)

        # Assert
        assert response is not None
        assert len(response.parse_result.entries) <= 3

    def test_execute_handles_parse_service_errors(self):
        """Test that use case handles errors from parse service"""
        # Arrange
        use_case = ParseUseCase()
        rules = [
            ParseRule(
                name="test",
                pattern=r"valid_pattern",
                mode=ParseMode.REGEX
            )
        ]
        text = "test text"
        request = ParseRequest(
            text=text,
            rules=rules,
            output_filter="all"
        )

        # Act - Should not raise, should handle gracefully
        response = use_case.execute(request)

        # Assert
        assert response is not None


class TestParseRequestValidationError:
    """Tests for custom ParseRequestValidationError exception"""

    def test_validation_error_creation(self):
        """Test creating ParseRequestValidationError"""
        # Arrange & Act
        error = ParseRequestValidationError("Test error message")

        # Assert
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_validation_error_with_cause(self):
        """Test validation error with cause chain"""
        # Arrange
        cause = ValueError("Original error")

        # Act
        try:
            raise ParseRequestValidationError("Validation failed") from cause
        except ParseRequestValidationError as e:
            # Assert
            assert str(e) == "Validation failed"
            assert e.__cause__ == cause


class TestParseUseCaseValidation:
    """Tests for ParseUseCase._validate_request()"""

    def test_validate_empty_rules_raises_error(self):
        """Test that empty rules list raises validation error"""
        use_case = ParseUseCase()

        # Create request with empty rules using Mock to bypass entity validation
        request = Mock()
        request.text = "Some text"
        request.rules = []

        with pytest.raises(ValueError, match="At least one parsing rule"):
            use_case._validate_request(request)

    def test_validate_rule_without_pattern_raises_error(self):
        """Test that rule without pattern raises validation error"""
        use_case = ParseUseCase()

        rule = Mock()
        rule.name = "bad_rule"
        rule.pattern = ""  # Empty pattern

        request = Mock()
        request.text = "Some text"
        request.rules = [rule]

        with pytest.raises(ValueError, match="missing required pattern"):
            use_case._validate_request(request)


class TestParseUseCaseErrorHandling:
    """Tests for error handling in ParseUseCase"""

    @patch.object(ParseUseCase, '_validate_request')
    def test_execute_value_error_from_service(self, mock_validate):
        """Test ValueError handling from parse service"""
        mock_validate.return_value = None  # Skip validation

        use_case = ParseUseCase()
        use_case.parse_service = Mock()
        use_case.parse_service.parse_text.side_effect = ValueError("Bad config")

        request = Mock()
        request.text = "text"
        request.rules = [Mock()]

        with pytest.raises(ParseRequestValidationError, match="Parsing failed"):
            use_case.execute(request)

    @patch.object(ParseUseCase, '_validate_request')
    def test_execute_runtime_error_from_service(self, mock_validate):
        """Test generic exception handling from parse service"""
        mock_validate.return_value = None

        use_case = ParseUseCase()
        use_case.parse_service = Mock()
        use_case.parse_service.parse_text.side_effect = RuntimeError("Service crashed")

        request = Mock()
        request.text = "text"
        request.rules = [Mock()]

        with pytest.raises(RuntimeError, match="Parsing operation failed"):
            use_case.execute(request)


class TestParseUseCaseInit:
    """Tests for ParseUseCase initialization"""

    def test_init_creates_parse_service(self):
        """Test that init creates ParseService"""
        use_case = ParseUseCase()

        assert use_case.parse_service is not None
