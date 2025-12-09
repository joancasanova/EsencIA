# tests/unit/application/use_cases/test_generate_text_use_case.py

import pytest
from unittest.mock import Mock, patch
from application.use_cases.generate_text_use_case import GenerateTextUseCase
from domain.model.entities.generation import GenerateTextRequest, GeneratedResult, GenerationMetadata
from datetime import datetime


def create_mock_result(content="Generated text", tokens=10):
    """Helper to create mock GeneratedResult"""
    return GeneratedResult(
        content=content,
        metadata=GenerationMetadata(
            model_name="test-model",
            system_prompt="sys",
            user_prompt="user",
            temperature=1.0,
            tokens_used=tokens,
            generation_time=0.5,
            timestamp=datetime.now()
        )
    )


class TestGenerateTextUseCaseExecute:
    """Tests for GenerateTextUseCase.execute()"""

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_execute_success(self, mock_gen_service_class):
        """Test successful text generation"""
        mock_gen_service = Mock()
        mock_gen_service_class.return_value = mock_gen_service
        mock_gen_service.generate.return_value = [create_mock_result()]

        use_case = GenerateTextUseCase("test-model")

        request = GenerateTextRequest(
            system_prompt="System",
            user_prompt="User",
            num_sequences=1,
            max_tokens=50,
            temperature=0.7
        )

        response = use_case.execute(request)

        assert len(response.generated_texts) == 1
        assert response.generated_texts[0].content == "Generated text"
        assert response.total_tokens == 10
        assert response.generation_time >= 0

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_execute_calculates_total_tokens(self, mock_gen_service_class):
        """Test that total tokens is sum of all results"""
        mock_gen_service = Mock()
        mock_gen_service_class.return_value = mock_gen_service
        mock_gen_service.generate.return_value = [
            create_mock_result("Result 1", tokens=10),
            create_mock_result("Result 2", tokens=15),
            create_mock_result("Result 3", tokens=25)
        ]

        use_case = GenerateTextUseCase("test-model")
        request = GenerateTextRequest(
            system_prompt="System",
            user_prompt="User",
            num_sequences=3
        )

        response = use_case.execute(request)

        assert response.total_tokens == 50  # 10 + 15 + 25

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_execute_value_error_from_service(self, mock_gen_service_class):
        """Test ValueError handling from service"""
        mock_gen_service = Mock()
        mock_gen_service_class.return_value = mock_gen_service
        mock_gen_service.generate.side_effect = ValueError("Invalid param")

        use_case = GenerateTextUseCase("test-model")
        request = GenerateTextRequest(
            system_prompt="System",
            user_prompt="User",
            num_sequences=1
        )

        with pytest.raises(ValueError, match="Generation failed"):
            use_case.execute(request)

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_execute_runtime_error_from_service(self, mock_gen_service_class):
        """Test generic exception handling from service"""
        mock_gen_service = Mock()
        mock_gen_service_class.return_value = mock_gen_service
        mock_gen_service.generate.side_effect = RuntimeError("Service crashed")

        use_case = GenerateTextUseCase("test-model")
        request = GenerateTextRequest(
            system_prompt="System",
            user_prompt="User",
            num_sequences=1
        )

        with pytest.raises(RuntimeError, match="Text generation failed"):
            use_case.execute(request)


class TestGenerateTextUseCaseValidation:
    """Tests for GenerateTextUseCase._validate_request()"""

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_validate_empty_system_prompt_raises_error(self, mock_gen_service_class):
        """Test that empty system prompt raises error"""
        mock_gen_service_class.return_value = Mock()
        use_case = GenerateTextUseCase("test-model")

        request = Mock()
        request.system_prompt = ""
        request.user_prompt = "User"

        with pytest.raises(ValueError, match="System prompt cannot be empty"):
            use_case._validate_request(request)

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_validate_whitespace_system_prompt_raises_error(self, mock_gen_service_class):
        """Test that whitespace-only system prompt raises error"""
        mock_gen_service_class.return_value = Mock()
        use_case = GenerateTextUseCase("test-model")

        request = Mock()
        request.system_prompt = "   \n\t  "
        request.user_prompt = "User"

        with pytest.raises(ValueError, match="System prompt cannot be empty"):
            use_case._validate_request(request)

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_validate_empty_user_prompt_raises_error(self, mock_gen_service_class):
        """Test that empty user prompt raises error"""
        mock_gen_service_class.return_value = Mock()
        use_case = GenerateTextUseCase("test-model")

        request = Mock()
        request.system_prompt = "System"
        request.user_prompt = ""

        with pytest.raises(ValueError, match="User prompt cannot be empty"):
            use_case._validate_request(request)

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_validate_whitespace_user_prompt_raises_error(self, mock_gen_service_class):
        """Test that whitespace-only user prompt raises error"""
        mock_gen_service_class.return_value = Mock()
        use_case = GenerateTextUseCase("test-model")

        request = Mock()
        request.system_prompt = "System"
        request.user_prompt = "   \t\n  "

        with pytest.raises(ValueError, match="User prompt cannot be empty"):
            use_case._validate_request(request)


class TestGenerateTextUseCaseInit:
    """Tests for GenerateTextUseCase initialization"""

    @patch('application.use_cases.generate_text_use_case.GenerateService')
    def test_init_creates_generate_service(self, mock_gen_service_class):
        """Test that init creates GenerateService with model name"""
        mock_gen_service_class.return_value = Mock()

        use_case = GenerateTextUseCase("custom-model")

        mock_gen_service_class.assert_called_once_with("custom-model")
        assert use_case.generate_service is not None
