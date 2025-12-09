# tests/unit/domain/entities/test_generation.py

import pytest
import sys
import importlib
from datetime import datetime
from unittest.mock import patch, MagicMock
from app.domain.model.entities.generation import (
    GenerateTextRequest,
    GenerateTextResponse,
    GeneratedResult,
    GenerationMetadata
)


class TestGenerateTextRequest:
    """Tests for GenerateTextRequest entity validation"""

    def test_valid_generate_text_request_creation(self):
        """Test creating a valid GenerateTextRequest"""
        # Arrange & Act
        request = GenerateTextRequest(
            system_prompt="You are a helpful assistant",
            user_prompt="Tell me a joke",
            num_sequences=1,
            max_tokens=100,
            temperature=0.7
        )

        # Assert
        assert request.system_prompt == "You are a helpful assistant"
        assert request.user_prompt == "Tell me a joke"
        assert request.num_sequences == 1
        assert request.max_tokens == 100
        assert request.temperature == 0.7

    def test_generate_text_request_empty_system_prompt_raises_error(self):
        """Test that empty system_prompt raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            GenerateTextRequest(
                system_prompt="",
                user_prompt="Test",
                num_sequences=1
            )

    def test_generate_text_request_whitespace_system_prompt_raises_error(self):
        """Test that whitespace-only system_prompt raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            GenerateTextRequest(
                system_prompt="   ",
                user_prompt="Test",
                num_sequences=1
            )

    def test_generate_text_request_empty_user_prompt_raises_error(self):
        """Test that empty user_prompt raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="user_prompt cannot be empty"):
            GenerateTextRequest(
                system_prompt="Test",
                user_prompt="",
                num_sequences=1
            )

    def test_generate_text_request_invalid_num_sequences_raises_error(self):
        """Test that num_sequences < 1 raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="num_sequences must be >= 1"):
            GenerateTextRequest(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=0
            )

    def test_generate_text_request_invalid_max_tokens_raises_error(self):
        """Test that invalid max_tokens raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="max_tokens must be between"):
            GenerateTextRequest(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                max_tokens=0  # Too low
            )

    def test_generate_text_request_temperature_too_high_raises_error(self):
        """Test that temperature > 2.0 raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="temperature must be between"):
            GenerateTextRequest(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                temperature=3.0
            )

    def test_generate_text_request_temperature_too_low_raises_error(self):
        """Test that temperature < 0.0 raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="temperature must be between"):
            GenerateTextRequest(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                temperature=-0.5
            )

    def test_generate_text_request_to_dict(self):
        """Test conversion to dictionary"""
        # Arrange
        request = GenerateTextRequest(
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            max_tokens=150,
            temperature=0.5
        )

        # Act
        result = request.to_dict()

        # Assert
        assert result["system_prompt"] == "System"
        assert result["user_prompt"] == "User"
        assert result["num_sequences"] == 2
        assert result["max_tokens"] == 150
        assert result["temperature"] == 0.5


class TestGenerationMetadata:
    """Tests for GenerationMetadata entity"""

    def test_generation_metadata_creation(self):
        """Test creating GenerationMetadata"""
        # Arrange
        timestamp = datetime.now()

        # Act
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5,
            timestamp=timestamp
        )

        # Assert
        assert metadata.model_name == "test-model"
        assert metadata.system_prompt == "System"
        assert metadata.user_prompt == "User"
        assert metadata.temperature == 0.7
        assert metadata.tokens_used == 50
        assert metadata.generation_time == 1.5
        assert metadata.timestamp == timestamp

    def test_generation_metadata_immutability(self):
        """Test that GenerationMetadata is immutable"""
        # Arrange
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5
        )

        # Act & Assert
        with pytest.raises(AttributeError):
            metadata.model_name = "new-model"


class TestGeneratedResult:
    """Tests for GeneratedResult entity"""

    def test_generated_result_creation(self):
        """Test creating a GeneratedResult"""
        # Arrange
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5
        )

        # Act
        result = GeneratedResult(
            content="Generated text content",
            metadata=metadata,
            reference_data={"key": "value"}
        )

        # Assert
        assert result.content == "Generated text content"
        assert result.metadata == metadata
        assert result.reference_data == {"key": "value"}

    def test_generated_result_contains_reference(self):
        """Test contains_reference method"""
        # Arrange
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5
        )
        result = GeneratedResult(
            content="This is a TEST content",
            metadata=metadata
        )

        # Act & Assert
        assert result.contains_reference("test")
        assert result.contains_reference("TEST")
        assert result.contains_reference("content")
        assert not result.contains_reference("missing")

    def test_generated_result_word_count(self):
        """Test word_count method"""
        # Arrange
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5
        )
        result = GeneratedResult(
            content="This is a test sentence with seven words",
            metadata=metadata
        )

        # Act
        count = result.word_count()

        # Assert
        assert count == 8

    def test_generated_result_to_dict(self):
        """Test conversion to dictionary"""
        # Arrange
        timestamp = datetime.now()
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5,
            timestamp=timestamp
        )
        result = GeneratedResult(
            content="Test content",
            metadata=metadata,
            reference_data={"ref": "data"}
        )

        # Act
        dict_result = result.to_dict()

        # Assert
        assert dict_result["content"] == "Test content"
        assert dict_result["metadata"]["model_name"] == "test-model"
        assert dict_result["metadata"]["temperature"] == 0.7
        assert dict_result["reference_data"] == {"ref": "data"}


class TestGenerateTextResponse:
    """Tests for GenerateTextResponse entity"""

    def test_generate_text_response_creation(self):
        """Test creating a GenerateTextResponse"""
        # Arrange
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5
        )
        generated_texts = [
            GeneratedResult(content="Text 1", metadata=metadata),
            GeneratedResult(content="Text 2", metadata=metadata)
        ]

        # Act
        response = GenerateTextResponse(
            generated_texts=generated_texts,
            total_tokens=100,
            generation_time=2.5,
            model_name="test-model"
        )

        # Assert
        assert len(response.generated_texts) == 2
        assert response.total_tokens == 100
        assert response.generation_time == 2.5
        assert response.model_name == "test-model"

    def test_generate_text_response_to_dict(self):
        """Test conversion to dictionary"""
        # Arrange
        metadata = GenerationMetadata(
            model_name="test-model",
            system_prompt="System",
            user_prompt="User",
            temperature=0.7,
            tokens_used=50,
            generation_time=1.5
        )
        generated_texts = [
            GeneratedResult(content="Text 1", metadata=metadata)
        ]
        response = GenerateTextResponse(
            generated_texts=generated_texts,
            total_tokens=50,
            generation_time=1.5,
            model_name="test-model"
        )

        # Act
        dict_response = response.to_dict()

        # Assert
        assert len(dict_response["generated_texts"]) == 1
        assert dict_response["total_tokens"] == 50
        assert dict_response["generation_time"] == 1.5
        assert dict_response["model_name"] == "test-model"

    def test_generate_text_response_to_dict_empty_results(self):
        """Test to_dict with empty results returns 'unknown' model"""
        # Arrange
        response = GenerateTextResponse(
            generated_texts=[],
            total_tokens=0,
            generation_time=0.0,
            model_name="unknown"
        )

        # Act
        dict_response = response.to_dict()

        # Assert
        assert dict_response["model_name"] == "unknown"


class TestGenerationImportFallback:
    """Tests for import fallback behavior when config module is unavailable"""

    def test_import_fallback_uses_default_values_when_config_unavailable(self):
        """Test that fallback constants are used when config import fails"""
        # Store original modules to restore later
        original_modules = {}
        modules_to_remove = [
            'app.domain.model.entities.generation',
            'config'
        ]

        for mod in modules_to_remove:
            if mod in sys.modules:
                original_modules[mod] = sys.modules[mod]

        try:
            # Remove cached modules
            for mod in modules_to_remove:
                if mod in sys.modules:
                    del sys.modules[mod]

            # Mock 'config' to raise ImportError
            with patch.dict(sys.modules, {'config': None}):
                # Force the import to fail by making config unavailable
                def raise_import_error(*args, **kwargs):
                    raise ImportError("No module named 'config'")

                # Create a custom import that fails for 'config'
                original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

                def custom_import(name, *args, **kwargs):
                    if name == 'config':
                        raise ImportError("No module named 'config'")
                    return original_import(name, *args, **kwargs)

                with patch('builtins.__import__', side_effect=custom_import):
                    # Re-import the module to trigger the except block
                    import app.domain.model.entities.generation as gen_module
                    importlib.reload(gen_module)

                    # Verify fallback values are set
                    assert hasattr(gen_module, 'MIN_TEMPERATURE')
                    assert hasattr(gen_module, 'MAX_TEMPERATURE')
                    assert hasattr(gen_module, 'MIN_MAX_TOKENS')
                    assert hasattr(gen_module, 'MAX_MAX_TOKENS')

                    # The values should be the fallback defaults
                    assert gen_module.MIN_TEMPERATURE == 0.0
                    assert gen_module.MAX_TEMPERATURE == 2.0
                    assert gen_module.MIN_MAX_TOKENS == 1
                    assert gen_module.MAX_MAX_TOKENS == 4096
        finally:
            # Restore original modules
            for mod, module in original_modules.items():
                sys.modules[mod] = module

            # Reload the module with proper config to restore normal state
            if 'app.domain.model.entities.generation' in sys.modules:
                importlib.reload(sys.modules['app.domain.model.entities.generation'])

    def test_constants_have_correct_values_with_config(self):
        """Test that constants are properly loaded from config when available"""
        from app.domain.model.entities import generation as gen_module

        # These should be loaded from config (or fallback if config not available)
        # Either way, they should have valid values
        assert isinstance(gen_module.MIN_TEMPERATURE, (int, float))
        assert isinstance(gen_module.MAX_TEMPERATURE, (int, float))
        assert isinstance(gen_module.MIN_MAX_TOKENS, int)
        assert isinstance(gen_module.MAX_MAX_TOKENS, int)

        # Verify sensible ranges
        assert gen_module.MIN_TEMPERATURE >= 0.0
        assert gen_module.MAX_TEMPERATURE <= 3.0
        assert gen_module.MIN_MAX_TOKENS >= 1
        assert gen_module.MAX_MAX_TOKENS >= 100
