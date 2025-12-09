# tests/unit/domain/services/test_generate_service.py

import pytest
from domain.services.generate_service import GenerateService, InsufficientResourcesError
from domain.model.entities.generation import GeneratedResult

# Modelo pequeño para tests (~2MB, se descarga rápido)
TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def generate_service():
    """Fixture que crea el servicio una vez por módulo para reutilizar"""
    return GenerateService(TINY_MODEL)


class TestGenerateServiceInit:
    """Tests for GenerateService initialization"""

    def test_init_with_tiny_model_success(self):
        """Test successful initialization with a real tiny model"""
        service = GenerateService(TINY_MODEL)

        assert service.model_name == TINY_MODEL
        assert service.device in ["cuda", "cpu"]
        assert service.instruct_mode == False
        assert service.model is not None
        assert service.tokenizer is not None

    def test_init_detects_instruct_mode_in_name(self, generate_service):
        """Test that instruct mode detection works based on model name"""
        # El modelo tiny-gpt2 no tiene "instruct" en el nombre
        assert generate_service.instruct_mode == False

    def test_init_with_empty_model_name_raises_error(self):
        """Test that empty model name raises ValueError"""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            GenerateService("")

    def test_init_with_whitespace_model_name_raises_error(self):
        """Test that whitespace-only model name raises ValueError"""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            GenerateService("   ")

    def test_init_with_invalid_model_raises_runtime_error(self):
        """Test that invalid model name raises RuntimeError when skipping resource check"""
        # skip_resource_check=True to test actual model loading error
        with pytest.raises(RuntimeError):
            GenerateService("modelo-que-no-existe-xyz123", skip_resource_check=True)

    def test_init_with_huge_model_raises_insufficient_resources_error(self):
        """Test that model requiring too many resources raises InsufficientResourcesError"""
        # A 70B model should fail resource validation on consumer hardware
        with pytest.raises(InsufficientResourcesError):
            GenerateService("meta-llama/Llama-2-70b")


class TestGenerateServiceGenerate:
    """Tests for GenerateService.generate() method"""

    def test_generate_single_sequence(self, generate_service):
        """Test generating a single sequence"""
        results = generate_service.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello.",
            num_sequences=1,
            max_tokens=10,
            temperature=1.0
        )

        assert len(results) == 1
        assert isinstance(results[0], GeneratedResult)
        assert results[0].content is not None
        assert len(results[0].content) > 0

    def test_generate_multiple_sequences(self, generate_service):
        """Test generating multiple sequences"""
        results = generate_service.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Count to three.",
            num_sequences=3,
            max_tokens=10,
            temperature=1.0
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, GeneratedResult)
            assert result.content is not None

    def test_generate_with_low_temperature(self, generate_service):
        """Test generation with low temperature (more deterministic)"""
        results = generate_service.generate(
            system_prompt="Complete the sentence.",
            user_prompt="The sky is",
            num_sequences=1,
            max_tokens=5,
            temperature=0.1
        )

        assert len(results) == 1
        assert results[0].content is not None

    def test_generate_returns_metadata(self, generate_service):
        """Test that generated results include proper metadata"""
        results = generate_service.generate(
            system_prompt="System prompt here.",
            user_prompt="User prompt here.",
            num_sequences=1,
            max_tokens=10,
            temperature=0.7
        )

        metadata = results[0].metadata
        assert metadata.model_name == TINY_MODEL
        assert metadata.system_prompt == "System prompt here."
        assert metadata.user_prompt == "User prompt here."
        assert metadata.temperature == 0.7
        assert metadata.tokens_used >= 0
        assert metadata.generation_time >= 0

    def test_generate_with_max_tokens_limit(self, generate_service):
        """Test that max_tokens parameter limits output length"""
        results = generate_service.generate(
            system_prompt="Write a long story.",
            user_prompt="Once upon a time",
            num_sequences=1,
            max_tokens=5,
            temperature=1.0
        )

        # El resultado no debería ser extremadamente largo
        assert len(results) == 1
        # Tokens generados deberían ser limitados (aproximadamente)
        assert results[0].metadata.tokens_used < 50

    def test_generate_with_invalid_temperature_too_high_raises_error(self, generate_service):
        """Test that temperature > 2.0 raises ValueError"""
        with pytest.raises(ValueError, match="Temperature must be between"):
            generate_service.generate(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                max_tokens=10,
                temperature=2.5
            )

    def test_generate_with_invalid_temperature_negative_raises_error(self, generate_service):
        """Test that negative temperature raises ValueError"""
        with pytest.raises(ValueError, match="Temperature must be between"):
            generate_service.generate(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                max_tokens=10,
                temperature=-0.5
            )

    def test_generate_with_invalid_max_tokens_zero_raises_error(self, generate_service):
        """Test that max_tokens = 0 raises ValueError"""
        with pytest.raises(ValueError, match="Max tokens must be between"):
            generate_service.generate(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                max_tokens=0,
                temperature=1.0
            )

    def test_generate_with_invalid_max_tokens_too_high_raises_error(self, generate_service):
        """Test that max_tokens > 4096 raises ValueError"""
        with pytest.raises(ValueError, match="Max tokens must be between"):
            generate_service.generate(
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=1,
                max_tokens=5000,
                temperature=1.0
            )

    def test_generate_with_boundary_temperature_values(self, generate_service):
        """Test that boundary values for temperature are accepted"""
        # Temperature 0.01 (near minimum - 0.0 is not valid with do_sample=True)
        results_min = generate_service.generate(
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=1,
            max_tokens=5,
            temperature=0.01
        )
        assert len(results_min) == 1

        # Temperature 2.0 (maximum valid)
        results_max = generate_service.generate(
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=1,
            max_tokens=5,
            temperature=2.0
        )
        assert len(results_max) == 1


class TestGenerateServiceGetTokenCount:
    """Tests for GenerateService.get_token_count() method"""

    def test_get_token_count_simple_text(self, generate_service):
        """Test token counting for simple text"""
        count = generate_service.get_token_count("Hello world")

        assert isinstance(count, int)
        assert count > 0

    def test_get_token_count_empty_string(self, generate_service):
        """Test token count for empty string"""
        count = generate_service.get_token_count("")

        assert count == 0

    def test_get_token_count_long_text(self, generate_service):
        """Test token count for longer text"""
        long_text = "This is a longer sentence with multiple words and tokens." * 5
        count = generate_service.get_token_count(long_text)

        assert count > 10  # Should be significantly more tokens

    def test_get_token_count_with_special_characters(self, generate_service):
        """Test token count with special characters"""
        text_with_special = "Hello! @#$% World? 123"
        count = generate_service.get_token_count(text_with_special)

        assert count > 0

    def test_get_token_count_with_non_string_raises_error(self, generate_service):
        """Test that non-string input raises ValueError"""
        with pytest.raises(ValueError, match="Text must be a string"):
            generate_service.get_token_count(123)

    def test_get_token_count_with_none_raises_error(self, generate_service):
        """Test that None input raises ValueError"""
        with pytest.raises(ValueError, match="Text must be a string"):
            generate_service.get_token_count(None)


class TestGenerateServiceTextProcessing:
    """Tests for text processing helper methods"""

    def test_extract_assistant_response_with_marker(self, generate_service):
        """Test extraction of assistant response from formatted text"""
        result = generate_service._extract_assistant_response(
            "Some preamble\nassistant\nActual response here"
        )

        assert result == "Actual response here"

    def test_extract_assistant_response_without_marker(self, generate_service):
        """Test extraction when no assistant marker present"""
        result = generate_service._extract_assistant_response("Just plain text response")

        assert result == "Just plain text response"

    def test_extract_assistant_response_multiline(self, generate_service):
        """Test extraction with multiline response"""
        text = "System\nassistant\nLine 1\nLine 2\nLine 3"
        result = generate_service._extract_assistant_response(text)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_trim_response_removes_prompt(self, generate_service):
        """Test that prompt is removed from output"""
        prompt = "This is the prompt. "
        output = "This is the prompt. And this is the generated response."

        result = generate_service._trim_response(prompt, output)

        assert result == "And this is the generated response."

    def test_trim_response_no_prompt_match(self, generate_service):
        """Test trim when output doesn't start with prompt"""
        result = generate_service._trim_response("Different prompt", "Output without prompt")

        assert result == "Output without prompt"

    def test_trim_response_exact_prompt(self, generate_service):
        """Test trim when output is exactly the prompt"""
        prompt = "Just the prompt"
        result = generate_service._trim_response(prompt, prompt)

        assert result == ""

    def test_process_output_base_model(self, generate_service):
        """Test output processing for base model (non-instruct)"""
        prompt = "Prompt text "
        output = "Prompt text Generated content"

        result = generate_service._process_output(output, prompt)

        assert result == "Generated content"


class TestGenerateServiceCreateResults:
    """Tests for _create_results method"""

    def test_create_results_single_output(self, generate_service):
        """Test creating results from single output"""
        from datetime import datetime

        outputs = ["Generated text here"]
        start_time = datetime.now()

        results = generate_service._create_results(
            outputs=outputs,
            prompt="full prompt",
            system_prompt="system",
            user_prompt="user",
            temperature=0.8,
            start_time=start_time
        )

        assert len(results) == 1
        assert results[0].metadata.system_prompt == "system"
        assert results[0].metadata.user_prompt == "user"
        assert results[0].metadata.temperature == 0.8

    def test_create_results_multiple_outputs(self, generate_service):
        """Test creating results from multiple outputs"""
        from datetime import datetime

        outputs = ["Output 1", "Output 2", "Output 3"]
        start_time = datetime.now()

        results = generate_service._create_results(
            outputs=outputs,
            prompt="prompt",
            system_prompt="sys",
            user_prompt="usr",
            temperature=1.0,
            start_time=start_time
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.metadata.model_name == TINY_MODEL


class TestGenerateServiceEdgeCases:
    """Edge case tests"""

    def test_generate_with_unicode_text(self, generate_service):
        """Test generation with unicode characters"""
        results = generate_service.generate(
            system_prompt="Responde en español.",
            user_prompt="¿Cómo estás? 你好 🎉",
            num_sequences=1,
            max_tokens=10,
            temperature=1.0
        )

        assert len(results) == 1
        assert results[0].content is not None

    def test_token_count_unicode(self, generate_service):
        """Test token count with unicode"""
        count = generate_service.get_token_count("Héllo wörld 中文 🎉")

        assert count > 0

    def test_generate_preserves_prompt_info(self, generate_service):
        """Test that original prompts are preserved in metadata"""
        system = "You are a test assistant"
        user = "This is a test query"

        results = generate_service.generate(
            system_prompt=system,
            user_prompt=user,
            num_sequences=1,
            max_tokens=5,
            temperature=1.0
        )

        assert results[0].metadata.system_prompt == system
        assert results[0].metadata.user_prompt == user


class TestGenerateServiceInstructMode:
    """Tests for instruct mode functionality with a real instruct model"""

    # Modelo instruct pequeño para tests (~500MB pero necesario para probar chat templates)
    TINY_INSTRUCT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"

    @pytest.fixture(scope="class")
    def instruct_service(self):
        """Fixture que crea el servicio con modelo instruct una vez por clase"""
        return GenerateService(self.TINY_INSTRUCT_MODEL)

    def test_instruct_mode_detected(self, instruct_service):
        """Test that instruct mode is correctly detected from model name"""
        assert instruct_service.instruct_mode == True

    def test_instruct_generate_single_sequence(self, instruct_service):
        """Test generating with instruct model"""
        results = instruct_service.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello.",
            num_sequences=1,
            max_tokens=10,
            temperature=1.0
        )

        assert len(results) == 1
        assert isinstance(results[0], GeneratedResult)
        assert results[0].content is not None

    def test_instruct_generate_uses_chat_template(self, instruct_service):
        """Test that instruct mode uses chat template formatting"""
        results = instruct_service.generate(
            system_prompt="You are a test assistant.",
            user_prompt="What is 2+2?",
            num_sequences=1,
            max_tokens=15,
            temperature=0.5
        )

        # El resultado debería ser procesado con _extract_assistant_response
        assert len(results) == 1
        # Verificar que el contenido no incluye tokens de chat crudos
        content = results[0].content
        # No debería contener marcadores de sistema sin procesar
        assert "<|system|>" not in content or "system" not in content.lower()[:20]

    def test_instruct_metadata_preserved(self, instruct_service):
        """Test that metadata is correctly preserved in instruct mode"""
        system = "You are helpful."
        user = "Help me."

        results = instruct_service.generate(
            system_prompt=system,
            user_prompt=user,
            num_sequences=1,
            max_tokens=5,
            temperature=1.0
        )

        assert results[0].metadata.system_prompt == system
        assert results[0].metadata.user_prompt == user
        assert results[0].metadata.model_name == self.TINY_INSTRUCT_MODEL

    def test_instruct_process_output_extracts_response(self, instruct_service):
        """Test that _process_output uses extract_assistant_response in instruct mode"""
        # Simular salida con marcador assistant
        output = "system\nYou are helpful\nassistant\nHere is my response"
        prompt = "ignored in instruct mode"

        result = instruct_service._process_output(output, prompt)

        # En modo instruct, debería extraer solo la respuesta del asistente
        assert "Here is my response" in result


class TestGenerateServiceRealModelEdgeCases:
    """Additional edge case tests with real models"""

    def test_generate_with_empty_system_prompt(self, generate_service):
        """Test generation with empty system prompt"""
        results = generate_service.generate(
            system_prompt="",
            user_prompt="Hello",
            num_sequences=1,
            max_tokens=5,
            temperature=1.0
        )

        assert len(results) == 1
        assert results[0].content is not None

    def test_generate_with_very_long_prompt(self, generate_service):
        """Test generation with a longer prompt"""
        long_prompt = "This is a test. " * 50  # ~200 tokens

        results = generate_service.generate(
            system_prompt="You are helpful.",
            user_prompt=long_prompt,
            num_sequences=1,
            max_tokens=5,
            temperature=1.0
        )

        assert len(results) == 1

    def test_generate_with_newlines_in_prompt(self, generate_service):
        """Test generation with newlines in prompts"""
        results = generate_service.generate(
            system_prompt="Line 1\nLine 2\nLine 3",
            user_prompt="Question:\nWhat is this?",
            num_sequences=1,
            max_tokens=5,
            temperature=1.0
        )

        assert len(results) == 1
        assert results[0].content is not None
