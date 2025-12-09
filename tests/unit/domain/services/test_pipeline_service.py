# tests/unit/domain/services/test_pipeline_service.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from domain.services.pipeline_service import PipelineService, PlaceholderDict
from domain.model.entities.pipeline import PipelineStep, PipelineExecutionContext
from domain.model.entities.generation import GenerateTextRequest
from domain.model.entities.parsing import ParseRequest, ParseRule, ParseMode, ParseResult
from domain.model.entities.verification import VerificationMethod, VerificationMode, VerifyRequest


def create_context_with_results(results_list):
    """Helper to create a context with pre-populated results."""
    context = PipelineExecutionContext()
    for i, (step_type, results) in enumerate(results_list):
        context = context.with_result(i, step_type, tuple(results))
    return context


class TestPlaceholderDict:
    """Tests for PlaceholderDict helper class"""

    def test_placeholder_dict_returns_value_when_key_exists(self):
        """Test that PlaceholderDict returns value for existing key"""
        d = PlaceholderDict({"name": "John", "age": "30"})

        assert d["name"] == "John"
        assert d["age"] == "30"

    def test_placeholder_dict_returns_placeholder_for_missing_key(self):
        """Test that PlaceholderDict returns placeholder format for missing key"""
        d = PlaceholderDict({"name": "John"})

        # Missing key should return {key} format
        assert d["missing"] == "{missing}"

    def test_placeholder_dict_empty(self):
        """Test PlaceholderDict with empty dict"""
        d = PlaceholderDict({})

        assert d["any_key"] == "{any_key}"


class TestPipelineServiceInit:
    """Tests for PipelineService initialization"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_init_success(self, mock_parse_service, mock_gen_service):
        """Test successful initialization"""
        mock_gen_service.return_value = Mock()
        mock_parse_service.return_value = Mock()

        service = PipelineService("test-model")

        assert service.default_model_name == "test-model"
        # Context is None before run_pipeline is called
        assert service._context is None
        # Properties return empty values when context is None
        assert service.global_references == {}
        assert service.confirmed_references == []
        assert service.to_verify_references == []

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_init_creates_parse_service(self, mock_parse_service, mock_gen_service):
        """Test that ParseService is created during init"""
        mock_gen_service.return_value = Mock()
        mock_parse_instance = Mock()
        mock_parse_service.return_value = mock_parse_instance

        service = PipelineService("test-model")

        mock_parse_service.assert_called_once()
        assert service.parse_service == mock_parse_instance


class TestPipelineServicePlaceholders:
    """Tests for placeholder processing methods"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_has_placeholders_with_placeholders(self, mock_parse, mock_gen):
        """Test detecting placeholders in text returns True"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result = service._has_placeholders("Hello {name}, your age is {age}")

        assert result is True

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_has_placeholders_without_placeholders(self, mock_parse, mock_gen):
        """Test text without placeholders returns False"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result = service._has_placeholders("Hello world, no placeholders here")

        assert result is False

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_placeholders_extracts_names(self, mock_parse, mock_gen):
        """Test extracting placeholder names from text"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result = service._get_placeholders("Hello {name}, your age is {age}")

        assert "name" in result
        assert "age" in result
        assert len(result) == 2

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_placeholders_empty_for_no_placeholders(self, mock_parse, mock_gen):
        """Test that _get_placeholders returns empty set when no placeholders"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result = service._get_placeholders("Hello world, no placeholders here")

        assert len(result) == 0
        assert isinstance(result, set)

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_replace_placeholders_success(self, mock_parse, mock_gen):
        """Test successful placeholder replacement"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result, was_replaced = service._replace_placeholders(
            "Hello {name}!",
            {"name": "John"}
        )

        assert result == "Hello John!"
        assert was_replaced is True

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_replace_placeholders_missing_key(self, mock_parse, mock_gen):
        """Test placeholder replacement with missing key"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result, was_replaced = service._replace_placeholders(
            "Hello {name}! Age: {age}",
            {"name": "John"}
        )

        assert result == "Hello John! Age: {age}"
        assert was_replaced is True

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_replace_placeholders_no_match(self, mock_parse, mock_gen):
        """Test placeholder replacement with no matching keys"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result, was_replaced = service._replace_placeholders(
            "Hello {name}!",
            {"other": "value"}
        )

        assert result == "Hello {name}!"
        assert was_replaced is False


class TestPipelineServiceValidation:
    """Tests for validation methods"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_step_references_no_references(self, mock_parse, mock_gen):
        """Test validation when step doesn't use references"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = False
        step.reference_step_numbers = []

        # Should not raise
        service._validate_step_references(step, 0)

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_step_references_with_global_references(self, mock_parse, mock_gen):
        """Test validation when step uses global references"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        service.global_references = {"key": "value"}

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = []

        # Should not raise when global references exist
        service._validate_step_references(step, 0)

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_step_references_invalid_reference(self, mock_parse, mock_gen):
        """Test validation with invalid reference step number"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [5]  # Invalid - references future step

        with pytest.raises(ValueError, match="invalid reference"):
            service._validate_step_references(step, 1)

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_references_valid(self, mock_parse, mock_gen):
        """Test _validate_references with valid references"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Set up context with results using immutable pattern
        service._context = create_context_with_results([
            ("generate", ["result1"]),
            ("parse", ["result2"])
        ])

        result = service._validate_references([0, 1], 2)

        assert result is True

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_references_invalid(self, mock_parse, mock_gen):
        """Test _validate_references with invalid reference"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Set up context with results using immutable pattern
        service._context = create_context_with_results([
            ("generate", ["result1"])
        ])

        result = service._validate_references([0, 5], 2)  # 5 doesn't exist

        assert result is False


class TestPipelineServiceStoreResult:
    """Tests for _store_result method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_store_result_first_result(self, mock_parse, mock_gen):
        """Test storing first result"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Initialize empty context
        service._context = PipelineExecutionContext()

        service._store_result(0, "generate", ["result1", "result2"])

        assert len(service._context.results) == 1
        assert service._context.results[0] == ("generate", ("result1", "result2"))

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_store_result_extends_existing(self, mock_parse, mock_gen):
        """Test storing result extends existing results"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Set up context with existing result
        service._context = create_context_with_results([
            ("generate", ["result1"])
        ])

        service._store_result(0, "generate", ["result2"])

        # Results are tuples in immutable context
        assert service._context.results[0] == ("generate", ("result1", "result2"))


class TestPipelineServiceGetResults:
    """Tests for get_results method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_results_empty(self, mock_parse, mock_gen):
        """Test get_results with no results"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        result = service.get_results()

        assert result == []

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_results_with_to_dict_items(self, mock_parse, mock_gen):
        """Test get_results serializes items with to_dict method"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        mock_item = Mock()
        mock_item.to_dict.return_value = {"key": "value"}
        # Set up context with results using immutable pattern
        service._context = create_context_with_results([
            ("generate", [mock_item])
        ])

        result = service.get_results()

        assert len(result) == 1
        assert result[0]["step_type"] == "generate"
        assert result[0]["step_data"] == [{"key": "value"}]


class TestPipelineServiceGetReferenceData:
    """Tests for _get_reference_data method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_reference_data_valid(self, mock_parse, mock_gen):
        """Test getting valid reference data"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Set up context with results using immutable pattern
        service._context = create_context_with_results([
            ("generate", ["gen_result"]),
            ("parse", ["parse_result"])
        ])

        result = service._get_reference_data([0, 1], 2)

        assert len(result) == 2
        # Note: results are now tuples in immutable context
        assert result[0] == (0, "generate", ("gen_result",))
        assert result[1] == (1, "parse", ("parse_result",))

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_reference_data_invalid_returns_empty(self, mock_parse, mock_gen):
        """Test getting invalid reference returns empty list"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Set up context with results using immutable pattern
        service._context = create_context_with_results([
            ("generate", ["result"])
        ])

        result = service._get_reference_data([5], 2)  # Invalid reference

        assert result == []


class TestPipelineServiceExecuteStep:
    """Tests for _execute_step method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_step_unknown_type(self, mock_parse, mock_gen):
        """Test execute step with unknown type returns empty list"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = False
        step.type = "unknown"

        result = service._execute_step(step, 0)

        assert result == []

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_step_with_invalid_reference(self, mock_parse, mock_gen):
        """Test execute step with invalid reference returns empty"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Initialize empty context
        service._context = PipelineExecutionContext()

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [0]  # Invalid reference - no results exist
        step.type = "generate"

        result = service._execute_step(step, 1)

        assert result == []


class TestPipelineServiceExecuteParse:
    """Tests for _execute_parse method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_parse_without_reference(self, mock_parse_class, mock_gen):
        """Test parse execution without reference"""
        mock_gen.return_value = Mock()
        mock_parse_instance = Mock()
        mock_parse_class.return_value = mock_parse_instance

        mock_parse_result = Mock()
        mock_parse_instance.parse_text.return_value = mock_parse_result
        mock_parse_instance.filter_entries.return_value = mock_parse_result

        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = False
        step.parameters = Mock()
        step.parameters.text = "Some text"
        step.parameters.rules = []
        step.parameters.output_filter = "all"
        step.parameters.output_limit = None

        result = service._execute_parse(step, 0)

        assert len(result) == 1
        mock_parse_instance.parse_text.assert_called_once()
        mock_parse_instance.filter_entries.assert_called_once()

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_parse_with_generate_reference(self, mock_parse_class, mock_gen):
        """Test parse execution with reference to generate step"""
        mock_gen.return_value = Mock()
        mock_parse_instance = Mock()
        mock_parse_class.return_value = mock_parse_instance

        mock_parse_result = Mock()
        mock_parse_instance.parse_text.return_value = mock_parse_result
        mock_parse_instance.filter_entries.return_value = mock_parse_result

        service = PipelineService("test-model")

        # Setup generate step result using immutable context
        gen_result = Mock()
        gen_result.content = "Generated content to parse"
        service._context = create_context_with_results([
            ("generate", [gen_result])
        ])

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [0]
        step.parameters = Mock()
        step.parameters.rules = []
        step.parameters.output_filter = "all"
        step.parameters.output_limit = None

        result = service._execute_parse(step, 1)

        assert len(result) == 1
        mock_parse_instance.parse_text.assert_called_once_with(text="Generated content to parse", rules=[])


class TestPipelineServiceSerializeStepItem:
    """Tests for _serialize_step_item method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_serialize_item_with_to_dict(self, mock_parse, mock_gen):
        """Test serialization of item with to_dict method"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        mock_item = Mock()
        mock_item.to_dict.return_value = {"key": "value"}

        result = service._serialize_step_item(mock_item, "generate")

        assert result == {"key": "value"}
        mock_item.to_dict.assert_called_once()

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_serialize_verify_item_with_verification_summary(self, mock_parse, mock_gen):
        """Test serialization of verify step item with verification_summary"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        mock_item = Mock(spec=[])  # No to_dict method
        mock_item.verification_summary = Mock()
        mock_item.verification_summary.to_dict.return_value = {"status": "confirmed"}

        result = service._serialize_step_item(mock_item, "verify")

        assert result == {"status": "confirmed"}

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_serialize_parse_item_with_parse_result(self, mock_parse, mock_gen):
        """Test serialization of parse step item with parse_result"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        mock_item = Mock(spec=[])  # No to_dict method
        mock_item.parse_result = Mock()
        mock_item.parse_result.to_list_of_dicts.return_value = [{"name": "John"}]

        result = service._serialize_step_item(mock_item, "parse")

        assert result == [{"name": "John"}]

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_serialize_plain_item(self, mock_parse, mock_gen):
        """Test serialization of plain item without serialization methods"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        plain_item = {"simple": "data"}

        result = service._serialize_step_item(plain_item, "generate")

        assert result == {"simple": "data"}


class TestPipelineServiceRunPipeline:
    """Tests for run_pipeline method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_run_pipeline_clears_previous_results(self, mock_parse, mock_gen):
        """Test that run_pipeline creates fresh context, clearing previous results"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        # Set up context with old data
        old_context = PipelineExecutionContext(
            global_references={"old": "ref"}
        )
        old_context = old_context.with_result(0, "old", ("data",))
        old_context = old_context.with_confirmed_reference({"old_ref": "value"})
        old_context = old_context.with_to_verify_reference({"old_ref": "value"})
        service._context = old_context

        # Empty steps list - no execution, but should create fresh context
        service.run_pipeline([])

        # New context should be empty
        assert service._context.results == ()
        assert service.confirmed_references == []
        assert service.to_verify_references == []

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_run_pipeline_raises_on_invalid_reference(self, mock_parse, mock_gen):
        """Test that run_pipeline raises error on invalid step reference"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [5]  # Invalid future reference

        with pytest.raises(ValueError):
            service.run_pipeline([step])


class TestPipelineServiceExecuteGenerate:
    """Tests for _execute_generate method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_generate_without_reference(self, mock_parse, mock_gen):
        """Test generate execution without reference"""
        mock_gen_instance = Mock()
        mock_gen.return_value = mock_gen_instance
        mock_gen_instance.generate.return_value = [Mock()]

        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = False
        step.reference_step_numbers = []
        step.llm_config = None
        step.parameters = Mock()
        step.parameters.system_prompt = "System prompt"
        step.parameters.user_prompt = "User prompt"
        step.parameters.num_sequences = 1
        step.parameters.max_tokens = 100
        step.parameters.temperature = 0.7

        result = service._execute_generate(step, 0)

        assert len(result) == 1
        mock_gen_instance.generate.assert_called_once()

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_generate_with_custom_model(self, mock_parse, mock_gen):
        """Test generate execution with custom model config"""
        mock_gen_instance = Mock()
        mock_gen.return_value = mock_gen_instance
        mock_gen_instance.generate.return_value = [Mock()]

        service = PipelineService("default-model")

        step = Mock()
        step.uses_reference = False
        step.reference_step_numbers = []
        step.llm_config = "custom-model"
        step.parameters = Mock()
        step.parameters.system_prompt = "System"
        step.parameters.user_prompt = "User"
        step.parameters.num_sequences = 1
        step.parameters.max_tokens = 50
        step.parameters.temperature = 0.5

        result = service._execute_generate(step, 0)

        # Should have cached the custom model
        assert "custom-model" in service._generate_services_cache


class TestPipelineServiceExecuteVerify:
    """Tests for _execute_verify method"""

    @patch('domain.services.pipeline_service.VerifierService')
    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_verify_without_reference(self, mock_parse, mock_gen, mock_verifier):
        """Test verify execution without reference"""
        mock_gen.return_value = Mock()
        mock_verifier_instance = Mock()
        mock_verifier.return_value = mock_verifier_instance
        mock_summary = Mock()
        mock_summary.final_status = "confirmed"
        mock_verifier_instance.verify.return_value = mock_summary

        service = PipelineService("test-model")

        step = Mock()
        step.uses_reference = False
        step.reference_step_numbers = []
        step.llm_config = None
        step.parameters = Mock()
        step.parameters.methods = []
        step.parameters.required_for_confirmed = 1
        step.parameters.required_for_review = 1

        result = service._execute_verify(step, 0)

        assert len(result) == 1
        mock_verifier_instance.verify.assert_called_once()


class TestPipelineServiceProcessPlaceholders:
    """Tests for _process_placeholders method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_process_placeholders_replaces_in_both_prompts(self, mock_parse, mock_gen):
        """Test placeholder replacement in both system and user prompts"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        system_prompt = "Hello {name}"
        user_prompt = "Age: {age}"
        references = {"name": "John", "age": "30"}
        reference_dict = {}

        new_sys, new_usr, ref_dict = service._process_placeholders(
            system_prompt, user_prompt, references, reference_dict
        )

        assert new_sys == "Hello John"
        assert new_usr == "Age: 30"
        assert ref_dict == {"name": "John", "age": "30"}

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_process_placeholders_tracks_used_references(self, mock_parse, mock_gen):
        """Test that only used references are tracked"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        system_prompt = "Hello {name}"
        user_prompt = "No placeholders here"
        references = {"name": "John", "unused": "value"}
        reference_dict = {}

        new_sys, new_usr, ref_dict = service._process_placeholders(
            system_prompt, user_prompt, references, reference_dict
        )

        assert "name" in ref_dict
        assert "unused" not in ref_dict


class TestPipelineServiceGetGenerateService:
    """Tests for _get_generate_service caching"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_generate_service_caches_instance(self, mock_parse, mock_gen):
        """Test that generate service is cached"""
        mock_gen_instance = Mock()
        mock_gen.return_value = mock_gen_instance

        service = PipelineService("test-model")

        # First call should create
        result1 = service._get_generate_service("test-model")
        # Second call should use cache
        result2 = service._get_generate_service("test-model")

        assert result1 == result2
        # Should only be called once for init + once for this model
        assert mock_gen.call_count == 1

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_generate_service_creates_new_for_different_model(self, mock_parse, mock_gen):
        """Test that different models get different services.

        Note: MAX_CACHED_MODELS=2 by default, so we test with only 2 models
        to avoid LRU eviction affecting the test.
        """
        mock_gen.return_value = Mock()

        service = PipelineService("model1")

        # Explicitly call _get_generate_service to add models to cache
        # (constructor only stores default_model_name, doesn't populate cache)
        service._get_generate_service("model1")
        service._get_generate_service("model2")

        assert "model1" in service._generate_services_cache
        assert "model2" in service._generate_services_cache


class TestPipelineServiceGenerateVariations:
    """Tests for _generate_variations method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_generate_variations_no_placeholders(self, mock_parse, mock_gen):
        """Test variations when no placeholders present"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        variations = service._generate_variations(
            system_prompt_template="Static system",
            user_prompt_template="Static user",
            reference_data=[],
            other_attributes=None
        )

        assert len(variations) == 1
        assert variations[0] == ("Static system", "Static user", {})

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_generate_variations_with_global_references(self, mock_parse, mock_gen):
        """Test variations with global references"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        service.global_references = {"name": "John"}

        variations = service._generate_variations(
            system_prompt_template="Hello {name}",
            user_prompt_template="User message",
            reference_data=[],
            other_attributes=None
        )

        assert len(variations) == 1
        assert variations[0][0] == "Hello John"
        assert "name" in variations[0][2]


class TestPipelineServiceValidateStepReferencesNoRefs:
    """Additional tests for _validate_step_references"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_step_references_uses_reference_no_data(self, mock_parse, mock_gen):
        """Test validation fails when uses_reference but no references provided"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")
        service.global_references = {}  # No global references

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = []  # No step references either

        with pytest.raises(ValueError, match="no references are provided"):
            service._validate_step_references(step, 0)


class TestPipelineServiceExecuteVerifyWithReferences:
    """Tests for _execute_verify method with references"""

    @patch('domain.services.pipeline_service.VerifierService')
    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_verify_with_reference_confirmed_status(self, mock_parse, mock_gen, mock_verifier):
        """Test verify with reference adds to confirmed_references"""
        mock_gen.return_value = Mock()
        mock_verifier_instance = Mock()
        mock_verifier.return_value = mock_verifier_instance

        # Create mock summary with confirmed status
        mock_summary = Mock()
        mock_summary.final_status = "confirmed"
        mock_summary.reference_data = None
        mock_verifier_instance.verify.return_value = mock_summary

        service = PipelineService("test-model")

        # Setup parse result as reference using immutable context
        mock_parse_result = Mock()
        mock_parse_result.entries = [{"name": "John", "age": "30"}]
        service._context = create_context_with_results([
            ("parse", [mock_parse_result])
        ])

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [0]
        step.llm_config = None
        step.parameters = Mock()
        step.parameters.methods = [
            Mock(
                system_prompt="Hello {name}",
                user_prompt="Age: {age}",
                mode="eliminatory",
                name="test",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1,
                max_tokens=10,
                temperature=0.7
            )
        ]
        step.parameters.required_for_confirmed = 1
        step.parameters.required_for_review = 0

        result = service._execute_verify(step, 1)

        assert len(result) >= 1
        assert len(service.confirmed_references) >= 1

    @patch('domain.services.pipeline_service.VerifierService')
    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_verify_with_reference_review_status(self, mock_parse, mock_gen, mock_verifier):
        """Test verify with reference adds to to_verify_references when review"""
        mock_gen.return_value = Mock()
        mock_verifier_instance = Mock()
        mock_verifier.return_value = mock_verifier_instance

        # Create mock summary with review status
        mock_summary = Mock()
        mock_summary.final_status = "review"
        mock_summary.reference_data = None
        mock_verifier_instance.verify.return_value = mock_summary

        service = PipelineService("test-model")

        # Setup parse result as reference using immutable context
        mock_parse_result = Mock()
        mock_parse_result.entries = [{"name": "Jane"}]
        service._context = create_context_with_results([
            ("parse", [mock_parse_result])
        ])

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [0]
        step.llm_config = None
        step.parameters = Mock()
        step.parameters.methods = [
            Mock(
                system_prompt="Hello {name}",
                user_prompt="Check user",
                mode="cumulative",
                name="test",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1,
                max_tokens=10,
                temperature=0.7
            )
        ]
        step.parameters.required_for_confirmed = 1
        step.parameters.required_for_review = 0

        result = service._execute_verify(step, 1)

        assert len(result) >= 1
        assert len(service.to_verify_references) >= 1

    @patch('domain.services.pipeline_service.VerifierService')
    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_execute_verify_with_reference_discarded_status(self, mock_parse, mock_gen, mock_verifier):
        """Test verify with reference - discarded status doesn't add to either list"""
        mock_gen.return_value = Mock()
        mock_verifier_instance = Mock()
        mock_verifier.return_value = mock_verifier_instance

        mock_summary = Mock()
        mock_summary.final_status = "discarded"
        mock_summary.reference_data = None
        mock_verifier_instance.verify.return_value = mock_summary

        service = PipelineService("test-model")

        # Setup parse result as reference using immutable context
        mock_parse_result = Mock()
        mock_parse_result.entries = [{"name": "Bob"}]
        service._context = create_context_with_results([
            ("parse", [mock_parse_result])
        ])

        step = Mock()
        step.uses_reference = True
        step.reference_step_numbers = [0]
        step.llm_config = None
        step.parameters = Mock()
        step.parameters.methods = [
            Mock(
                system_prompt="System",
                user_prompt="User {name}",
                mode="eliminatory",
                name="test",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1,
                max_tokens=10,
                temperature=0.7
            )
        ]
        step.parameters.required_for_confirmed = 1
        step.parameters.required_for_review = 0

        result = service._execute_verify(step, 1)

        assert len(result) >= 1
        assert len(service.confirmed_references) == 0
        assert len(service.to_verify_references) == 0


class TestPipelineServiceCreateMethodsVariations:
    """Tests for _create_methods_variations method"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_create_methods_variations_single_method(self, mock_parse, mock_gen):
        """Test creating variations for single method"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        original_request = Mock()
        original_request.methods = [
            Mock(
                system_prompt="Hello {name}",
                user_prompt="Age: {age}",
                mode=VerificationMode.ELIMINATORY,
                name="test",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1,
                max_tokens=10,
                temperature=0.7
            )
        ]
        original_request.required_for_confirmed = 1
        original_request.required_for_review = 0

        # Reference data with parse results
        mock_parse_result = Mock()
        mock_parse_result.entries = [
            {"name": "John", "age": "30"},
            {"name": "Jane", "age": "25"}
        ]
        reference_data = [(0, "parse", [mock_parse_result])]

        variations = service._create_methods_variations(original_request, reference_data)

        # Should have 2 variations (one per parse entry)
        assert len(variations) == 2

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_create_methods_variations_multiple_methods(self, mock_parse, mock_gen):
        """Test creating variations for multiple methods"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        original_request = Mock()
        original_request.methods = [
            Mock(
                system_prompt="Method1 {name}",
                user_prompt="User1",
                mode=VerificationMode.ELIMINATORY,
                name="method1",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1,
                max_tokens=10,
                temperature=0.7
            ),
            Mock(
                system_prompt="Method2",
                user_prompt="User2 {age}",
                mode=VerificationMode.CUMULATIVE,
                name="method2",
                num_sequences=2,
                valid_responses=["true"],
                required_matches=1,
                max_tokens=20,
                temperature=0.5
            )
        ]
        original_request.required_for_confirmed = 2
        original_request.required_for_review = 1

        mock_parse_result = Mock()
        mock_parse_result.entries = [{"name": "Alice", "age": "28"}]
        reference_data = [(0, "parse", [mock_parse_result])]

        variations = service._create_methods_variations(original_request, reference_data)

        # Each variation should have 2 methods
        assert len(variations) >= 1
        for verify_request, ref_dict in variations:
            assert len(verify_request.methods) == 2


class TestPipelineServiceGenerateVariationsWithMethods:
    """Tests for _generate_variations with other_attributes (method creation)"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_generate_variations_creates_verification_method(self, mock_parse, mock_gen):
        """Test that variations create VerificationMethod when other_attributes provided"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        other_attrs = {
            "mode": VerificationMode.ELIMINATORY,
            "name": "test_method",
            "num_sequences": 2,
            "valid_responses": ["yes", "true"],
            "required_matches": 1,
            "max_tokens": 50,
            "temperature": 0.8
        }

        variations = service._generate_variations(
            system_prompt_template="Static system",
            user_prompt_template="Static user",
            reference_data=[],
            other_attributes=other_attrs
        )

        assert len(variations) == 1
        method, ref_dict = variations[0]
        assert isinstance(method, VerificationMethod)
        assert method.system_prompt == "Static system"
        assert method.user_prompt == "Static user"

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_generate_variations_with_generate_reference(self, mock_parse, mock_gen):
        """Test variations with generate step reference"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Mock generate result
        gen_result = Mock()
        gen_result.content = "Generated content for testing"
        reference_data = [(0, "generate", [gen_result])]

        variations = service._generate_variations(
            system_prompt_template="System with {output_1}",
            user_prompt_template="User prompt",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt, user_prompt, ref_dict = variations[0]
        assert "Generated content for testing" in sys_prompt

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_generate_variations_with_parse_reference(self, mock_parse, mock_gen):
        """Test variations with parse step reference"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Mock parse result
        mock_parse_result = Mock()
        mock_parse_result.entries = [
            {"field1": "value1"},
            {"field1": "value2"}
        ]
        reference_data = [(0, "parse", [mock_parse_result])]

        variations = service._generate_variations(
            system_prompt_template="System with {field1}",
            user_prompt_template="User prompt",
            reference_data=reference_data,
            other_attributes=None
        )

        # Should have 2 variations (one per parse entry)
        assert len(variations) == 2
        values_found = [v[0] for v in variations]
        assert "System with value1" in values_found
        assert "System with value2" in values_found

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_generate_variations_with_unknown_step_type(self, mock_parse, mock_gen):
        """Test variations with unknown step type passes through"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Mock unknown step type
        reference_data = [(0, "unknown", ["some_result"])]

        variations = service._generate_variations(
            system_prompt_template="Static system",
            user_prompt_template="Static user",
            reference_data=reference_data,
            other_attributes=None
        )

        # Should still produce a variation (falls through to next index)
        assert len(variations) >= 1


class TestPipelineServiceRunPipelineException:
    """Tests for run_pipeline exception handling"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_run_pipeline_logs_and_raises_on_step_error(self, mock_parse, mock_gen):
        """Test that pipeline logs error and re-raises on step failure"""
        mock_gen_instance = Mock()
        mock_gen.return_value = mock_gen_instance
        mock_gen_instance.generate.side_effect = RuntimeError("Generation failed")

        service = PipelineService("test-model")

        step = PipelineStep(
            type="generate",
            parameters=GenerateTextRequest(
                system_prompt="System",
                user_prompt="User",
                num_sequences=1
            ),
            uses_reference=False,
            reference_step_numbers=[]
        )

        with pytest.raises(RuntimeError, match="Generation failed"):
            service.run_pipeline([step])


# =============================================================================
# Tests for Multiple Step References
# =============================================================================

class TestMultipleStepReferences:
    """
    Tests for referencing MULTIPLE previous steps in a single pipeline step.

    Verifies that:
    - {output_1} and {output_2} can be used together (two generate steps)
    - Fields from multiple parse steps can be combined
    - Generate and parse references can be mixed
    - Cartesian product is correctly generated for multiple results
    """

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_multiple_generate_step_references(self, mock_parse, mock_gen):
        """Test using {output_1} and {output_2} from two generate steps"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Setup two generate step results
        gen_result_0 = Mock()
        gen_result_0.content = "First generated content"
        gen_result_1 = Mock()
        gen_result_1.content = "Second generated content"

        service._context = create_context_with_results([
            ("generate", [gen_result_0]),
            ("generate", [gen_result_1])
        ])

        # Reference data from both steps
        reference_data = [
            (0, "generate", (gen_result_0,)),
            (1, "generate", (gen_result_1,))
        ]

        # Template using both {output_1} and {output_2}
        variations = service._generate_variations(
            system_prompt_template="Compare: {output_1} vs {output_2}",
            user_prompt_template="Analysis of both contents",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt, user_prompt, ref_dict = variations[0]

        # Both contents should be substituted
        assert "First generated content" in sys_prompt
        assert "Second generated content" in sys_prompt
        assert "output_1" in ref_dict
        assert "output_2" in ref_dict

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_multiple_parse_step_references(self, mock_parse, mock_gen):
        """Test combining fields from two different parse steps"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Setup two parse step results with different fields
        parse_result_0 = Mock()
        parse_result_0.entries = [{"name": "John", "age": "30"}]
        parse_result_1 = Mock()
        parse_result_1.entries = [{"city": "Madrid", "country": "Spain"}]

        service._context = create_context_with_results([
            ("parse", [parse_result_0]),
            ("parse", [parse_result_1])
        ])

        reference_data = [
            (0, "parse", (parse_result_0,)),
            (1, "parse", (parse_result_1,))
        ]

        # Template using fields from both parse steps
        variations = service._generate_variations(
            system_prompt_template="{name} lives in {city}",
            user_prompt_template="Age: {age}, Country: {country}",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt, user_prompt, ref_dict = variations[0]

        assert sys_prompt == "John lives in Madrid"
        assert user_prompt == "Age: 30, Country: Spain"
        assert ref_dict == {"name": "John", "age": "30", "city": "Madrid", "country": "Spain"}

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_mixed_generate_and_parse_references(self, mock_parse, mock_gen):
        """Test mixing generate step reference with parse step reference"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Setup generate and parse results
        gen_result = Mock()
        gen_result.content = "Generated text here"
        parse_result = Mock()
        parse_result.entries = [{"extracted_name": "Alice"}]

        service._context = create_context_with_results([
            ("generate", [gen_result]),
            ("parse", [parse_result])
        ])

        reference_data = [
            (0, "generate", (gen_result,)),
            (1, "parse", (parse_result,))
        ]

        # Template using both output_1 (generate) and extracted_name (parse)
        variations = service._generate_variations(
            system_prompt_template="Content: {output_1}",
            user_prompt_template="Name found: {extracted_name}",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt, user_prompt, ref_dict = variations[0]

        assert sys_prompt == "Content: Generated text here"
        assert user_prompt == "Name found: Alice"
        assert "output_1" in ref_dict
        assert "extracted_name" in ref_dict

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_cartesian_product_multiple_results(self, mock_parse, mock_gen):
        """Test that multiple results from multiple steps create Cartesian product"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Setup two generate steps, each with 2 results
        gen_result_0a = Mock()
        gen_result_0a.content = "Content 0A"
        gen_result_0b = Mock()
        gen_result_0b.content = "Content 0B"

        gen_result_1a = Mock()
        gen_result_1a.content = "Content 1A"
        gen_result_1b = Mock()
        gen_result_1b.content = "Content 1B"

        service._context = create_context_with_results([
            ("generate", [gen_result_0a, gen_result_0b]),
            ("generate", [gen_result_1a, gen_result_1b])
        ])

        reference_data = [
            (0, "generate", (gen_result_0a, gen_result_0b)),
            (1, "generate", (gen_result_1a, gen_result_1b))
        ]

        variations = service._generate_variations(
            system_prompt_template="{output_1} + {output_2}",
            user_prompt_template="Combined",
            reference_data=reference_data,
            other_attributes=None
        )

        # Should be 2 x 2 = 4 variations (Cartesian product)
        assert len(variations) == 4

        # Check all combinations are present
        combinations = set()
        for sys_prompt, _, _ in variations:
            combinations.add(sys_prompt)

        assert "Content 0A + Content 1A" in combinations
        assert "Content 0A + Content 1B" in combinations
        assert "Content 0B + Content 1A" in combinations
        assert "Content 0B + Content 1B" in combinations

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_parse_entries_cartesian_product(self, mock_parse, mock_gen):
        """Test Cartesian product with multiple parse entries from multiple steps"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        # Setup two parse steps with multiple entries each
        parse_result_0 = Mock()
        parse_result_0.entries = [
            {"person": "Alice"},
            {"person": "Bob"}
        ]
        parse_result_1 = Mock()
        parse_result_1.entries = [
            {"item": "Book"},
            {"item": "Pen"}
        ]

        service._context = create_context_with_results([
            ("parse", [parse_result_0]),
            ("parse", [parse_result_1])
        ])

        reference_data = [
            (0, "parse", (parse_result_0,)),
            (1, "parse", (parse_result_1,))
        ]

        variations = service._generate_variations(
            system_prompt_template="{person} has {item}",
            user_prompt_template="Check ownership",
            reference_data=reference_data,
            other_attributes=None
        )

        # Should be 2 x 2 = 4 variations
        assert len(variations) == 4

        prompts = [sys for sys, _, _ in variations]
        assert "Alice has Book" in prompts
        assert "Alice has Pen" in prompts
        assert "Bob has Book" in prompts
        assert "Bob has Pen" in prompts

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_three_step_references(self, mock_parse, mock_gen):
        """Test referencing three previous steps simultaneously"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        gen_result = Mock()
        gen_result.content = "Generated intro"
        parse_result_1 = Mock()
        parse_result_1.entries = [{"subject": "Math"}]
        parse_result_2 = Mock()
        parse_result_2.entries = [{"level": "Advanced"}]

        service._context = create_context_with_results([
            ("generate", [gen_result]),
            ("parse", [parse_result_1]),
            ("parse", [parse_result_2])
        ])

        reference_data = [
            (0, "generate", (gen_result,)),
            (1, "parse", (parse_result_1,)),
            (2, "parse", (parse_result_2,))
        ]

        variations = service._generate_variations(
            system_prompt_template="{output_1}: {subject} - {level}",
            user_prompt_template="Full context",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt, _, ref_dict = variations[0]

        assert sys_prompt == "Generated intro: Math - Advanced"
        assert len(ref_dict) == 3
        assert "output_1" in ref_dict
        assert "subject" in ref_dict
        assert "level" in ref_dict

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_get_reference_data_multiple_steps(self, mock_parse, mock_gen):
        """Test _get_reference_data correctly retrieves multiple step results"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        gen_result = Mock()
        gen_result.content = "Gen content"
        parse_result = Mock()
        parse_result.entries = [{"key": "value"}]

        service._context = create_context_with_results([
            ("generate", [gen_result]),
            ("parse", [parse_result])
        ])

        # Reference both steps 0 and 1 from step 2
        ref_data = service._get_reference_data([0, 1], 2)

        assert len(ref_data) == 2
        assert ref_data[0][0] == 0  # step index
        assert ref_data[0][1] == "generate"  # step type
        assert ref_data[1][0] == 1
        assert ref_data[1][1] == "parse"

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_references_multiple_valid(self, mock_parse, mock_gen):
        """Test _validate_references with multiple valid references"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        service._context = create_context_with_results([
            ("generate", ["result1"]),
            ("parse", ["result2"]),
            ("generate", ["result3"])
        ])

        # Reference steps 0, 1, 2 from step 3
        result = service._validate_references([0, 1, 2], 3)
        assert result is True

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_validate_references_multiple_one_invalid(self, mock_parse, mock_gen):
        """Test _validate_references fails if any reference is invalid"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        service._context = create_context_with_results([
            ("generate", ["result1"]),
            ("parse", ["result2"])
        ])

        # Reference steps 0, 1, 5 (5 doesn't exist)
        result = service._validate_references([0, 1, 5], 3)
        assert result is False

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_reference_order_preserved(self, mock_parse, mock_gen):
        """Test that reference processing order is preserved"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        gen_result_0 = Mock()
        gen_result_0.content = "FIRST"
        gen_result_1 = Mock()
        gen_result_1.content = "SECOND"

        service._context = create_context_with_results([
            ("generate", [gen_result_0]),
            ("generate", [gen_result_1])
        ])

        # Reference in order [0, 1]
        reference_data = [
            (0, "generate", (gen_result_0,)),
            (1, "generate", (gen_result_1,))
        ]

        variations = service._generate_variations(
            system_prompt_template="Order: {output_1} then {output_2}",
            user_prompt_template="Test",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt = variations[0][0]
        assert sys_prompt == "Order: FIRST then SECOND"


class TestMultipleReferencesWithGlobalReferences:
    """Tests for combining multiple step references with global references"""

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_step_refs_with_global_refs(self, mock_parse, mock_gen):
        """Test combining step references with global references"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        gen_result = Mock()
        gen_result.content = "Generated text"

        service._context = create_context_with_results([
            ("generate", [gen_result])
        ])
        # Add global references
        service._context = service._context.with_global_references({"global_var": "GLOBAL_VALUE"})

        reference_data = [
            (0, "generate", (gen_result,))
        ]

        variations = service._generate_variations(
            system_prompt_template="Content: {output_1}, Global: {global_var}",
            user_prompt_template="Combined",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt, _, ref_dict = variations[0]

        assert sys_prompt == "Content: Generated text, Global: GLOBAL_VALUE"
        assert "output_1" in ref_dict
        assert "global_var" in ref_dict

    @patch('domain.services.pipeline_service.GenerateService')
    @patch('domain.services.pipeline_service.ParseService')
    def test_global_refs_override_step_refs(self, mock_parse, mock_gen):
        """Test that global references can fill placeholders not covered by step refs"""
        mock_gen.return_value = Mock()
        service = PipelineService("test-model")

        parse_result = Mock()
        parse_result.entries = [{"name": "StepName"}]

        service._context = create_context_with_results([
            ("parse", [parse_result])
        ])
        # Global ref with different key
        service._context = service._context.with_global_references({"extra_info": "Extra"})

        reference_data = [
            (0, "parse", (parse_result,))
        ]

        variations = service._generate_variations(
            system_prompt_template="{name} - {extra_info}",
            user_prompt_template="Test",
            reference_data=reference_data,
            other_attributes=None
        )

        assert len(variations) == 1
        sys_prompt = variations[0][0]
        assert sys_prompt == "StepName - Extra"
