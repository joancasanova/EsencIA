# tests/unit/domain/entities/test_pipeline.py

import pytest
from domain.model.entities.pipeline import (
    PipelineStep, PipelineRequest, PipelineResponse, PipelineExecutionError,
    PipelineExecutionContext, BenchmarkExecutionContext
)
from domain.model.entities.generation import GenerateTextRequest
from domain.model.entities.verification import VerificationMethod, VerificationMode, VerifyRequest
from domain.model.entities.parsing import ParseRule, ParseMode, ParseRequest


class TestPipelineStep:
    """Tests for PipelineStep entity"""

    def test_pipeline_step_creation_generate(self):
        """Test PipelineStep creation for generate type"""
        # Arrange & Act
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

        # Assert
        assert step.type == "generate"
        assert step.uses_reference == False
        assert len(step.reference_step_numbers) == 0

    def test_pipeline_step_creation_verify(self):
        """Test PipelineStep creation for verify type"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            valid_responses=["yes"],
            required_matches=1
        )

        # Act
        step = PipelineStep(
            type="verify",
            parameters=VerifyRequest(
                methods=[method],
                required_for_confirmed=1,
                required_for_review=1
            ),
            uses_reference=True,
            reference_step_numbers=[0]
        )

        # Assert
        assert step.type == "verify"
        assert step.uses_reference == True
        assert 0 in step.reference_step_numbers

    def test_pipeline_step_creation_parse(self):
        """Test PipelineStep creation for parse type"""
        # Arrange
        rule = ParseRule(
            name="test",
            pattern=r"\d+",
            mode=ParseMode.REGEX
        )

        # Act
        step = PipelineStep(
            type="parse",
            parameters=ParseRequest(
                rules=[rule],
                text="test text"
            ),
            uses_reference=False,
            reference_step_numbers=[]
        )

        # Assert
        assert step.type == "parse"


class TestPipelineRequest:
    """Tests for PipelineRequest entity"""

    def test_pipeline_request_creation(self):
        """Test PipelineRequest creation"""
        # Arrange
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

        # Act
        request = PipelineRequest(
            steps=[step],
            global_references={"key": "value"}
        )

        # Assert
        assert len(request.steps) == 1
        assert "key" in request.global_references

    def test_pipeline_request_with_multiple_steps(self):
        """Test PipelineRequest with multiple steps"""
        # Arrange
        step1 = PipelineStep(
            type="generate",
            parameters=GenerateTextRequest(
                system_prompt="System",
                user_prompt="User",
                num_sequences=1
            ),
            uses_reference=False,
            reference_step_numbers=[]
        )

        rule = ParseRule(
            name="test_rule",
            pattern=r"\d+",
            mode=ParseMode.REGEX
        )

        step2 = PipelineStep(
            type="parse",
            parameters=ParseRequest(
                rules=[rule],
                text="test 123"
            ),
            uses_reference=True,
            reference_step_numbers=[0]
        )

        # Act
        request = PipelineRequest(
            steps=[step1, step2],
            global_references={}
        )

        # Assert
        assert len(request.steps) == 2


class TestPipelineResponse:
    """Tests for PipelineResponse entity"""

    def test_pipeline_response_creation_success(self):
        """Test PipelineResponse creation with successful execution"""
        # Arrange & Act
        response = PipelineResponse(
            step_results=[
                {"step_type": "generate", "step_data": []},
                {"step_type": "parse", "step_data": []}
            ],
            verification_references={"confirmed": [], "to_verify": []},
            total_entries=10,
            successful_entries=8,
            failed_entries=2,
            errors=[]
        )

        # Assert
        assert len(response.step_results) == 2
        assert response.total_entries == 10
        assert response.successful_entries == 8
        assert response.failed_entries == 2

    def test_pipeline_response_with_errors(self):
        """Test PipelineResponse creation with errors"""
        # Arrange
        error = PipelineExecutionError(
            entry_index=0,
            error_type="ValueError",
            error_message="Test error",
            entry_data={"test": "data"},
            traceback="Traceback..."
        )

        # Act
        response = PipelineResponse(
            step_results=[],
            verification_references={"confirmed": [], "to_verify": []},
            total_entries=1,
            successful_entries=0,
            failed_entries=1,
            errors=[error]
        )

        # Assert
        assert len(response.errors) == 1
        assert response.errors[0].error_type == "ValueError"

    def test_pipeline_response_to_dict(self):
        """Test PipelineResponse serialization to dict"""
        # Arrange
        response = PipelineResponse(
            step_results=[{"step_type": "generate", "data": []}],
            verification_references={"confirmed": ["ref1"], "to_verify": []},
            total_entries=5,
            successful_entries=4,
            failed_entries=1,
            errors=[]
        )

        # Act
        result_dict = response.to_dict()

        # Assert
        assert "step_results" in result_dict
        assert "verification_references" in result_dict
        assert "execution_summary" in result_dict
        assert result_dict["execution_summary"]["total_entries"] == 5


class TestPipelineExecutionError:
    """Tests for PipelineExecutionError entity"""

    def test_pipeline_error_creation(self):
        """Test PipelineExecutionError creation"""
        # Arrange & Act
        error = PipelineExecutionError(
            entry_index=5,
            error_type="RuntimeError",
            error_message="Something went wrong",
            entry_data={"input": "test"},
            traceback="Traceback:\n  Line 1\n  Line 2"
        )

        # Assert
        assert error.entry_index == 5
        assert error.error_type == "RuntimeError"
        assert error.error_message == "Something went wrong"
        assert "input" in error.entry_data
        assert "Traceback" in error.traceback

    def test_pipeline_error_to_dict(self):
        """Test PipelineExecutionError serialization to dict"""
        # Arrange
        error = PipelineExecutionError(
            entry_index=1,
            error_type="ValueError",
            error_message="Invalid value",
            entry_data={"field": "value"},
            traceback=None
        )

        # Act
        error_dict = error.to_dict()

        # Assert
        assert error_dict["entry_index"] == 1
        assert error_dict["error_type"] == "ValueError"
        assert error_dict["entry_data"] == {"field": "value"}


class TestPipelineExecutionContext:
    """Tests for PipelineExecutionContext immutable context"""

    def test_creation_with_defaults(self):
        """Test creating context with default values"""
        context = PipelineExecutionContext()

        assert context.results == ()
        assert context.global_references == {}
        assert context.confirmed_references == ()
        assert context.to_verify_references == ()

    def test_creation_with_values(self):
        """Test creating context with initial values"""
        context = PipelineExecutionContext(
            results=(("generate", ("result1", "result2")),),
            global_references={"key": "value"},
            confirmed_references=({"ref": "confirmed"},),
            to_verify_references=({"ref": "pending"},)
        )

        assert len(context.results) == 1
        assert context.global_references["key"] == "value"
        assert len(context.confirmed_references) == 1
        assert len(context.to_verify_references) == 1

    def test_with_result_adds_new_result(self):
        """Test with_result adds a new result at specified step"""
        context = PipelineExecutionContext()

        new_context = context.with_result(
            step_number=0,
            step_type="generate",
            step_result=("result1", "result2")
        )

        # Original context unchanged (immutable)
        assert context.results == ()

        # New context has the result
        assert len(new_context.results) == 1
        assert new_context.results[0] == ("generate", ("result1", "result2"))

    def test_with_result_extends_list_for_higher_step_number(self):
        """Test with_result extends results list when step_number > len(results)"""
        context = PipelineExecutionContext()

        # Add result at step 2 (skipping 0 and 1)
        new_context = context.with_result(
            step_number=2,
            step_type="parse",
            step_result=("parsed_data",)
        )

        # Should have extended to 3 elements (indices 0, 1, 2)
        assert len(new_context.results) == 3
        assert new_context.results[0] is None
        assert new_context.results[1] is None
        assert new_context.results[2] == ("parse", ("parsed_data",))

    def test_with_result_extends_existing_result(self):
        """Test with_result extends existing result at same step"""
        context = PipelineExecutionContext(
            results=(("generate", ("result1",)),)
        )

        # Add more results to step 0
        new_context = context.with_result(
            step_number=0,
            step_type="generate",
            step_result=("result2", "result3")
        )

        # Should have combined results
        assert len(new_context.results) == 1
        step_type, step_results = new_context.results[0]
        assert step_type == "generate"
        assert step_results == ("result1", "result2", "result3")

    def test_with_result_preserves_other_fields(self):
        """Test with_result preserves global_references and other fields"""
        context = PipelineExecutionContext(
            global_references={"key": "value"},
            confirmed_references=({"confirmed": "ref"},),
            to_verify_references=({"pending": "ref"},)
        )

        new_context = context.with_result(
            step_number=0,
            step_type="verify",
            step_result=("verified",)
        )

        assert new_context.global_references == {"key": "value"}
        assert new_context.confirmed_references == ({"confirmed": "ref"},)
        assert new_context.to_verify_references == ({"pending": "ref"},)

    def test_with_global_references_sets_references(self):
        """Test with_global_references creates new context with references"""
        context = PipelineExecutionContext()

        new_context = context.with_global_references({"text": "hello", "number": "42"})

        # Original unchanged
        assert context.global_references == {}

        # New context has references
        assert new_context.global_references == {"text": "hello", "number": "42"}

    def test_with_global_references_preserves_other_fields(self):
        """Test with_global_references preserves results and other references"""
        context = PipelineExecutionContext(
            results=(("generate", ("r1",)),),
            confirmed_references=({"c": "1"},),
            to_verify_references=({"v": "2"},)
        )

        new_context = context.with_global_references({"new": "refs"})

        assert new_context.results == (("generate", ("r1",)),)
        assert new_context.confirmed_references == ({"c": "1"},)
        assert new_context.to_verify_references == ({"v": "2"},)

    def test_with_confirmed_reference_adds_reference(self):
        """Test with_confirmed_reference adds a new confirmed reference"""
        context = PipelineExecutionContext()

        new_context = context.with_confirmed_reference({"status": "confirmed", "data": "test"})

        # Original unchanged
        assert context.confirmed_references == ()

        # New context has the reference
        assert len(new_context.confirmed_references) == 1
        assert new_context.confirmed_references[0] == {"status": "confirmed", "data": "test"}

    def test_with_confirmed_reference_appends_to_existing(self):
        """Test with_confirmed_reference appends to existing references"""
        context = PipelineExecutionContext(
            confirmed_references=({"first": "ref"},)
        )

        new_context = context.with_confirmed_reference({"second": "ref"})

        assert len(new_context.confirmed_references) == 2
        assert new_context.confirmed_references[0] == {"first": "ref"}
        assert new_context.confirmed_references[1] == {"second": "ref"}

    def test_with_confirmed_reference_preserves_other_fields(self):
        """Test with_confirmed_reference preserves other context fields"""
        context = PipelineExecutionContext(
            results=(("generate", ("r",)),),
            global_references={"g": "r"},
            to_verify_references=({"t": "v"},)
        )

        new_context = context.with_confirmed_reference({"new": "confirmed"})

        assert new_context.results == (("generate", ("r",)),)
        assert new_context.global_references == {"g": "r"}
        assert new_context.to_verify_references == ({"t": "v"},)

    def test_with_to_verify_reference_adds_reference(self):
        """Test with_to_verify_reference adds a new to-verify reference"""
        context = PipelineExecutionContext()

        new_context = context.with_to_verify_reference({"status": "pending", "data": "check"})

        # Original unchanged
        assert context.to_verify_references == ()

        # New context has the reference
        assert len(new_context.to_verify_references) == 1
        assert new_context.to_verify_references[0] == {"status": "pending", "data": "check"}

    def test_with_to_verify_reference_appends_to_existing(self):
        """Test with_to_verify_reference appends to existing references"""
        context = PipelineExecutionContext(
            to_verify_references=({"first": "verify"},)
        )

        new_context = context.with_to_verify_reference({"second": "verify"})

        assert len(new_context.to_verify_references) == 2
        assert new_context.to_verify_references[0] == {"first": "verify"}
        assert new_context.to_verify_references[1] == {"second": "verify"}

    def test_with_to_verify_reference_preserves_other_fields(self):
        """Test with_to_verify_reference preserves other context fields"""
        context = PipelineExecutionContext(
            results=(("parse", ("p",)),),
            global_references={"global": "ref"},
            confirmed_references=({"confirmed": "yes"},)
        )

        new_context = context.with_to_verify_reference({"new": "pending"})

        assert new_context.results == (("parse", ("p",)),)
        assert new_context.global_references == {"global": "ref"}
        assert new_context.confirmed_references == ({"confirmed": "yes"},)

    def test_immutability_frozen_dataclass(self):
        """Test that PipelineExecutionContext is immutable (frozen)"""
        context = PipelineExecutionContext(
            global_references={"key": "value"}
        )

        # Should raise FrozenInstanceError (a subclass of AttributeError)
        with pytest.raises(AttributeError):
            context.global_references = {"new": "value"}

    # Tests for getter methods: has_result_at_step, get_step_type, get_step_result, get_all_results_for_step

    def test_has_result_at_step_true_when_result_exists(self):
        """Test has_result_at_step returns True when result exists"""
        context = PipelineExecutionContext(
            results=(("generate", ("result1",)),)
        )

        assert context.has_result_at_step(0) is True

    def test_has_result_at_step_false_for_empty_context(self):
        """Test has_result_at_step returns False for empty context"""
        context = PipelineExecutionContext()

        assert context.has_result_at_step(0) is False

    def test_has_result_at_step_false_for_out_of_bounds(self):
        """Test has_result_at_step returns False for out of bounds index"""
        context = PipelineExecutionContext(
            results=(("generate", ("result1",)),)
        )

        assert context.has_result_at_step(5) is False

    def test_has_result_at_step_false_for_negative_index(self):
        """Test has_result_at_step returns False for negative index"""
        context = PipelineExecutionContext(
            results=(("generate", ("result1",)),)
        )

        assert context.has_result_at_step(-1) is False

    def test_has_result_at_step_false_for_none_slot(self):
        """Test has_result_at_step returns False when slot contains None"""
        context = PipelineExecutionContext(
            results=(None, ("parse", ("result",)))
        )

        assert context.has_result_at_step(0) is False
        assert context.has_result_at_step(1) is True

    def test_get_step_type_returns_type_string(self):
        """Test get_step_type returns the step type string"""
        context = PipelineExecutionContext(
            results=(
                ("generate", ("gen_result",)),
                ("parse", ("parse_result",)),
                ("verify", ("verify_result",))
            )
        )

        assert context.get_step_type(0) == "generate"
        assert context.get_step_type(1) == "parse"
        assert context.get_step_type(2) == "verify"

    def test_get_step_type_returns_none_for_empty_context(self):
        """Test get_step_type returns None for empty context"""
        context = PipelineExecutionContext()

        assert context.get_step_type(0) is None

    def test_get_step_type_returns_none_for_out_of_bounds(self):
        """Test get_step_type returns None for out of bounds index"""
        context = PipelineExecutionContext(
            results=(("generate", ("result1",)),)
        )

        assert context.get_step_type(10) is None

    def test_get_step_type_returns_none_for_none_slot(self):
        """Test get_step_type returns None when slot contains None"""
        context = PipelineExecutionContext(
            results=(None, ("parse", ("result",)))
        )

        assert context.get_step_type(0) is None
        assert context.get_step_type(1) == "parse"

    def test_get_step_result_returns_results_tuple(self):
        """Test get_step_result returns the results tuple"""
        context = PipelineExecutionContext(
            results=(
                ("generate", ("result1", "result2")),
                ("parse", ("parsed_data",))
            )
        )

        assert context.get_step_result(0) == ("result1", "result2")
        assert context.get_step_result(1) == ("parsed_data",)

    def test_get_step_result_returns_none_for_empty_context(self):
        """Test get_step_result returns None for empty context"""
        context = PipelineExecutionContext()

        assert context.get_step_result(0) is None

    def test_get_step_result_returns_none_for_out_of_bounds(self):
        """Test get_step_result returns None for out of bounds index"""
        context = PipelineExecutionContext(
            results=(("generate", ("result",)),)
        )

        assert context.get_step_result(5) is None

    def test_get_step_result_returns_none_for_none_slot(self):
        """Test get_step_result returns None when slot contains None"""
        context = PipelineExecutionContext(
            results=(None, ("verify", ("verified",)))
        )

        assert context.get_step_result(0) is None
        assert context.get_step_result(1) == ("verified",)

    def test_get_all_results_for_step_returns_results(self):
        """Test get_all_results_for_step returns the results tuple"""
        context = PipelineExecutionContext(
            results=(("generate", ("r1", "r2", "r3")),)
        )

        assert context.get_all_results_for_step(0) == ("r1", "r2", "r3")

    def test_get_all_results_for_step_returns_empty_tuple_for_empty_context(self):
        """Test get_all_results_for_step returns empty tuple for empty context"""
        context = PipelineExecutionContext()

        assert context.get_all_results_for_step(0) == ()

    def test_get_all_results_for_step_returns_empty_tuple_for_out_of_bounds(self):
        """Test get_all_results_for_step returns empty tuple for out of bounds"""
        context = PipelineExecutionContext(
            results=(("generate", ("result",)),)
        )

        assert context.get_all_results_for_step(100) == ()

    def test_get_all_results_for_step_returns_empty_tuple_for_none_slot(self):
        """Test get_all_results_for_step returns empty tuple when slot is None"""
        context = PipelineExecutionContext(
            results=(None, ("parse", ("data",)))
        )

        assert context.get_all_results_for_step(0) == ()
        assert context.get_all_results_for_step(1) == ("data",)

    def test_getter_methods_with_complex_results(self):
        """Test getter methods work with complex result objects"""
        mock_result = {"content": "generated text", "tokens": 100}
        context = PipelineExecutionContext(
            results=(("generate", (mock_result,)),)
        )

        assert context.has_result_at_step(0) is True
        assert context.get_step_type(0) == "generate"
        assert context.get_step_result(0) == (mock_result,)
        assert context.get_all_results_for_step(0)[0]["content"] == "generated text"

    def test_chained_operations(self):
        """Test chaining multiple with_* operations"""
        context = (
            PipelineExecutionContext()
            .with_global_references({"input": "data"})
            .with_result(0, "generate", ("gen1",))
            .with_confirmed_reference({"confirmed": "ref1"})
            .with_to_verify_reference({"pending": "ref2"})
            .with_result(1, "parse", ("parsed1",))
        )

        assert context.global_references == {"input": "data"}
        assert len(context.results) == 2
        assert context.results[0] == ("generate", ("gen1",))
        assert context.results[1] == ("parse", ("parsed1",))
        assert len(context.confirmed_references) == 1
        assert len(context.to_verify_references) == 1


class TestBenchmarkExecutionContext:
    """Tests for BenchmarkExecutionContext immutable context"""

    def test_creation_with_defaults(self):
        """Test creating context with default values"""
        context = BenchmarkExecutionContext()

        assert context.results == ()
        assert context.global_references == {}

    def test_creation_with_values(self):
        """Test creating context with initial values"""
        context = BenchmarkExecutionContext(
            results=("result1", "result2"),
            global_references={"field1": "value1", "field2": "value2"}
        )

        assert len(context.results) == 2
        assert context.global_references["field1"] == "value1"

    def test_with_result_adds_result(self):
        """Test with_result adds a new result"""
        context = BenchmarkExecutionContext()

        new_context = context.with_result("benchmark_result_1")

        # Original unchanged
        assert context.results == ()

        # New context has the result
        assert len(new_context.results) == 1
        assert new_context.results[0] == "benchmark_result_1"

    def test_with_result_appends_to_existing(self):
        """Test with_result appends to existing results"""
        context = BenchmarkExecutionContext(
            results=("first_result",)
        )

        new_context = context.with_result("second_result")

        assert len(new_context.results) == 2
        assert new_context.results[0] == "first_result"
        assert new_context.results[1] == "second_result"

    def test_with_result_preserves_global_references(self):
        """Test with_result preserves global_references"""
        context = BenchmarkExecutionContext(
            global_references={"entry_field": "entry_value"}
        )

        new_context = context.with_result("new_result")

        assert new_context.global_references == {"entry_field": "entry_value"}

    def test_with_global_references_sets_references(self):
        """Test with_global_references creates new context with references"""
        context = BenchmarkExecutionContext()

        new_context = context.with_global_references({
            "original": "texto original",
            "expected": "texto esperado"
        })

        # Original unchanged
        assert context.global_references == {}

        # New context has references
        assert new_context.global_references["original"] == "texto original"
        assert new_context.global_references["expected"] == "texto esperado"

    def test_with_global_references_preserves_results(self):
        """Test with_global_references preserves results"""
        context = BenchmarkExecutionContext(
            results=("existing_result",)
        )

        new_context = context.with_global_references({"new": "refs"})

        assert new_context.results == ("existing_result",)

    def test_immutability_frozen_dataclass(self):
        """Test that BenchmarkExecutionContext is immutable (frozen)"""
        context = BenchmarkExecutionContext(
            results=("result",),
            global_references={"key": "value"}
        )

        # Should raise FrozenInstanceError (a subclass of AttributeError)
        with pytest.raises(AttributeError):
            context.results = ("new_result",)

    def test_chained_operations(self):
        """Test chaining multiple with_* operations"""
        context = (
            BenchmarkExecutionContext()
            .with_global_references({"text": "original text", "label": "expected"})
            .with_result("result1")
            .with_result("result2")
            .with_result("result3")
        )

        assert context.global_references == {"text": "original text", "label": "expected"}
        assert len(context.results) == 3
        assert context.results == ("result1", "result2", "result3")

    def test_with_result_accepts_complex_objects(self):
        """Test with_result can store complex objects"""
        context = BenchmarkExecutionContext()

        complex_result = {
            "score": 0.95,
            "details": {"precision": 0.9, "recall": 1.0},
            "metadata": ["tag1", "tag2"]
        }

        new_context = context.with_result(complex_result)

        assert new_context.results[0] == complex_result
        assert new_context.results[0]["score"] == 0.95
