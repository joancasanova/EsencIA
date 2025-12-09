# domain/model/entities/pipeline.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from domain.model.entities.generation import GenerateTextRequest
from domain.model.entities.parsing import ParseRequest
from domain.model.entities.verification import VerifyRequest


@dataclass(frozen=True)
class PipelineExecutionContext:
    """
    Immutable execution context for pipeline operations.

    Instead of storing mutable state in the service, each execution
    creates a new context that is passed through methods. This ensures
    thread-safety and prevents state leakage between executions.

    Attributes:
        results: Tuple of step results (step_type, list_of_results)
        global_references: Immutable reference data available to all steps
        confirmed_references: Tuple of confirmed verification references
        to_verify_references: Tuple of references pending verification
    """
    results: Tuple[Optional[Tuple[str, Tuple[Any, ...]]], ...] = field(default_factory=tuple)
    global_references: Dict[str, str] = field(default_factory=dict)
    confirmed_references: Tuple[Dict[str, str], ...] = field(default_factory=tuple)
    to_verify_references: Tuple[Dict[str, str], ...] = field(default_factory=tuple)

    def with_result(
        self,
        step_number: int,
        step_type: str,
        step_result: Tuple[Any, ...]
    ) -> "PipelineExecutionContext":
        """
        Create a new context with an added/updated result.

        Args:
            step_number: Index of the step
            step_type: Type of the step ('generate', 'parse', 'verify')
            step_result: Tuple of results from the step

        Returns:
            New PipelineExecutionContext with the updated results
        """
        results_list = list(self.results)

        # Extend list if necessary
        while len(results_list) <= step_number:
            results_list.append(None)

        if results_list[step_number] is None:
            results_list[step_number] = (step_type, step_result)
        else:
            _, existing_results = results_list[step_number]
            results_list[step_number] = (step_type, existing_results + step_result)

        return PipelineExecutionContext(
            results=tuple(results_list),
            global_references=self.global_references,
            confirmed_references=self.confirmed_references,
            to_verify_references=self.to_verify_references
        )

    def with_global_references(self, references: Dict[str, str]) -> "PipelineExecutionContext":
        """
        Create a new context with updated global references.

        Args:
            references: New global references to set

        Returns:
            New PipelineExecutionContext with updated global references
        """
        return PipelineExecutionContext(
            results=self.results,
            global_references=references,
            confirmed_references=self.confirmed_references,
            to_verify_references=self.to_verify_references
        )

    def with_confirmed_reference(self, reference: Dict[str, str]) -> "PipelineExecutionContext":
        """
        Create a new context with an added confirmed reference.

        Args:
            reference: Reference data to add as confirmed

        Returns:
            New PipelineExecutionContext with the added confirmed reference
        """
        return PipelineExecutionContext(
            results=self.results,
            global_references=self.global_references,
            confirmed_references=self.confirmed_references + (reference,),
            to_verify_references=self.to_verify_references
        )

    def with_to_verify_reference(self, reference: Dict[str, str]) -> "PipelineExecutionContext":
        """
        Create a new context with an added to-verify reference.

        Args:
            reference: Reference data to add for verification

        Returns:
            New PipelineExecutionContext with the added to-verify reference
        """
        return PipelineExecutionContext(
            results=self.results,
            global_references=self.global_references,
            confirmed_references=self.confirmed_references,
            to_verify_references=self.to_verify_references + (reference,)
        )

    def has_result_at_step(self, step_number: int) -> bool:
        """
        Check if a result exists at the given step number.

        Args:
            step_number: Index of the step to check

        Returns:
            True if a result exists at the step, False otherwise
        """
        if step_number < 0 or step_number >= len(self.results):
            return False
        return self.results[step_number] is not None

    def get_step_type(self, step_number: int) -> Optional[str]:
        """
        Get the type of a step at the given step number.

        Args:
            step_number: Index of the step

        Returns:
            The step type string ('generate', 'parse', 'verify') or None if not found
        """
        if not self.has_result_at_step(step_number):
            return None
        step_type, _ = self.results[step_number]
        return step_type

    def get_step_result(self, step_number: int) -> Optional[Tuple[Any, ...]]:
        """
        Get the results of a step at the given step number.

        Args:
            step_number: Index of the step

        Returns:
            Tuple of results from the step or None if not found
        """
        if not self.has_result_at_step(step_number):
            return None
        _, step_results = self.results[step_number]
        return step_results

    def get_all_results_for_step(self, step_number: int) -> Tuple[Any, ...]:
        """
        Get all results for a step, returning empty tuple if step doesn't exist.

        This is a convenience method that never returns None, making it
        easier to iterate over results without null checks.

        Args:
            step_number: Index of the step

        Returns:
            Tuple of results from the step, or empty tuple if not found
        """
        result = self.get_step_result(step_number)
        return result if result is not None else ()


@dataclass(frozen=True)
class BenchmarkExecutionContext:
    """
    Immutable execution context for benchmark operations.

    Maintains state for a single benchmark execution without
    storing mutable state in the service.

    Attributes:
        results: Tuple of benchmark results collected during execution
        global_references: Entry data available for placeholder substitution
    """
    results: Tuple[Any, ...] = field(default_factory=tuple)
    global_references: Dict[str, str] = field(default_factory=dict)

    def with_result(self, result: Any) -> "BenchmarkExecutionContext":
        """
        Create a new context with an added result.

        Args:
            result: Benchmark result to add

        Returns:
            New BenchmarkExecutionContext with the added result
        """
        return BenchmarkExecutionContext(
            results=self.results + (result,),
            global_references=self.global_references
        )

    def with_global_references(self, references: Dict[str, str]) -> "BenchmarkExecutionContext":
        """
        Create a new context with updated global references.

        Args:
            references: Entry data for placeholder substitution

        Returns:
            New BenchmarkExecutionContext with updated references
        """
        return BenchmarkExecutionContext(
            results=self.results,
            global_references=references
        )

# Valid step types for pipeline execution
VALID_STEP_TYPES = frozenset({"generate", "parse", "verify"})

# Mapping of step types to their expected parameter types
STEP_TYPE_PARAMETERS = {
    "generate": GenerateTextRequest,
    "parse": ParseRequest,
    "verify": VerifyRequest,
}


@dataclass
class PipelineStep:
    """
    Represents a single step in the processing pipeline.

    Attributes:
        type: The type of processing step. Valid values:
              - 'generate': Text generation step
              - 'parse': Text parsing step
              - 'verify': Result verification step
        parameters: Configuration specific to the step type. Must match:
                   - GenerateTextRequest for 'generate' steps
                   - ParseRequest for 'parse' steps
                   - VerifyRequest for 'verify' steps
        uses_reference: Flag indicating if this step uses reference data
                       from previous steps
        reference_step_numbers: List of step indices (0-based) providing
                               reference data. Ordered by priority.
        llm_config: Optional configuration for the LLM to use for this step.

    Raises:
        ValueError: If type is invalid, reference_step_numbers contains negatives,
                   or parameters type doesn't match step type.
    """
    type: str
    parameters: Union[GenerateTextRequest, ParseRequest, VerifyRequest]
    uses_reference: bool = False
    reference_step_numbers: Optional[List[int]] = None
    llm_config: Optional[str] = None

    def __post_init__(self):
        """Validates step configuration after initialization."""
        # Validate step type
        if self.type not in VALID_STEP_TYPES:
            raise ValueError(
                f"Invalid step type '{self.type}'. "
                f"Must be one of: {', '.join(sorted(VALID_STEP_TYPES))}"
            )

        # Validate parameters type matches step type
        expected_type = STEP_TYPE_PARAMETERS[self.type]
        if not isinstance(self.parameters, expected_type):
            raise ValueError(
                f"Step type '{self.type}' requires {expected_type.__name__} parameters, "
                f"got {type(self.parameters).__name__}"
            )

        # Validate reference_step_numbers if provided
        if self.reference_step_numbers is not None:
            if not isinstance(self.reference_step_numbers, list):
                raise ValueError(
                    f"reference_step_numbers must be a list, "
                    f"got {type(self.reference_step_numbers).__name__}"
                )
            for idx, ref in enumerate(self.reference_step_numbers):
                if not isinstance(ref, int):
                    raise ValueError(
                        f"reference_step_numbers[{idx}] must be an integer, "
                        f"got {type(ref).__name__}"
                    )
                if ref < 0:
                    raise ValueError(
                        f"reference_step_numbers[{idx}] cannot be negative, got {ref}"
                    )

        # Validate uses_reference consistency
        if self.uses_reference and not self.reference_step_numbers:
            # This is allowed if global_references will be used,
            # but we can't validate that here (done at runtime)
            pass

@dataclass
class PipelineRequest:
    """
    Complete configuration for executing a processing pipeline.

    Attributes:
        steps: Ordered list of PipelineStep objects defining the 
              processing workflow
        global_references: Optional shared reference data available 
                          to all steps through {placeholder} syntax
    """
    steps: List[PipelineStep]
    global_references: Optional[Dict[str, str]] = None

@dataclass
class PipelineExecutionError:
    """
    Represents an error that occurred during pipeline execution.

    Attributes:
        entry_index: Index of the entry that failed
        entry_data: The data that caused the error
        error_type: Type of the exception
        error_message: Error message
        traceback: Optional full traceback
    """
    entry_index: int
    entry_data: Dict[str, Any]
    error_type: str
    error_message: str
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error to dictionary."""
        return {
            "entry_index": self.entry_index,
            "entry_data": self.entry_data,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback
        }


@dataclass
class PipelineResponse:
    """
    Aggregated results from executing a processing pipeline.

    Attributes:
        step_results: List of dictionaries containing raw outputs
                     from each step, preserving execution order
        verification_references: Categorized references from
                                verification steps with keys:
                                - 'confirmed': Verified valid results
                                - 'to_verify': Results needing manual review
        errors: List of errors that occurred during execution
        total_entries: Total number of entries processed
        successful_entries: Number of successfully processed entries
        failed_entries: Number of failed entries
    """
    step_results: List[Dict[str, Any]]
    verification_references: Dict[str, List]
    errors: List[PipelineExecutionError] = field(default_factory=list)
    total_entries: int = 0
    successful_entries: int = 0
    failed_entries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the pipeline response to a dictionary format.

        Returns:
            Dictionary with keys:
            - step_results: Serialized step outputs
            - verification_references: Direct reference to verification categories
            - errors: List of error dictionaries
            - execution_summary: Summary of execution statistics
        """
        return {
            "step_results": self.step_results,
            "verification_references": self.verification_references,
            "errors": [error.to_dict() for error in self.errors],
            "execution_summary": {
                "total_entries": self.total_entries,
                "successful_entries": self.successful_entries,
                "failed_entries": self.failed_entries,
                "success_rate": (
                    self.successful_entries / self.total_entries
                    if self.total_entries > 0
                    else 0.0
                )
            }
        }