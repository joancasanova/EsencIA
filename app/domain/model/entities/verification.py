# domain/model/entities/verification.py

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
from enum import Enum
from datetime import datetime

from config.settings import (
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_MAX_TOKENS,
    MAX_MAX_TOKENS
)

class VerificationMode(Enum):
    """
    Determines how verification method outcomes affect final status.
    
    Options:
        ELIMINATORY: Fails entire verification if any method fails (all-or-nothing)
        CUMULATIVE: Allows partial successes contributing to final score
    """
    ELIMINATORY = "eliminatory"
    CUMULATIVE = "cumulative"

@dataclass(frozen=True)
class VerificationThresholds:
    """
    Defines acceptable ranges for verification metrics.
    
    Attributes:
        lower_bound: Minimum acceptable value (inclusive)
        upper_bound: Maximum acceptable value (inclusive)
        target_value: Optional ideal value for optimization purposes
    """
    lower_bound: float
    upper_bound: float
    target_value: Optional[float] = None

    def is_within_bounds(self, value: float) -> bool:
        """Validates if value falls within defined range."""
        return self.lower_bound <= value <= self.upper_bound

@dataclass
class VerificationMethod:
    """
    Configuration for a single verification technique.

    Attributes:
        mode: Verification strategy type (ELIMINATORY/CUMULATIVE)
        name: Unique identifier for the method
        system_prompt: LLM context/instructions for verification
        user_prompt: Specific question/request to verify
        num_sequences: Number of LLM responses to generate
        valid_responses: Acceptable answer patterns
        required_matches: Minimum valid responses needed
        max_tokens: Limit for LLM response length
        temperature: Sampling creativity control (0.0-2.0)

    Raises:
        ValueError: If any parameter is invalid or out of range
    """
    mode: VerificationMode
    name: str
    system_prompt: str
    user_prompt: str
    num_sequences: int
    valid_responses: List[str]
    required_matches: int
    max_tokens: int = 100
    temperature: float = 1.0
    ignore_case: bool = True
    llm_config: Optional[str] = None

    def __post_init__(self):
        """Validates method parameters after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("name cannot be empty")

        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        if not self.user_prompt or not self.user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        if self.num_sequences < 1:
            raise ValueError(f"num_sequences must be >= 1, got {self.num_sequences}")

        if self.required_matches < 1:
            raise ValueError(f"required_matches must be >= 1, got {self.required_matches}")

        if self.required_matches > self.num_sequences:
            raise ValueError(
                f"required_matches ({self.required_matches}) cannot exceed "
                f"num_sequences ({self.num_sequences})"
            )

        if not (MIN_MAX_TOKENS <= self.max_tokens <= MAX_MAX_TOKENS):
            raise ValueError(
                f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}, "
                f"got {self.max_tokens}"
            )

        if not (MIN_TEMPERATURE <= self.temperature <= MAX_TEMPERATURE):
            raise ValueError(
                f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
                f"got {self.temperature}"
            )

@dataclass(frozen=True)
class VerificationResult:
    """
    Record of verification method execution outcome.
    
    Attributes:
        method: Reference to VerificationMethod configuration
        passed: Success/failure status
        score: Optional quantitative measure (e.g., 0.75 = 75% match)
        details: Raw data supporting the result
        timestamp: Execution completion time
    """
    method: VerificationMethod
    passed: bool
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        """Serializes result for storage/analysis."""
        return {
            "method_name": self.method.name,
            "mode": self.method.mode.value,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class VerificationSummary:
    """
    Aggregated results of multiple verification methods.
    
    Attributes:
        results: All method execution outcomes
        final_status: Overall verification conclusion
        reference_data: Contextual data used during verification
    """
    results: List[VerificationResult]
    final_status: str
    reference_data: Optional[Dict[str, str]] = None
    
    def to_dict(self):
        """Serializes summary with computed success rate."""
        return {
            "results": [result.to_dict() for result in self.results],
            "final_status": self.final_status,
            "reference_data": self.reference_data,
            "success_rate": self.success_rate
        }

    @property
    def passed_methods(self) -> List[str]:
        """Names of successful verification methods."""
        return [result.method.name for result in self.results if result.passed]
    
    @property
    def failed_methods(self) -> List[str]:
        """Names of failed verification methods."""
        return [result.method.name for result in self.results if not result.passed]
    
    @property
    def success_rate(self) -> float:
        """Ratio of passed methods to total methods (0.0-1.0)."""
        return len(self.passed_methods) / len(self.results) if self.results else 0.0

    @property
    def scores(self) -> List[Optional[float]]:
        """Collection of individual method scores for analysis."""
        return [result.score for result in self.results]

@dataclass
class VerifyRequest:
    """
    Request to execute verification workflow.

    Attributes:
        methods: Verification techniques to apply
        required_for_confirmed: Minimum successes needed for 'confirmed' status
        required_for_review: Minimum successes needed for 'review' status

    Raises:
        ValueError: If validation constraints are violated
    """
    methods: List[VerificationMethod]
    required_for_confirmed: int
    required_for_review: int

    def __post_init__(self):
        """Validates request constraints after initialization."""
        if not self.methods:
            raise ValueError("methods list cannot be empty")

        if self.required_for_confirmed < 0:
            raise ValueError(
                f"required_for_confirmed must be >= 0, got {self.required_for_confirmed}"
            )

        if self.required_for_review < 0:
            raise ValueError(
                f"required_for_review must be >= 0, got {self.required_for_review}"
            )

        if self.required_for_confirmed > len(self.methods):
            raise ValueError(
                f"required_for_confirmed ({self.required_for_confirmed}) cannot exceed "
                f"number of methods ({len(self.methods)})"
            )

        if self.required_for_review > len(self.methods):
            raise ValueError(
                f"required_for_review ({self.required_for_review}) cannot exceed "
                f"number of methods ({len(self.methods)})"
            )

        if self.required_for_review > self.required_for_confirmed:
            raise ValueError(
                f"required_for_review ({self.required_for_review}) cannot exceed "
                f"required_for_confirmed ({self.required_for_confirmed})"
            )

@dataclass
class VerifyResponse:
    """
    Final output of verification process.
    
    Attributes:
        verification_summary: Aggregated results and status
        execution_time: Total processing time in seconds
        success_rate: Overall success percentage
    """
    verification_summary: VerificationSummary
    execution_time: float
    success_rate: float

@dataclass(frozen=True)
class VerificationStatus:
    """
    Standardized verification status container.
    
    Provides factory methods for common statuses:
    - confirmed: Successful verification
    - discarded: Failed verification
    - review: Requires human evaluation
    """
    id: str  # Unique status identifier
    status: str  # Human-readable status

    @classmethod
    def confirmed(cls):
        return cls(id="CONFIRMED", status="confirmed")

    @classmethod
    def discarded(cls):
        return cls(id="DISCARDED", status="discarded")

    @classmethod
    def review(cls):
        return cls(id="REVIEW", status="review")
    
    @classmethod
    def from_string(cls, status: str) -> Optional['VerificationStatus']:
        """Creates status from string (case-insensitive)."""
        status = status.lower()
        return {
            'confirmed': cls.confirmed(),
            'discarded': cls.discarded(),
            'review': cls.review()
        }.get(status)
        
    def is_final(self) -> bool:
        """Determines if status is terminal (no further action needed)."""
        return self.id in ["CONFIRMED", "DISCARDED"]

    def requires_review(self) -> bool:
        """Checks if status requires human intervention."""
        return self.id == "REVIEW"