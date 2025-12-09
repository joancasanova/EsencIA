# domain/model/entities/generation.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

from config.settings import (
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_MAX_TOKENS,
    MAX_MAX_TOKENS
)

@dataclass(frozen=True)
class GenerationMetadata:
    """
    Immutable record of generation process metadata.
    
    Attributes:
        model_name: Identifier of the LLM used for generation
        system_prompt: System-level instructions/context provided to the model
        user_prompt: User's input/question/request to the model
        temperature: Sampling temperature used (0.0-2.0, controls randomness)
        tokens_used: Total tokens consumed in the generation
        generation_time: Time taken for generation in seconds
        timestamp: Exact datetime of generation completion
    """
    model_name: str
    system_prompt: str
    user_prompt: str
    temperature: float
    tokens_used: int
    generation_time: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class GeneratedResult:
    """
    Complete record of a single generated text output with context.
    
    Attributes:
        content: The generated text output
        metadata: Technical details about the generation process
        reference_data: Optional contextual data used in prompt templates
        
    Methods:
        to_dict: Serializes object for storage/transmission
        contains_reference: Checks for specific text in content
        word_count: Estimates word count using simple whitespace splitting
    """
    content: str
    metadata: GenerationMetadata
    reference_data: Optional[Dict[str, str]] = None
    
    def to_dict(self):
        """Serializes object to JSON-friendly dictionary format."""
        return {
            "content": self.content,
            "metadata": {
                "model_name": self.metadata.model_name,
                "system_prompt": self.metadata.system_prompt,
                "user_prompt": self.metadata.user_prompt,
                "temperature": self.metadata.temperature,  # Fixed from original code
                "tokens_used": self.metadata.tokens_used,
                "generation_time": self.metadata.generation_time,
                "timestamp": self.metadata.timestamp.isoformat()
            },
            "reference_data": self.reference_data
        }
    
    def contains_reference(self, text: str) -> bool:
        """
        Case-insensitive check for text presence in generated content.
        
        Args:
            text: Substring to search for in content
            
        Returns:
            bool: True if text found in content (case-insensitive)
        """
        return text.lower() in self.content.lower()
    
    def word_count(self) -> int:
        """
        Estimates word count using whitespace splitting.
        
        Note: This is a simple heuristic and might not handle:
        - Multiple whitespace characters
        - Hyphenated words
        - Non-English languages
        
        Returns:
            int: Number of whitespace-separated tokens
        """
        return len(self.content.split())

@dataclass
class GenerateTextRequest:
    """
    Parameters for controlling text generation process.

    Attributes:
        system_prompt: High-level instructions/context for the model
        user_prompt: Specific input/question to generate response for
        num_sequences: Number of variations to generate (1-10 typical)
        max_tokens: Maximum length of generated text (1-4096 typical)
        temperature: Sampling temperature (0.0=deterministic, 1.0=default, 2.0=creative)

    Raises:
        ValueError: If any parameter is outside valid range or empty
    """
    system_prompt: str
    user_prompt: str
    num_sequences: int = 1
    max_tokens: int = 200
    temperature: float = 1.0

    def __post_init__(self):
        """Validates request parameters after initialization."""
        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        if not self.user_prompt or not self.user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        if self.num_sequences < 1:
            raise ValueError(f"num_sequences must be >= 1, got {self.num_sequences}")

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

    def to_dict(self):
        """Serializes to API-friendly format."""
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "num_sequences": self.num_sequences,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

@dataclass
class GenerateTextResponse:
    """
    Complete output of a text generation request.
    
    Attributes:
        generated_texts: List of generated results with metadata
        total_tokens: Sum of tokens used across all generations
        generation_time: Total wall-clock time for all generations
        model_name: Name of model used (from first result if available)
    """
    generated_texts: List[GeneratedResult]
    total_tokens: int
    generation_time: float
    model_name: str

    def to_dict(self):
        """Serializes response to API-friendly format."""
        return {
            "generated_texts": [result.to_dict() for result in self.generated_texts],
            "total_tokens": self.total_tokens,
            "generation_time": self.generation_time,
            # Handle empty results case safely
            "model_name": self.generated_texts[0].metadata.model_name if self.generated_texts else "unknown"
        }