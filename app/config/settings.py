# config/settings.py
"""
Central configuration file for EsencIA project.
All constants and default values should be defined here.
"""

import os
from pathlib import Path
from typing import Final

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory (parent of 'app')
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent

# Configuration directories
CONFIG_DIR: Final[Path] = PROJECT_ROOT / os.getenv("ESENCIA_CONFIG_DIR", "config")
PIPELINE_CONFIG_DIR: Final[Path] = CONFIG_DIR / "pipeline"
BENCHMARK_CONFIG_DIR: Final[Path] = CONFIG_DIR / "benchmark"

# Output directories
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / os.getenv("ESENCIA_OUTPUT_DIR", "out")
PIPELINE_OUTPUT_DIR: Final[Path] = OUTPUT_DIR / "pipeline"
BENCHMARK_OUTPUT_DIR: Final[Path] = OUTPUT_DIR / "benchmark"

# ============================================================================
# DEFAULT FILE PATHS
# ============================================================================

# Pipeline configuration files
DEFAULT_PIPELINE_CONFIG: Final[str] = str(PIPELINE_CONFIG_DIR / "pipeline_config.json")
DEFAULT_PIPELINE_REFERENCE_DATA: Final[str] = str(PIPELINE_CONFIG_DIR / "pipeline_reference_data.json")

# Benchmark configuration files
DEFAULT_BENCHMARK_CONFIG: Final[str] = str(BENCHMARK_CONFIG_DIR / "benchmark_config.json")
DEFAULT_BENCHMARK_ENTRIES: Final[str] = str(BENCHMARK_CONFIG_DIR / "benchmark_entries.json")

# Output file paths
PIPELINE_RESULTS_DIR: Final[str] = str(PIPELINE_OUTPUT_DIR / "results")
PIPELINE_RESULTS_FILE: Final[str] = "pipeline_results.json"

PIPELINE_VERIFICATION_DIR: Final[str] = str(PIPELINE_OUTPUT_DIR / "verification")

BENCHMARK_RESULTS_DIR: Final[str] = str(BENCHMARK_OUTPUT_DIR / "results")
BENCHMARK_RESULTS_PREFIX: Final[str] = "benchmark_results"

BENCHMARK_MISCLASSIFIED_DIR: Final[str] = str(BENCHMARK_OUTPUT_DIR / "misclassified")
BENCHMARK_MISCLASSIFIED_PREFIX: Final[str] = "misclassified"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Default language model
DEFAULT_MODEL_NAME: Final[str] = os.getenv(
    "ESENCIA_DEFAULT_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct"
)

# Generation defaults
DEFAULT_NUM_SEQUENCES: Final[int] = 1
DEFAULT_MAX_TOKENS: Final[int] = 100
DEFAULT_TEMPERATURE: Final[float] = 1.0

# Verification defaults
DEFAULT_VERIFICATION_MAX_TOKENS: Final[int] = 10

# ============================================================================
# PARSING CONFIGURATION
# ============================================================================

# Parse output filter options
PARSE_FILTER_ALL: Final[str] = "all"
PARSE_FILTER_SUCCESSFUL: Final[str] = "successful"
PARSE_FILTER_FIRST_N: Final[str] = "first_n"

DEFAULT_PARSE_FILTER: Final[str] = PARSE_FILTER_ALL
DEFAULT_PARSE_LIMIT: Final[int | None] = None

# Parse safety limits
MAX_PARSE_ITERATIONS: Final[int] = 10000  # Maximum iterations for keyword extraction loops
MAX_MATCHES_PER_RULE: Final[int] = 5000   # Maximum matches per parsing rule
MAX_PARSE_ENTRIES: Final[int] = 10000     # Maximum entries that can be built

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

DEFAULT_LOG_LEVEL: Final[str] = os.getenv("ESENCIA_LOG_LEVEL", "INFO")
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# ============================================================================
# VALIDATION LIMITS
# ============================================================================

# Model parameter limits
MIN_TEMPERATURE: Final[float] = 0.0
MAX_TEMPERATURE: Final[float] = 2.0
MIN_MAX_TOKENS: Final[int] = 1
MAX_MAX_TOKENS: Final[int] = 4096

# Pipeline limits
MAX_PIPELINE_STEPS: Final[int] = 100
MAX_REFERENCE_DEPTH: Final[int] = 10
MAX_VARIATIONS: Final[int] = 10000  # Maximum variations generated in pipeline

# Model cache limits (prevents CUDA OOM when using multiple models)
MAX_CACHED_MODELS: Final[int] = int(os.getenv("ESENCIA_MAX_CACHED_MODELS", "2"))

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features via environment variables
ENABLE_CACHING: Final[bool] = os.getenv("ESENCIA_ENABLE_CACHING", "true").lower() == "true"
ENABLE_VERBOSE_LOGGING: Final[bool] = os.getenv("ESENCIA_VERBOSE", "false").lower() == "true"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pipeline_verification_output_dir(result_type: str) -> str:
    """
    Get the output directory for pipeline verification results.

    Args:
        result_type: Type of verification result ('confirmed' or 'to_verify')

    Returns:
        Path to the output directory
    """
    return str(PIPELINE_OUTPUT_DIR / "verification" / result_type)


def get_pipeline_verification_filename(result_type: str) -> str:
    """
    Get the filename for pipeline verification results.

    Args:
        result_type: Type of verification result ('confirmed' or 'to_verify')

    Returns:
        Filename for the results
    """
    return f"{result_type}.json"


def validate_temperature(temperature: float) -> None:
    """
    Validate temperature parameter.

    Args:
        temperature: Temperature value to validate

    Raises:
        ValueError: If temperature is out of valid range
    """
    if not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
        raise ValueError(
            f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
            f"got {temperature}"
        )


def validate_max_tokens(max_tokens: int) -> None:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Max tokens value to validate

    Raises:
        ValueError: If max_tokens is out of valid range
    """
    if not (MIN_MAX_TOKENS <= max_tokens <= MAX_MAX_TOKENS):
        raise ValueError(
            f"Max tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}, "
            f"got {max_tokens}"
        )
