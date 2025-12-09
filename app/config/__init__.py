# config/__init__.py
"""Configuration module for EsencIA project."""

from .settings import (
    # Paths
    PROJECT_ROOT,
    CONFIG_DIR,
    OUTPUT_DIR,
    PIPELINE_CONFIG_DIR,
    BENCHMARK_CONFIG_DIR,
    PIPELINE_OUTPUT_DIR,
    BENCHMARK_OUTPUT_DIR,

    # Default file paths
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_PIPELINE_REFERENCE_DATA,
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_BENCHMARK_ENTRIES,
    PIPELINE_RESULTS_DIR,
    PIPELINE_RESULTS_FILE,
    BENCHMARK_RESULTS_DIR,
    BENCHMARK_RESULTS_PREFIX,
    BENCHMARK_MISCLASSIFIED_DIR,
    BENCHMARK_MISCLASSIFIED_PREFIX,

    # Model configuration
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_SEQUENCES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_VERIFICATION_MAX_TOKENS,

    # Parsing configuration
    PARSE_FILTER_ALL,
    PARSE_FILTER_SUCCESSFUL,
    PARSE_FILTER_FIRST_N,
    DEFAULT_PARSE_FILTER,
    DEFAULT_PARSE_LIMIT,

    # Logging
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,

    # Validation limits
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_MAX_TOKENS,
    MAX_MAX_TOKENS,
    MAX_PIPELINE_STEPS,
    MAX_REFERENCE_DEPTH,

    # Feature flags
    ENABLE_CACHING,
    ENABLE_VERBOSE_LOGGING,

    # Helper functions
    get_pipeline_verification_output_dir,
    get_pipeline_verification_filename,
    validate_temperature,
    validate_max_tokens,
)

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "PIPELINE_CONFIG_DIR",
    "BENCHMARK_CONFIG_DIR",
    "PIPELINE_OUTPUT_DIR",
    "BENCHMARK_OUTPUT_DIR",

    # Default file paths
    "DEFAULT_PIPELINE_CONFIG",
    "DEFAULT_PIPELINE_REFERENCE_DATA",
    "DEFAULT_BENCHMARK_CONFIG",
    "DEFAULT_BENCHMARK_ENTRIES",
    "PIPELINE_RESULTS_DIR",
    "PIPELINE_RESULTS_FILE",
    "BENCHMARK_RESULTS_DIR",
    "BENCHMARK_RESULTS_PREFIX",
    "BENCHMARK_MISCLASSIFIED_DIR",
    "BENCHMARK_MISCLASSIFIED_PREFIX",

    # Model configuration
    "DEFAULT_MODEL_NAME",
    "DEFAULT_NUM_SEQUENCES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_VERIFICATION_MAX_TOKENS",

    # Parsing configuration
    "PARSE_FILTER_ALL",
    "PARSE_FILTER_SUCCESSFUL",
    "PARSE_FILTER_FIRST_N",
    "DEFAULT_PARSE_FILTER",
    "DEFAULT_PARSE_LIMIT",

    # Logging
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FORMAT",

    # Validation limits
    "MIN_TEMPERATURE",
    "MAX_TEMPERATURE",
    "MIN_MAX_TOKENS",
    "MAX_MAX_TOKENS",
    "MAX_PIPELINE_STEPS",
    "MAX_REFERENCE_DEPTH",

    # Feature flags
    "ENABLE_CACHING",
    "ENABLE_VERBOSE_LOGGING",

    # Helper functions
    "get_pipeline_verification_output_dir",
    "get_pipeline_verification_filename",
    "validate_temperature",
    "validate_max_tokens",
]
