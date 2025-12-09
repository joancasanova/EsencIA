# tests/conftest.py
"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the app directory to Python path for imports
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))


@pytest.fixture
def sample_parse_rules():
    """Fixture providing sample parse rules for testing"""
    from domain.model.entities.parsing import ParseRule, ParseMode

    return [
        ParseRule(
            name="name",
            pattern=r"Name:\s*(\w+)",
            mode=ParseMode.REGEX
        ),
        ParseRule(
            name="age",
            pattern=r"Age:\s*(\d+)",
            mode=ParseMode.REGEX,
            fallback_value="N/A"
        ),
        ParseRule(
            name="city",
            pattern="City:",
            mode=ParseMode.KEYWORD,
            secondary_pattern="\n",
            fallback_value="Unknown"
        )
    ]


@pytest.fixture
def sample_verification_methods():
    """Fixture providing sample verification methods for testing"""
    from domain.model.entities.verification import VerificationMethod, VerificationMode

    return [
        VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="consensus_check",
            system_prompt="You are a helpful assistant",
            user_prompt="Is this valid?",
            num_sequences=3,
            valid_responses=["yes", "valid", "correct"],
            required_matches=2,
            max_tokens=10,
            temperature=0.7
        ),
        VerificationMethod(
            mode=VerificationMode.CUMULATIVE,
            name="quality_check",
            system_prompt="You are a quality checker",
            user_prompt="Is the quality acceptable?",
            num_sequences=5,
            valid_responses=["yes", "acceptable"],
            required_matches=3,
            max_tokens=10,
            temperature=0.5
        )
    ]


@pytest.fixture
def sample_generation_request():
    """Fixture providing a sample generation request"""
    from domain.model.entities.generation import GenerateTextRequest

    return GenerateTextRequest(
        system_prompt="You are a helpful assistant",
        user_prompt="Tell me a short joke",
        num_sequences=1,
        max_tokens=50,
        temperature=0.7
    )


@pytest.fixture
def sample_generation_metadata():
    """Fixture providing sample generation metadata"""
    from datetime import datetime
    from domain.model.entities.generation import GenerationMetadata

    return GenerationMetadata(
        model_name="test-model",
        system_prompt="System prompt",
        user_prompt="User prompt",
        temperature=0.7,
        tokens_used=50,
        generation_time=1.5,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_generated_result(sample_generation_metadata):
    """Fixture providing a sample generated result"""
    from domain.model.entities.generation import GeneratedResult

    return GeneratedResult(
        content="This is a test generated text",
        metadata=sample_generation_metadata,
        reference_data={"key": "value"}
    )


@pytest.fixture
def sample_parse_result():
    """Fixture providing a sample parse result"""
    from domain.model.entities.parsing import ParseResult

    return ParseResult(entries=[
        {"name": "John", "age": "30", "city": "NYC"},
        {"name": "Jane", "age": "25", "city": "LA"},
        {"name": "Bob", "age": "35", "city": "Chicago"}
    ])


@pytest.fixture
def sample_json_data():
    """Fixture providing sample JSON data for file operations"""
    return {
        "name": "Test User",
        "email": "test@example.com",
        "age": 30,
        "hobbies": ["reading", "coding", "gaming"],
        "address": {
            "street": "123 Main St",
            "city": "Test City",
            "country": "Test Country"
        }
    }


@pytest.fixture
def sample_pipeline_steps():
    """Fixture providing sample pipeline steps"""
    from domain.model.entities.pipeline import PipelineStep
    from domain.model.entities.generation import GenerateTextRequest

    return [
        PipelineStep(
            type="generate",
            parameters=GenerateTextRequest(
                system_prompt="You are a helpful assistant",
                user_prompt="Generate test data",
                num_sequences=1,
                max_tokens=50,
                temperature=0.7
            ),
            uses_reference=False,
            reference_step_numbers=[]
        )
    ]


# Configuration for pytest
def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Pytest collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add 'unit' marker to all tests in 'unit' directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add 'integration' marker to all tests in 'integration' directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
