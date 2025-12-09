# tests/unit/domain/entities/test_benchmark.py

import pytest
from datetime import datetime
from domain.model.entities.benchmark import (
    BenchmarkConfig, BenchmarkEntry, BenchmarkResult, BenchmarkMetrics
)
from domain.model.entities.pipeline import PipelineStep
from domain.model.entities.generation import GenerateTextRequest


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig entity"""

    def test_benchmark_config_creation_success(self):
        """Test successful BenchmarkConfig creation"""
        # Arrange & Act
        config = BenchmarkConfig(
            model_name="test-model",
            pipeline_steps=[
                PipelineStep(
                    type="generate",
                    parameters=GenerateTextRequest(
                        system_prompt="System",
                        user_prompt="User",
                        num_sequences=1
                    ),
                    uses_reference=False,
                    reference_step_numbers=[]
                )
            ],
            label_key="label",
            label_value="positive"
        )

        # Assert
        assert config.model_name == "test-model"
        assert config.label_key == "label"
        assert config.label_value == "positive"

    def test_benchmark_config_empty_model_name_raises_error(self):
        """Test that empty model_name raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            BenchmarkConfig(
                model_name="",
                pipeline_steps=[Mock()],
                label_key="label",
                label_value="positive"
            )

    def test_benchmark_config_empty_pipeline_steps_raises_error(self):
        """Test that empty pipeline_steps raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="pipeline_steps cannot be empty"):
            BenchmarkConfig(
                model_name="test-model",
                pipeline_steps=[],
                label_key="label",
                label_value="positive"
            )

    def test_benchmark_config_empty_label_key_raises_error(self):
        """Test that empty label_key raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="label_key cannot be empty"):
            BenchmarkConfig(
                model_name="test-model",
                pipeline_steps=[Mock()],
                label_key="",
                label_value="positive"
            )

    def test_benchmark_config_empty_label_value_raises_error(self):
        """Test that empty label_value raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError, match="label_value cannot be empty"):
            BenchmarkConfig(
                model_name="test-model",
                pipeline_steps=[Mock()],
                label_key="label",
                label_value=""
            )


class TestBenchmarkEntry:
    """Tests for BenchmarkEntry entity"""

    def test_benchmark_entry_creation(self):
        """Test successful BenchmarkEntry creation"""
        # Arrange & Act
        entry = BenchmarkEntry(
            input_data={"text": "test data"},
            expected_label="positive"
        )

        # Assert
        assert entry.input_data == {"text": "test data"}
        assert entry.expected_label == "positive"


class TestBenchmarkResult:
    """Tests for BenchmarkResult entity"""

    def test_benchmark_result_creation(self):
        """Test successful BenchmarkResult creation"""
        # Arrange & Act
        result = BenchmarkResult(
            input_data={"text": "test"},
            predicted_label="positive",
            actual_label="positive",
            timestamp=datetime.now()
        )

        # Assert
        assert result.predicted_label == "positive"
        assert result.actual_label == "positive"

    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult serialization to dict"""
        # Arrange
        timestamp = datetime.now()
        result = BenchmarkResult(
            input_data={"text": "test"},
            predicted_label="positive",
            actual_label="positive",
            timestamp=timestamp
        )

        # Act
        result_dict = result.to_dict()

        # Assert
        assert result_dict["predicted_label"] == "positive"
        assert result_dict["actual_label"] == "positive"
        assert "input_data" in result_dict


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics entity"""

    def test_benchmark_metrics_creation(self):
        """Test successful BenchmarkMetrics creation"""
        # Arrange & Act
        metrics = BenchmarkMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            confusion_matrix={
                "true_positive": 95,
                "false_positive": 2,
                "true_negative": 1,
                "false_negative": 2
            },
            misclassified=[]
        )

        # Assert
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.94
        assert metrics.f1_score == 0.95

    def test_benchmark_metrics_to_dict(self):
        """Test BenchmarkMetrics serialization to dict"""
        # Arrange
        metrics = BenchmarkMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            confusion_matrix={
                "true_positive": 95,
                "false_positive": 2,
                "true_negative": 1,
                "false_negative": 2
            },
            misclassified=[]
        )

        # Act
        metrics_dict = metrics.to_dict()

        # Assert
        assert metrics_dict["accuracy"] == 0.95
        assert "confusion_matrix" in metrics_dict
        assert "misclassified_count" in metrics_dict


from unittest.mock import Mock
