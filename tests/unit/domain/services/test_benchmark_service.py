# tests/unit/domain/services/test_benchmark_service.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from domain.services.benchmark_service import BenchmarkService
from domain.model.entities.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkMetrics
from domain.model.entities.pipeline import PipelineStep


class TestBenchmarkServiceInit:
    """Tests for BenchmarkService initialization"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_init_creates_pipeline_service(self, mock_pipeline_service):
        """Test that initialization creates PipelineService"""
        # Act
        service = BenchmarkService("test-model")

        # Assert
        mock_pipeline_service.assert_called_once_with("test-model")


class TestBenchmarkServiceCalculateMetrics:
    """Tests for BenchmarkService.calculate_metrics method"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_perfect_accuracy(self, mock_pipeline_service):
        """Test metrics calculation with perfect predictions"""
        # Arrange
        service = BenchmarkService("test-model")

        results = [
            BenchmarkResult(
                input_data={"text": "test1"},
                predicted_label="confirmed",
                actual_label="confirmed",
                timestamp=datetime.now()
            ),
            BenchmarkResult(
                input_data={"text": "test2"},
                predicted_label="confirmed",
                actual_label="confirmed",
                timestamp=datetime.now()
            )
        ]

        # Act
        metrics = service.calculate_metrics(results, "confirmed")

        # Assert
        assert metrics.accuracy == 1.0
        # Use approximate comparison due to smoothing factor (1e-10)
        assert metrics.precision > 0.999
        assert metrics.recall > 0.999
        assert metrics.f1_score > 0.999
        assert len(metrics.misclassified) == 0

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_with_misclassifications(self, mock_pipeline_service):
        """Test metrics calculation with some misclassifications"""
        # Arrange
        service = BenchmarkService("test-model")

        results = [
            BenchmarkResult(
                input_data={"text": "test1"},
                predicted_label="confirmed",
                actual_label="confirmed",
                timestamp=datetime.now()
            ),
            BenchmarkResult(
                input_data={"text": "test2"},
                predicted_label="not_confirmed",
                actual_label="confirmed",
                timestamp=datetime.now()
            ),
            BenchmarkResult(
                input_data={"text": "test3"},
                predicted_label="confirmed",
                actual_label="not_confirmed",
                timestamp=datetime.now()
            )
        ]

        # Act
        metrics = service.calculate_metrics(results, "confirmed")

        # Assert
        assert metrics.accuracy == 1/3  # Only 1 correct
        assert len(metrics.misclassified) == 2

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_zero_division_protection(self, mock_pipeline_service):
        """Test that zero division is handled gracefully"""
        # Arrange
        service = BenchmarkService("test-model")

        # All negative predictions
        results = [
            BenchmarkResult(
                input_data={"text": "test1"},
                predicted_label="not_confirmed",
                actual_label="not_confirmed",
                timestamp=datetime.now()
            )
        ]

        # Act
        metrics = service.calculate_metrics(results, "confirmed")

        # Assert - Should not crash
        assert metrics.precision >= 0.0
        assert metrics.recall >= 0.0


class TestBenchmarkServiceExecutePipeline:
    """Tests for BenchmarkService.execute_pipeline_for_entry method"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_execute_pipeline_success(self, mock_pipeline_service):
        """Test successful pipeline execution for entry"""
        # Arrange
        mock_pipeline = mock_pipeline_service.return_value
        mock_pipeline.run_pipeline.return_value = None
        mock_pipeline.get_results.return_value = [{"step_type": "generate", "step_data": []}]

        service = BenchmarkService("test-model")

        config = Mock()
        config.pipeline_steps = []
        entry = {"text": "test"}

        # Act
        result = service.execute_pipeline_for_entry(config, entry)

        # Assert
        assert result is not None
        mock_pipeline.run_pipeline.assert_called_once()
        mock_pipeline.get_results.assert_called_once()

    @patch('domain.services.benchmark_service.PipelineService')
    def test_execute_pipeline_failure_returns_none(self, mock_pipeline_service):
        """Test that pipeline failure returns None in the result tuple"""
        # Arrange
        mock_pipeline = mock_pipeline_service.return_value
        mock_pipeline.run_pipeline.side_effect = Exception("Pipeline crashed")

        service = BenchmarkService("test-model")

        config = Mock()
        config.pipeline_steps = []
        entry = {"text": "test"}

        # Act
        result, context = service.execute_pipeline_for_entry(config, entry)

        # Assert
        assert result is None
        assert context is not None  # Context should still be returned

    @patch('domain.services.benchmark_service.PipelineService')
    def test_execute_pipeline_sets_global_references(self, mock_pipeline_service):
        """Test that entry data is set as global references"""
        # Arrange
        mock_pipeline = mock_pipeline_service.return_value
        mock_pipeline.run_pipeline.return_value = None
        mock_pipeline.get_results.return_value = []

        service = BenchmarkService("test-model")

        config = Mock()
        config.pipeline_steps = []
        entry = {"text": "test", "label": "positive"}

        # Act
        service.execute_pipeline_for_entry(config, entry)

        # Assert
        assert mock_pipeline.global_references == entry


class TestBenchmarkServiceConfigureSteps:
    """Tests for BenchmarkService._configure_steps method"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_configure_steps_deep_copies(self, mock_pipeline_service):
        """Test that steps are deep copied"""
        # Arrange
        service = BenchmarkService("test-model")

        mock_step = Mock()
        mock_step.parameters = Mock()
        mock_step.parameters.system_prompt = "Original {text}"
        mock_step.parameters.user_prompt = "User {text}"

        steps = [mock_step]
        entry = {"text": "replaced"}

        # Act
        result = service._configure_steps(steps, entry)

        # Assert
        assert len(result) == 1
        # Original should not be modified
        assert mock_step.parameters.system_prompt == "Original {text}"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_configure_steps_multiple_steps(self, mock_pipeline_service):
        """Test configuring multiple steps"""
        # Arrange
        service = BenchmarkService("test-model")

        step1 = Mock()
        step1.parameters = Mock()
        step1.parameters.system_prompt = "Step1 {key}"
        step1.parameters.user_prompt = ""

        step2 = Mock()
        step2.parameters = Mock()
        step2.parameters.system_prompt = "Step2 {key}"
        step2.parameters.user_prompt = ""

        steps = [step1, step2]
        entry = {"key": "value"}

        # Act
        result = service._configure_steps(steps, entry)

        # Assert
        assert len(result) == 2


class TestBenchmarkServiceSubstitutePlaceholders:
    """Tests for BenchmarkService._substitute_placeholders method"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_substitute_system_prompt(self, mock_pipeline_service):
        """Test substituting placeholders in system_prompt"""
        # Arrange
        service = BenchmarkService("test-model")

        params = Mock()
        params.system_prompt = "Hello {name}"
        params.user_prompt = "User message"

        data = {"name": "World"}

        # Act
        service._substitute_placeholders(params, data)

        # Assert
        assert params.system_prompt == "Hello World"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_substitute_user_prompt(self, mock_pipeline_service):
        """Test substituting placeholders in user_prompt"""
        # Arrange
        service = BenchmarkService("test-model")

        params = Mock()
        params.system_prompt = "System"
        params.user_prompt = "Tell me about {topic}"

        data = {"topic": "AI"}

        # Act
        service._substitute_placeholders(params, data)

        # Assert
        assert params.user_prompt == "Tell me about AI"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_substitute_text_field(self, mock_pipeline_service):
        """Test substituting placeholders in text field"""
        # Arrange
        service = BenchmarkService("test-model")

        params = Mock()
        params.text = "Parse this: {content}"

        data = {"content": "Some text to parse"}

        # Act
        service._substitute_placeholders(params, data)

        # Assert
        assert params.text == "Parse this: Some text to parse"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_substitute_missing_attribute(self, mock_pipeline_service):
        """Test substitution when attribute doesn't exist"""
        # Arrange
        service = BenchmarkService("test-model")

        params = Mock(spec=[])  # No attributes
        data = {"key": "value"}

        # Act - Should not raise
        service._substitute_placeholders(params, data)

        # Assert - No error


class TestBenchmarkServiceReplaceInTemplate:
    """Tests for BenchmarkService._replace_in_template method"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_replace_single_placeholder(self, mock_pipeline_service):
        """Test replacing a single placeholder"""
        # Arrange
        service = BenchmarkService("test-model")

        template = "Hello {name}!"
        data = {"name": "John"}

        # Act
        result = service._replace_in_template(template, data)

        # Assert
        assert result == "Hello John!"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_replace_multiple_placeholders(self, mock_pipeline_service):
        """Test replacing multiple placeholders"""
        # Arrange
        service = BenchmarkService("test-model")

        template = "{greeting} {name}, you are {age} years old"
        data = {"greeting": "Hello", "name": "John", "age": 30}

        # Act
        result = service._replace_in_template(template, data)

        # Assert
        assert result == "Hello John, you are 30 years old"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_replace_no_placeholders(self, mock_pipeline_service):
        """Test template without placeholders"""
        # Arrange
        service = BenchmarkService("test-model")

        template = "No placeholders here"
        data = {"unused": "value"}

        # Act
        result = service._replace_in_template(template, data)

        # Assert
        assert result == "No placeholders here"

    @patch('domain.services.benchmark_service.PipelineService')
    def test_replace_missing_placeholder(self, mock_pipeline_service):
        """Test template with placeholder not in data"""
        # Arrange
        service = BenchmarkService("test-model")

        template = "Hello {name}, your id is {id}"
        data = {"name": "John"}  # Missing "id"

        # Act
        result = service._replace_in_template(template, data)

        # Assert
        assert result == "Hello John, your id is {id}"


class TestBenchmarkServiceCalculateMetricsEdgeCases:
    """Edge case tests for calculate_metrics"""

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_empty_results(self, mock_pipeline_service):
        """Test metrics with empty results list"""
        # Arrange
        service = BenchmarkService("test-model")

        # Act
        metrics = service.calculate_metrics([], "positive")

        # Assert
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert len(metrics.misclassified) == 0

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_all_true_negatives(self, mock_pipeline_service):
        """Test metrics when all predictions are true negatives"""
        # Arrange
        service = BenchmarkService("test-model")

        results = [
            BenchmarkResult(
                input_data={"text": "test"},
                predicted_label="not_confirmed",
                actual_label="negative",
                timestamp=datetime.now()
            )
        ] * 3

        # Act
        metrics = service.calculate_metrics(results, "positive")

        # Assert
        assert metrics.accuracy == 1.0  # All correct
        assert metrics.confusion_matrix["true_negative"] == 3

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_all_false_positives(self, mock_pipeline_service):
        """Test metrics when all predictions are false positives"""
        # Arrange
        service = BenchmarkService("test-model")

        results = [
            BenchmarkResult(
                input_data={"text": "test"},
                predicted_label="confirmed",
                actual_label="negative",
                timestamp=datetime.now()
            )
        ] * 2

        # Act
        metrics = service.calculate_metrics(results, "positive")

        # Assert
        assert metrics.confusion_matrix["false_positive"] == 2
        assert len(metrics.misclassified) == 2

    @patch('domain.services.benchmark_service.PipelineService')
    def test_calculate_metrics_all_false_negatives(self, mock_pipeline_service):
        """Test metrics when all predictions are false negatives"""
        # Arrange
        service = BenchmarkService("test-model")

        results = [
            BenchmarkResult(
                input_data={"text": "test"},
                predicted_label="not_confirmed",
                actual_label="positive",
                timestamp=datetime.now()
            )
        ] * 2

        # Act
        metrics = service.calculate_metrics(results, "positive")

        # Assert
        assert metrics.confusion_matrix["false_negative"] == 2
        assert len(metrics.misclassified) == 2
