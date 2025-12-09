# tests/unit/application/use_cases/test_benchmark_use_case_full.py

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from application.use_cases.benchmark_use_case import BenchmarkUseCase
from domain.model.entities.benchmark import BenchmarkConfig, BenchmarkEntry, BenchmarkMetrics
from domain.model.entities.pipeline import PipelineStep, BenchmarkExecutionContext
from domain.model.entities.generation import GenerateTextRequest


class TestBenchmarkUseCase:
    """Tests for BenchmarkUseCase"""

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_run_benchmark_success(self, mock_benchmark_service_class):
        """Test successful benchmark execution"""
        # Arrange
        mock_benchmark_service = Mock()
        mock_benchmark_service_class.return_value = mock_benchmark_service

        mock_metrics = BenchmarkMetrics(
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
        mock_benchmark_service.calculate_metrics.return_value = mock_metrics
        # execute_pipeline_for_entry returns a tuple (results, context)
        mock_benchmark_service.execute_pipeline_for_entry.return_value = (
            [{"step_type": "verify", "step_data": [{"final_status": "confirmed"}]}],
            BenchmarkExecutionContext()
        )

        use_case = BenchmarkUseCase("test-model")

        # Create a valid pipeline step to pass validation
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

        config = BenchmarkConfig(
            model_name="test-model",
            pipeline_steps=[step],
            label_key="label",
            label_value="positive"
        )

        entries = [
            BenchmarkEntry(
                input_data={"text": "test1"},
                expected_label="positive"
            )
        ]

        # Act
        metrics = use_case.run_benchmark(config, entries)

        # Assert
        assert metrics.accuracy == 0.95

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_run_benchmark_with_empty_entries_raises_error(self, mock_benchmark_service_class):
        """Test that empty entries list raises ValueError"""
        # Arrange
        use_case = BenchmarkUseCase("test-model")

        # Create a valid pipeline step to pass validation
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

        config = BenchmarkConfig(
            model_name="test-model",
            pipeline_steps=[step],
            label_key="label",
            label_value="positive"
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Benchmark entries list cannot be empty"):
            use_case.run_benchmark(config, [])

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_run_benchmark_with_all_failures_raises_error(self, mock_benchmark_service_class):
        """Test that all failed entries raises ValueError"""
        # Arrange
        mock_benchmark_service = Mock()
        mock_benchmark_service_class.return_value = mock_benchmark_service
        # execute_pipeline_for_entry returns a tuple (None, context) on failure
        mock_benchmark_service.execute_pipeline_for_entry.return_value = (
            None, BenchmarkExecutionContext()
        )

        use_case = BenchmarkUseCase("test-model")

        # Create a valid pipeline step to pass validation
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

        config = BenchmarkConfig(
            model_name="test-model",
            pipeline_steps=[step],
            label_key="label",
            label_value="positive"
        )

        entries = [
            BenchmarkEntry(
                input_data={"text": "test1"},
                expected_label="positive"
            )
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="Benchmark produced no valid results"):
            use_case.run_benchmark(config, entries)

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_run_benchmark_metrics_calculation_error(self, mock_benchmark_service_class):
        """Test RuntimeError when metrics calculation fails"""
        mock_benchmark_service = Mock()
        mock_benchmark_service_class.return_value = mock_benchmark_service
        # execute_pipeline_for_entry returns a tuple (results, context)
        mock_benchmark_service.execute_pipeline_for_entry.return_value = (
            [{"step_type": "verify", "step_data": [{"final_status": "confirmed"}]}],
            BenchmarkExecutionContext()
        )
        mock_benchmark_service.calculate_metrics.side_effect = Exception("Calculation error")

        use_case = BenchmarkUseCase("test-model")

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

        config = BenchmarkConfig(
            model_name="test-model",
            pipeline_steps=[step],
            label_key="label",
            label_value="positive"
        )

        entries = [
            BenchmarkEntry(input_data={"text": "test"}, expected_label="positive")
        ]

        with pytest.raises(RuntimeError, match="Metrics calculation failed"):
            use_case.run_benchmark(config, entries)

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_run_benchmark_entry_exception_continues(self, mock_benchmark_service_class):
        """Test that exception during entry processing continues to next entry"""
        mock_benchmark_service = Mock()
        mock_benchmark_service_class.return_value = mock_benchmark_service

        # First entry fails, second succeeds
        # execute_pipeline_for_entry returns tuples (results, context)
        mock_benchmark_service.execute_pipeline_for_entry.side_effect = [
            Exception("Entry failed"),
            (
                [{"step_type": "verify", "step_data": [{"final_status": "confirmed"}]}],
                BenchmarkExecutionContext()
            )
        ]

        mock_metrics = BenchmarkMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            confusion_matrix={"true_positive": 1},
            misclassified=[]
        )
        mock_benchmark_service.calculate_metrics.return_value = mock_metrics

        use_case = BenchmarkUseCase("test-model")

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

        config = BenchmarkConfig(
            model_name="test-model",
            pipeline_steps=[step],
            label_key="label",
            label_value="positive"
        )

        entries = [
            BenchmarkEntry(input_data={"text": "fail"}, expected_label="positive"),
            BenchmarkEntry(input_data={"text": "success"}, expected_label="positive")
        ]

        # Should not raise - should continue after first entry fails
        metrics = use_case.run_benchmark(config, entries)
        assert metrics is not None


class TestBenchmarkUseCaseProcessPrediction:
    """Tests for BenchmarkUseCase._process_prediction()"""

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_process_prediction_no_verify_step(self, mock_benchmark_service_class):
        """Test that missing verify step returns None"""
        mock_benchmark_service_class.return_value = Mock()
        use_case = BenchmarkUseCase("test-model")

        entry = BenchmarkEntry(input_data={"text": "test"}, expected_label="positive")
        pipeline_response = [{"step_type": "generate", "step_data": []}]

        result = use_case._process_prediction(pipeline_response, entry)

        assert result is None

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_process_prediction_empty_step_data(self, mock_benchmark_service_class):
        """Test that empty step_data returns None"""
        mock_benchmark_service_class.return_value = Mock()
        use_case = BenchmarkUseCase("test-model")

        entry = BenchmarkEntry(input_data={"text": "test"}, expected_label="positive")
        pipeline_response = [{"step_type": "verify", "step_data": []}]

        result = use_case._process_prediction(pipeline_response, entry)

        assert result is None

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_process_prediction_confirmed_status(self, mock_benchmark_service_class):
        """Test prediction with confirmed status"""
        mock_benchmark_service_class.return_value = Mock()
        use_case = BenchmarkUseCase("test-model")

        entry = BenchmarkEntry(input_data={"text": "test"}, expected_label="positive")
        pipeline_response = [{"step_type": "verify", "step_data": [{"final_status": "confirmed"}]}]

        result = use_case._process_prediction(pipeline_response, entry)

        assert result is not None
        assert result.predicted_label == "confirmed"

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_process_prediction_not_confirmed_status(self, mock_benchmark_service_class):
        """Test prediction with non-confirmed status"""
        mock_benchmark_service_class.return_value = Mock()
        use_case = BenchmarkUseCase("test-model")

        entry = BenchmarkEntry(input_data={"text": "test"}, expected_label="positive")
        pipeline_response = [{"step_type": "verify", "step_data": [{"final_status": "discarded"}]}]

        result = use_case._process_prediction(pipeline_response, entry)

        assert result is not None
        assert result.predicted_label == "not_confirmed"


class TestBenchmarkUseCaseInit:
    """Tests for BenchmarkUseCase initialization"""

    @patch('application.use_cases.benchmark_use_case.BenchmarkService')
    def test_init_creates_benchmark_service(self, mock_benchmark_service_class):
        """Test that init creates BenchmarkService"""
        mock_benchmark_service_class.return_value = Mock()

        use_case = BenchmarkUseCase("custom-model")

        mock_benchmark_service_class.assert_called_once_with("custom-model")
        assert use_case.benchmark_service is not None
