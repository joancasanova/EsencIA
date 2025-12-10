# tests/unit/application/use_cases/test_pipeline_use_case.py

import pytest
from unittest.mock import Mock, patch, PropertyMock
from application.use_cases.pipeline_use_case import PipelineUseCase
from domain.model.entities.pipeline import PipelineRequest, PipelineStep, PipelineResponse
from domain.model.entities.generation import GenerateTextRequest


class TestPipelineUseCase:
    """Tests for PipelineUseCase"""

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_execute_success(self, mock_pipeline_service_class):
        """Test successful pipeline execution"""
        # Arrange
        mock_pipeline_service = Mock()
        mock_pipeline_service_class.return_value = mock_pipeline_service

        # Mock the methods called by _execute
        mock_pipeline_service.run_pipeline.return_value = None
        mock_pipeline_service.get_results.return_value = [{"step_type": "generate", "data": []}]
        mock_pipeline_service.confirmed_references = ["ref1"]
        mock_pipeline_service.to_verify_references = []

        use_case = PipelineUseCase("test-model")

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

        request = PipelineRequest(
            steps=[step],
            global_references={}
        )

        # Act
        response = use_case._execute(request)

        # Assert
        assert response is not None
        assert isinstance(response, PipelineResponse)
        assert "confirmed" in response.verification_references

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_execute_with_references_success(self, mock_pipeline_service_class):
        """Test pipeline execution with multiple reference entries"""
        # Arrange
        mock_pipeline_service = Mock()
        mock_pipeline_service_class.return_value = mock_pipeline_service

        mock_pipeline_service.run_pipeline.return_value = None
        mock_pipeline_service.get_results.return_value = [{"step_type": "generate", "data": []}]
        mock_pipeline_service.confirmed_references = ["ref1"]
        mock_pipeline_service.to_verify_references = []

        use_case = PipelineUseCase("test-model")

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

        request = PipelineRequest(
            steps=[step],
            global_references={}
        )

        reference_entries = [
            {"key1": "value1"},
            {"key1": "value2"}
        ]

        # Act
        response = use_case.execute_with_references(request, reference_entries)

        # Assert
        assert response.total_entries == 2
        assert response.successful_entries == 2
        assert response.failed_entries == 0

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_execute_value_error(self, mock_pipeline_service_class):
        """Test ValueError handling in _execute"""
        mock_pipeline_service = Mock()
        mock_pipeline_service_class.return_value = mock_pipeline_service
        mock_pipeline_service.run_pipeline.side_effect = ValueError("Invalid config")

        use_case = PipelineUseCase("test-model")

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

        request = PipelineRequest(steps=[step], global_references={})

        with pytest.raises(ValueError, match="Pipeline execution failed due to invalid configuration"):
            use_case._execute(request)

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_execute_runtime_error(self, mock_pipeline_service_class):
        """Test generic exception handling in _execute"""
        mock_pipeline_service = Mock()
        mock_pipeline_service_class.return_value = mock_pipeline_service
        mock_pipeline_service.run_pipeline.side_effect = RuntimeError("Service crashed")

        use_case = PipelineUseCase("test-model")

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

        request = PipelineRequest(steps=[step], global_references={})

        with pytest.raises(RuntimeError, match="Pipeline execution failed"):
            use_case._execute(request)

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_execute_with_references_partial_failure(self, mock_pipeline_service_class):
        """Test execute_with_references handles partial failures"""
        mock_pipeline_service = Mock()
        mock_pipeline_service_class.return_value = mock_pipeline_service

        # First entry fails, second succeeds
        mock_pipeline_service.run_pipeline.side_effect = [
            Exception("First entry failed"),
            None
        ]
        mock_pipeline_service.get_results.return_value = [{"step_type": "generate", "data": []}]
        mock_pipeline_service.confirmed_references = []
        mock_pipeline_service.to_verify_references = []

        use_case = PipelineUseCase("test-model")

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

        request = PipelineRequest(steps=[step], global_references={})
        reference_entries = [{"key": "fail"}, {"key": "success"}]

        response = use_case.execute_with_references(request, reference_entries)

        assert response.total_entries == 2
        assert response.successful_entries == 1
        assert response.failed_entries == 1
        assert len(response.errors) == 1
        assert response.errors[0].entry_index == 0

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_execute_with_global_references(self, mock_pipeline_service_class):
        """Test _execute loads global references"""
        mock_pipeline_service = Mock()
        mock_pipeline_service_class.return_value = mock_pipeline_service
        mock_pipeline_service.run_pipeline.return_value = None
        mock_pipeline_service.get_results.return_value = []
        mock_pipeline_service.confirmed_references = []
        mock_pipeline_service.to_verify_references = []

        use_case = PipelineUseCase("test-model")

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

        global_refs = {"name": "John", "age": "30"}
        request = PipelineRequest(steps=[step], global_references=global_refs)

        use_case._execute(request)

        # Verify global references were set
        assert mock_pipeline_service.global_references == global_refs


class TestPipelineUseCaseInit:
    """Tests for PipelineUseCase initialization"""

    @patch('application.use_cases.pipeline_use_case.PipelineService')
    def test_init_creates_services(self, mock_pipeline_service_class):
        """Test that init creates required services"""
        mock_pipeline_service_class.return_value = Mock()

        use_case = PipelineUseCase("custom-model")

        mock_pipeline_service_class.assert_called_once_with("custom-model", progress_callback=None)
        assert use_case.service is not None
        assert use_case.file_repo is not None
