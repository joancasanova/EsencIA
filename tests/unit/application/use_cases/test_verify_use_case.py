# tests/unit/application/use_cases/test_verify_use_case.py

import pytest
from unittest.mock import Mock, patch
from application.use_cases.verify_use_case import VerifyUseCase
from domain.model.entities.verification import VerifyRequest, VerificationMethod, VerificationMode, VerificationSummary


def create_valid_method(name="test", num_sequences=2, required_matches=1):
    """Helper to create a valid VerificationMethod"""
    return VerificationMethod(
        mode=VerificationMode.ELIMINATORY,
        name=name,
        system_prompt="System prompt",
        user_prompt="User prompt",
        num_sequences=num_sequences,
        valid_responses=["yes"],
        required_matches=required_matches
    )


class TestVerifyUseCaseExecute:
    """Tests for VerifyUseCase.execute()"""

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_execute_success(self, mock_verifier_service_class):
        """Test successful verification"""
        mock_verifier_service = Mock()
        mock_verifier_service_class.return_value = mock_verifier_service

        mock_summary = VerificationSummary(results=[], final_status="confirmed")
        mock_verifier_service.verify.return_value = mock_summary

        use_case = VerifyUseCase("test-model")

        method1 = create_valid_method("test1")
        method2 = create_valid_method("test2")

        request = VerifyRequest(
            methods=[method1, method2],
            required_for_confirmed=2,
            required_for_review=1
        )

        response = use_case.execute(request)

        assert response.verification_summary.final_status == "confirmed"
        assert response.execution_time >= 0
        assert response.success_rate >= 0

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_execute_returns_execution_time(self, mock_verifier_service_class):
        """Test that execution time is calculated"""
        mock_verifier_service = Mock()
        mock_verifier_service_class.return_value = mock_verifier_service

        mock_summary = VerificationSummary(results=[], final_status="confirmed")
        mock_verifier_service.verify.return_value = mock_summary

        use_case = VerifyUseCase("test-model")
        method = create_valid_method()

        request = VerifyRequest(
            methods=[method, create_valid_method("m2")],
            required_for_confirmed=2,
            required_for_review=1
        )

        response = use_case.execute(request)

        assert isinstance(response.execution_time, float)
        assert response.execution_time >= 0

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_execute_value_error_from_service(self, mock_verifier_service_class):
        """Test ValueError handling from service"""
        mock_verifier_service = Mock()
        mock_verifier_service_class.return_value = mock_verifier_service
        mock_verifier_service.verify.side_effect = ValueError("Invalid param")

        use_case = VerifyUseCase("test-model")
        request = VerifyRequest(
            methods=[create_valid_method(), create_valid_method("m2")],
            required_for_confirmed=2,
            required_for_review=1
        )

        with pytest.raises(ValueError, match="Verification failed"):
            use_case.execute(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_execute_runtime_error_from_service(self, mock_verifier_service_class):
        """Test generic exception handling from service"""
        mock_verifier_service = Mock()
        mock_verifier_service_class.return_value = mock_verifier_service
        mock_verifier_service.verify.side_effect = RuntimeError("Service crashed")

        use_case = VerifyUseCase("test-model")
        request = VerifyRequest(
            methods=[create_valid_method(), create_valid_method("m2")],
            required_for_confirmed=2,
            required_for_review=1
        )

        with pytest.raises(RuntimeError, match="Verification process failed"):
            use_case.execute(request)


class TestVerifyUseCaseValidation:
    """Tests for VerifyUseCase._validate_request()"""

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_confirmed_threshold_zero_raises_error(self, mock_verifier_service_class):
        """Test that confirmed threshold <= 0 raises error"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        method = create_valid_method()

        # Create request bypassing VerifyRequest validation
        request = Mock()
        request.methods = [method]
        request.required_for_confirmed = 0
        request.required_for_review = 1

        with pytest.raises(ValueError, match="Confirmed threshold must be positive"):
            use_case._validate_request(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_review_threshold_zero_raises_error(self, mock_verifier_service_class):
        """Test that review threshold <= 0 raises error"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        method = create_valid_method()

        request = Mock()
        request.methods = [method]
        request.required_for_confirmed = 2
        request.required_for_review = 0

        with pytest.raises(ValueError, match="Review threshold must be positive"):
            use_case._validate_request(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_confirmed_less_than_review_raises_error(self, mock_verifier_service_class):
        """Test that confirmed < review raises error (equality is allowed)"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        method = create_valid_method()

        request = Mock()
        request.methods = [method]
        request.required_for_confirmed = 1
        request.required_for_review = 2

        with pytest.raises(ValueError, match="Confirmed threshold must be greater than or equal to review threshold"):
            use_case._validate_request(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_confirmed_equal_to_review_is_allowed(self, mock_verifier_service_class):
        """Test that confirmed == review is valid (no error raised)"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        method = create_valid_method()

        request = Mock()
        request.methods = [method]
        request.required_for_confirmed = 2
        request.required_for_review = 2  # Equal to confirmed - should be allowed

        # Should not raise any error
        use_case._validate_request(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_empty_methods_raises_error(self, mock_verifier_service_class):
        """Test that empty methods list raises error"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        request = Mock()
        request.methods = []
        request.required_for_confirmed = 2
        request.required_for_review = 1

        with pytest.raises(ValueError, match="At least one verification method required"):
            use_case._validate_request(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_method_required_matches_zero_raises_error(self, mock_verifier_service_class):
        """Test that method with required_matches <= 0 raises error"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        method = Mock()
        method.name = "bad_method"
        method.required_matches = 0
        method.num_sequences = 2

        request = Mock()
        request.methods = [method]
        request.required_for_confirmed = 2
        request.required_for_review = 1

        with pytest.raises(ValueError, match="Required matches must be positive"):
            use_case._validate_request(request)

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_validate_method_required_matches_exceeds_sequences_raises_error(self, mock_verifier_service_class):
        """Test that required_matches > num_sequences raises error"""
        mock_verifier_service_class.return_value = Mock()
        use_case = VerifyUseCase("test-model")

        method = Mock()
        method.name = "bad_method"
        method.required_matches = 5
        method.num_sequences = 2

        request = Mock()
        request.methods = [method]
        request.required_for_confirmed = 2
        request.required_for_review = 1

        with pytest.raises(ValueError, match="exceed available sequences"):
            use_case._validate_request(request)

    # Note: test_validate_method_required_matches_none_skips_validation was removed
    # because it tested an impossible scenario. VerificationMethod.__post_init__
    # already validates that required_matches >= 1, so None is never a valid value.
    # Using Mock() to bypass dataclass validation creates artificial test cases
    # that don't reflect real-world usage.


class TestVerifyUseCaseInit:
    """Tests for VerifyUseCase initialization"""

    @patch('application.use_cases.verify_use_case.VerifierService')
    def test_init_creates_verifier_service(self, mock_verifier_service_class):
        """Test that init creates VerifierService"""
        mock_verifier_service_class.return_value = Mock()

        use_case = VerifyUseCase("custom-model")

        mock_verifier_service_class.assert_called_once_with(model_name="custom-model")
        assert use_case.verifier_service is not None
