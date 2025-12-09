# tests/unit/domain/entities/test_verification.py

import pytest
from datetime import datetime
from domain.model.entities.verification import (
    VerificationMode,
    VerificationMethod,
    VerificationResult,
    VerificationSummary,
    VerificationStatus,
    VerificationThresholds,
    VerifyRequest,
    VerifyResponse
)


class TestVerificationMethod:
    """Tests for VerificationMethod entity validation"""

    def test_valid_verification_method_creation(self):
        """Test creating a valid VerificationMethod"""
        # Arrange & Act
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test_method",
            system_prompt="You are a helpful assistant",
            user_prompt="Verify this data",
            num_sequences=3,
            valid_responses=["yes", "no"],
            required_matches=2,
            max_tokens=100,
            temperature=0.7
        )

        # Assert
        assert method.name == "test_method"
        assert method.mode == VerificationMode.ELIMINATORY
        assert method.num_sequences == 3
        assert method.required_matches == 2

    def test_verification_method_empty_name_raises_error(self):
        """Test that empty name raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="name cannot be empty"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )

    def test_verification_method_empty_system_prompt_raises_error(self):
        """Test that empty system_prompt raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )

    def test_verification_method_empty_user_prompt_raises_error(self):
        """Test that empty user_prompt raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="user_prompt cannot be empty"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )

    def test_verification_method_invalid_num_sequences_raises_error(self):
        """Test that num_sequences < 1 raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="num_sequences must be >= 1"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=0,
                valid_responses=["yes"],
                required_matches=1
            )

    def test_verification_method_required_matches_exceeds_sequences_raises_error(self):
        """Test that required_matches > num_sequences raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="required_matches.*cannot exceed.*num_sequences"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=5
            )

    def test_verification_method_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="temperature must be between"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2,
                temperature=3.0  # Too high
            )

    def test_verification_method_required_matches_zero_raises_error(self):
        """Test that required_matches < 1 raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="required_matches must be >= 1"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=0
            )

    def test_verification_method_invalid_max_tokens_raises_error(self):
        """Test that invalid max_tokens raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="max_tokens must be between"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2,
                max_tokens=0  # Too low
            )

    def test_verification_method_max_tokens_too_high_raises_error(self):
        """Test that max_tokens above limit raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="max_tokens must be between"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2,
                max_tokens=99999999  # Too high
            )

    def test_verification_method_temperature_negative_raises_error(self):
        """Test that negative temperature raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="temperature must be between"):
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2,
                temperature=-0.5
            )


class TestVerifyRequest:
    """Tests for VerifyRequest entity validation"""

    def test_valid_verify_request_creation(self):
        """Test creating a valid VerifyRequest"""
        # Arrange
        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Act
        request = VerifyRequest(
            methods=methods,
            required_for_confirmed=1,
            required_for_review=1
        )

        # Assert
        assert len(request.methods) == 1
        assert request.required_for_confirmed == 1

    def test_verify_request_empty_methods_raises_error(self):
        """Test that empty methods list raises ValueError"""
        # Arrange, Act & Assert
        with pytest.raises(ValueError, match="methods list cannot be empty"):
            VerifyRequest(
                methods=[],
                required_for_confirmed=1,
                required_for_review=0
            )

    def test_verify_request_negative_required_for_confirmed_raises_error(self):
        """Test that negative required_for_confirmed raises ValueError"""
        # Arrange
        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="required_for_confirmed must be >= 0"):
            VerifyRequest(
                methods=methods,
                required_for_confirmed=-1,
                required_for_review=0
            )

    def test_verify_request_review_exceeds_confirmed_raises_error(self):
        """Test that required_for_review > required_for_confirmed raises ValueError"""
        # Arrange - use enough methods so the validation checks review vs confirmed
        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test1",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            ),
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test2",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            ),
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test3",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="required_for_review.*cannot exceed.*required_for_confirmed"):
            VerifyRequest(
                methods=methods,
                required_for_confirmed=1,
                required_for_review=2
            )

    def test_verify_request_negative_required_for_review_raises_error(self):
        """Test that negative required_for_review raises ValueError"""
        # Arrange
        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="required_for_review must be >= 0"):
            VerifyRequest(
                methods=methods,
                required_for_confirmed=1,
                required_for_review=-1
            )

    def test_verify_request_confirmed_exceeds_methods_raises_error(self):
        """Test that required_for_confirmed > len(methods) raises ValueError"""
        # Arrange
        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="required_for_confirmed.*cannot exceed.*number of methods"):
            VerifyRequest(
                methods=methods,
                required_for_confirmed=5,  # More than 1 method
                required_for_review=0
            )

    def test_verify_request_review_exceeds_methods_raises_error(self):
        """Test that required_for_review > len(methods) raises ValueError"""
        # Arrange
        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="test",
                system_prompt="Test",
                user_prompt="Test",
                num_sequences=3,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="required_for_review.*cannot exceed.*number of methods"):
            VerifyRequest(
                methods=methods,
                required_for_confirmed=1,
                required_for_review=5  # More than 1 method
            )


class TestVerificationStatus:
    """Tests for VerificationStatus entity"""

    def test_verification_status_confirmed(self):
        """Test creating confirmed status"""
        # Arrange & Act
        status = VerificationStatus.confirmed()

        # Assert
        assert status.id == "CONFIRMED"
        assert status.status == "confirmed"

    def test_verification_status_discarded(self):
        """Test creating discarded status"""
        # Arrange & Act
        status = VerificationStatus.discarded()

        # Assert
        assert status.id == "DISCARDED"
        assert status.status == "discarded"

    def test_verification_status_review(self):
        """Test creating review status"""
        # Arrange & Act
        status = VerificationStatus.review()

        # Assert
        assert status.id == "REVIEW"
        assert status.status == "review"

    def test_verification_status_from_string(self):
        """Test creating status from string"""
        # Arrange & Act
        confirmed = VerificationStatus.from_string("confirmed")
        discarded = VerificationStatus.from_string("DISCARDED")
        review = VerificationStatus.from_string("Review")

        # Assert
        assert confirmed.status == "confirmed"
        assert discarded.status == "discarded"
        assert review.status == "review"

    def test_verification_status_is_final(self):
        """Test is_final method"""
        # Arrange
        confirmed = VerificationStatus.confirmed()
        discarded = VerificationStatus.discarded()
        review = VerificationStatus.review()

        # Act & Assert
        assert confirmed.is_final()
        assert discarded.is_final()
        assert not review.is_final()

    def test_verification_status_requires_review(self):
        """Test requires_review method"""
        # Arrange
        confirmed = VerificationStatus.confirmed()
        review = VerificationStatus.review()

        # Act & Assert
        assert not confirmed.requires_review()
        assert review.requires_review()


class TestVerificationSummary:
    """Tests for VerificationSummary entity"""

    def test_verification_summary_success_rate(self):
        """Test success_rate calculation"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        results = [
            VerificationResult(method=method, passed=True, score=0.9),
            VerificationResult(method=method, passed=False, score=0.3),
            VerificationResult(method=method, passed=True, score=0.8)
        ]

        # Act
        summary = VerificationSummary(
            results=results,
            final_status="confirmed"
        )

        # Assert
        assert summary.success_rate == pytest.approx(2/3)

    def test_verification_summary_passed_methods(self):
        """Test passed_methods property"""
        # Arrange
        method1 = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="method1",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )
        method2 = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="method2",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        results = [
            VerificationResult(method=method1, passed=True),
            VerificationResult(method=method2, passed=False)
        ]

        # Act
        summary = VerificationSummary(
            results=results,
            final_status="review"
        )

        # Assert
        assert summary.passed_methods == ["method1"]
        assert summary.failed_methods == ["method2"]


class TestVerificationThresholds:
    """Tests for VerificationThresholds entity"""

    def test_verification_thresholds_is_within_bounds(self):
        """Test is_within_bounds method"""
        # Arrange
        thresholds = VerificationThresholds(
            lower_bound=0.5,
            upper_bound=1.0,
            target_value=0.8
        )

        # Act & Assert
        assert thresholds.is_within_bounds(0.7)
        assert thresholds.is_within_bounds(0.5)
        assert thresholds.is_within_bounds(1.0)
        assert not thresholds.is_within_bounds(0.3)
        assert not thresholds.is_within_bounds(1.5)


class TestVerificationResult:
    """Tests for VerificationResult entity"""

    def test_verification_result_to_dict(self):
        """Test to_dict serialization"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test_method",
            system_prompt="Test system",
            user_prompt="Test user",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        result = VerificationResult(
            method=method,
            passed=True,
            score=0.85,
            details={"positive_responses": 2, "total_responses": 3}
        )

        # Act
        result_dict = result.to_dict()

        # Assert
        assert result_dict["method_name"] == "test_method"
        assert result_dict["mode"] == "eliminatory"
        assert result_dict["passed"] == True
        assert result_dict["score"] == 0.85
        assert result_dict["details"]["positive_responses"] == 2
        assert "timestamp" in result_dict

    def test_verification_result_to_dict_without_optional_fields(self):
        """Test to_dict with minimal result"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.CUMULATIVE,
            name="minimal_test",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=1,
            valid_responses=["yes"],
            required_matches=1
        )

        result = VerificationResult(
            method=method,
            passed=False
        )

        # Act
        result_dict = result.to_dict()

        # Assert
        assert result_dict["method_name"] == "minimal_test"
        assert result_dict["mode"] == "cumulative"
        assert result_dict["passed"] == False
        assert result_dict["score"] is None
        assert result_dict["details"] is None


class TestVerificationSummaryMethods:
    """Additional tests for VerificationSummary methods"""

    def test_verification_summary_to_dict(self):
        """Test to_dict serialization"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        results = [
            VerificationResult(method=method, passed=True, score=0.9),
            VerificationResult(method=method, passed=False, score=0.3)
        ]

        summary = VerificationSummary(
            results=results,
            final_status="review",
            reference_data={"name": "John", "age": "30"}
        )

        # Act
        summary_dict = summary.to_dict()

        # Assert
        assert len(summary_dict["results"]) == 2
        assert summary_dict["final_status"] == "review"
        assert summary_dict["reference_data"] == {"name": "John", "age": "30"}
        assert summary_dict["success_rate"] == pytest.approx(0.5)

    def test_verification_summary_scores_property(self):
        """Test scores property returns all scores"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        results = [
            VerificationResult(method=method, passed=True, score=0.9),
            VerificationResult(method=method, passed=False, score=0.3),
            VerificationResult(method=method, passed=True, score=None)  # No score
        ]

        summary = VerificationSummary(
            results=results,
            final_status="confirmed"
        )

        # Act
        scores = summary.scores

        # Assert
        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.3
        assert scores[2] is None

    def test_verification_summary_empty_results(self):
        """Test summary with empty results"""
        # Arrange
        summary = VerificationSummary(
            results=[],
            final_status="discarded"
        )

        # Act & Assert
        assert summary.success_rate == 0.0
        assert summary.passed_methods == []
        assert summary.failed_methods == []
        assert summary.scores == []

    def test_verification_summary_failed_methods(self):
        """Test failed_methods property"""
        # Arrange
        method1 = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="failed_method",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        results = [
            VerificationResult(method=method1, passed=False, score=0.2)
        ]

        summary = VerificationSummary(
            results=results,
            final_status="discarded"
        )

        # Act & Assert
        assert summary.failed_methods == ["failed_method"]
        assert summary.passed_methods == []


class TestVerificationStatusEdgeCases:
    """Edge case tests for VerificationStatus"""

    def test_verification_status_from_string_unknown(self):
        """Test from_string with unknown status returns None"""
        # Act
        result = VerificationStatus.from_string("unknown")

        # Assert
        assert result is None

    def test_verification_status_from_string_case_insensitive(self):
        """Test from_string handles different cases"""
        # Act & Assert
        assert VerificationStatus.from_string("CONFIRMED").status == "confirmed"
        assert VerificationStatus.from_string("Discarded").status == "discarded"
        assert VerificationStatus.from_string("REVIEW").status == "review"


class TestVerifyResponse:
    """Tests for VerifyResponse entity"""

    def test_verify_response_creation(self):
        """Test creating VerifyResponse"""
        # Arrange
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="Test",
            user_prompt="Test",
            num_sequences=3,
            valid_responses=["yes"],
            required_matches=2
        )

        results = [VerificationResult(method=method, passed=True)]
        summary = VerificationSummary(results=results, final_status="confirmed")

        # Act
        response = VerifyResponse(
            verification_summary=summary,
            execution_time=1.5,
            success_rate=1.0
        )

        # Assert
        assert response.execution_time == 1.5
        assert response.success_rate == 1.0
        assert response.verification_summary.final_status == "confirmed"
