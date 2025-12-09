# tests/unit/domain/services/test_verifier_service.py

import pytest
from unittest.mock import Mock, patch
from domain.services.verifier_service import VerifierService
from domain.model.entities.verification import (
    VerificationMethod, VerificationMode, VerificationStatus
)
from domain.model.entities.generation import GeneratedResult, GenerationMetadata
from datetime import datetime


class TestVerifierServiceInit:
    """Tests for VerifierService initialization"""

    def test_init_without_arguments_raises_error(self):
        """Test that initialization without model_name or generate_service raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            VerifierService()

        assert "Either model_name or generate_service must be provided" in str(exc_info.value)

    def test_init_with_custom_model(self):
        """Test initialization with custom model name"""
        # Act
        with patch('domain.services.verifier_service.GenerateService') as mock_gen_service:
            service = VerifierService(model_name="custom-model")

            # Assert
            mock_gen_service.assert_called_once_with("custom-model")

    def test_init_with_provided_generate_service(self):
        """Test initialization with provided GenerateService"""
        # Arrange
        mock_gen_service = Mock()

        # Act
        service = VerifierService(generate_service=mock_gen_service)

        # Assert
        assert service.generate_service == mock_gen_service


class TestVerifierServiceVerify:
    """Tests for VerifierService.verify method"""

    def test_verify_all_eliminatory_pass(self):
        """Test verification when all ELIMINATORY methods pass"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="check1",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=1
            ),
            VerificationMethod(
                mode=VerificationMode.CUMULATIVE,
                name="check2",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=1
            )
        ]

        # Mock generation results
        mock_results = [
            GeneratedResult(
                content="yes",
                metadata=GenerationMetadata(
                    model_name="test",
                    system_prompt="s",
                    user_prompt="u",
                    temperature=1.0,
                    tokens_used=10,
                    generation_time=0.5,
                    timestamp=datetime.now()
                )
            )
        ] * 2

        mock_gen_service.generate.return_value = mock_results

        # Act
        result = service.verify(methods, required_for_confirmed=2, required_for_review=1)

        # Assert
        assert result.final_status == "confirmed"
        assert len(result.results) == 2

    def test_verify_eliminatory_fails(self):
        """Test verification when ELIMINATORY method fails"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="critical_check",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # Mock generation results that don't match
        mock_results = [
            GeneratedResult(
                content="no",
                metadata=GenerationMetadata(
                    model_name="test",
                    system_prompt="s",
                    user_prompt="u",
                    temperature=1.0,
                    tokens_used=10,
                    generation_time=0.5,
                    timestamp=datetime.now()
                )
            )
        ] * 2

        mock_gen_service.generate.return_value = mock_results

        # Act
        result = service.verify(methods, required_for_confirmed=1, required_for_review=1)

        # Assert
        assert result.final_status == "discarded"
        assert len(result.results) == 1
        assert result.results[0].passed == False

    def test_verify_cumulative_for_review(self):
        """Test verification resulting in review status"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        methods = [
            VerificationMethod(
                mode=VerificationMode.CUMULATIVE,
                name="check1",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=1
            ),
            VerificationMethod(
                mode=VerificationMode.CUMULATIVE,
                name="check2",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=1
            )
        ]

        # First method passes, second fails
        mock_gen_service.generate.side_effect = [
            [GeneratedResult(content="yes", metadata=Mock())],
            [GeneratedResult(content="no", metadata=Mock())]
        ]

        # Act
        result = service.verify(methods, required_for_confirmed=2, required_for_review=1)

        # Assert
        assert result.final_status == "review"

    def test_verify_cumulative_discarded(self):
        """Test verification resulting in discarded status"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        methods = [
            VerificationMethod(
                mode=VerificationMode.CUMULATIVE,
                name="check1",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=2
            )
        ]

        # All fail
        mock_gen_service.generate.return_value = [
            GeneratedResult(content="no", metadata=Mock())
        ] * 2

        # Act
        result = service.verify(methods, required_for_confirmed=1, required_for_review=1)

        # Assert
        assert result.final_status == "discarded"


class TestVerifierServiceVerifyConsensus:
    """Tests for VerifierService._verify_consensus method"""

    def test_verify_consensus_missing_valid_responses(self):
        """Test that missing valid_responses raises ValueError"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            valid_responses=None,  # Invalid!
            required_matches=1
        )

        # Act & Assert
        with pytest.raises(ValueError, match="valid responses"):
            service._verify_consensus(method)

    def test_verify_consensus_generation_value_error(self):
        """Test that generation ValueError is properly handled"""
        # Arrange
        mock_gen_service = Mock()
        mock_gen_service.generate.side_effect = ValueError("Invalid parameters")
        service = VerifierService(generate_service=mock_gen_service)

        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            valid_responses=["yes"],
            required_matches=1
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Verification failed"):
            service._verify_consensus(method)

    def test_verify_consensus_generation_runtime_error(self):
        """Test that generation RuntimeError is properly handled"""
        # Arrange
        mock_gen_service = Mock()
        mock_gen_service.generate.side_effect = RuntimeError("Model failed")
        service = VerifierService(generate_service=mock_gen_service)

        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            valid_responses=["yes"],
            required_matches=1
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Verification execution failed"):
            service._verify_consensus(method)

    def test_verify_consensus_case_insensitive_matching(self):
        """Test that valid response matching is case-insensitive"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            valid_responses=["YES"],
            required_matches=1
        )

        # Mock generation with lowercase response
        mock_gen_service.generate.return_value = [
            GeneratedResult(content="yes, this is valid", metadata=Mock()),
            GeneratedResult(content="no", metadata=Mock())
        ]

        # Act
        result = service._verify_consensus(method)

        # Assert
        assert result.passed == True
        assert result.score == 0.5  # 1 out of 2

    def test_verify_consensus_missing_required_matches(self):
        """Test that missing required_matches raises ValueError"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        # Use Mock to bypass entity validation
        method = Mock()
        method.name = "test"
        method.system_prompt = "System"
        method.user_prompt = "User"
        method.num_sequences = 2
        method.valid_responses = ["yes"]
        method.required_matches = None  # Invalid!

        # Act & Assert
        with pytest.raises(ValueError, match="required_matches"):
            service._verify_consensus(method)

    def test_verify_consensus_num_sequences_less_than_required(self):
        """Test that num_sequences < required_matches raises ValueError"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        # Use Mock to bypass entity validation
        method = Mock()
        method.name = "test"
        method.system_prompt = "System"
        method.user_prompt = "User"
        method.num_sequences = 1  # Less than required_matches
        method.valid_responses = ["yes"]
        method.required_matches = 3

        # Act & Assert
        with pytest.raises(ValueError, match="num_sequences must be >= required_matches"):
            service._verify_consensus(method)

    def test_verify_consensus_empty_responses(self):
        """Test handling of empty response list"""
        # Arrange
        mock_gen_service = Mock()
        mock_gen_service.generate.return_value = []  # Empty response list
        service = VerifierService(generate_service=mock_gen_service)

        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=2,
            valid_responses=["yes"],
            required_matches=1
        )

        # Act
        result = service._verify_consensus(method)

        # Assert
        assert result.passed == False
        assert result.score == 0.0

    def test_verify_consensus_multiple_valid_responses(self):
        """Test matching against multiple valid responses"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=3,
            valid_responses=["yes", "true", "correct"],
            required_matches=2
        )

        # Mock generation with different valid responses
        mock_gen_service.generate.return_value = [
            GeneratedResult(content="yes", metadata=Mock()),
            GeneratedResult(content="true", metadata=Mock()),
            GeneratedResult(content="no", metadata=Mock())
        ]

        # Act
        result = service._verify_consensus(method)

        # Assert
        assert result.passed == True
        assert result.details["positive_responses"] == 2

    def test_verify_consensus_passes_temperature_parameter(self):
        """Test that temperature parameter from method is passed to generate"""
        # Arrange
        mock_gen_service = Mock()
        mock_gen_service.generate.return_value = [
            GeneratedResult(content="yes", metadata=Mock())
        ]
        service = VerifierService(generate_service=mock_gen_service)

        custom_temperature = 0.5
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=1,
            valid_responses=["yes"],
            required_matches=1,
            temperature=custom_temperature
        )

        # Act
        service._verify_consensus(method)

        # Assert - verify temperature was passed to generate()
        mock_gen_service.generate.assert_called_once()
        call_kwargs = mock_gen_service.generate.call_args.kwargs
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == custom_temperature

    def test_verify_consensus_uses_default_temperature(self):
        """Test that default temperature (1.0) is used when not specified"""
        # Arrange
        mock_gen_service = Mock()
        mock_gen_service.generate.return_value = [
            GeneratedResult(content="yes", metadata=Mock())
        ]
        service = VerifierService(generate_service=mock_gen_service)

        # Method without explicit temperature (defaults to 1.0)
        method = VerificationMethod(
            mode=VerificationMode.ELIMINATORY,
            name="test",
            system_prompt="System",
            user_prompt="User",
            num_sequences=1,
            valid_responses=["yes"],
            required_matches=1
        )

        # Act
        service._verify_consensus(method)

        # Assert - verify default temperature was passed
        call_kwargs = mock_gen_service.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0


class TestVerifierServiceVerifyEdgeCases:
    """Edge case tests for VerifierService.verify method"""

    def test_verify_empty_methods_list(self):
        """Test verification with empty methods list"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        # Act
        result = service.verify(
            methods=[],
            required_for_confirmed=0,
            required_for_review=0
        )

        # Assert - with 0 passes and 0 required, should be confirmed
        assert result.final_status == "confirmed"

    def test_verify_eliminatory_pass_counts_as_cumulative(self):
        """Test that passing ELIMINATORY method also counts toward cumulative"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="elim_check",
                system_prompt="System",
                user_prompt="User",
                num_sequences=2,
                valid_responses=["yes"],
                required_matches=1
            )
        ]

        mock_gen_service.generate.return_value = [
            GeneratedResult(content="yes", metadata=Mock()),
            GeneratedResult(content="yes", metadata=Mock())
        ]

        # Act - need 1 for confirmed
        result = service.verify(methods, required_for_confirmed=1, required_for_review=0)

        # Assert - eliminatory pass should count
        assert result.final_status == "confirmed"

    def test_verify_mixed_modes_confirmed(self):
        """Test verification with mixed modes resulting in confirmed"""
        # Arrange
        mock_gen_service = Mock()
        service = VerifierService(generate_service=mock_gen_service)

        methods = [
            VerificationMethod(
                mode=VerificationMode.ELIMINATORY,
                name="elim",
                system_prompt="S",
                user_prompt="U",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1
            ),
            VerificationMethod(
                mode=VerificationMode.CUMULATIVE,
                name="cumul",
                system_prompt="S",
                user_prompt="U",
                num_sequences=1,
                valid_responses=["yes"],
                required_matches=1
            )
        ]

        mock_gen_service.generate.return_value = [
            GeneratedResult(content="yes", metadata=Mock())
        ]

        # Act
        result = service.verify(methods, required_for_confirmed=2, required_for_review=1)

        # Assert
        assert result.final_status == "confirmed"
        assert len(result.results) == 2
