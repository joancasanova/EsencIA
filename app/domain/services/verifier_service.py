# domain/services/verifier_service.py

import logging
from typing import List, Optional

from domain.model.entities.verification import (
    VerificationMethod, VerificationMode,
    VerificationResult, VerificationSummary, VerificationStatus
)
from domain.services.generate_service import GenerateService

logger = logging.getLogger(__name__)

class VerifierService:
    """
    Service that performs verification checks on text/data by leveraging
    a provided LLM. Supports both ELIMINATORY and CUMULATIVE verification modes.
    """
    def __init__(self, model_name: Optional[str] = None, generate_service: Optional[GenerateService] = None):
        """
        Initialize VerifierService with either a model name or a GenerateService.

        Args:
            model_name: Name of the model to use for verification (creates new GenerateService)
            generate_service: Pre-configured GenerateService instance

        Raises:
            ValueError: If neither model_name nor generate_service is provided
        """
        if generate_service:
            self.generate_service = generate_service
        elif model_name:
            self.generate_service = GenerateService(model_name)
        else:
            raise ValueError("Either model_name or generate_service must be provided")

    def verify(
        self,
        methods: List[VerificationMethod],
        required_for_confirmed: int,
        required_for_review: int
    ) -> VerificationSummary:
        """
        Verifies the provided methods. If any ELIMINATORY method fails, we discard.
        Otherwise, count cumulative successes and compare with required_for_confirmed/review.
        """
        logger.info("Starting verification process in VerifierService.")
        results: List[VerificationResult] = []
        cumulative_passes = 0
        final_status = None

        for method in methods:
            logger.debug(f"Verifying method '{method.name}' in mode '{method.mode}'.")
            result = self._verify_consensus(method)
            results.append(result)

            if not result.passed and method.mode == VerificationMode.ELIMINATORY:
                logger.info(f"Method '{method.name}' failed in ELIMINATORY mode. Discarding.")
                final_status = VerificationStatus.discarded()
                break
            elif result.passed and method.mode == VerificationMode.ELIMINATORY:
                # ELIMINATORY methods that pass also count toward cumulative threshold
                cumulative_passes += 1

            if result.passed and method.mode == VerificationMode.CUMULATIVE:
                cumulative_passes += 1

        else:  # Only executes if no break occurred in the for-loop
            if cumulative_passes >= required_for_confirmed:
                final_status = VerificationStatus.confirmed()
            elif cumulative_passes >= required_for_review:
                final_status = VerificationStatus.review()
            else:
                final_status = VerificationStatus.discarded()
        
        if not final_status:
            final_status = VerificationStatus.discarded()

        logger.info(f"Verification results concluded with status '{final_status.status}'.")
        return VerificationSummary(
            results=results,
            final_status=final_status.status
        )
    
    def _verify_consensus(self, method: VerificationMethod) -> VerificationResult:
        """
        Conduct a 'consensus' check: LLM generates multiple sequences;
        we count how many responses match a valid_responses list.

        Args:
            method: VerificationMethod configuration

        Returns:
            VerificationResult with verification outcome

        Raises:
            ValueError: If method configuration is invalid
            RuntimeError: If generation or processing fails
        """
        logger.debug(f"Consensus verification for method '{method.name}'.")

        # Validate method configuration
        if not method.valid_responses:
            logger.error("Valid responses not defined for consensus verification.")
            raise ValueError("Consensus verification requires valid responses")
        if method.required_matches is None:
            logger.error("required_matches not set for consensus verification.")
            raise ValueError("Consensus verification requires 'required_matches' to be set.")
        if method.num_sequences < method.required_matches:
            logger.error("num_sequences is smaller than required_matches.")
            raise ValueError("num_sequences must be >= required_matches for consensus verification.")

        system_prompt = method.system_prompt
        user_prompt = method.user_prompt

        # Generate responses with error handling
        try:
            logger.debug(f"Generating {method.num_sequences} sequence(s) for verification method '{method.name}'.")
            responses = self.generate_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                num_sequences=method.num_sequences,
                max_tokens=method.max_tokens,
                temperature=method.temperature
            )
        except ValueError as e:
            logger.error(f"Invalid parameters for verification method '{method.name}': {e}")
            raise ValueError(f"Verification failed for method '{method.name}': {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during generation for method '{method.name}'")
            raise RuntimeError(f"Verification execution failed for method '{method.name}': {e}") from e

        # Process responses with error handling
        try:
            generated_responses = [response.content for response in responses]

            positive_responses = sum(
                1 for r in responses
                if any(vr.lower() in r.content.strip().lower() for vr in method.valid_responses)
            )
            passed = positive_responses >= method.required_matches

            logger.debug(f"Method '{method.name}' => {positive_responses}/{len(responses)} positive responses. Passed={passed}")

            return VerificationResult(
                method=method,
                passed=passed,
                score=positive_responses / len(responses) if len(responses) > 0 else 0.0,
                details={
                    "positive_responses": positive_responses,
                    "generated_responses": generated_responses,
                }
            )
        except Exception as e:
            logger.exception(f"Error processing verification results for method '{method.name}'")
            raise RuntimeError(f"Failed to process verification results: {e}") from e
