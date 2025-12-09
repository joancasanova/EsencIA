# application/use_cases/pipeline_use_case.py

import json
import logging
import traceback
from typing import List

from config import DEFAULT_MODEL_NAME
from domain.model.entities.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineExecutionError
)
from domain.services.pipeline_service import PipelineService
from infrastructure.file_repository import FileRepository

logger = logging.getLogger(__name__)

class PipelineUseCase:
    """
    Orchestrates pipeline execution and handles reference data processing.

    Responsibilities:
    - Manage complete pipeline lifecycle
    - Handle single and multi-reference executions
    - Coordinate result aggregation and storage
    - Maintain data consistency across executions
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initializes pipeline components with model configuration.

        Args:
            model_name: Model identifier for all pipeline operations (generation and verification)
        """
        self.service = PipelineService(model_name)
        self.file_repo = FileRepository()
        logger.debug("Initialized PipelineUseCase with model: %s", model_name)

    def _execute(self, request: PipelineRequest) -> PipelineResponse:
        """
        Executes a single pipeline run with provided configuration.
        
        Args:
            request: Pipeline configuration and input data
            
        Returns:
            PipelineResponse: Consolidated results from all steps
        """
        logger.info("Starting single pipeline execution")
        try:
            if request.global_references:
                logger.debug("Loading %d global references", len(request.global_references))
                self.service.global_references = request.global_references

            self.service.run_pipeline(request.steps)
            
            return PipelineResponse(
                step_results=self.service.get_results(),
                verification_references={
                    'confirmed': self.service.confirmed_references,
                    'to_verify': self.service.to_verify_references
                }
            )
            
        except ValueError as e:
            logger.error(f"Invalid pipeline configuration: {e}")
            raise ValueError(f"Pipeline execution failed due to invalid configuration: {e}") from e
        except Exception as e:
            logger.exception(f"Pipeline execution failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"Pipeline execution failed: {e}") from e

    def execute_with_references(self,
                               request: PipelineRequest,
                               reference_entries: List[dict]) -> PipelineResponse:
        """
        Executes pipeline for multiple reference entries with comprehensive error tracking.

        Args:
            request: Pipeline configuration
            reference_entries: List of reference data dictionaries

        Returns:
            PipelineResponse with results, errors, and execution statistics
        """
        logger.info("Starting multi-reference pipeline execution with %d entries", len(reference_entries))

        cumulative_response = PipelineResponse(
            step_results=[],
            verification_references={'confirmed': [], 'to_verify': []},
            total_entries=len(reference_entries),
            successful_entries=0,
            failed_entries=0
        )

        for idx, entry in enumerate(reference_entries):
            try:
                logger.debug("Processing reference entry %d/%d", idx+1, len(reference_entries))
                result = self._process_single_entry(request, entry)

                # Accumulate successful results
                cumulative_response.step_results.extend(result.step_results)
                cumulative_response.verification_references['confirmed'].extend(
                    result.verification_references['confirmed']
                )
                cumulative_response.verification_references['to_verify'].extend(
                    result.verification_references['to_verify']
                )
                cumulative_response.successful_entries += 1

                logger.debug("Successfully processed entry %d", idx+1)

            except Exception as e:
                # Capture full error information
                error_traceback = traceback.format_exc()
                error = PipelineExecutionError(
                    entry_index=idx,
                    entry_data=entry,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=error_traceback
                )
                cumulative_response.errors.append(error)
                cumulative_response.failed_entries += 1

                logger.error(
                    "Failed processing entry %d/%d: %s: %s",
                    idx+1, len(reference_entries),
                    type(e).__name__, str(e),
                    exc_info=True
                )

        # Log final summary
        logger.info(
            "Pipeline execution completed: %d/%d successful, %d failed",
            cumulative_response.successful_entries,
            cumulative_response.total_entries,
            cumulative_response.failed_entries
        )

        if cumulative_response.failed_entries > 0:
            logger.warning(
                "Pipeline completed with %d errors. Check response.errors for details.",
                cumulative_response.failed_entries
            )

        return cumulative_response

    def _process_single_entry(self, 
                             base_request: PipelineRequest,
                             entry_data: dict) -> PipelineResponse:
        """
        Processes a single reference entry through the pipeline.
        
        Args:
            base_request: Original pipeline configuration
            entry_data: Specific reference data for this execution
            
        Returns:
            PipelineResponse: Results for this single entry
        """
        modified_request = PipelineRequest(
            steps=base_request.steps,
            global_references=entry_data
        )
        
        return self._execute(modified_request)
