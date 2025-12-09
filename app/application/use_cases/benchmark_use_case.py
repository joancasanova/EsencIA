# application/use_cases/benchmark_use_case.py

import logging
from datetime import datetime
from typing import Dict, List, Optional
from domain.model.entities.benchmark import BenchmarkConfig, BenchmarkEntry, BenchmarkMetrics, BenchmarkResult
from domain.services.benchmark_service import BenchmarkService

logger = logging.getLogger(__name__)

class BenchmarkUseCase:
    """
    Coordinates the benchmarking process for evaluating pipeline performance.
    
    Responsibilities:
    - Executes the pipeline for each benchmark entry
    - Processes pipeline responses to extract predictions
    - Calculates performance metrics based on results
    """
    
    def __init__(self, model_name: str):
        """Initializes the use case with required services.
        
        Args:
            model_name: Identifier for the ML model to benchmark
        """
        self.benchmark_service = BenchmarkService(model_name)

    def run_benchmark(self, config: BenchmarkConfig, entries: List[BenchmarkEntry]) -> BenchmarkMetrics:
        """Executes the full benchmarking workflow.

        Args:
            config: Benchmark configuration parameters
            entries: List of input entries with expected labels

        Returns:
            BenchmarkMetrics: Calculated performance metrics

        Raises:
            ValueError: If entries list is empty or no valid results obtained
            RuntimeError: If metrics calculation fails
        """
        # Validate inputs
        if not entries:
            logger.error("Cannot run benchmark with empty entries list")
            raise ValueError("Benchmark entries list cannot be empty")

        results = []
        failed_entries = []

        # Process each benchmark entry through the pipeline
        for idx, entry in enumerate(entries):
            try:
                logger.debug(f"Running pipeline for entry {idx + 1}/{len(entries)}: {entry.input_data}")

                # Execute pipeline and get response (returns tuple of result and context)
                pipeline_response, _ = self.benchmark_service.execute_pipeline_for_entry(
                    config, entry.input_data
                )
                logger.debug(pipeline_response)

                # Extract prediction from pipeline response
                prediction = self._process_prediction(pipeline_response, entry)
                if prediction:
                    results.append(prediction)
                else:
                    logger.warning(f"No valid prediction obtained for entry {idx + 1}")
                    failed_entries.append(idx + 1)
            except Exception as e:
                logger.warning(f"Failed to process entry {idx + 1}: {e}")
                failed_entries.append(idx + 1)
                continue

        # Log summary of failures
        if failed_entries:
            logger.warning(f"Failed to process {len(failed_entries)}/{len(entries)} entries: {failed_entries}")

        # Ensure we have at least some results
        if not results:
            logger.error("No valid results obtained from benchmark run")
            raise ValueError("Benchmark produced no valid results. All entries failed to process.")

        # Calculate final performance metrics with error handling
        try:
            metrics = self.benchmark_service.calculate_metrics(results, config.label_value)
            logger.info(f"Benchmark completed: {len(results)}/{len(entries)} entries processed successfully")
            return metrics
        except Exception as e:
            logger.exception("Failed to calculate benchmark metrics")
            raise RuntimeError(f"Metrics calculation failed: {e}") from e

    def _process_prediction(self, pipeline_response: Optional[Dict], entry: BenchmarkEntry) -> Optional[BenchmarkResult]:
        """Extracts prediction from pipeline response and validates results.
        
        Args:
            pipeline_response: Raw output from pipeline execution
            entry: Original benchmark entry with expected label
            
        Returns:
            BenchmarkResult if valid prediction found, None otherwise
        """
        # Handle empty pipeline response (None or empty list)
        if not pipeline_response:
            logger.debug(f"Pipeline response is empty for entry: {entry.input_data}")
            return None

        # Find verification step in pipeline results
        verify_step = next(
            (s for s in pipeline_response
             if s["step_type"] == "verify"),
            None
        )
        
        # Validate verification step existence
        if not verify_step:
            return None

        # Check for valid step data
        step_data = verify_step.get("step_data", [])
        if not step_data:
            logger.debug(f"Verify step data is empty for entry: {entry.input_data}")
            return None

        # Determine prediction based on verification outcome
        final_status = step_data[0].get("final_status", "").lower()
        return BenchmarkResult(
            input_data=entry.input_data,
            predicted_label="confirmed" if final_status == "confirmed" else "not_confirmed",
            actual_label=entry.expected_label,
            timestamp=datetime.now()
        )