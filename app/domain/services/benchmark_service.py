# domain/services/benchmark_service.py

import logging
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional

from domain.model.entities.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkMetrics
from domain.model.entities.pipeline import BenchmarkExecutionContext
from domain.services.pipeline_service import PipelineService

logger = logging.getLogger(__name__)


class BenchmarkService:
    """
    Service handling benchmark execution and metric calculations.

    Thread-safe: Uses execution contexts instead of mutable instance state
    and locks for shared resource access.

    Responsibilities:
    - Execute pipeline for benchmark entries
    - Manage pipeline step configuration
    - Handle placeholder substitution
    - Calculate performance metrics
    - Track misclassified cases
    """

    def __init__(self, model_name: str):
        """
        Initialize benchmark components.

        Args:
            model_name: Name of the model being benchmarked
        """
        self.model_name = model_name
        self.pipeline_service = PipelineService(model_name)
        self._execution_lock = threading.Lock()

    def execute_pipeline_for_entry(
        self,
        config: BenchmarkConfig,
        entry: Dict,
        context: Optional[BenchmarkExecutionContext] = None
    ) -> tuple[Optional[List[Dict[str, Any]]], BenchmarkExecutionContext]:
        """
        Execute configured pipeline for a single benchmark entry.

        Thread-safe: Uses a lock to prevent concurrent pipeline executions
        and an immutable context to track execution state.

        Args:
            config: Benchmark configuration
            entry: Input data for pipeline execution
            context: Optional execution context to use (creates new if None)

        Returns:
            Tuple of (pipeline results or None if failed, updated context)
        """
        if context is None:
            context = BenchmarkExecutionContext()

        # Update context with entry references (convert values to strings for consistency)
        string_entry = {k: str(v) for k, v in entry.items()}
        context = context.with_global_references(string_entry)

        try:
            with self._execution_lock:
                # Set entry data as global references for placeholder substitution
                self.pipeline_service.global_references = dict(entry)

                # Configure pipeline steps with entry-specific data
                configured_steps = self._configure_steps(config.pipeline_steps, entry)

                # Execute pipeline and return results
                self.pipeline_service.run_pipeline(configured_steps)
                return self.pipeline_service.get_results(), context
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return None, context

    def _configure_steps(self, steps: List, entry: Dict) -> List:
        """
        Clone and configure pipeline steps with entry data.
        
        Args:
            steps: Original pipeline steps
            entry: Data for placeholder substitution
            
        Returns:
            List of configured pipeline steps
        """
        configured_steps = []
        for step in steps:
            # Deep copy to prevent mutation of original steps
            cloned_step = deepcopy(step)
            # Inject entry data into step parameters
            self._substitute_placeholders(cloned_step.parameters, entry)
            configured_steps.append(cloned_step)
        return configured_steps

    def _substitute_placeholders(self, parameters: Any, data: Dict):
        """
        Replace placeholders in parameters with actual values from data.
        
        Handles parameters with:
        - system_prompt
        - user_prompt 
        - text
        """
        if hasattr(parameters, 'system_prompt'):
            parameters.system_prompt = self._replace_in_template(parameters.system_prompt, data)
        if hasattr(parameters, 'user_prompt'):
            parameters.user_prompt = self._replace_in_template(parameters.user_prompt, data)
        if hasattr(parameters, 'text'):
            parameters.text = self._replace_in_template(parameters.text, data)

    def _replace_in_template(self, template: str, data: Dict) -> str:
        """
        Replace {placeholder} patterns in template with actual values.
        
        Args:
            template: String with {key} placeholders
            data: Dictionary of key-value replacements
            
        Returns:
            String with substituted values
        """
        for key, value in data.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    def calculate_metrics(self, results: List[BenchmarkResult], label_value: str) -> BenchmarkMetrics:
        """
        Calculate performance metrics from benchmark results.
        
        Args:
            results: Collected benchmark results
            label_value: Positive class identifier
            
        Returns:
            BenchmarkMetrics: Calculated performance metrics
        """
        # Initialize confusion matrix counters
        confusion_matrix = {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0
        }
        misclassified = []
        
        # Categorize each result
        for result in results:
            actual = result.actual_label
            predicted = result.predicted_label
            
            is_actual_positive = (actual == label_value)
            is_predicted_positive = (predicted == "confirmed")

            # Update confusion matrix
            if is_actual_positive and is_predicted_positive:
                confusion_matrix["true_positive"] += 1
            elif not is_actual_positive and not is_predicted_positive:
                confusion_matrix["true_negative"] += 1
            elif is_actual_positive and not is_predicted_positive:
                confusion_matrix["false_negative"] += 1
                misclassified.append(result)
            else:
                confusion_matrix["false_positive"] += 1
                misclassified.append(result)

        total = len(results)

        # Handle empty results case
        if total == 0:
            return BenchmarkMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                confusion_matrix=confusion_matrix,
                misclassified=misclassified
            )
        
        # Calculate metrics with proper division by zero handling
        accuracy = (confusion_matrix["true_positive"] + confusion_matrix["true_negative"]) / total

        precision_denominator = confusion_matrix["true_positive"] + confusion_matrix["false_positive"]
        precision = confusion_matrix["true_positive"] / precision_denominator if precision_denominator > 0 else 0.0

        recall_denominator = confusion_matrix["true_positive"] + confusion_matrix["false_negative"]
        recall = confusion_matrix["true_positive"] / recall_denominator if recall_denominator > 0 else 0.0

        f1_denominator = precision + recall
        f1 = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0.0

        return BenchmarkMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix,
            misclassified=misclassified
        )