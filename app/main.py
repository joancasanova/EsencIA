import argparse
import json
import logging
import os
from typing import Dict, Any, List, Callable

from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_PIPELINE_REFERENCE_DATA,
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_BENCHMARK_ENTRIES,
    PIPELINE_RESULTS_DIR,
    PIPELINE_RESULTS_FILE,
    BENCHMARK_RESULTS_DIR,
    BENCHMARK_RESULTS_PREFIX,
    BENCHMARK_MISCLASSIFIED_DIR,
    BENCHMARK_MISCLASSIFIED_PREFIX,
    get_pipeline_verification_output_dir,
    get_pipeline_verification_filename,
)
from domain.model.entities.benchmark import BenchmarkConfig, BenchmarkEntry, BenchmarkMetrics
from infrastructure.file_repository import FileRepository

# Basic logging configuration with validation
_log_level = getattr(logging, DEFAULT_LOG_LEVEL.upper(), None)
if _log_level is None:
    _valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    print(f"WARNING: Invalid log level '{DEFAULT_LOG_LEVEL}'. Valid options: {_valid_levels}. Using INFO.")
    _log_level = logging.INFO

logging.basicConfig(
    level=_log_level,
    format=DEFAULT_LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Imports of use cases and models
from application.use_cases.benchmark_use_case import BenchmarkUseCase
from application.use_cases.generate_text_use_case import GenerateTextRequest
from application.use_cases.parse_use_case import ParseRequest
from application.use_cases.verify_use_case import VerifyRequest
from domain.model.entities.parsing import ParseMode, ParseRule
from domain.model.entities.verification import (
    VerificationMethod,
    VerificationMode
)
from domain.model.entities.pipeline import (
    PipelineResponse,
    PipelineStep,
    PipelineRequest
)
from application.use_cases.pipeline_use_case import PipelineUseCase

# Type for command handlers
CommandHandler = Callable[[argparse.Namespace], None]

class CommandProcessor:
    """Base class for command processing with common utilities"""
    
    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """
        Loads a JSON file and returns a dictionary.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dict containing the loaded JSON data

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            IOError: If file cannot be read
        """
        # Validate file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        # Validate it's a file (not a directory)
        if not os.path.isfile(file_path):
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a valid file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise json.JSONDecodeError(
                f"File contains invalid JSON: {e.msg}",
                e.doc,
                e.pos
            ) from e
        except IOError as e:
            logger.error(f"Cannot read file {file_path}: {e}")
            raise IOError(f"Cannot read file '{file_path}': {e}") from e

    @staticmethod
    def parse_rules(rules_data: List[Dict]) -> List[ParseRule]:
        """Converts JSON data into ParseRule objects.

        Note: Creates copies to avoid mutating original config data.
        """
        rules = []
        for idx, rule_data in enumerate(rules_data):
            # Create a copy to avoid mutating the original
            rule_copy = dict(rule_data)

            # Validate required 'mode' field
            if "mode" not in rule_copy:
                raise ValueError(f"Parse rule {idx + 1} is missing required field 'mode'")

            mode_str = rule_copy.pop("mode").upper()
            try:
                mode = ParseMode[mode_str]
            except KeyError:
                valid_modes = [m.name for m in ParseMode]
                raise ValueError(
                    f"Parse rule {idx + 1} has invalid mode '{mode_str}'. "
                    f"Valid modes: {valid_modes}"
                )

            rules.append(ParseRule(mode=mode, **rule_copy))
        return rules

    @staticmethod
    def parse_verification_methods(methods_data: List[Dict]) -> List[VerificationMethod]:
        """Converts JSON data into VerificationMethod objects.

        Note: Does not mutate original config data.
        """
        methods = []
        for idx, method_data in enumerate(methods_data):
            # Validate required 'mode' field
            if "mode" not in method_data:
                raise ValueError(f"Verification method {idx + 1} is missing required field 'mode'")

            mode_str = method_data["mode"].upper()
            try:
                mode = VerificationMode[mode_str]
            except KeyError:
                valid_modes = [m.name for m in VerificationMode]
                raise ValueError(
                    f"Verification method {idx + 1} has invalid mode '{mode_str}'. "
                    f"Valid modes: {valid_modes}"
                )

            # Validate other required fields
            required_fields = ["name", "system_prompt", "user_prompt", "valid_responses"]
            for field in required_fields:
                if field not in method_data:
                    raise ValueError(f"Verification method {idx + 1} is missing required field '{field}'")

            num_sequences = method_data.get("num_sequences", 3)
            # Default required_matches to majority vote (ceil of half + 1)
            default_required_matches = (num_sequences // 2) + 1
            methods.append(VerificationMethod(
                mode=mode,
                name=method_data["name"],
                system_prompt=method_data["system_prompt"],
                user_prompt=method_data["user_prompt"],
                valid_responses=method_data["valid_responses"],
                num_sequences=num_sequences,
                required_matches=method_data.get("required_matches", default_required_matches)
            ))
        return methods

    @staticmethod
    def parse_pipeline_steps(config: Dict) -> List[PipelineStep]:
        """Parses pipeline configuration into a list of steps."""
        steps = []
        for step_data in config["steps"]:
            step_type = step_data["type"]
            
            # Create parameters based on step type
            if step_type == "generate":
                parameters = GenerateTextRequest(**step_data["parameters"])
            elif step_type == "parse":
                parameters = ParseRequest(
                    rules=CommandProcessor.parse_rules(step_data["parameters"]["rules"]),
                    text=step_data["parameters"].get("text"),
                    output_filter=step_data["parameters"].get("output_filter", "all"),
                    output_limit=step_data["parameters"].get("output_limit")
                )
            elif step_type == "verify":
                parameters = VerifyRequest(
                    methods=CommandProcessor.parse_verification_methods(
                        step_data["parameters"]["methods"]
                    ),
                    required_for_confirmed=step_data["parameters"]["required_for_confirmed"],
                    required_for_review=step_data["parameters"]["required_for_review"]
                )
            else:
                raise ValueError(f"Invalid step type: {step_type}")

            steps.append(PipelineStep(
                type=step_type,
                parameters=parameters,
                uses_reference=step_data.get("uses_reference", False),
                reference_step_numbers=step_data.get("reference_step_numbers", []),
                llm_config=step_data.get("llm_config")
            ))
        return steps

class OutputFormatter:
    """Class for formatting different types of outputs"""

    @staticmethod
    def print_pipeline_results(response: PipelineResponse) -> None:
        """Prints pipeline results in a readable format."""
        # Print execution summary
        if response.total_entries > 0:
            print("\n" + "="*60)
            print("PIPELINE EXECUTION SUMMARY")
            print("="*60)
            print(f"Total entries processed: {response.total_entries}")
            print(f"✓ Successful: {response.successful_entries}")
            print(f"✗ Failed: {response.failed_entries}")
            if response.total_entries > 0:
                success_rate = (response.successful_entries / response.total_entries) * 100
                print(f"Success rate: {success_rate:.1f}%")
            print("="*60)

        # Print step results
        for i, step_result in enumerate(response.step_results):
            print(f"\n--- Step {i}: {step_result['step_type']} ---")
            OutputFormatter._print_step_data(step_result['step_data'])

        # Print errors if any
        if response.errors:
            print("\n" + "="*60)
            print(f"ERRORS ({len(response.errors)} total)")
            print("="*60)
            for error in response.errors:
                OutputFormatter._print_error(error)
            print("="*60)

    @staticmethod
    def _print_error(error: Any) -> None:
        """Prints detailed error information."""
        print(f"\n  Error at entry {error.entry_index}:")
        print(f"    Type: {error.error_type}")
        print(f"    Message: {error.error_message}")
        print(f"    Entry data: {error.entry_data}")
        if error.traceback:
            print(f"    Traceback preview:")
            # Print only last 3 lines of traceback
            traceback_lines = error.traceback.strip().split('\n')
            for line in traceback_lines[-3:]:
                print(f"      {line}")

    @staticmethod
    def _print_step_data(step_data: List[Any]) -> None:
        """Handles printing of different data types in steps"""
        for i, item in enumerate(step_data, 1):
            if isinstance(item, dict):
                OutputFormatter._print_dict_item(item, i)
            else:
                print(f"  Result {i}: {item}")

    @staticmethod
    def _print_dict_item(item: Dict, index: int) -> None:
        """Handles printing of specific dictionary items"""
        if 'content' in item:
            OutputFormatter._print_generation_result(item, index)
        elif 'final_status' in item:
            OutputFormatter._print_verification_result(item, index)
        elif 'entries' in item:
            OutputFormatter._print_parsing_result(item, index)

    @staticmethod
    def _print_generation_result(item: Dict, index: int) -> None:
        """Prints generation results"""
        print(f"\n  Result {index}:")
        print(f"    - Content: {item['content']}")
        if 'metadata' in item and isinstance(item['metadata'], dict):
            metadata = item['metadata']
            print("    - Metadata:")
            if 'system_prompt' in metadata:
                print(f"      -- System prompt: {metadata['system_prompt']}")
            if 'user_prompt' in metadata:
                print(f"      -- User prompt: {metadata['user_prompt']}")
        if 'reference_data' in item and item['reference_data']:
            print("    - Reference data:")
            for k, v in item['reference_data'].items():
                print(f"      -- {k}: {v}")

    @staticmethod
    def _print_verification_result(item: Dict, index: int) -> None:
        """Prints verification results"""
        print(f"  Verification result {index}:")
        print(f"    Final status: {item['final_status']}")
        print(f"    Success rate: {item['success_rate']:.2f}")
        print(f"    Reference data: {item['reference_data']}")
        print("    Results:")
        for result in item['results']:
            print(f"      Method: {result['method_name']}")
            print(f"        Mode: {result['mode']}")
            print(f"        Passed: {result['passed']}")
            print(f"        Score: {result['score']:.2f}")
            print(f"        Timestamp: {result['timestamp']}")
            print(f"        Details: {result['details']}")

    @staticmethod
    def _print_parsing_result(item: Dict, index: int) -> None:
        """Prints parsing results"""
        print(f"  Parsing result {index}:")
        for j, entry in enumerate(item['entries'], 1):
            print(f"    Entry {j}:")
            for key, value in entry.items():
                print(f"      {key}: {value}")

    @staticmethod
    def print_benchmark_results(metrics: BenchmarkMetrics) -> None:
        print("\n=== Benchmark Results ===")
        print(f"• Accuracy: {metrics.accuracy:.2%}")
        print(f"• Precision: {metrics.precision:.2%}")
        print(f"• Recall: {metrics.recall:.2%}")
        print(f"• F1-Score: {metrics.f1_score:.2%}")
        print("\nConfusion Matrix:")
        print(f"True Positives: {metrics.confusion_matrix['true_positive']}")
        print(f"False Positives: {metrics.confusion_matrix['false_positive']}")
        print(f"True Negatives: {metrics.confusion_matrix['true_negative']}")
        print(f"False Negatives: {metrics.confusion_matrix['false_negative']}")
        # Total = TP + FP + TN + FN (misclassified already included in FP and FN)
        total_cases = sum(metrics.confusion_matrix.values())
        print(f"Total cases evaluated: {total_cases}")
        print(f"\nMisclassified cases saved in: misclassified_*.json")

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Text Processing Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command registration
    commands = {
        "pipeline": setup_pipeline_parser,
        "benchmark": setup_benchmark_parser
    }

    for cmd, setup_fn in commands.items():
        subparser = subparsers.add_parser(cmd, help=f"{cmd} command")
        setup_fn(subparser)

    return parser

def setup_pipeline_parser(parser: argparse.ArgumentParser) -> None:
    """Sets up the parser for the pipeline command"""
    parser.add_argument("--config",
                      default=DEFAULT_PIPELINE_CONFIG,
                      help="Path to the pipeline configuration file")
    parser.add_argument(
        "--pipeline-model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model to use for all pipeline operations"
    )

def setup_benchmark_parser(parser: argparse.ArgumentParser) -> None:
    """Sets up the parser for the benchmark command"""
    parser.add_argument("--config",
                      default=DEFAULT_BENCHMARK_CONFIG,
                      help="Path to the benchmark configuration file")
    parser.add_argument("--entries",
                      default=DEFAULT_BENCHMARK_ENTRIES,
                      help="Path to the benchmark entries file")

def handle_pipeline(args: argparse.Namespace) -> None:
    """Handler for the pipeline command"""
    # Load and validate pipeline configuration
    config = CommandProcessor.load_json_file(args.config)
    pipeline_steps = CommandProcessor.parse_pipeline_steps(config)

    # Load reference data with validation
    if not os.path.exists(DEFAULT_PIPELINE_REFERENCE_DATA):
        logger.error(f"Reference data file not found: {DEFAULT_PIPELINE_REFERENCE_DATA}")
        raise FileNotFoundError(f"Reference data file not found: {DEFAULT_PIPELINE_REFERENCE_DATA}")

    try:
        with open(DEFAULT_PIPELINE_REFERENCE_DATA, 'r', encoding='utf-8') as f:
            reference_entries = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in reference data file: {e}")
        raise json.JSONDecodeError(
            f"Reference data contains invalid JSON: {e.msg}",
            e.doc,
            e.pos
        ) from e
    except IOError as e:
        logger.error(f"Cannot read reference data file: {e}")
        raise IOError(f"Cannot read reference data file: {e}") from e

    response = PipelineUseCase(
        args.pipeline_model_name
    ).execute_with_references(
        PipelineRequest(
            steps=pipeline_steps,
            global_references=config.get("global_references", {})
        ),
        reference_entries=reference_entries
    )

    FileRepository.append(
        data=response.step_results,
        output_dir=PIPELINE_RESULTS_DIR,
        filename=PIPELINE_RESULTS_FILE
    )

    for result_type in ['confirmed', 'to_verify']:
        entries = response.verification_references.get(result_type, [])
        if entries:
            output_dir = get_pipeline_verification_output_dir(result_type)
            filename = get_pipeline_verification_filename(result_type)

            for entry in entries:
                FileRepository.append(
                    data=entry,
                    output_dir=output_dir,
                    filename=filename
                )

    OutputFormatter.print_pipeline_results(response)

def handle_benchmark(args: argparse.Namespace) -> None:
    """Handler for the benchmark command"""
    config_data = CommandProcessor.load_json_file(args.config)
    entries_data = CommandProcessor.load_json_file(args.entries)
    
    # Convert to entities
    benchmark_config = BenchmarkConfig(
        model_name=config_data.get("model_name", DEFAULT_MODEL_NAME),
        pipeline_steps=CommandProcessor.parse_pipeline_steps(config_data),
        label_key=config_data["label_key"],
        label_value=config_data["label_value"]
    )

    benchmark_entries = [
        BenchmarkEntry(
            input_data={k: v for k, v in entry.items() if k != benchmark_config.label_key},
            expected_label=entry.get(benchmark_config.label_key, "")
        )
        for entry in entries_data
    ]

    use_case = BenchmarkUseCase(benchmark_config.model_name)
    metrics = use_case.run_benchmark(benchmark_config, benchmark_entries)  # Capture metrics

    # Save results
    FileRepository.save(
        metrics.to_dict(),
        output_dir=BENCHMARK_RESULTS_DIR,
        filename_prefix=BENCHMARK_RESULTS_PREFIX
    )

    if metrics.misclassified:
        FileRepository.save(
            [result.to_dict() for result in metrics.misclassified],
            output_dir=BENCHMARK_MISCLASSIFIED_DIR,
            filename_prefix=BENCHMARK_MISCLASSIFIED_PREFIX
        )
    
    OutputFormatter.print_benchmark_results(metrics)

def main() -> None:
    """Main function of the program"""
    command_handlers: Dict[str, CommandHandler] = {
        "pipeline": handle_pipeline,
        "benchmark": handle_benchmark
    }

    parser = setup_arg_parser()
    args = parser.parse_args()

    try:
        logger.info("Starting execution of command: %s", args.command)
        command_handlers[args.command](args)
    except Exception as e:
        logger.exception("Error during command execution")
        print(f"\nERROR: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()