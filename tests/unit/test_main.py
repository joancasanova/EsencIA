# tests/unit/test_main.py
"""
Unit tests for the main CLI module.
Following TDD methodology - tests written first.
"""

import pytest
import json
import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))

from main import (
    CommandProcessor,
    OutputFormatter,
    setup_arg_parser,
    setup_pipeline_parser,
    setup_benchmark_parser,
    handle_pipeline,
    handle_benchmark,
    main
)


# =============================================================================
# Tests for CommandProcessor.load_json_file()
# =============================================================================

class TestCommandProcessorLoadJsonFile:
    """Tests for CommandProcessor.load_json_file()"""

    def test_load_valid_json_file(self, tmp_path):
        """Test loading a valid JSON file"""
        # Arrange
        json_content = {"key": "value", "number": 42}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(json_content), encoding="utf-8")

        # Act
        result = CommandProcessor.load_json_file(str(json_file))

        # Assert
        assert result == json_content

    def test_load_json_file_not_found_raises_error(self):
        """Test that loading non-existent file raises FileNotFoundError"""
        # Arrange
        non_existent_path = "/non/existent/path/file.json"

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            CommandProcessor.load_json_file(non_existent_path)
        assert "not found" in str(exc_info.value).lower()

    def test_load_json_file_directory_raises_error(self, tmp_path):
        """Test that passing a directory raises ValueError"""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.load_json_file(str(tmp_path))
        assert "not a valid file" in str(exc_info.value).lower()

    def test_load_invalid_json_raises_error(self, tmp_path):
        """Test that invalid JSON raises JSONDecodeError"""
        # Arrange
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{invalid json content", encoding="utf-8")

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            CommandProcessor.load_json_file(str(json_file))

    def test_load_json_file_with_utf8_content(self, tmp_path):
        """Test loading JSON with UTF-8 characters"""
        # Arrange
        json_content = {"mensaje": "Hola, ¿cómo estás?", "emoji": "🎉"}
        json_file = tmp_path / "utf8.json"
        json_file.write_text(json.dumps(json_content, ensure_ascii=False), encoding="utf-8")

        # Act
        result = CommandProcessor.load_json_file(str(json_file))

        # Assert
        assert result == json_content


# =============================================================================
# Tests for CommandProcessor.parse_rules()
# =============================================================================

class TestCommandProcessorParseRules:
    """Tests for CommandProcessor.parse_rules()"""

    def test_parse_rules_with_keyword_mode(self):
        """Test parsing rules with KEYWORD mode"""
        # Arrange
        rules_data = [
            {
                "name": "content",
                "mode": "KEYWORD",
                "pattern": "Content:",
                "secondary_pattern": ".",
                "fallback_value": "not_found"
            }
        ]

        # Act
        result = CommandProcessor.parse_rules(rules_data)

        # Assert
        assert len(result) == 1
        assert result[0].name == "content"
        assert result[0].pattern == "Content:"

    def test_parse_rules_with_regex_mode(self):
        """Test parsing rules with REGEX mode"""
        # Arrange
        rules_data = [
            {
                "name": "number",
                "mode": "REGEX",
                "pattern": r"\d+"
            }
        ]

        # Act
        result = CommandProcessor.parse_rules(rules_data)

        # Assert
        assert len(result) == 1
        assert result[0].name == "number"

    def test_parse_rules_case_insensitive_mode(self):
        """Test that mode parsing is case insensitive"""
        # Arrange
        rules_data = [
            {"name": "test", "mode": "keyword", "pattern": "test"}
        ]

        # Act
        result = CommandProcessor.parse_rules(rules_data)

        # Assert
        assert len(result) == 1

    def test_parse_rules_missing_mode_raises_error(self):
        """Test that missing mode raises ValueError"""
        # Arrange
        rules_data = [
            {"name": "test", "pattern": "test"}  # Missing 'mode'
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.parse_rules(rules_data)
        assert "missing required field 'mode'" in str(exc_info.value).lower()

    def test_parse_rules_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        # Arrange
        rules_data = [
            {"name": "test", "mode": "INVALID_MODE", "pattern": "test"}
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.parse_rules(rules_data)
        assert "invalid mode" in str(exc_info.value).lower()

    def test_parse_rules_does_not_mutate_input(self):
        """Test that parse_rules does not mutate the input data"""
        # Arrange
        rules_data = [
            {"name": "test", "mode": "KEYWORD", "pattern": "test"}
        ]
        original_data = json.dumps(rules_data)

        # Act
        CommandProcessor.parse_rules(rules_data)

        # Assert
        assert json.dumps(rules_data) == original_data


# =============================================================================
# Tests for CommandProcessor.parse_verification_methods()
# =============================================================================

class TestCommandProcessorParseVerificationMethods:
    """Tests for CommandProcessor.parse_verification_methods()"""

    def test_parse_verification_methods_cumulative(self):
        """Test parsing cumulative verification method"""
        # Arrange
        methods_data = [
            {
                "mode": "cumulative",
                "name": "test_method",
                "system_prompt": "System prompt",
                "user_prompt": "User prompt",
                "valid_responses": ["yes", "si"],
                "num_sequences": 3,
                "required_matches": 2
            }
        ]

        # Act
        result = CommandProcessor.parse_verification_methods(methods_data)

        # Assert
        assert len(result) == 1
        assert result[0].name == "test_method"
        assert result[0].num_sequences == 3

    def test_parse_verification_methods_eliminatory(self):
        """Test parsing eliminatory verification method"""
        # Arrange
        methods_data = [
            {
                "mode": "ELIMINATORY",
                "name": "strict_check",
                "system_prompt": "Be strict",
                "user_prompt": "Check this",
                "valid_responses": ["pass"],
                "required_matches": 1
            }
        ]

        # Act
        result = CommandProcessor.parse_verification_methods(methods_data)

        # Assert
        assert len(result) == 1

    def test_parse_verification_methods_missing_mode_raises_error(self):
        """Test that missing mode raises ValueError"""
        # Arrange
        methods_data = [
            {
                "name": "test",
                "system_prompt": "sys",
                "user_prompt": "user",
                "valid_responses": ["yes"]
            }
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.parse_verification_methods(methods_data)
        assert "missing required field 'mode'" in str(exc_info.value).lower()

    def test_parse_verification_methods_missing_required_field_raises_error(self):
        """Test that missing required fields raises ValueError"""
        # Arrange
        methods_data = [
            {
                "mode": "cumulative",
                "name": "test"
                # Missing system_prompt, user_prompt, valid_responses
            }
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.parse_verification_methods(methods_data)
        assert "missing required field" in str(exc_info.value).lower()

    def test_parse_verification_methods_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        # Arrange
        methods_data = [
            {
                "mode": "INVALID",
                "name": "test",
                "system_prompt": "sys",
                "user_prompt": "user",
                "valid_responses": ["yes"]
            }
        ]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.parse_verification_methods(methods_data)
        assert "invalid mode" in str(exc_info.value).lower()

    def test_parse_verification_methods_default_num_sequences(self):
        """Test that num_sequences defaults to 3 when not provided"""
        # Arrange
        methods_data = [
            {
                "mode": "cumulative",
                "name": "test",
                "system_prompt": "sys",
                "user_prompt": "user",
                "valid_responses": ["yes"]
            }
        ]

        # Act
        result = CommandProcessor.parse_verification_methods(methods_data)

        # Assert
        assert result[0].num_sequences == 3
        # With 3 sequences, default required_matches should be majority (2)
        assert result[0].required_matches == 2


# =============================================================================
# Tests for CommandProcessor.parse_pipeline_steps()
# =============================================================================

class TestCommandProcessorParsePipelineSteps:
    """Tests for CommandProcessor.parse_pipeline_steps()"""

    def test_parse_generate_step(self):
        """Test parsing a generate step"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "generate",
                    "parameters": {
                        "system_prompt": "You are helpful",
                        "user_prompt": "Generate text",
                        "num_sequences": 2,
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                }
            ]
        }

        # Act
        result = CommandProcessor.parse_pipeline_steps(config)

        # Assert
        assert len(result) == 1
        assert result[0].type == "generate"

    def test_parse_parse_step(self):
        """Test parsing a parse step"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "parse",
                    "parameters": {
                        "rules": [
                            {"name": "content", "mode": "KEYWORD", "pattern": "Content:"}
                        ],
                        "output_filter": "all"
                    }
                }
            ]
        }

        # Act
        result = CommandProcessor.parse_pipeline_steps(config)

        # Assert
        assert len(result) == 1
        assert result[0].type == "parse"

    def test_parse_verify_step(self):
        """Test parsing a verify step"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "verify",
                    "parameters": {
                        "methods": [
                            {
                                "mode": "cumulative",
                                "name": "check",
                                "system_prompt": "sys",
                                "user_prompt": "user",
                                "valid_responses": ["yes"],
                                "required_matches": 1
                            }
                        ],
                        "required_for_confirmed": 1,
                        "required_for_review": 0
                    }
                }
            ]
        }

        # Act
        result = CommandProcessor.parse_pipeline_steps(config)

        # Assert
        assert len(result) == 1
        assert result[0].type == "verify"

    def test_parse_invalid_step_type_raises_error(self):
        """Test that invalid step type raises ValueError"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "invalid_type",
                    "parameters": {}
                }
            ]
        }

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            CommandProcessor.parse_pipeline_steps(config)
        assert "invalid step type" in str(exc_info.value).lower()

    def test_parse_multiple_steps(self):
        """Test parsing multiple steps"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "generate",
                    "parameters": {
                        "system_prompt": "sys",
                        "user_prompt": "user",
                        "num_sequences": 1,
                        "max_tokens": 50,
                        "temperature": 0.5
                    }
                },
                {
                    "type": "parse",
                    "parameters": {
                        "rules": [{"name": "r", "mode": "KEYWORD", "pattern": "p"}],
                        "output_filter": "all"
                    }
                }
            ]
        }

        # Act
        result = CommandProcessor.parse_pipeline_steps(config)

        # Assert
        assert len(result) == 2
        assert result[0].type == "generate"
        assert result[1].type == "parse"

    def test_parse_step_with_uses_reference(self):
        """Test parsing step with uses_reference flag"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "generate",
                    "parameters": {
                        "system_prompt": "sys",
                        "user_prompt": "user",
                        "num_sequences": 1,
                        "max_tokens": 50,
                        "temperature": 0.5
                    },
                    "uses_reference": True,
                    "reference_step_numbers": [0, 1]
                }
            ]
        }

        # Act
        result = CommandProcessor.parse_pipeline_steps(config)

        # Assert
        assert result[0].uses_reference is True
        assert result[0].reference_step_numbers == [0, 1]


# =============================================================================
# Tests for OutputFormatter
# =============================================================================

class TestOutputFormatter:
    """Tests for OutputFormatter class"""

    def test_print_pipeline_results_with_entries(self, capsys):
        """Test printing pipeline results with entries"""
        # Arrange
        mock_response = Mock()
        mock_response.total_entries = 10
        mock_response.successful_entries = 8
        mock_response.failed_entries = 2
        mock_response.step_results = []
        mock_response.errors = []

        # Act
        OutputFormatter.print_pipeline_results(mock_response)

        # Assert
        captured = capsys.readouterr()
        assert "PIPELINE EXECUTION SUMMARY" in captured.out
        assert "10" in captured.out
        assert "80.0%" in captured.out

    def test_print_pipeline_results_with_errors(self, capsys):
        """Test printing pipeline results with errors"""
        # Arrange
        mock_error = Mock()
        mock_error.entry_index = 0
        mock_error.error_type = "TestError"
        mock_error.error_message = "Test error message"
        mock_error.entry_data = {"key": "value"}
        mock_error.traceback = "Line 1\nLine 2\nLine 3"

        mock_response = Mock()
        mock_response.total_entries = 1
        mock_response.successful_entries = 0
        mock_response.failed_entries = 1
        mock_response.step_results = []
        mock_response.errors = [mock_error]

        # Act
        OutputFormatter.print_pipeline_results(mock_response)

        # Assert
        captured = capsys.readouterr()
        assert "ERRORS" in captured.out
        assert "TestError" in captured.out

    def test_print_benchmark_results(self, capsys):
        """Test printing benchmark results"""
        # Arrange
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.85
        mock_metrics.precision = 0.90
        mock_metrics.recall = 0.80
        mock_metrics.f1_score = 0.8485
        mock_metrics.confusion_matrix = {
            'true_positive': 80,
            'false_positive': 10,
            'true_negative': 85,
            'false_negative': 25
        }

        # Act
        OutputFormatter.print_benchmark_results(mock_metrics)

        # Assert
        captured = capsys.readouterr()
        assert "Benchmark Results" in captured.out
        assert "85.00%" in captured.out
        assert "True Positives: 80" in captured.out


# =============================================================================
# Tests for Argument Parser
# =============================================================================

class TestArgumentParser:
    """Tests for argument parser setup"""

    def test_setup_arg_parser_creates_parser(self):
        """Test that setup_arg_parser creates a valid parser"""
        # Act
        parser = setup_arg_parser()

        # Assert
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_requires_command(self):
        """Test that parser requires a command"""
        # Arrange
        parser = setup_arg_parser()

        # Act & Assert
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_pipeline_command(self):
        """Test that parser accepts pipeline command"""
        # Arrange
        parser = setup_arg_parser()

        # Act
        args = parser.parse_args(["pipeline"])

        # Assert
        assert args.command == "pipeline"

    def test_parser_accepts_benchmark_command(self):
        """Test that parser accepts benchmark command"""
        # Arrange
        parser = setup_arg_parser()

        # Act
        args = parser.parse_args(["benchmark"])

        # Assert
        assert args.command == "benchmark"

    def test_pipeline_command_accepts_config_option(self):
        """Test that pipeline command accepts --config option"""
        # Arrange
        parser = setup_arg_parser()

        # Act
        args = parser.parse_args(["pipeline", "--config", "/path/to/config.json"])

        # Assert
        assert args.config == "/path/to/config.json"

    def test_pipeline_command_accepts_model_name_option(self):
        """Test that pipeline command accepts --pipeline-model-name option"""
        # Arrange
        parser = setup_arg_parser()

        # Act
        args = parser.parse_args(["pipeline", "--pipeline-model-name", "test-model"])

        # Assert
        assert args.pipeline_model_name == "test-model"

    def test_benchmark_command_accepts_entries_option(self):
        """Test that benchmark command accepts --entries option"""
        # Arrange
        parser = setup_arg_parser()

        # Act
        args = parser.parse_args(["benchmark", "--entries", "/path/to/entries.json"])

        # Assert
        assert args.entries == "/path/to/entries.json"


# =============================================================================
# Tests for Command Handlers (with mocks)
# =============================================================================

class TestHandlePipeline:
    """Tests for handle_pipeline function"""

    @patch('main.PipelineUseCase')
    @patch('main.FileRepository')
    @patch('main.CommandProcessor.load_json_file')
    @patch('builtins.open', create=True)
    @patch('os.path.exists')
    def test_handle_pipeline_success(
        self,
        mock_exists,
        mock_open,
        mock_load_json,
        mock_file_repo,
        mock_use_case
    ):
        """Test successful pipeline execution"""
        # Arrange
        mock_exists.return_value = True
        mock_load_json.return_value = {
            "steps": [
                {
                    "type": "generate",
                    "parameters": {
                        "system_prompt": "sys",
                        "user_prompt": "user",
                        "num_sequences": 1,
                        "max_tokens": 50,
                        "temperature": 0.5
                    }
                }
            ],
            "global_references": {}
        }
        mock_open.return_value.__enter__ = Mock(return_value=Mock(
            read=Mock(return_value='[]')
        ))
        mock_open.return_value.__exit__ = Mock(return_value=False)

        # Configure mock file read
        mock_file = MagicMock()
        mock_file.read.return_value = '[]'
        mock_open.return_value.__enter__.return_value = mock_file

        mock_response = Mock()
        mock_response.step_results = []
        mock_response.verification_references = {}
        mock_response.total_entries = 0
        mock_response.successful_entries = 0
        mock_response.failed_entries = 0
        mock_response.errors = []

        mock_use_case_instance = Mock()
        mock_use_case_instance.execute_with_references.return_value = mock_response
        mock_use_case.return_value = mock_use_case_instance

        args = argparse.Namespace(
            config="test_config.json",
            pipeline_model_name="test-model"
        )

        # Act & Assert - should not raise
        with patch('json.load', return_value=[]):
            handle_pipeline(args)

    @patch('main.CommandProcessor.load_json_file')
    def test_handle_pipeline_config_not_found(self, mock_load_json):
        """Test pipeline with missing config file"""
        # Arrange
        mock_load_json.side_effect = FileNotFoundError("Config not found")

        args = argparse.Namespace(
            config="nonexistent.json",
            pipeline_model_name="test-model"
        )

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            handle_pipeline(args)


class TestHandleBenchmark:
    """Tests for handle_benchmark function"""

    @patch('main.BenchmarkUseCase')
    @patch('main.FileRepository')
    @patch('main.CommandProcessor.load_json_file')
    def test_handle_benchmark_success(
        self,
        mock_load_json,
        mock_file_repo,
        mock_use_case
    ):
        """Test successful benchmark execution"""
        # Arrange
        mock_load_json.side_effect = [
            # Config data
            {
                "model_name": "test-model",
                "label_key": "expected",
                "label_value": "positive",
                "steps": [
                    {
                        "type": "generate",
                        "parameters": {
                            "system_prompt": "sys",
                            "user_prompt": "user",
                            "num_sequences": 1,
                            "max_tokens": 50,
                            "temperature": 0.5
                        }
                    }
                ]
            },
            # Entries data
            [
                {"input": "test1", "expected": "positive"},
                {"input": "test2", "expected": "negative"}
            ]
        ]

        mock_metrics = Mock()
        mock_metrics.to_dict.return_value = {}
        mock_metrics.misclassified = []
        mock_metrics.accuracy = 0.8
        mock_metrics.precision = 0.8
        mock_metrics.recall = 0.8
        mock_metrics.f1_score = 0.8
        mock_metrics.confusion_matrix = {
            'true_positive': 1,
            'false_positive': 0,
            'true_negative': 1,
            'false_negative': 0
        }

        mock_use_case_instance = Mock()
        mock_use_case_instance.run_benchmark.return_value = mock_metrics
        mock_use_case.return_value = mock_use_case_instance

        args = argparse.Namespace(
            config="config.json",
            entries="entries.json"
        )

        # Act & Assert - should not raise
        handle_benchmark(args)


# =============================================================================
# Tests for main() function
# =============================================================================

class TestMainFunction:
    """Tests for the main() function"""

    @patch('main.handle_pipeline')
    @patch('sys.argv', ['main.py', 'pipeline'])
    def test_main_calls_pipeline_handler(self, mock_handle_pipeline):
        """Test that main() routes to pipeline handler"""
        # Act
        main()

        # Assert
        mock_handle_pipeline.assert_called_once()

    @patch('main.handle_benchmark')
    @patch('sys.argv', ['main.py', 'benchmark'])
    def test_main_calls_benchmark_handler(self, mock_handle_benchmark):
        """Test that main() routes to benchmark handler"""
        # Act
        main()

        # Assert
        mock_handle_benchmark.assert_called_once()

    @patch('main.handle_pipeline')
    @patch('sys.argv', ['main.py', 'pipeline'])
    def test_main_exits_on_error(self, mock_handle_pipeline):
        """Test that main() exits with code 1 on error"""
        # Arrange
        mock_handle_pipeline.side_effect = Exception("Test error")

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_parse_empty_rules_list(self):
        """Test parsing empty rules list"""
        # Act
        result = CommandProcessor.parse_rules([])

        # Assert
        assert result == []

    def test_parse_empty_verification_methods_list(self):
        """Test parsing empty verification methods list"""
        # Act
        result = CommandProcessor.parse_verification_methods([])

        # Assert
        assert result == []

    def test_parse_pipeline_steps_with_llm_config(self):
        """Test parsing step with custom llm_config"""
        # Arrange
        config = {
            "steps": [
                {
                    "type": "generate",
                    "parameters": {
                        "system_prompt": "sys",
                        "user_prompt": "user",
                        "num_sequences": 1,
                        "max_tokens": 50,
                        "temperature": 0.5
                    },
                    "llm_config": {"custom_key": "custom_value"}
                }
            ]
        }

        # Act
        result = CommandProcessor.parse_pipeline_steps(config)

        # Assert
        assert result[0].llm_config == {"custom_key": "custom_value"}
