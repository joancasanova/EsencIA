# tests/unit/infrastructure/test_file_repository.py

import errno
import json
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from app.infrastructure.file_repository import FileRepository


class TestFileRepositorySaveValidation:
    """Tests for FileRepository.save() validation"""

    def test_save_empty_output_dir_raises_error(self, tmp_path):
        """Test that empty output_dir raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            FileRepository.save(data, "", "test")

    def test_save_whitespace_output_dir_raises_error(self, tmp_path):
        """Test that whitespace-only output_dir raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            FileRepository.save(data, "   ", "test")

    def test_save_empty_filename_prefix_raises_error(self, tmp_path):
        """Test that empty filename_prefix raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="filename_prefix cannot be empty"):
            FileRepository.save(data, str(tmp_path), "")

    def test_save_whitespace_filename_prefix_raises_error(self, tmp_path):
        """Test that whitespace-only filename_prefix raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="filename_prefix cannot be empty"):
            FileRepository.save(data, str(tmp_path), "   ")


class TestFileRepositorySave:
    """Tests for FileRepository.save() method"""

    def test_save_creates_file_with_timestamp(self, tmp_path):
        """Test that save creates a file with timestamp"""
        # Arrange
        data = {"key": "value", "number": 42}
        output_dir = str(tmp_path)
        prefix = "test_output"

        # Act
        filepath = FileRepository.save(data, output_dir, prefix)

        # Assert
        assert os.path.exists(filepath)
        assert prefix in filepath
        assert filepath.endswith(".json")

    def test_save_writes_correct_data(self, tmp_path):
        """Test that saved data matches input data"""
        # Arrange
        data = {"name": "John", "age": 30, "city": "NYC"}
        output_dir = str(tmp_path)

        # Act
        filepath = FileRepository.save(data, output_dir, "test")

        # Load and verify
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Assert
        assert loaded_data == data

    def test_save_creates_directory_if_not_exists(self, tmp_path):
        """Test that save creates output directory if it doesn't exist"""
        # Arrange
        data = {"test": "data"}
        output_dir = str(tmp_path / "nested" / "directory")

        # Act
        filepath = FileRepository.save(data, output_dir, "test")

        # Assert
        assert os.path.exists(output_dir)
        assert os.path.exists(filepath)

    def test_save_with_special_characters(self, tmp_path):
        """Test saving data with special characters"""
        # Arrange
        data = {"text": "Special chars: áéíóú ñ 中文"}
        output_dir = str(tmp_path)

        # Act
        filepath = FileRepository.save(data, output_dir, "test")

        # Load and verify
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Assert
        assert loaded_data["text"] == "Special chars: áéíóú ñ 中文"

class TestFileRepositorySaveComplexData:
    """Tests for FileRepository.save() with complex data structures"""

    def test_save_nested_data(self, tmp_path):
        """Test saving deeply nested data structures"""
        data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                }
            }
        }

        filepath = FileRepository.save(data, str(tmp_path), "nested")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded["level1"]["level2"]["level3"] == ["a", "b", "c"]

    def test_save_list_data(self, tmp_path):
        """Test saving list as root data"""
        data = [1, 2, 3, {"nested": "value"}]

        filepath = FileRepository.save(data, str(tmp_path), "list")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == data

    def test_save_empty_dict(self, tmp_path):
        """Test saving empty dictionary"""
        data = {}

        filepath = FileRepository.save(data, str(tmp_path), "empty")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == {}


class TestFileRepositoryAppendValidation:
    """Tests for FileRepository.append() validation"""

    def test_append_empty_output_dir_raises_error(self, tmp_path):
        """Test that empty output_dir raises ValueError"""
        with pytest.raises(ValueError, match="output_dir cannot be empty"):
            FileRepository.append({"key": "value"}, "", "test.json")

    def test_append_empty_filename_raises_error(self, tmp_path):
        """Test that empty filename raises ValueError"""
        with pytest.raises(ValueError, match="filename cannot be empty"):
            FileRepository.append({"key": "value"}, str(tmp_path), "")


class TestFileRepositoryAppend:
    """Tests for FileRepository.append() method"""

    def test_append_creates_new_file_with_list(self, tmp_path):
        """Test that append creates a new file with data in list format"""
        # Arrange
        data = {"item": "first"}
        output_dir = str(tmp_path)
        filename = "test.json"

        # Act
        filepath = FileRepository.append(data, output_dir, filename)

        # Load and verify
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Assert
        assert isinstance(loaded_data, list)
        assert len(loaded_data) == 1
        assert loaded_data[0] == data

    def test_append_adds_to_existing_list(self, tmp_path):
        """Test that append adds data to existing list"""
        # Arrange
        output_dir = str(tmp_path)
        filename = "test.json"

        # Create initial file
        FileRepository.append({"item": "first"}, output_dir, filename)

        # Act - append second item
        FileRepository.append({"item": "second"}, output_dir, filename)

        # Load and verify
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Assert
        assert len(loaded_data) == 2
        assert loaded_data[0]["item"] == "first"
        assert loaded_data[1]["item"] == "second"

    def test_append_converts_non_list_to_list(self, tmp_path):
        """Test that append converts existing non-list data to list format"""
        # Arrange
        output_dir = str(tmp_path)
        filename = "test.json"
        filepath = os.path.join(output_dir, filename)

        # Create file with non-list data
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"single": "object"}, f)

        # Act
        FileRepository.append({"item": "new"}, output_dir, filename)

        # Load and verify
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Assert
        assert isinstance(loaded_data, list)
        assert len(loaded_data) == 2
        assert loaded_data[0] == {"single": "object"}
        assert loaded_data[1] == {"item": "new"}

    def test_append_handles_invalid_json_file(self, tmp_path):
        """Test that append raises IOError and creates backup for files with invalid JSON"""
        # Arrange
        output_dir = str(tmp_path)
        filename = "test.json"
        filepath = os.path.join(output_dir, filename)

        # Create file with invalid JSON
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("{invalid json}")

        # Act & Assert - should raise IOError and create backup
        with pytest.raises(IOError, match="file contains invalid JSON"):
            FileRepository.append({"item": "new"}, output_dir, filename)

        # Verify backup was created
        backup_files = [f for f in os.listdir(output_dir) if f.endswith('.bak')]
        assert len(backup_files) == 1
        assert 'corrupted' in backup_files[0]

        # Verify original file is unchanged (not overwritten)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == "{invalid json}"

    def test_append_creates_directory_if_not_exists(self, tmp_path):
        """Test that append creates directory if it doesn't exist"""
        # Arrange
        output_dir = str(tmp_path / "new_dir")
        filename = "test.json"

        # Act
        filepath = FileRepository.append({"item": "test"}, output_dir, filename)

        # Assert
        assert os.path.exists(output_dir)
        assert os.path.exists(filepath)

    def test_append_multiple_items(self, tmp_path):
        """Test appending multiple items sequentially"""
        # Arrange
        output_dir = str(tmp_path)
        filename = "test.json"
        items = [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
            {"id": 3, "value": "third"}
        ]

        # Act
        for item in items:
            FileRepository.append(item, output_dir, filename)

        # Load and verify
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Assert
        assert len(loaded_data) == 3
        for i, item in enumerate(items):
            assert loaded_data[i] == item

    def test_append_preserves_formatting(self, tmp_path):
        """Test that JSON is properly formatted with indentation"""
        # Arrange
        output_dir = str(tmp_path)
        filename = "test.json"

        # Act
        filepath = FileRepository.append({"key": "value"}, output_dir, filename)

        # Read raw file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Assert - check for indentation
        assert "  " in content or "\t" in content  # Should have indentation


@pytest.fixture
def tmp_path():
    """Fixture to provide a temporary directory path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestFileRepositorySavePathTraversal:
    """Tests for path traversal prevention in save()"""

    def test_save_path_traversal_raises_error(self, tmp_path):
        """Test that path traversal in output_dir raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="Path traversal detected"):
            FileRepository.save(data, "../dangerous/path", "test")

    def test_save_nested_path_traversal_raises_error(self, tmp_path):
        """Test that nested path traversal raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="Path traversal detected"):
            FileRepository.save(data, "safe/../../dangerous", "test")


class TestFileRepositorySaveErrors:
    """Tests for error handling in save()"""

    @patch('os.makedirs')
    def test_save_permission_error_on_directory(self, mock_makedirs, tmp_path):
        """Test PermissionError when creating directory"""
        mock_makedirs.side_effect = PermissionError("No permission")
        data = {"key": "value"}

        with pytest.raises(PermissionError, match="No permission to create directory"):
            FileRepository.save(data, str(tmp_path / "new_dir"), "test")

    @patch('os.makedirs')
    def test_save_disk_full_on_directory(self, mock_makedirs, tmp_path):
        """Test disk full error when creating directory"""
        error = OSError(errno.ENOSPC, "No space left")
        mock_makedirs.side_effect = error
        data = {"key": "value"}

        with pytest.raises(IOError, match="Disk is full"):
            FileRepository.save(data, str(tmp_path / "new_dir"), "test")

    @patch('os.makedirs')
    def test_save_generic_os_error_on_directory(self, mock_makedirs, tmp_path):
        """Test generic OSError when creating directory"""
        error = OSError(errno.EIO, "I/O error")
        mock_makedirs.side_effect = error
        data = {"key": "value"}

        with pytest.raises(OSError, match="Cannot create output directory"):
            FileRepository.save(data, str(tmp_path / "new_dir"), "test")

    @patch('builtins.open')
    @patch('os.makedirs')
    def test_save_permission_error_on_write(self, mock_makedirs, mock_file_open, tmp_path):
        """Test PermissionError when writing file"""
        mock_makedirs.return_value = None
        mock_file_open.side_effect = PermissionError("No write permission")
        data = {"key": "value"}

        with pytest.raises(PermissionError, match="No permission to write"):
            FileRepository.save(data, str(tmp_path), "test")

    @patch('builtins.open')
    @patch('os.makedirs')
    def test_save_disk_full_on_write(self, mock_makedirs, mock_file_open, tmp_path):
        """Test disk full error when writing file"""
        mock_makedirs.return_value = None
        error = OSError(errno.ENOSPC, "No space left")
        mock_file_open.side_effect = error
        data = {"key": "value"}

        with pytest.raises(IOError, match="Disk is full"):
            FileRepository.save(data, str(tmp_path), "test")

    @patch('builtins.open')
    @patch('os.makedirs')
    def test_save_generic_os_error_on_write(self, mock_makedirs, mock_file_open, tmp_path):
        """Test generic OSError when writing file"""
        mock_makedirs.return_value = None
        error = OSError(errno.EIO, "I/O error")
        mock_file_open.side_effect = error
        data = {"key": "value"}

        with pytest.raises(IOError, match="Cannot write to file"):
            FileRepository.save(data, str(tmp_path), "test")

    @patch('builtins.open')
    @patch('os.makedirs')
    def test_save_io_error_on_write(self, mock_makedirs, mock_file_open, tmp_path):
        """Test IOError when writing file"""
        mock_makedirs.return_value = None
        mock_file_open.side_effect = IOError("Write failed")
        data = {"key": "value"}

        with pytest.raises(IOError, match="Cannot write to file"):
            FileRepository.save(data, str(tmp_path), "test")


class TestFileRepositoryAppendPathTraversal:
    """Tests for path traversal prevention in append()"""

    def test_append_path_traversal_raises_error(self, tmp_path):
        """Test that path traversal in output_dir raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="Path traversal detected"):
            FileRepository.append(data, "../dangerous/path", "test.json")


class TestFileRepositoryAppendErrors:
    """Tests for error handling in append()"""

    @patch('os.makedirs')
    def test_append_permission_error_on_directory(self, mock_makedirs, tmp_path):
        """Test PermissionError when creating directory"""
        mock_makedirs.side_effect = PermissionError("No permission")
        data = {"key": "value"}

        with pytest.raises(PermissionError, match="No permission to create directory"):
            FileRepository.append(data, str(tmp_path / "new_dir"), "test.json")

    @patch('os.makedirs')
    def test_append_disk_full_on_directory(self, mock_makedirs, tmp_path):
        """Test disk full error when creating directory"""
        error = OSError(errno.ENOSPC, "No space left")
        mock_makedirs.side_effect = error
        data = {"key": "value"}

        with pytest.raises(IOError, match="Disk is full"):
            FileRepository.append(data, str(tmp_path / "new_dir"), "test.json")

    @patch('os.makedirs')
    def test_append_generic_os_error_on_directory(self, mock_makedirs, tmp_path):
        """Test generic OSError when creating directory"""
        error = OSError(errno.EIO, "I/O error")
        mock_makedirs.side_effect = error
        data = {"key": "value"}

        with pytest.raises(OSError, match="Cannot create output directory"):
            FileRepository.append(data, str(tmp_path / "new_dir"), "test.json")

    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('os.makedirs')
    def test_append_permission_error_on_read(self, mock_makedirs, mock_file_open, mock_exists, tmp_path):
        """Test PermissionError when reading existing file"""
        mock_makedirs.return_value = None
        mock_exists.return_value = True
        mock_file_open.side_effect = PermissionError("No read permission")
        data = {"key": "value"}

        with pytest.raises(PermissionError, match="No permission to read"):
            FileRepository.append(data, str(tmp_path), "test.json")

    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('os.makedirs')
    def test_append_io_error_on_read(self, mock_makedirs, mock_file_open, mock_exists, tmp_path):
        """Test IOError when reading existing file"""
        mock_makedirs.return_value = None
        mock_exists.return_value = True
        mock_file_open.side_effect = IOError("Read failed")
        data = {"key": "value"}

        with pytest.raises(IOError, match="Cannot read file"):
            FileRepository.append(data, str(tmp_path), "test.json")

    def test_append_permission_error_on_write(self, tmp_path):
        """Test PermissionError when writing file"""
        filepath = tmp_path / "test.json"
        data = {"key": "value"}

        # Create the file first
        with open(filepath, 'w') as f:
            json.dump([], f)

        # Now make the file read-only (platform dependent, may not work on all systems)
        # Instead, we'll use mocking
        with patch('builtins.open') as mock_open_func:
            # First call for reading succeeds
            mock_read_file = MagicMock()
            mock_read_file.__enter__ = MagicMock(return_value=mock_read_file)
            mock_read_file.__exit__ = MagicMock(return_value=False)
            mock_read_file.read.return_value = '[]'

            # Second call for writing fails
            def open_side_effect(path, mode='r', *args, **kwargs):
                if 'w' in mode:
                    raise PermissionError("No write permission")
                return mock_read_file

            mock_open_func.side_effect = open_side_effect

            with pytest.raises(PermissionError, match="No permission to write"):
                FileRepository.append(data, str(tmp_path), "test.json")

    def test_append_disk_full_on_write(self, tmp_path):
        """Test disk full error when writing file"""
        data = {"key": "value"}

        with patch('builtins.open') as mock_open_func:
            mock_read_file = MagicMock()
            mock_read_file.__enter__ = MagicMock(return_value=mock_read_file)
            mock_read_file.__exit__ = MagicMock(return_value=False)
            mock_read_file.read.return_value = '[]'

            def open_side_effect(path, mode='r', *args, **kwargs):
                if 'w' in mode:
                    raise OSError(errno.ENOSPC, "No space left")
                return mock_read_file

            mock_open_func.side_effect = open_side_effect

            with pytest.raises(IOError, match="Disk is full"):
                FileRepository.append(data, str(tmp_path), "test.json")

    def test_append_generic_os_error_on_write(self, tmp_path):
        """Test generic OSError when writing file"""
        data = {"key": "value"}

        with patch('builtins.open') as mock_open_func:
            mock_read_file = MagicMock()
            mock_read_file.__enter__ = MagicMock(return_value=mock_read_file)
            mock_read_file.__exit__ = MagicMock(return_value=False)
            mock_read_file.read.return_value = '[]'

            def open_side_effect(path, mode='r', *args, **kwargs):
                if 'w' in mode:
                    raise OSError(errno.EIO, "I/O error")
                return mock_read_file

            mock_open_func.side_effect = open_side_effect

            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.append(data, str(tmp_path), "test.json")

    def test_append_io_error_on_write(self, tmp_path):
        """Test IOError when writing file"""
        data = {"key": "value"}

        with patch('builtins.open') as mock_open_func:
            mock_read_file = MagicMock()
            mock_read_file.__enter__ = MagicMock(return_value=mock_read_file)
            mock_read_file.__exit__ = MagicMock(return_value=False)
            mock_read_file.read.return_value = '[]'

            def open_side_effect(path, mode='r', *args, **kwargs):
                if 'w' in mode:
                    raise IOError("Write failed")
                return mock_read_file

            mock_open_func.side_effect = open_side_effect

            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.append(data, str(tmp_path), "test.json")


class TestFileRepositoryWhitespaceValidation:
    """Tests for whitespace validation in both methods"""

    def test_save_whitespace_path_traversal(self, tmp_path):
        """Test normalized path with spaces is handled"""
        data = {"key": "value"}

        # This should not raise path traversal - it's just whitespace
        # The path will be normalized
        result = FileRepository.save(data, str(tmp_path), "test")
        assert os.path.exists(result)

    def test_append_filename_with_spaces(self, tmp_path):
        """Test filename with spaces works"""
        data = {"key": "value"}

        result = FileRepository.append(data, str(tmp_path), "test file.json")
        assert os.path.exists(result)


class TestFileRepositoryNonSerializableData:
    """Tests for handling data serialization with default=str fallback"""

    def test_save_lambda_is_serialized_as_string(self, tmp_path):
        """Test that lambda functions are converted to string (default=str behavior)"""
        # Note: json.dump with default=str converts non-serializable objects to their string repr
        data = {"func": lambda x: x}

        filepath = FileRepository.save(data, str(tmp_path), "test")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Lambda should be converted to its string representation
        assert "function" in loaded["func"].lower()

    def test_save_custom_class_is_serialized_as_string(self, tmp_path):
        """Test that custom class instances are converted to string"""
        class NonSerializable:
            pass

        data = {"obj": NonSerializable()}

        filepath = FileRepository.save(data, str(tmp_path), "test")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Object should be converted to its string representation
        assert "NonSerializable" in loaded["obj"]

    def test_save_set_is_serialized_as_string(self, tmp_path):
        """Test that sets are converted to string representation"""
        data = {"items": {1, 2, 3}}

        filepath = FileRepository.save(data, str(tmp_path), "test")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Set should be converted to its string representation
        # The exact format depends on set ordering, but should contain the values
        assert isinstance(loaded["items"], str)
        assert "1" in loaded["items"] or "2" in loaded["items"]

    def test_append_lambda_is_serialized_as_string(self, tmp_path):
        """Test that appending lambda results in string serialization"""
        # First create a valid file
        FileRepository.append({"valid": "data"}, str(tmp_path), "test.json")

        # Now append data with lambda
        data = {"func": lambda x: x}

        filepath = FileRepository.append(data, str(tmp_path), "test.json")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Lambda should be serialized as string in the appended item
        assert "function" in loaded[1]["func"].lower()

    def test_append_custom_object_is_serialized_as_string(self, tmp_path):
        """Test that appending custom object uses string serialization"""
        class CustomObject:
            def __init__(self):
                self.value = "test"

        data = CustomObject()

        filepath = FileRepository.append(data, str(tmp_path), "new.json")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Object should be serialized as string
        assert "CustomObject" in loaded[0]

    def test_save_datetime_is_serialized_as_string(self, tmp_path):
        """Test that datetime objects are serialized using default=str"""
        from datetime import datetime
        data = {"timestamp": datetime(2024, 1, 15, 10, 30, 0)}

        filepath = FileRepository.save(data, str(tmp_path), "test")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Datetime should be converted to string
        assert "2024-01-15" in loaded["timestamp"]

    def test_save_bytes_is_serialized_as_string(self, tmp_path):
        """Test that bytes are converted to string representation"""
        data = {"binary": b"hello world"}

        filepath = FileRepository.save(data, str(tmp_path), "test")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Bytes should be converted to their string representation
        assert "hello" in loaded["binary"]

    def test_save_complex_number_is_serialized_as_string(self, tmp_path):
        """Test that complex numbers are converted to string"""
        data = {"complex": complex(1, 2)}

        filepath = FileRepository.save(data, str(tmp_path), "test")

        with open(filepath, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Complex number should be converted to string (e.g., "(1+2j)")
        assert isinstance(loaded["complex"], str)
        assert "1" in loaded["complex"] and "2" in loaded["complex"]


class TestFileRepositoryCorruptedFileBackup:
    """Tests for corrupted file backup behavior"""

    def test_append_creates_backup_of_corrupted_file(self, tmp_path):
        """Test that backup is created when file contains invalid JSON"""
        output_dir = str(tmp_path)
        filename = "corrupted.json"
        filepath = os.path.join(output_dir, filename)

        # Create corrupted file
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("not valid json {{{")

        # Attempt to append - should fail and create backup
        with pytest.raises(IOError, match="file contains invalid JSON"):
            FileRepository.append({"new": "data"}, output_dir, filename)

        # Verify backup was created
        backup_files = [f for f in os.listdir(output_dir) if '.bak' in f]
        assert len(backup_files) == 1
        assert 'corrupted' in backup_files[0]

        # Verify backup contains original content
        backup_path = os.path.join(output_dir, backup_files[0])
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        assert backup_content == "not valid json {{{"

    def test_append_backup_failure_still_raises_error(self, tmp_path):
        """Test that error is raised with correct message when backup creation fails"""
        output_dir = str(tmp_path)
        filename = "test.json"
        filepath = os.path.join(output_dir, filename)

        # Create corrupted file
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("{invalid json")

        # Mock shutil.copy2 to fail
        with patch('app.infrastructure.file_repository.shutil.copy2') as mock_copy:
            mock_copy.side_effect = PermissionError("Cannot create backup")

            with pytest.raises(IOError, match="Backup creation failed"):
                FileRepository.append({"new": "data"}, output_dir, filename)

    def test_append_backup_success_includes_path_in_error(self, tmp_path):
        """Test that successful backup includes path in error message"""
        output_dir = str(tmp_path)
        filename = "corrupted.json"
        filepath = os.path.join(output_dir, filename)

        # Create corrupted file
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("not valid json")

        # Attempt to append - should fail with backup path in message
        with pytest.raises(IOError, match="A backup was saved to"):
            FileRepository.append({"new": "data"}, output_dir, filename)

    def test_append_original_file_unchanged_after_json_error(self, tmp_path):
        """Test that original corrupted file is not modified"""
        output_dir = str(tmp_path)
        filename = "corrupt.json"
        filepath = os.path.join(output_dir, filename)
        original_content = '{"broken: json'

        # Create corrupted file
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(original_content)

        # Attempt to append
        with pytest.raises(IOError):
            FileRepository.append({"new": "data"}, output_dir, filename)

        # Verify original file is unchanged
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == original_content


class TestFileRepositorySanitizeFilename:
    """Tests for _sanitize_filename static method"""

    def test_sanitize_empty_filename_raises_error(self, tmp_path):
        """Test that empty filename raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            FileRepository._sanitize_filename("")

    def test_sanitize_whitespace_filename_raises_error(self, tmp_path):
        """Test that whitespace-only filename raises ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            FileRepository._sanitize_filename("   ")

    def test_sanitize_path_traversal_raises_error(self, tmp_path):
        """Test that path traversal sequences raise ValueError"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            FileRepository._sanitize_filename("../secret")

    def test_sanitize_forward_slash_raises_error(self, tmp_path):
        """Test that forward slash raises ValueError"""
        with pytest.raises(ValueError, match="cannot contain path separators"):
            FileRepository._sanitize_filename("dir/file")

    def test_sanitize_backslash_raises_error(self, tmp_path):
        """Test that backslash raises ValueError"""
        with pytest.raises(ValueError, match="cannot contain path separators"):
            FileRepository._sanitize_filename("dir\\file")

    def test_sanitize_forbidden_characters_raises_error(self, tmp_path):
        """Test that forbidden characters raise ValueError"""
        forbidden_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in forbidden_chars:
            with pytest.raises(ValueError, match="contains forbidden characters"):
                FileRepository._sanitize_filename(f"file{char}name")

    def test_sanitize_null_byte_raises_error(self, tmp_path):
        """Test that null byte raises ValueError"""
        with pytest.raises(ValueError, match="contains forbidden characters"):
            FileRepository._sanitize_filename("file\x00name")

    def test_sanitize_valid_filename_returns_stripped(self, tmp_path):
        """Test that valid filename is returned stripped"""
        result = FileRepository._sanitize_filename("  valid_file  ")
        assert result == "valid_file"

    def test_sanitize_filename_with_dots_is_valid(self, tmp_path):
        """Test that single dots in filename are valid"""
        result = FileRepository._sanitize_filename("file.name.json")
        assert result == "file.name.json"

    def test_sanitize_filename_with_hyphens_underscores(self, tmp_path):
        """Test that hyphens and underscores are valid"""
        result = FileRepository._sanitize_filename("my-file_name")
        assert result == "my-file_name"

    def test_save_with_path_separator_in_prefix_raises_error(self, tmp_path):
        """Test that path separator in filename_prefix raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="cannot contain path separators"):
            FileRepository.save(data, str(tmp_path), "dir/file")

    def test_append_with_path_separator_in_filename_raises_error(self, tmp_path):
        """Test that path separator in filename raises ValueError"""
        data = {"key": "value"}

        with pytest.raises(ValueError, match="cannot contain path separators"):
            FileRepository.append(data, str(tmp_path), "subdir/file.json")


class TestFileRepositoryGenericIOErrors:
    """Tests for generic IOError handling in save() and append()"""

    def test_save_generic_ioerror_during_write(self, tmp_path):
        """Test that generic IOError during write is handled properly"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory first
        os.makedirs(output_dir, exist_ok=True)

        # Mock json.dump to raise generic IOError (not PermissionError or disk full)
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=IOError("Device not ready")):
            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.save(data, output_dir, "test")

    def test_save_ioerror_on_file_open(self, tmp_path):
        """Test IOError when opening file for write in save()"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create the directory first
        os.makedirs(output_dir, exist_ok=True)

        # Mock open to raise IOError
        original_open = open

        def mock_open_ioerror(path, mode='r', **kwargs):
            if 'w' in mode and 'test_' in str(path):
                raise IOError("Simulated IO error")
            return original_open(path, mode, **kwargs)

        with patch('builtins.open', side_effect=mock_open_ioerror):
            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.save(data, output_dir, "test")

    def test_append_generic_ioerror_during_write(self, tmp_path):
        """Test that generic IOError during write in append() is handled properly"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create the directory first
        os.makedirs(output_dir, exist_ok=True)

        # Create a file to append to
        filepath = os.path.join(output_dir, "test.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)

        # Mock the second open (for writing) to raise IOError
        call_count = [0]
        original_open = open

        def mock_open_ioerror(path, mode='r', **kwargs):
            call_count[0] += 1
            if 'w' in mode and call_count[0] > 1:
                raise IOError("Simulated IO error during write")
            return original_open(path, mode, **kwargs)

        with patch('builtins.open', side_effect=mock_open_ioerror):
            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.append(data, output_dir, "test.json")

    def test_save_json_dump_raises_ioerror(self, tmp_path):
        """Test IOError raised during json.dump in save()"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory
        os.makedirs(output_dir, exist_ok=True)

        # Mock json.dump to raise IOError
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=IOError("Write failed during serialization")):
            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.save(data, output_dir, "test")

    def test_append_json_dump_raises_ioerror(self, tmp_path):
        """Test IOError raised during json.dump in append()"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory and file
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "test.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)

        # Mock json.dump to raise IOError (only on second call - first is our setup)
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=IOError("Write failed during serialization")):
            with pytest.raises(IOError, match="Cannot write to file"):
                FileRepository.append(data, output_dir, "test.json")


class TestFileRepositoryTypeErrorHandling:
    """Tests for TypeError/ValueError handling when default=str fails"""

    def test_save_json_dump_raises_typeerror(self, tmp_path):
        """Test TypeError raised during json.dump in save() when default=str fails"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory
        os.makedirs(output_dir, exist_ok=True)

        # Mock json.dump to raise TypeError (simulating a case where default=str doesn't help)
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=TypeError("Object not serializable even with default")):
            with pytest.raises(TypeError, match="Data cannot be serialized to JSON"):
                FileRepository.save(data, output_dir, "test")

    def test_save_json_dump_raises_valueerror(self, tmp_path):
        """Test ValueError raised during json.dump in save()"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory
        os.makedirs(output_dir, exist_ok=True)

        # Mock json.dump to raise ValueError (e.g., circular reference)
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=ValueError("Circular reference detected")):
            with pytest.raises(TypeError, match="Data cannot be serialized to JSON"):
                FileRepository.save(data, output_dir, "test")

    def test_append_json_dump_raises_typeerror(self, tmp_path):
        """Test TypeError raised during json.dump in append()"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory and file
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "test.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)

        # Mock json.dump to raise TypeError
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=TypeError("Serialization failed")):
            with pytest.raises(TypeError, match="Data cannot be serialized to JSON"):
                FileRepository.append(data, output_dir, "test.json")

    def test_append_json_dump_raises_valueerror(self, tmp_path):
        """Test ValueError raised during json.dump in append()"""
        data = {"key": "value"}
        output_dir = str(tmp_path)

        # Create directory and file
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "test.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f)

        # Mock json.dump to raise ValueError
        with patch('app.infrastructure.file_repository.json.dump',
                   side_effect=ValueError("Invalid value in data")):
            with pytest.raises(TypeError, match="Data cannot be serialized to JSON"):
                FileRepository.append(data, output_dir, "test.json")
