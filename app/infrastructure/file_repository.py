# infrastructure/file_repository.py

import errno
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

class FileRepository:
    """
    Handles file system operations for data persistence.

    Provides methods for saving data with timestamped filenames
    and appending data to existing files.
    """

    @staticmethod
    def _sanitize_filename(filename: str, param_name: str = "filename") -> str:
        """
        Sanitizes filenames to prevent path traversal attacks.

        Args:
            filename: The filename or prefix to sanitize
            param_name: Name of the parameter for error messages

        Returns:
            str: The sanitized filename

        Raises:
            ValueError: If filename contains dangerous characters or patterns
        """
        if not filename or not filename.strip():
            raise ValueError(f"{param_name} cannot be empty")

        filename = filename.strip()

        # Reject path traversal sequences
        if ".." in filename:
            raise ValueError(f"Path traversal detected in {param_name}: {filename}")

        # Reject path separators
        if "/" in filename or "\\" in filename:
            raise ValueError(f"{param_name} cannot contain path separators: {filename}")

        # Reject dangerous characters (Windows and Unix)
        if re.search(r'[<>:"|?*\x00-\x1f]', filename):
            raise ValueError(f"{param_name} contains forbidden characters: {filename}")

        return filename

    @staticmethod
    def save(data: dict, output_dir: str, filename_prefix: str) -> str:
        """
        Saves data to a new timestamped JSON file.

        Args:
            data: Dictionary containing data to save
            output_dir: Target directory for the file
            filename_prefix: Base name for the file (timestamp will be added)

        Returns:
            str: Full path to the created file

        Raises:
            ValueError: If parameters are invalid
            OSError: If directory creation fails
            PermissionError: If lacking write permissions
            IOError: If disk is full or write operation fails
            TypeError: If data is not JSON-serializable

        Example:
            save(data, "results", "experiment") -> "results/experiment_20230101_123456.json"
        """
        # Validate inputs
        if not output_dir or not output_dir.strip():
            raise ValueError("output_dir cannot be empty")

        # Sanitize filename_prefix to prevent path traversal
        filename_prefix = FileRepository._sanitize_filename(filename_prefix, "filename_prefix")

        # Sanitize path to prevent path traversal (check BEFORE normalization)
        if ".." in output_dir:
            raise ValueError(f"Path traversal detected in output_dir: {output_dir}")
        output_dir = os.path.normpath(output_dir)

        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating directory {output_dir}: {e}")
            raise PermissionError(
                f"No permission to create directory '{output_dir}'. "
                f"Check file system permissions."
            ) from e
        except OSError as e:
            if e.errno == errno.ENOSPC:
                logger.error(f"Disk full while creating directory {output_dir}: {e}")
                raise IOError("Disk is full. Free up space and try again.") from e
            logger.error(f"Failed to create directory {output_dir}: {e}")
            raise OSError(f"Cannot create output directory '{output_dir}': {e}") from e

        # Generate filepath
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Write data to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        except PermissionError as e:
            logger.error(f"Permission denied writing to {filepath}: {e}")
            raise PermissionError(
                f"No permission to write to '{filepath}'. "
                f"Check file system permissions."
            ) from e
        except OSError as e:
            if e.errno == errno.ENOSPC:
                logger.error(f"Disk full while writing to {filepath}: {e}")
                raise IOError("Disk is full. Free up space and try again.") from e
            logger.error(f"OS error writing to {filepath}: {e}")
            raise IOError(f"Cannot write to file '{filepath}': {e}") from e
        except IOError as e:
            logger.error(f"Failed to write to file {filepath}: {e}")
            raise IOError(f"Cannot write to file '{filepath}': {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Data is not JSON-serializable: {e}")
            raise TypeError(f"Data cannot be serialized to JSON: {e}") from e

        logger.info(f"Successfully saved data to {filepath}")
        return filepath
    
    @staticmethod
    def append(data: Any, output_dir: str, filename: str) -> str:
        """
        Appends data to an existing JSON file or creates a new one.

        Handles:
        - Creating directories if needed
        - Initializing empty files
        - Converting existing content to list format
        - Maintaining JSON integrity

        Args:
            data: Data to append (any JSON-serializable type)
            output_dir: Target directory for the file
            filename: Exact filename to use (no timestamp)

        Returns:
            str: Full path to the modified file

        Raises:
            ValueError: If parameters are invalid
            OSError: If directory creation fails
            PermissionError: If lacking read/write permissions
            IOError: If disk is full or file operations fail
            json.JSONDecodeError: If existing file contains invalid JSON
            TypeError: If data is not JSON-serializable
        """
        # Validate inputs
        if not output_dir or not output_dir.strip():
            raise ValueError("output_dir cannot be empty")

        # Sanitize filename to prevent path traversal
        filename = FileRepository._sanitize_filename(filename, "filename")

        # Sanitize path to prevent path traversal (check BEFORE normalization)
        if ".." in output_dir:
            raise ValueError(f"Path traversal detected in output_dir: {output_dir}")
        output_dir = os.path.normpath(output_dir)

        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating directory {output_dir}: {e}")
            raise PermissionError(
                f"No permission to create directory '{output_dir}'. "
                f"Check file system permissions."
            ) from e
        except OSError as e:
            if e.errno == errno.ENOSPC:
                logger.error(f"Disk full while creating directory {output_dir}: {e}")
                raise IOError("Disk is full. Free up space and try again.") from e
            logger.error(f"Failed to create directory {output_dir}: {e}")
            raise OSError(f"Cannot create output directory '{output_dir}': {e}") from e

        filepath = os.path.join(output_dir, filename)

        # Read existing data if file exists
        existing_data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except PermissionError as e:
                logger.error(f"Permission denied reading file {filepath}: {e}")
                raise PermissionError(
                    f"No permission to read '{filepath}'. "
                    f"Check file system permissions."
                ) from e
            except IOError as e:
                logger.error(f"Failed to read file {filepath}: {e}")
                raise IOError(f"Cannot read file '{filepath}': {e}") from e
            except json.JSONDecodeError as e:
                # Create backup of corrupted file to prevent data loss
                backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{filepath}.corrupted_{backup_timestamp}.bak"
                backup_created = False
                try:
                    shutil.copy2(filepath, backup_path)
                    backup_created = True
                    logger.error(
                        f"File {filepath} contains invalid JSON (position {e.pos}). "
                        f"Backup created at {backup_path}"
                    )
                except Exception as backup_error:
                    logger.error(
                        f"File {filepath} contains invalid JSON and backup failed: {backup_error}"
                    )

                if backup_created:
                    raise IOError(
                        f"Cannot append to '{filepath}': file contains invalid JSON at position {e.pos}. "
                        f"A backup was saved to '{backup_path}'. "
                        f"Please fix or remove the corrupted file manually."
                    ) from e
                else:
                    raise IOError(
                        f"Cannot append to '{filepath}': file contains invalid JSON at position {e.pos}. "
                        f"Backup creation failed. "
                        f"Please fix or remove the corrupted file manually."
                    ) from e

        # Ensure we're working with a list to enable appending
        if not isinstance(existing_data, list):
            logger.info(f"Converting existing data to list format in {filepath}")
            existing_data = [existing_data]

        existing_data.append(data)

        # Write updated data
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, default=str, ensure_ascii=False)
        except PermissionError as e:
            logger.error(f"Permission denied writing to {filepath}: {e}")
            raise PermissionError(
                f"No permission to write to '{filepath}'. "
                f"Check file system permissions."
            ) from e
        except OSError as e:
            if e.errno == errno.ENOSPC:
                logger.error(f"Disk full while writing to {filepath}: {e}")
                raise IOError("Disk is full. Free up space and try again.") from e
            logger.error(f"OS error writing to {filepath}: {e}")
            raise IOError(f"Cannot write to file '{filepath}': {e}") from e
        except IOError as e:
            logger.error(f"Failed to write to file {filepath}: {e}")
            raise IOError(f"Cannot write to file '{filepath}': {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Data is not JSON-serializable: {e}")
            raise TypeError(f"Data cannot be serialized to JSON: {e}") from e

        logger.debug(f"Successfully appended data to {filepath}")
        return filepath