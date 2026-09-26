import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
from backend.app.config.settings import Settings, get_settings
from backend.app.utils.logger import logger


class BaseStorageService(ABC):
    """Abstract interface for document file storage operations."""

    @abstractmethod
    def save_file(
        self, file_content: bytes, document_id: str, original_filename: str
    ) -> Tuple[str, str, int]:
        """Saves file content securely and returns (storage_path, normalized_filename, size_bytes)."""
        pass

    @abstractmethod
    def get_file_path(self, storage_path: str) -> Path:
        """Returns the verified system Path for a stored file."""
        pass

    @abstractmethod
    def read_file(self, storage_path: str) -> bytes:
        """Reads and returns the raw file bytes."""
        pass

    @abstractmethod
    def delete_file(self, storage_path: str) -> bool:
        """Deletes the stored file and its parent document directory if empty."""
        pass

    @abstractmethod
    def file_exists(self, storage_path: str) -> bool:
        """Returns True if the file exists on storage."""
        pass


class LocalStorageService(BaseStorageService):
    """
    Secure local filesystem storage service.
    Implements path traversal prevention, filename sanitization,
    and isolated document-level folder structures.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_dir = Path(self.settings.UPLOAD_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitizes an untrusted filename, neutralizing directory traversal attempts
        and removing unsafe non-alphanumeric characters while preserving extension.
        """
        # Strip path delimiters and null bytes
        clean = filename.replace("\\", "/").split("/")[-1].replace("\0", "").strip()

        # Split base name and extension
        if "." in clean:
            parts = clean.rsplit(".", 1)
            name_part = parts[0]
            ext_part = parts[1].lower()
        else:
            name_part = clean
            ext_part = ""

        # Remove everything except alphanumeric, dashes, underscores, and dots
        safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name_part)
        safe_name = re.sub(r"_+", "_", safe_name).strip("_")

        if not safe_name:
            safe_name = "unnamed_document"

        # Cap length to 100 characters to prevent filesystem overflow
        safe_name = safe_name[:100]

        if ext_part:
            safe_ext = re.sub(r"[^a-zA-Z0-9]", "", ext_part)[:10]
            return f"{safe_name}.{safe_ext}"
        return safe_name

    def _resolve_and_verify_path(self, storage_path: str) -> Path:
        """
        Resolves the given storage path and ensures it does NOT escape base_dir.
        Raises ValueError if a directory traversal attempt is detected.
        """
        path = Path(storage_path)
        if not path.is_absolute():
            resolved = (self.base_dir.parent / path).resolve()
        else:
            resolved = path.resolve()

        # Strict containment check
        try:
            resolved.relative_to(self.base_dir)
        except ValueError:
            logger.error(f"Directory traversal attempt detected: {storage_path}")
            raise ValueError(f"Security error: Invalid storage path '{storage_path}'")

        return resolved

    def save_file(
        self, file_content: bytes, document_id: str, original_filename: str
    ) -> Tuple[str, str, int]:
        """
        Saves raw bytes into an isolated document folder:
        <base_dir>/<document_id>/original.<extension>
        """
        normalized_filename = self.sanitize_filename(original_filename)
        ext = normalized_filename.split(".")[-1] if "." in normalized_filename else "bin"

        # Dedicated isolated folder for this document ID
        doc_dir = self.base_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        target_file = doc_dir / f"original.{ext}"
        target_file.write_bytes(file_content)

        size_bytes = len(file_content)
        # Store relative path from workspace root for portability
        relative_storage_path = str(target_file.relative_to(self.base_dir.parent)).replace("\\", "/")

        logger.info(f"Stored file for document_id={document_id} at {relative_storage_path} ({size_bytes} bytes)")
        return relative_storage_path, normalized_filename, size_bytes

    def get_file_path(self, storage_path: str) -> Path:
        """Returns the verified system Path for a stored file."""
        return self._resolve_and_verify_path(storage_path)

    def read_file(self, storage_path: str) -> bytes:
        """Reads raw bytes from storage."""
        path = self._resolve_and_verify_path(storage_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stored file not found: {storage_path}")
        return path.read_bytes()

    def delete_file(self, storage_path: str) -> bool:
        """Deletes file and cleans up the document folder."""
        try:
            path = self._resolve_and_verify_path(storage_path)
            if path.is_file():
                path.unlink()
                # Attempt to remove the parent document directory if empty
                parent_dir = path.parent
                if parent_dir != self.base_dir and parent_dir.is_dir() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                logger.info(f"Deleted storage file: {storage_path}")
                return True
            return False
        except Exception as exc:
            logger.error(f"Error deleting storage file {storage_path}: {exc}")
            return False

    def file_exists(self, storage_path: str) -> bool:
        """Checks existence of file on storage."""
        try:
            path = self._resolve_and_verify_path(storage_path)
            return path.is_file()
        except Exception:
            return False


_storage_service_instance: Optional[BaseStorageService] = None


def get_storage_service() -> BaseStorageService:
    """Returns the singleton storage service instance."""
    global _storage_service_instance
    if _storage_service_instance is None:
        _storage_service_instance = LocalStorageService()
    return _storage_service_instance
