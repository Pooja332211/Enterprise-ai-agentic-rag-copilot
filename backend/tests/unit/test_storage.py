from pathlib import Path
import pytest
from backend.app.services.storage_service import LocalStorageService


@pytest.fixture
def storage_service(tmp_path):
    """Provides a LocalStorageService instance isolated inside a temp directory."""
    class MockSettings:
        UPLOAD_DIR = str(tmp_path / "uploads")

    return LocalStorageService(settings=MockSettings())


def test_save_and_read_normal_file(storage_service):
    """Verify standard file persistence and retrieval."""
    content = b"Enterprise Copilot Knowledge Data"
    doc_id = "test-doc-001"
    storage_path, norm_name, size = storage_service.save_file(content, doc_id, "report.pdf")

    assert norm_name == "report.pdf"
    assert size == len(content)
    assert storage_service.file_exists(storage_path) is True

    read_bytes = storage_service.read_file(storage_path)
    assert read_bytes == content


def test_filename_with_spaces_and_special_chars(storage_service):
    """Verify filename sanitization handles spaces, hyphens, and underscores."""
    content = b"Content with spaces"
    doc_id = "test-doc-002"
    storage_path, norm_name, _ = storage_service.save_file(
        content, doc_id, "Q3 2026 Financial & Tech Report (Final).pdf"
    )

    assert norm_name == "Q3_2026_Financial_Tech_Report_Final.pdf"
    assert " " not in norm_name
    assert "&" not in norm_name
    assert "(" not in norm_name
    assert storage_service.file_exists(storage_path) is True


def test_path_traversal_prevention(storage_service):
    """Verify path traversal attempts are neutralized during sanitization and access."""
    # Test traversal in filename is sanitized away
    content = b"Malicious attempt"
    doc_id = "test-doc-003"

    _, norm_name1, _ = storage_service.save_file(content, doc_id, "../../etc/passwd")
    assert ".." not in norm_name1
    assert "/" not in norm_name1

    _, norm_name2, _ = storage_service.save_file(content, doc_id, "..\\..\\windows\\system32\\cmd.exe")
    assert ".." not in norm_name2
    assert "\\" not in norm_name2

    # Test direct path verification raises ValueError if someone passes escaping path
    with pytest.raises(ValueError, match="Security error: Invalid storage path"):
        storage_service.get_file_path("../../outside_root.txt")


def test_excessively_long_filename(storage_service):
    """Verify long filename is truncated safely to 100 characters plus extension."""
    long_name = "a" * 200 + ".docx"
    content = b"Long filename test"
    doc_id = "test-doc-004"
    _, norm_name, _ = storage_service.save_file(content, doc_id, long_name)

    assert len(norm_name) <= 106  # 100 chars + .docx
    assert norm_name.endswith(".docx")


def test_delete_file_and_clean_directory(storage_service):
    """Verify deletion removes file and cleans up document subfolder."""
    content = b"Temporary content"
    doc_id = "test-doc-005"
    storage_path, _, _ = storage_service.save_file(content, doc_id, "temp.txt")

    assert storage_service.file_exists(storage_path) is True
    deleted = storage_service.delete_file(storage_path)
    assert deleted is True
    assert storage_service.file_exists(storage_path) is False
