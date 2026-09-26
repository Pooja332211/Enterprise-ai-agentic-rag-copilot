import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database.base import Base, TimestampMixin


def generate_uuid() -> str:
    """Generates a string representation of a standard UUID4."""
    return str(uuid.uuid4())


class Document(Base, TimestampMixin):
    """
    Document entity tracking uploaded enterprise files, storage location,
    structural metrics, and overall ingestion lifecycle state.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_filename: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False, index=True
    )  # pending, processing, completed, failed
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    jobs: Mapped[list["IngestionJob"]] = relationship(
        "IngestionJob", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status} chunks={self.chunk_count}>"


class IngestionJob(Base):
    """
    Tracks state machine lifecycle and progress of asynchronous/synchronous ingestion jobs.
    """
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(50), default="queued", nullable=False, index=True
    )  # queued, processing, completed, failed
    stage: Mapped[str] = mapped_column(
        String(50), default="VALIDATION", nullable=False
    )  # VALIDATION, PARSING, CHUNKING, PERSISTING, COMPLETED, FAILED
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<IngestionJob id={self.id} document_id={self.document_id} stage={self.stage} status={self.status}>"


class DocumentChunk(Base):
    """
    Structure-aware document chunk preserving section hierarchy, page numbers,
    and granular text content ready for downstream embedding, BM25, and citations.
    """
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    section_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_document_chunks_doc_chunk", "document_id", "chunk_index"),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} doc={self.document_id} idx={self.chunk_index} "
            f"page={self.page_number} chars={self.character_count}>"
        )


class SystemAuditLog(Base):
    """
    Audit log recording system migrations, health checks, and lifecycle events.
    """
    __tablename__ = "system_audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<SystemAuditLog id={self.id} event_type={self.event_type}>"
