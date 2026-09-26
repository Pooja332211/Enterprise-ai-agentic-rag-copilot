"""document_ingestion_schema

Revision ID: 0002_document_ingestion_schema
Revises: 0001_initial_schema
Create Date: 2026-09-20 14:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0002_document_ingestion_schema"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Expand documents table with ingestion metrics and normalized naming
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("normalized_filename", sa.String(length=255), server_default="", nullable=False)
        )
        batch_op.add_column(
            sa.Column("file_extension", sa.String(length=20), server_default="", nullable=False)
        )
        batch_op.add_column(
            sa.Column("page_count", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False)
        )

    # 2. Expand ingestion_jobs with stage tracking and updated_at timestamp
    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("stage", sa.String(length=50), server_default="VALIDATION", nullable=False)
        )
        batch_op.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            )
        )

    # 3. Create document_chunks table
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.String(length=255), nullable=True),
        sa.Column("section_path", sa.String(length=1024), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_document_chunks_doc_chunk", "document_chunks", ["document_id", "chunk_index"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_doc_chunk", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")

    with op.batch_alter_table("ingestion_jobs") as batch_op:
        batch_op.drop_column("updated_at")
        batch_op.drop_column("stage")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("chunk_count")
        batch_op.drop_column("page_count")
        batch_op.drop_column("file_extension")
        batch_op.drop_column("normalized_filename")
