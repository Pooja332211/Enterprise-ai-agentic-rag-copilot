import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.config.settings import get_settings
from backend.app.database.base import Base
from backend.app.database import models  # Ensures all models are registered

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata
target_metadata = Base.metadata

# Dynamically set database URL from application settings
settings = get_settings()
db_url = settings.sync_database_url

# Fallback check for local development
if settings.DATABASE_SQLITE_FALLBACK and "sqlite" in settings.async_database_url:
    db_url = f"sqlite:///{settings.DATABASE_SQLITE_PATH}"

config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        connection = connectable.connect()
    except Exception as exc:
        if settings.DATABASE_SQLITE_FALLBACK:
            import os
            os.makedirs(os.path.dirname(settings.DATABASE_SQLITE_PATH), exist_ok=True)
            fallback_url = f"sqlite:///{settings.DATABASE_SQLITE_PATH}"
            configuration["sqlalchemy.url"] = fallback_url
            connectable = engine_from_config(
                configuration,
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
            )
            connection = connectable.connect()
        else:
            raise exc

    with connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
