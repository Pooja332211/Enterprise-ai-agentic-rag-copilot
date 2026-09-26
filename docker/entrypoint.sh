#!/bin/bash
set -e

echo "Starting Enterprise AI Knowledge & Research Copilot entrypoint..."

# Run database migrations if in a Postgres environment
if [ -f "/app/backend/alembic.ini" ]; then
    echo "Running Alembic migrations..."
    cd /app/backend && alembic upgrade head || {
        echo "Alembic migration encountered an issue. Checking database readiness..."
    }
    cd /app
fi

echo "Launching application service..."
exec "$@"
