.PHONY: help install test lint run up down clean migrate

help:
	@echo "Enterprise AI Knowledge & Research Copilot - Command Menu"
	@echo ""
	@echo "  install      Install backend dependencies"
	@echo "  test         Run test suite with pytest"
	@echo "  run          Run backend development server locally"
	@echo "  up           Start Docker Compose services (Postgres, Qdrant, Backend)"
	@echo "  down         Stop Docker Compose services"
	@echo "  migrate      Run database migrations"
	@echo "  clean        Clean build, cache, and bytecode artifacts"

install:
	pip install -r backend/requirements.txt

test:
	pytest backend/tests -v

run:
	python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

up:
	docker compose up -d

down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
