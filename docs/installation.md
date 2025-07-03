# Installation Guide

This guide will help you set up Vectorize for development or production use. Choose the method that best fits your needs.

## Quick Start

### Prerequisites

- **Python 3.13+**
- **Git** for version control
- **Docker** (optional, for containerized setup)

### Method 1: Development Setup (Recommended for Contributors)

1. **Clone the Repository**

```bash
git clone https://github.com/yukasama/vectorize.git
cd vectorize
```

2. **Install UV Package Manager**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv
```

3. **Install Dependencies**

```bash
# Install all dependencies including dev tools
uv sync

# Or for production only
uv sync --no-dev
```

4. **Configure Environment**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
# Minimum required settings:
cat > .env << EOF
DATABASE_URL=sqlite+aiosqlite:///app.db
LOG_LEVEL=DEBUG
CLEAR_DB_ON_RESTART=true
EOF
```

5. **Start the Server**

```bash
# Start development server
uv run app

# Or with hot reload
uv run uvicorn vectorize.app:app --reload --host 0.0.0.0 --port 8000
```

6. **Verify Installation**

```bash
# Check if server is running
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# View API documentation
open http://localhost:8000/docs
```

### Method 2: Docker Setup (Recommended for Production)

1. **Clone and Setup**

```bash
git clone https://github.com/yukasama/vectorize.git
cd vectorize
cp .env.example .env
```

2. **Configure Environment**

```bash
# Edit .env for Docker environment to your needs
cat > .env << EOF
ENV: production
DATABASE_URL: sqlite+aiosqlite:///db/app.db
REDIS_URL: redis://redis:6379
DATASETS_DIR: /app/data/datasets
MODELS_DIR: /app/data/models
DB_DIR: /app/db
TZ: Europe/Berlin
LOG_LEVEL: INFO
HF_HOME: /app/data/hf_home
GH_HOME: /app/data/gh_home
EOF
```

3. **Start with Docker Compose**

```bash
# Start all services (Note that the Frontend image has to be built before this step)
docker compose up

# Start without Grafana
docker compose up vectorize vectorize_web dramatiq_worker dramatiq_training_worker dramatiq_evaluation_worker redis caddy

# Or run in background
docker compose up -d

# When you have to rebuild the image
docker compose up --build

# View logs
docker compose logs -f vectorize
```

4. **Access the Application**

- **API**: https://localhost/v1/api
- **API Docs**: https://localhost/v1/api/docs
- **Health Check**: https://localhost/v1/api/health

## Environment Configuration

### Essential Environment Variables

Create a `.env` file with these required settings:

```bash
# Application Environment
ENV=development|testing|production

# Database Configuration
DATABASE_URL=sqlite+aiosqlite:///app.db

# Logging
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL

# Development Settings
CLEAR_DB_ON_RESTART=true    # Reset DB on startup (dev only)
```

For complete configuration options, see the [Configuration Guide](configuration.md).

## Running Vectorize

### Development Mode

```bash
# Standard development server
uv run app

# With auto-reload on file changes
uv run uvicorn vectorize.app:app --reload

# Custom host and port
uv run uvicorn vectorize.app:app --host 0.0.0.0 --port 8080

# With debug logging
LOG_LEVEL=DEBUG uv run app
```

### Production Mode

```bash
# Using Docker (recommended)
docker compose up -d

# Or direct Python with production settings
ENV=production uv run uvicorn vectorize.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Background Services

For full functionality, you'll need these Backend services:

```bash
# Start all services with Docker
docker compose up vectorize vectorize_web dramatiq_worker dramatiq_training_worker dramatiq_evaluation_worker redis caddy

# Or individually
docker compose up -d redis        # Task queue
docker compose up -d vectorize    # Main application
docker compose up -d dramatiq_worker  # Background worker
docker compose up -d dramatiq_training_worker  # Background training worker
docker compose up -d dramatiq_evaluation_worker  # Background evaluation worker
docker compose up -d caddy        # Reverse proxy
```

## Verification and Testing

### Running Tests

```bash
# Run all tests (coverage enabled by default)
uv run pytest
# Note: On some systems, we experienced timeout errors regarding "dataset_hf" tests

# Load testing with Locust
uvx locust -f scripts/locust.py --host http://localhost:8000
```

### Sample API Calls

```bash
# List available endpoints
curl https://localhost/v1/api/docs

# Get models
curl https://localhost/v1/api/models

# Get datasets
curl https://localhost/v1/api/datasets

# Get background tasks
curl https://localhost/v1/api/tasks
```

## Development Tools Setup

### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):

```json
{
  "python.interpreter": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff"
}
```

**PyCharm Setup**:

1. Open project in PyCharm
2. Configure Python interpreter to `.venv/bin/python`
3. Enable Ruff for linting and formatting

### Git Hooks (Optional)

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## Troubleshooting

### Common Issues

**1. UV Installation Issues**

```bash
# If uv command not found after install
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc
```

**2. Database Connection Errors**

```bash
# Check database path exists
mkdir -p $(dirname $(echo $DATABASE_URL | sed 's/.*:\/\/\///'))

# Reset database
rm app.db  # Remove existing database
uv run app  # Restart to recreate
```

**3. Port Already in Use**

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uv run uvicorn vectorize.app:app --port 8001
```

**4. Docker Permission Issues**

```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Reset Docker
docker system prune -a
```

**5. Memory Issues with Large Models**

```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory

# Or use smaller models for development
```

### Getting Help

- **Documentation**: Check other docs in this directory
- **Issues**: [Report bugs on GitHub](https://github.com/yukasama/vectorize/issues)
- **Discussions**: [Ask questions on GitHub](https://github.com/yukasama/vectorize/discussions)

### Performance Optimization

```bash
# Profile startup time
python -m cProfile -o startup.prof src/vectorize/app.py
uv run snakeviz startup.prof

# Monitor resource usage
docker stats vectorize
```

## Next Steps

After successful installation:

1. **[Read the Configuration Guide](configuration.md)** - Learn about all available settings
2. **[Explore the API](api.md)** - Understand available endpoints
3. **[Contributing Guide](contributing.md)** - Start contributing to the project

---

**Installation complete!** You're now ready to start using Vectorize for your text embedding workflows.
