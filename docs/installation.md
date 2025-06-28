# 📦 Installation Guide

This guide will help you set up Vectorize for development or production use. Choose the method that best fits your needs.

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (recommended: Python 3.12)
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
uv sync --all-extras --dev

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
ENV=development
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
# Edit .env for Docker setup
cat > .env << EOF
DATABASE_URL=sqlite+aiosqlite:///app/data/app.db
LOG_LEVEL=INFO
ENV=production
CLEAR_DB_ON_RESTART=false
EOF
```

3. **Start with Docker Compose**

```bash
# Start all services
docker compose up

# Or run in background
docker compose up -d

# View logs
docker compose logs -f vectorize
```

4. **Access the Application**

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔧 Environment Configuration

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
SEED_DB_ON_START=true       # Add sample data (dev only)
```

### Optional Configuration

```bash
# Directory Overrides
UPLOAD_DIR=/custom/path/datasets
MODELS_DIR=/custom/path/models
LOG_DIR=/custom/path/logs

# Performance Tuning
MAX_UPLOAD_SIZE=53687091200  # 50GB in bytes
POOL_SIZE=10                 # Database connection pool

# Security (Production)
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

For complete configuration options, see the [Configuration Guide](configuration.md).

## 🚀 Running Vectorize

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

For full functionality, you'll need these services:

```bash
# Start all services with Docker
docker compose up vectorize dramatiq_worker redis caddy

# Or individually
docker compose up -d redis        # Task queue
docker compose up -d vectorize    # Main application
docker compose up -d dramatiq_worker  # Background worker
docker compose up -d caddy        # Reverse proxy
```

## 🧪 Verification and Testing

### Health Checks

```bash
# Application health
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Database connectivity
curl http://localhost:8000/health/database
# Expected: {"database": "connected"}

# Background tasks
curl http://localhost:8000/health/tasks
# Expected: {"tasks": "running"}
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/vectorize --cov-report=html

# Run specific test categories
uv run pytest -m "not slow"        # Skip slow tests
uv run pytest tests/unit/          # Unit tests only
uv run pytest tests/integration/   # Integration tests only

# Load testing with Locust
uvx locust -f scripts/locust.py --host http://localhost:8000
```

### Sample API Calls

```bash
# List available endpoints
curl http://localhost:8000/

# Get models
curl http://localhost:8000/models

# Get datasets
curl http://localhost:8000/datasets

# Get background tasks
curl http://localhost:8000/tasks
```

## 🐳 Docker Setup Details

### Production Docker Build

```bash
# Build production image
docker build -t vectorize:latest .

# Run with custom configuration
docker run -d \
  --name vectorize \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env \
  vectorize:latest
```

### Docker Compose Services

```yaml
# docker-compose.yml overview
services:
  vectorize: # Main application
  dramatiq_worker: # Background task processor
  redis: # Task queue and cache
  caddy: # Reverse proxy and HTTPS
```

### Persistent Data

```bash
# Create data directories
mkdir -p data/{datasets,models,db,logs}

# Set permissions
chmod 755 data/
chmod 644 data/db/
```

## 🔧 Development Tools Setup

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

## 🐛 Troubleshooting

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

- **📚 Documentation**: Check other docs in this directory
- **🐛 Issues**: [Report bugs on GitHub](https://github.com/yukasama/vectorize/issues)
- **💬 Discussions**: [Ask questions on GitHub](https://github.com/yukasama/vectorize/discussions)

### Performance Optimization

```bash
# Profile startup time
python -m cProfile -o startup.prof src/vectorize/app.py
uv run snakeviz startup.prof

# Monitor resource usage
docker stats vectorize
```

## ✅ Next Steps

After successful installation:

1. **📖 [Read the Configuration Guide](configuration.md)** - Learn about all available settings
2. **🔌 [Explore the API](api.md)** - Understand available endpoints
4. **🤝 [Contributing Guide](contributing.md)** - Start contributing to the project

---

**Installation complete!** 🎉 You're now ready to start using Vectorize for your text embedding workflows.
