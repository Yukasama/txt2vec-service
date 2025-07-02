
# Contributing to Vectorize

Thank you for your interest in contributing to Vectorize! This guide will help you get started with contributing to our text embedding service platform.


## Getting Started

Check out our [Installation Guide](installation.md) for setup instructions. Use **Method 1: Development Setup** which includes all dev tools and dependencies.

```bash
# Fork and clone your repository
git clone https://github.com/your-username/vectorize.git
cd vectorize

# Install with dev dependencies
uv sync

# Set up development environment
cp .env.example .env
# Edit .env with: DATABASE_URL, LOG_LEVEL=DEBUG

# Verify setup
uv run pytest
uv run app
```


## Development Workflow

### Code Quality Tools

```bash
# Format and lint code
ruff format .
ruff check .

# Type checking
pyright

# Run tests with coverage
uv run pytest --cov=src/vectorize
```

### Module Structure

```
module/
├── __init__.py     # Public interface
├── models.py       # Data models (SQLModel/Pydantic)
├── schemas.py      # API request/response schemas
├── repository.py   # Data access layer
├── service.py      # Business logic
├── router.py       # FastAPI endpoints
├── tasks.py        # Background tasks
└── exceptions.py   # Module-specific exceptions
└── utils           # Utils for the module
```

### Testing

- Write unit tests for business logic
- Write integration tests for API endpoints
- Use descriptive test names: `test_upload_dataset_with_valid_csv`
- Group related tests in classes

### Database Changes

- Use SQLModel for all database models
- Include proper type hints and validation
- Test schema changes thoroughly
- Consider backward compatibility


## Submitting Changes

### Pull Request Process

1. **Create Feature Branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Run Quality Checks**

```bash
uv run pytest
ruff check .
ruff format .
pyright
```

3. **Commit with Convention**

```bash
git commit -m "feat: add new feature description"
```

**Commit Conventions:**

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring

4. **Create PR with:**

- Clear title and description
- Link to related issues
- Test coverage information
- Breaking change notes (if any)

### PR Requirements

**Automated Checks:**

- All tests pass
- Code coverage maintained
- Linting passes
- Type checking passes

**Manual Review:**

- Code quality and maintainability
- Test completeness
- Documentation updates


## Development Tools

### Running Services

```bash
# Development server
uv run app

# With hot reload
uv run uvicorn vectorize.app:app --reload

# With Docker
docker compose up vectorize dramatiq_worker redis caddy
```

### Load Testing

```bash
# Interactive load testing
uvx locust -f scripts/locust.py

# Headless load testing
uvx locust -f scripts/locust.py --host=https://localhost/v1/api/health --headless --users 10 --run-time 1m
```

### Database Management

```bash
# Reset database (dev only)
# Set CLEAR_DB_ON_RESTART=true in .env and restart

# View database
sqlite3 app.db
```


## Issue Reporting

### Bug Reports Include:

- **Environment**: Python version, OS, Vectorize version
- **Reproduction Steps**: Clear step-by-step instructions
- **Expected vs Actual**: What should happen vs what happens
- **Logs**: Error messages and stack traces (remove sensitive data)

### Feature Requests Include:

- **Problem**: What problem does this solve?
- **Solution**: Detailed feature description
- **Alternatives**: Other approaches considered


## Getting Help

- **Issues**: Search [GitHub issues](https://github.com/yukasama/vectorize/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/yukasama/vectorize/discussions)
- **Bugs**: Create [new issue](https://github.com/yukasama/vectorize/issues/new)


## Code of Conduct

- Be respectful and constructive
- Help newcomers learn
- Follow project standards
- Keep discussions focused and relevant

---

**Thank you for contributing to Vectorize!**
