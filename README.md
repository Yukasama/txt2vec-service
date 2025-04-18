# AWP Projekt für Text Embedding Service

## Setup

### Dependency management

#### Installation

```bash
# Install uv for command line (MacOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh

# Install uv for command line (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Install dependencies and generate lock file
uv sync

## Update dependencies
uv sync --upgrade

# Add dependencies
uv add <package>

# Remove dependencies
uv remove <package>
```

##### Create .env

```bash
# .env
DATABASE_URL=sqlite+aiosqlite:///./app.db
```

#### Fix lock file

Error: Failed to parse `uv.lock`

1. Delete `uv.lock` file
2. Run `uv sync` or `uv lock`

Note: Do **not** edit the `uv.lock`-File yourself.

### Start server

```bash
uv run app
```

### Build

```bash
# Build project
uv build
```

### Run tests

`Note: The server must be running.`

```bash
# Run all tests
uv run pytest
```

### Build Docker Image

```bash
# Build Docker image
docker build -t txt2vec:prod .
```

## Workflow

### Run CI locally

```bash
# Install act Docker container
brew install act

# Run CI locally
act
```
