
# Local CI with Act

Learn how to run GitHub Actions workflows locally using Act, enabling you to test CI/CD changes before pushing to GitHub.


## What is Act?

[Act](https://github.com/nektos/act) runs your GitHub Actions locally using Docker. This allows you to:

- **Test workflows before pushing** - Catch issues early
- **Faster iteration** - No need to push and wait for GitHub runners
- **Debug workflow issues** - Interactive debugging capabilities
- **Save GitHub Action minutes** - Especially important for private repos


## Installation

### macOS

```bash
# Using Homebrew (recommended)
brew install act

# Using MacPorts
sudo port install act
```

### Linux

```bash
# Using curl
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Using package managers
# Ubuntu/Debian
sudo apt install act

# Arch Linux
yay -S act
```

### Windows

```bash
# Using Scoop
scoop install act

# Using Chocolatey
choco install act-cli

# Using WinGet
winget install nektos.act
```


## Basic Usage

### Run All Workflows

```bash
# Run all workflows for push event
act

# Run workflows for pull request event
act pull_request

# Run specific workflow
act -W .github/workflows/main.yaml
```

### Run Specific Jobs

```bash
# Run only the test job
act -j test

# Run only the lint job
act -j lint

# Run multiple specific jobs
act -j test -j lint
```


## Vectorize-Specific Commands

### Main CI Workflow

```bash
# Run the complete main CI pipeline
act -W .github/workflows/main.yaml

# Run with secrets (GitHub token required for some actions)
act -s GITHUB_TOKEN="$(gh auth token)" -W .github/workflows/main.yaml

# Run with artifact support (saves artifacts locally)
act --artifact-server-path $PWD/.artifacts -W .github/workflows/main.yaml
```

### Publish Workflow

```bash
# Test the publish workflow (won't actually publish)
act -W .github/workflows/publish.yaml

# With full GitHub integration
act -s GITHUB_TOKEN="$(gh auth token)" \
    --artifact-server-path $PWD/.artifacts \
    -W .github/workflows/publish.yaml
```

### Debugging Workflows

```bash
# Run with verbose output
act --verbose

# Run with debug information
act --env ACT_DEBUG=true

# Interactive debugging (if workflow fails)
act --interactive

# Dry run (shows what would execute)
act --dry-run
```


## Performance Tips

### Speed Optimization

```bash
# Use cached runners (much faster)
act --use-gitignore=false --cache

# Skip Docker image pulls if already present
act --pull=false

# Use smaller base images for faster startup
act --platform ubuntu-latest=node:18-alpine
```

### Resource Management

```bash
# Limit resource usage
act --container-cap-add SYS_PTRACE --container-cap-drop ALL

# Clean up after runs
docker system prune -f
```


## Useful Resources

- **[Act Documentation](https://github.com/nektos/act)** - Official documentation
- **[Available Images](https://github.com/catthehacker/docker_images)** - Act-optimized Docker images
- **[GitHub Actions Docs](https://docs.github.com/en/actions)** - Understanding GitHub Actions

---


**Happy local testing!**

Act helps you catch issues early and iterate faster on your CI/CD workflows. Remember that while Act is very accurate, the final test is always running on GitHub's actual runners.
