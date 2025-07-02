# Logging

The Vectorize application uses a structured logging system built on Loguru that adapts to different environments and provides comprehensive log management.

## Logging Architecture

### Core Components

1. **Loguru Logger**: Primary logging interface with advanced features
2. **Environment Adaptation**: Different logging behavior for development vs production
3. **Loki Integration**: Production logs are forwarded to Loki for centralized collection
4. **Intercept Handler**: Captures standard Python logging and routes it through Loguru

## Environment-Specific Behavior

### Development Mode

- **Console Output**: Colored logs to stdout for easy reading
- **File Logging**: Detailed logs written to rotating files with compression
- **Rich Context**: Includes function names, line numbers, and stack traces
- **Debug Information**: Full backtrace and diagnostic information available

### Production Mode

- **Structured Logging**: Optimized format for log processing systems
- **Loki Integration**: Logs forwarded to Loki with metadata labels
- **Performance Optimized**: Minimal overhead with async log processing
- **Security Focused**: Reduced verbose information to prevent data leakage

## Configuration

### Log Levels

Configure via the `LOG_LEVEL` environment variable:

```bash
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
```
