# Database

The Vectorize application uses SQLite with SQLModel/SQLAlchemy for database operations, providing a lightweight yet robust data persistence layer.

## Database Architecture

### Core Components

1. **SQLite Database**: File-based database for development and production deployments
2. **SQLModel Integration**: Type-safe ORM built on SQLAlchemy for Python data models
3. **Async Operations**: Full async/await support for non-blocking database operations
4. **Connection Pooling**: Managed connection pool for optimal performance

## Connection Management

### Engine Configuration

The database engine is configured with production-optimized settings:

- **Connection Pooling**: Base pool of 5 connections with up to 10 overflow connections
- **Connection Recycling**: Connections recycled every 300 seconds to prevent timeouts
- **Pre-ping Validation**: Connections validated before use to handle stale connections
- **Timeout Handling**: 30-second timeout for database operations

### Session Management

Database sessions use the async context manager pattern:

```python
async def get_session() -> AsyncGenerator[AsyncSession]:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```
