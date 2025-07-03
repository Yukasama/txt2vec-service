"""Common test fixtures for the application."""

import os

os.environ.setdefault("ENV", "testing")

import shutil
import signal
import subprocess  # noqa: S404
import sys
import time
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
import redis
from fastapi.testclient import TestClient
from loguru import logger
from redis import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession
from testcontainers.redis import RedisContainer

from vectorize.app import app
from vectorize.config.db import get_session
from vectorize.config.seed import seed_db

REDIS_TEST_PORT = 56379


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db() -> Generator[None]:
    """Clean up test database before and after each test."""
    db_path = Path("test.db")
    if db_path.exists():
        db_path.unlink()
    yield
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(autouse=True)
def flush_redis() -> None:
    """Flush Redis before each test to ensure a clean state."""
    redis_url = f"redis://localhost:{REDIS_TEST_PORT}/0"
    os.environ["REDIS_URL"] = redis_url
    r = redis.from_url(redis_url)
    r.flushall()
    r.script_flush()


@pytest.fixture(scope="session", autouse=True)
def redis_container() -> Generator[RedisContainer]:
    """Fixture to start a Redis container for testing."""
    container: RedisContainer = (
        RedisContainer("redis:7-alpine")
        .with_bind_ports(6379, REDIS_TEST_PORT)
        .with_env("REDIS_REPLICATION_MODE", "master")
        .with_env("save", "")
        .with_env("appendonly", "no")
        .start()
    )

    redis_url = f"redis://localhost:{REDIS_TEST_PORT}/0"
    os.environ["REDIS_URL"] = redis_url
    Redis.from_url(redis_url).ping()

    yield container
    container.stop()


@pytest.fixture(scope="session", autouse=True)
def dramatiq_worker(redis_container: RedisContainer) -> Generator[None]:  # noqa: ARG001
    """Fixture to start a Dramatiq worker for testing."""
    cmd = [sys.executable, "-m", "dramatiq", "vectorize.tasks", "-p", "2", "-t", "8"]
    worker = subprocess.Popen(cmd, env=os.environ.copy())  # noqa: S603
    time.sleep(2)

    if worker.poll() is not None:
        raise RuntimeError("Dramatiq worker crashed")

    yield
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            worker.send_signal(sig)
            worker.wait(timeout=10)
            break
        except Exception:
            logger.info("Dramatiq worker couldn't be stopped gracefully", exc_info=True)
    else:
        worker.kill()


@pytest.fixture(scope="session")
async def session() -> AsyncGenerator[AsyncSession]:
    """Create a test database engine.

    Returns:
        AsyncSession: SQLModel async session for database operations.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///test.db",
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            "timeout": 60,
            "isolation_level": None,
        },
        echo=False,
        pool_pre_ping=True,
        pool_recycle=300,
    )

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(test_engine) as session:
        await seed_db(session)

    async with AsyncSession(test_engine) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture(name="client")
def client_fixture(session: AsyncSession) -> Generator[TestClient]:
    """Create a test client for the FastAPI app.

    Args:
        session: Database session fixture.

    Returns:
        TestClient: Configured FastAPI test client.
    """

    def get_session_override() -> AsyncSession:
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app, base_url="http://testserver")  # NOSONAR
    yield client

    app.dependency_overrides.clear()


@pytest.fixture(scope="session", autouse=True)
def copy_training_datasets() -> Generator[None]:
    """Copy training datasets from the test data directory to the datasets directory."""
    src = Path("test_data/training/datasets")
    dst = Path("test_data/datasets")

    if not src.exists():
        return

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
        elif item.is_dir():
            target_dir = dst / item.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(item, target_dir)
    yield
