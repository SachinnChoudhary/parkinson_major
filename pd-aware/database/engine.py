"""Database engine and session management for PD-Aware."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config.settings import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    """Get a new database session."""
    session = SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def init_db():
    """Create all tables."""
    from database.models import Base
    Base.metadata.create_all(bind=engine)
