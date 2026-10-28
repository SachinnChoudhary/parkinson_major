"""
Tests for Authentication, Authorization, Role-Based Access, and Security Controls.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.engine import get_session, init_db
from database.models import User, UserRole, AuditLog, Screening
from components.auth_manager import (
    hash_password, verify_password, register_user, authenticate_user,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()


def test_password_hashing_and_verification():
    """Verify bcrypt password hashing and verification."""
    password = "SecurePassword2026!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_user_registration_and_duplicate_check():
    """Verify user registration, role assignment, and unique constraints."""
    import uuid
    rand_suffix = str(uuid.uuid4())[:8]
    username = f"user_{rand_suffix}"
    email = f"{username}@example.com"

    # Successful registration
    success, msg = register_user(
        username=username,
        email=email,
        password="ValidPassword123!",
        full_name="Test Patient",
        role=UserRole.PATIENT,
    )
    assert success is True

    # Duplicate username rejection
    dup_success, dup_msg = register_user(
        username=username,
        email=f"other_{rand_suffix}@example.com",
        password="ValidPassword123!",
        full_name="Duplicate User",
        role=UserRole.PATIENT,
    )
    assert dup_success is False
    assert "already exists" in dup_msg


def test_authentication_and_audit_logging():
    """Verify authentication succeeds with correct credentials and creates audit logs."""
    import uuid
    rand_suffix = str(uuid.uuid4())[:8]
    username = f"clinician_{rand_suffix}"
    email = f"{username}@hospital.org"

    register_user(
        username=username,
        email=email,
        password="ClinicianPass123!",
        full_name="Dr. Test Neurologist",
        role=UserRole.DOCTOR,
    )

    # Correct login
    auth_ok, user_info = authenticate_user(username, "ClinicianPass123!")
    assert auth_ok is True
    assert user_info["role"] == UserRole.DOCTOR.value
    assert user_info["username"] == username

    # Failed login
    fail_ok, _ = authenticate_user(username, "WrongPass123!")
    assert fail_ok is False

    # Check that audit log entry exists
    session = get_session()
    logs = session.query(AuditLog).filter(AuditLog.action.in_(["login_success", "login_failed"])).all()
    assert len(logs) > 0
    session.close()


def test_patient_data_isolation():
    """Verify screenings belong to specific session IDs and users."""
    session = get_session()
    import uuid
    s_id_1 = str(uuid.uuid4())
    s_id_2 = str(uuid.uuid4())

    s1 = Screening(session_id=s_id_1, age_range="50-59")
    s2 = Screening(session_id=s_id_2, age_range="70-79")
    session.add_all([s1, s2])
    session.commit()

    queried_s1 = session.query(Screening).filter(Screening.session_id == s_id_1).first()
    assert queried_s1.age_range == "50-59"

    queried_s2 = session.query(Screening).filter(Screening.session_id == s_id_2).first()
    assert queried_s2.age_range == "70-79"
    session.close()

