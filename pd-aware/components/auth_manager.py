"""
Authentication manager for PD-Aware.

Handles user registration, login, role-based access, and session management.
Uses bcrypt for password hashing and Streamlit session state.
"""

import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import bcrypt
import streamlit as st

from database.engine import get_session
from database.models import User, UserRole, AuditLog

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def register_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: UserRole = UserRole.PATIENT,
) -> Tuple[bool, str]:
    """Register a new user."""
    session = get_session()
    try:
        # Check existing
        if session.query(User).filter(User.username == username).first():
            return False, "Username already exists"
        if session.query(User).filter(User.email == email).first():
            return False, "Email already registered"

        # Validate
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if not username.strip() or not email.strip():
            return False, "Username and email are required"

        user = User(
            username=username.strip(),
            email=email.strip().lower(),
            password_hash=hash_password(password),
            full_name=full_name.strip(),
            role=role,
            is_verified=role == UserRole.PATIENT,  # Doctors need verification
        )
        session.add(user)

        # Audit log
        log = AuditLog(
            user_id=user.id,
            action="user_registered",
            resource_type="user",
            resource_id=user.id,
            details={"role": role.value},
        )
        session.add(log)
        session.commit()

        logger.info(f"User registered: {username} ({role.value})")
        return True, "Registration successful"

    except Exception as e:
        session.rollback()
        logger.error(f"Registration failed: {e}")
        return False, f"Registration failed: {str(e)}"
    finally:
        session.close()


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict]]:
    """Authenticate a user and return their info."""
    session = get_session()
    try:
        user = session.query(User).filter(User.username == username).first()

        if not user:
            return False, None

        if not user.is_active:
            return False, None

        if not verify_password(password, user.password_hash):
            # Log failed attempt
            log = AuditLog(
                action="login_failed",
                resource_type="user",
                details={"username": username},
            )
            session.add(log)
            session.commit()
            return False, None

        # Update last login
        user.last_login = datetime.now(timezone.utc)

        # Audit log
        log = AuditLog(
            user_id=user.id,
            action="login_success",
            resource_type="user",
            resource_id=user.id,
        )
        session.add(log)
        session.commit()

        user_info = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "is_verified": user.is_verified,
        }

        return True, user_info

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False, None
    finally:
        session.close()


def init_session_state():
    """Initialize authentication-related session state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def login_form():
    """Render login/register form in sidebar."""
    init_session_state()

    if st.session_state.authenticated:
        user = st.session_state.user
        st.sidebar.markdown(f"**{user['full_name']}**")
        st.sidebar.markdown(f"Role: `{user['role']}`")
        if st.sidebar.button("Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
        return True

    with st.sidebar.expander("Account Access", expanded=False):
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login")

                if submitted and username and password:
                    success, user_info = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = user_info
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email", key="reg_email")
                new_name = st.text_input("Full Name", key="reg_name")
                new_password = st.text_input("Password", type="password", key="reg_password")
                new_role = st.selectbox(
                    "Account Type",
                    options=["Patient", "Doctor"],
                    key="reg_role",
                )
                submitted = st.form_submit_button("Register")

                if submitted and new_username and new_email and new_password:
                    role = UserRole.DOCTOR if new_role == "Doctor" else UserRole.PATIENT
                    success, msg = register_user(
                        new_username, new_email, new_password, new_name, role,
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

    return st.session_state.authenticated


def require_role(required_role: UserRole) -> bool:
    """Check if the current user has the required role."""
    if not st.session_state.get("authenticated"):
        st.warning("Authentication required: Please log in to access this feature.")
        return False

    user = st.session_state.get("user", {})
    user_role = user.get("role", "")

    if user_role == "admin":
        return True  # Admin can access everything

    if user_role != required_role.value:
        st.error(f"Access restricted: This feature requires {required_role.value} access.")
        return False

    if required_role == UserRole.DOCTOR and not user.get("is_verified"):
        st.warning(
            "Account pending verification: Your clinician account is awaiting "
            "administrator review."
        )
        return False

    return True


def get_current_user() -> Optional[Dict]:
    """Get the current authenticated user."""
    if st.session_state.get("authenticated"):
        return st.session_state.get("user")
    return None
