"""SQLAlchemy ORM models for PD-Aware."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime,
    JSON, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


def gen_uuid():
    return str(uuid.uuid4())


# ─── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class ScreeningStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ModelStatus(str, enum.Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PUBLISHED = "published"
    RETIRED = "retired"


# ─── User ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.PATIENT, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)

    # Relationships
    screenings = relationship("Screening", back_populates="user", cascade="all, delete-orphan")
    clinician_notes = relationship("ClinicianNote", back_populates="doctor",
                                   foreign_keys="ClinicianNote.doctor_id")
    audit_logs = relationship("AuditLog", back_populates="user")

    def __repr__(self):
        return f"<User {self.username} ({self.role.value})>"


# ─── Consent ──────────────────────────────────────────────────────────────────

class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    session_id = Column(String, nullable=False, index=True)
    consent_type = Column(String(50), nullable=False)  # screening, storage, research
    consented = Column(Boolean, nullable=False)
    consent_text = Column(Text, nullable=False)
    ip_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    revoked_at = Column(DateTime, nullable=True)


# ─── Screening Session ───────────────────────────────────────────────────────

class Screening(Base):
    __tablename__ = "screenings"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    session_id = Column(String, nullable=False, index=True)
    status = Column(SAEnum(ScreeningStatus), default=ScreeningStatus.IN_PROGRESS)

    # Patient info
    age_range = Column(String(20), nullable=True)
    symptom_duration = Column(String(50), nullable=True)
    existing_diagnosis = Column(String(50), nullable=True)
    family_history = Column(String(50), nullable=True)

    # Symptom data
    natural_language_input = Column(Text, nullable=True)
    extracted_symptoms = Column(JSON, nullable=True)  # NLP extracted
    user_reviewed_symptoms = Column(JSON, nullable=True)  # After user review
    questionnaire_responses = Column(JSON, nullable=True)

    # Voice data reference (if uploaded)
    voice_file_path = Column(String(500), nullable=True)
    voice_features = Column(JSON, nullable=True)

    # Consent
    consent_given = Column(Boolean, default=False)
    consent_timestamp = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="screenings")
    prediction = relationship("Prediction", back_populates="screening",
                              uselist=False, cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="screening", cascade="all, delete-orphan")
    clinician_notes = relationship("ClinicianNote", back_populates="screening",
                                   cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Screening {self.id[:8]}... ({self.status.value})>"


# ─── Prediction ───────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=gen_uuid)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=False, unique=True)

    # Model info
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # e.g., "random_forest"
    feature_schema_version = Column(String(50), nullable=False)
    preprocessing_version = Column(String(50), nullable=False)

    # Prediction output
    prediction_label = Column(Integer, nullable=False)  # 0 or 1
    prediction_score = Column(Float, nullable=True)  # probability if calibrated
    prediction_category = Column(String(50), nullable=True)  # human-readable

    # Features used
    input_features = Column(JSON, nullable=False)
    feature_names = Column(JSON, nullable=False)

    # Explanations
    shap_values = Column(JSON, nullable=True)
    feature_importances = Column(JSON, nullable=True)

    # Metadata
    inference_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    screening = relationship("Screening", back_populates="prediction")

    def __repr__(self):
        return f"<Prediction {self.id[:8]}... label={self.prediction_label}>"


# ─── Report ───────────────────────────────────────────────────────────────────

class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=gen_uuid)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=False)
    report_type = Column(String(50), default="screening_summary")
    file_path = Column(String(500), nullable=True)
    include_symptoms = Column(Boolean, default=True)
    include_prediction = Column(Boolean, default=True)
    include_explanation = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    screening = relationship("Screening", back_populates="reports")


# ─── Clinician Note ───────────────────────────────────────────────────────────

class ClinicianNote(Base):
    __tablename__ = "clinician_notes"

    id = Column(String, primary_key=True, default=gen_uuid)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=False)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False)
    note_text = Column(Text, nullable=False)
    note_type = Column(String(50), default="observation")  # observation, recommendation, follow_up
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    screening = relationship("Screening", back_populates="clinician_notes")
    doctor = relationship("User", back_populates="clinician_notes", foreign_keys=[doctor_id])


# ─── Model Registry ──────────────────────────────────────────────────────────

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String, primary_key=True, default=gen_uuid)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)
    status = Column(SAEnum(ModelStatus), default=ModelStatus.DRAFT)

    # Training info
    dataset_name = Column(String(200), nullable=True)
    dataset_provenance = Column(Text, nullable=True)
    training_date = Column(DateTime, nullable=True)
    feature_schema = Column(JSON, nullable=True)  # List of feature names
    preprocessing_version = Column(String(50), nullable=True)

    # Evaluation metrics
    metrics = Column(JSON, nullable=True)  # accuracy, f1, auc, etc.
    validation_protocol = Column(String(100), nullable=True)

    # File paths
    model_file_path = Column(String(500), nullable=True)
    scaler_file_path = Column(String(500), nullable=True)
    metadata_file_path = Column(String(500), nullable=True)

    # Documentation
    intended_use = Column(Text, nullable=True)
    known_limitations = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    published_at = Column(DateTime, nullable=True)
    retired_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<ModelRegistry {self.model_name} {self.version} ({self.status.value})>"


# ─── Audit Log ────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog {self.action} at {self.created_at}>"
