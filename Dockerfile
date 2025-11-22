# Multi-stage Dockerfile for Gen Z Clinical Workflow Assistant
# Production-ready with HIPAA compliance and security hardening

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Builder
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim as builder

LABEL maintainer="GenZ Healthcare Team <support@example.com>"
LABEL version="2.0.0"
LABEL description="Clinical Workflow Assistant with RAG, Guardrails, and HITL"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY gen_z_agent/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim

# Security: Run as non-root user
RUN groupadd -r healthcare && useradd -r -g healthcare healthcare

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY gen_z_agent/ /app/gen_z_agent/
COPY docs/ /app/docs/
COPY LICENSE /app/
COPY README.md /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/gen_z_agent/healthcare/{fhir_data,clinical_reports,audit_logs,vector_db,reviews,fine_tuned_models,clinical_trials} && \
    chown -R healthcare:healthcare /app

# Security: Set restrictive permissions
RUN chmod 700 /app/gen_z_agent/healthcare && \
    chmod -R 600 /app/gen_z_agent/healthcare/*

# Switch to non-root user
USER healthcare

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from gen_z_agent.clinical_rag import get_rag_system; rag = get_rag_system(); print('Healthy')" || exit 1

# Expose port (if running as API server)
EXPOSE 8000

# Default command
CMD ["python", "-m", "gen_z_agent.healthcare_agents", "patient_risk_assessment", "--help"]
